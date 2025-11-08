# review_label_and_export_frames.py
import cv2 as cv
import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# ============= CONFIG =============
BASE = Path(__file__).parent.parent.resolve()
print(f"[INFO] Base path: {BASE}")
INPUT_VIDEOS_DIR = BASE / "create_datasets" / "original_videos_annotations" / "videos"
OUTPUT_ROOT      = BASE / "create_datasets" / "datasets" / "classified_frames"
STATE_DIR        = OUTPUT_ROOT / "_labels"  # autosave labels here (JSON per video)

# Class folders will be: OUTPUT_ROOT / eye, half_blink, full_blink
CLASS_FOLDERS = {0: "eye", 1: "half_blink", 2: "full_blink"}

# Video extensions to consider
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Playback speed: half-speed
SPEED_FACTOR = 0.5  # 0.5 => half speed (slower), 1.0 => normal, 0.25 => quarter speed

# UI
WINDOW = "Frame Class Labeller (space=pause/play, 0/1/2=class, ←/→=step, A/D=±5, J/L=±30, Enter=finish, Q=quit)"
FONT = cv.FONT_HERSHEY_SIMPLEX

# Export image format
IMG_EXT = ".png"  # change to ".png" if you want PNG
# ==================================


def list_videos(root: Path) -> List[Path]:
    vids = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            vids.append(p)
    return vids


def draw_overlay(frame: np.ndarray, text_lines: List[str]) -> np.ndarray:
    img = frame.copy()
    y = 28
    for t in text_lines:
        cv.putText(img, t, (10, y), FONT, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(img, t, (10, y), FONT, 0.7, (255, 255, 255), 1, cv.LINE_AA)
        y += 26
    return img


def clamp_frame_index(idx: int, total: int) -> int:
    return max(0, min(idx, max(0, total - 1)))


def load_labels(state_json: Path, total_frames: int) -> Dict[int, int]:
    """Load labels {frame_index: class_id}; default class is 0 for unlabeled frames."""
    if not state_json.exists():
        return {}
    try:
        obj = json.loads(state_json.read_text(encoding="utf-8"))
        lab = {int(k): int(v) for k, v in obj.get("labels", {}).items()}
        # clamp keys
        return {clamp_frame_index(k, total_frames): int(v) for k, v in lab.items()}
    except Exception:
        return {}


def save_labels(state_json: Path, labels: Dict[int, int], meta: dict) -> None:
    state_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labels": labels, "meta": meta}
    state_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_frames(video_path: Path, labels: Dict[int, int]) -> None:
    """Re-open the video and export every frame to its class folder."""
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot re-open for export: {video_path}")
        return

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    w  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    stem = video_path.stem

    # Prepare output dirs
    out_dirs = {}
    for cid, cname in CLASS_FOLDERS.items():
        d = OUTPUT_ROOT / cname
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[cid] = d

    print(f"[EXPORT] {video_path.name} → {OUTPUT_ROOT}  ({w}x{h}, {total} frames)")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cls_id = int(labels.get(idx, 0))
        dst_dir = out_dirs.get(cls_id, out_dirs[0])
        out_file = dst_dir / f"{stem}_{idx:06d}{IMG_EXT}"

        # write image robustly
        ok2, buf = cv.imencode(IMG_EXT, frame)
        if ok2:
            buf.tofile(os.fspath(out_file))
        else:
            print(f"[WARN] Failed to write frame {idx} -> {out_file}")

        if idx % 500 == 0 and idx > 0:
            print(f"  ... {idx}/{total} frames exported")

        idx += 1

    cap.release()
    print(f"[DONE] Exported {idx} frames for {video_path.name}")


def review_video(video_path: Path) -> None:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open: {video_path}")
        return

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    fps   = float(cap.get(cv.CAP_PROP_FPS) or 25.0)
    delay_ms = max(1, int(1000 / (fps * SPEED_FACTOR)))  # half speed

    # prepare state json (resume support)
    state_json = STATE_DIR / f"{video_path.stem}.json"
    labels: Dict[int, int] = load_labels(state_json, total)
    meta = {"video": str(video_path), "total_frames": total, "fps": fps}

    # window
    cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)

    # start at saved position if present, else 0
    cur = max(labels.keys(), default=0)
    cur = clamp_frame_index(cur, total)

    playing = True
    finished = False

    def set_pos(frame_index: int):
        nonlocal cur
        cur = clamp_frame_index(frame_index, total)
        cap.set(cv.CAP_PROP_POS_FRAMES, cur)

    set_pos(cur)

    while True:
        if playing:
            ok, frame = cap.read()
            if not ok:
                finished = True
            else:
                cur = int(cap.get(cv.CAP_PROP_POS_FRAMES)) - 1  # current position after read
        else:
            # paused: fetch current frame
            cap.set(cv.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok:
                finished = True

        if finished:
            # End of this video -> export frames
            print(f"[INFO] Finished video: {video_path.name}. Exporting frames...")
            save_labels(state_json, labels, meta)
            export_frames(video_path, labels)
            break

        # current class for overlay
        current_cls = int(labels.get(cur, 0))

        # overlay
        text = [
            f"{video_path.name}  frame {cur+1}/{total}  fps={fps:.2f}  speed={SPEED_FACTOR}x",
            f"current class: {current_cls}    keys: [space] play/pause  [0/1/2] set class",
            f"[←/→] step ±1   [A/D] ±5   [J/L] ±30   [Enter] finish+export   [Q] quit (without export)"
        ]
        shown = draw_overlay(frame, text)
        cv.imshow(WINDOW, shown)

        key = cv.waitKey(delay_ms if playing else 0) & 0xFFFFFFFF

        # No key or continue while playing
        if key == 0xFFFFFFFF and playing:
            continue

        # Key handling

        ENTER = 13

        if key in (ord('q'), 27):  # Q or ESC -> quit current video WITHOUT export
            print(f"[INFO] Quit without export: {video_path.name}")
            save_labels(state_json, labels, meta)
            break
        elif key == ord(' '):  # space: toggle play/pause
            playing = not playing
        elif key in (ENTER,):  # Enter: finish + export now
            print(f"[INFO] Manual finish: {video_path.name}. Exporting frames...")
            save_labels(state_json, labels, meta)
            export_frames(video_path, labels)
            break
        elif key in (ord('0'),):
            labels[cur] = 0
            save_labels(state_json, labels, meta)
        elif key in (ord('1'),):
            labels[cur] = 1
            save_labels(state_json, labels, meta)
        elif key in (ord('2'),):
            labels[cur] = 2
            save_labels(state_json, labels, meta)

        elif key in (ord('a'), ord('A')):
            playing = False
            set_pos(cur - 1)
        elif key in (ord('d'), ord('D')):
            playing = False
            set_pos(cur + 1)
        elif key in (ord('j'), ord('J')):
            playing = False
            set_pos(cur - 30)
        elif key in (ord('l'), ord('L')):
            playing = False
            set_pos(cur + 30)
        # else: ignore other keys

    cap.release()
    cv.destroyAllWindows()


def main():
    videos = list_videos(INPUT_VIDEOS_DIR)
    if not videos:
        print(f"[ERROR] No videos found under: {INPUT_VIDEOS_DIR}")
        return

    print(f"[INFO] Found {len(videos)} videos.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    for i, v in enumerate(videos):
        #if i == 13: # 13 total
        print(f"\n=== Reviewing: {v} ===")
        review_video(v)

    print("\n[ALL DONE] Processed all videos.")


if __name__ == "__main__":
    main()
