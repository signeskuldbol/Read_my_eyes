

# extract_frames_by_subject.py
# Edit INPUT_DIR, OUTPUT_DIR, SAVE_SCHEME. Then run:  python extract_frames_by_subject.py


from collections import defaultdict
import re
import cv2
from pathlib import Path

# ==== EDIT THESE ====
WORKSPACE_PATH = Path(__file__).parent
INPUT_DIR = WORKSPACE_PATH/ "video_data"/ "blink"
OUTPUT_DIR = Path(r"C:/frames_out")  
SAVE_SCHEME = "subject_video"  # "subject_video" | "subject_only" | "label_subject_video"
SAVE_EVERY_N = 1               # 1 = every frame, 5 = every 5th frame
JPG_QUALITY = 90
# =====================

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}

# FIXED: match "_S<digits>" followed by dot/underscore/hyphen or end (no word-boundary)
# Also accept lowercase s and both '_' and '-' as separators.
SUBJECT_RE = re.compile(r"[_-][sS](?P<sid>\d+)(?=\.|_|-|$)")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_videos(root: Path):
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

def extract_subject(name: str) -> str | None:
    m = SUBJECT_RE.search(name)
    return f"S{m.group('sid')}" if m else None

def write_jpg_safe(img, out_path: Path, quality: int = 90) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return False
    try:
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"[ERROR] Exception writing {out_path}: {e}")
        return False

def main():
    in_root = INPUT_DIR.resolve()
    out_root = OUTPUT_DIR.resolve()
    ensure_dir(out_root)

    videos = find_videos(in_root)
    if not videos:
        print(f"[ERROR] No videos found in {in_root}")
        return

    print(f"Found {len(videos)} video(s). Writing frames into subject folders under: {out_root}")

    # continuous counter per subject across all that subject's videos
    subject_counters = defaultdict(int)

    for vp in videos:
        subject = extract_subject(vp.name)
        if subject is None:
            print(f"[WARN] Could not find subject in filename, skipping: {vp.name}")
            continue

        out_dir = out_root / subject
        ensure_dir(out_dir)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] Could not open: {vp}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"\n{vp.name}: {w}x{h} @ {fps:.2f} fps | frames: {nframes if nframes>=0 else 'unknown'} -> {out_dir}")

        saved = 0
        local_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if local_idx % max(1, SAVE_EVERY_N) == 0:
                global_idx = subject_counters[subject]
                fname = f"frame_{global_idx:06d}.jpg"
                out_path = out_dir / fname
                ok = write_jpg_safe(frame, out_path, JPG_QUALITY)
                if not ok:
                    print(f"[WARN] Failed to write: {out_path}")
                else:
                    subject_counters[subject] += 1
                    saved += 1

            local_idx += 1

        cap.release()
        print(f"Saved {saved} frame(s) from {vp.name}")

    print("\nDone. Per-subject frame counts:")
    for subj, count in sorted(subject_counters.items()):
        print(f"  {subj}: {count} frames")

if __name__ == "__main__":
    main()
