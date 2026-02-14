import json
from pathlib import Path
import cv2 as cv

"""
Filtered YOLO review tool for correcting inconsistent HALF blink annotations.
This version EDITS THE ORIGINAL REVIEW JSONS IN-PLACE (overwrites them).
NO backups are made.

Modes:
- HALF_ONLY: show ONLY frames that contain at least one half-blink detection (class_id==1)
- TWO_BBOX:  show ONLY frames that contain exactly 2 detections (any classes)

Controls:
    Space: play/pause (plays through FILTERED frames only)
    A/D: step back/forward 1 filtered frame
    J/L: step back/forward 30 filtered frames
    TAB: cycle selected box
    0/1/2: set class of selected box
    B: edit selected box (or add if none)
    N: add new box
    Y: delete selected box
    S: save changes
    Q: next video
    X: stop all
"""

# ---------------- CONFIG ----------------
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()

# Folder with reviewed JSONs (these will be overwritten!)
JSON_DIR = WORKSPACE_ROOT / "yolo_approach" / "yolo_labeling" / "labels_extra_blink_only "

CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]

# Choose one:
MODE = "HALF_ONLY"   # "HALF_ONLY" or "TWO_BBOX"
HALF_CLASS_ID = 1

WINDOW_NAME = (
    "Filtered Review (Space play/pause, A/D step, J/L ±30, TAB select box, "
    "0/1/2 set class, B edit/add, N add, Y delete, S save, Q next video, X stop all)"
)
FONT = cv.FONT_HERSHEY_SIMPLEX

AUTOSAVE_EVERY_CHANGES = 200
speed = 1.0
# --------------------------------------


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def draw_label(img, text, x, y):
    cv.putText(img, text, (x, y), FONT, 0.7, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (x, y), FONT, 0.7, (255, 255, 255), 1, cv.LINE_AA)

def step_to(cap: cv.VideoCapture, frame_idx: int):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

def get_frame(cap: cv.VideoCapture, frame_idx: int):
    step_to(cap, frame_idx)
    ok, fr = cap.read()
    return ok, fr

def select_roi_scaled_xyxy(frame, max_w=2500, max_h=1500):
    H, W = frame.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)

    if scale < 1.0:
        small = cv.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv.INTER_AREA)
    else:
        small = frame

    win = "Select BBox (scaled) (ENTER=OK, ESC=cancel)"
    roi = cv.selectROI(win, small, fromCenter=False, showCrosshair=True)
    cv.destroyWindow(win)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None

    inv = 1.0 / scale
    x1 = int(x * inv)
    y1 = int(y * inv)
    x2 = int((x + w) * inv)
    y2 = int((y + h) * inv)

    x1 = clamp(x1, 0, W - 1)
    x2 = clamp(x2, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    y2 = clamp(y2, 0, H - 1)

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    return [x1, y1, x2, y2]

def mark_progress(data: dict, video_frame_idx: int, filtered_pos: int, filtered_len: int, mode: str):
    data.setdefault("review_meta", {})
    data["review_meta"]["reviewed"] = True
    data["review_meta"]["mode"] = mode
    data["review_meta"]["last_frame"] = int(video_frame_idx)
    data["review_meta"]["filtered_pos"] = int(filtered_pos)
    data["review_meta"]["filtered_len"] = int(filtered_len)

def build_filtered_indices(frames: list, mode: str) -> list[int]:
    idxs = []
    for i, rec in enumerate(frames):
        dets = rec.get("detections", []) or []
        if mode == "HALF_ONLY":
            if any(int(d.get("class_id", -1)) == HALF_CLASS_ID for d in dets):
                idxs.append(i)
        elif mode == "TWO_BBOX":
            if len(dets) == 2:
                idxs.append(i)
        else:
            raise ValueError(f"Unknown MODE: {mode}")
    return idxs

def color_for_class(cid: int):
    if cid == 0: return (255, 0, 0)     # eye
    if cid == 1: return (0, 255, 0)     # half
    if cid == 2: return (0, 0, 255)     # full
    return (200, 200, 200)

def main():
    stop_all = False
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in: {JSON_DIR}")
        return

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    for jf in json_files:
        if stop_all:
            break

        data = load_json(jf)

        video_path = Path(data["meta"]["video"])
        if not video_path.exists():
            print(f"[WARN] Missing video for {jf.name}: {video_path}")
            continue

        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            continue

        total_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv.CAP_PROP_FPS) or data["meta"].get("fps") or 25.0)

        frames = data.get("frames", [])
        total_json = len(frames)
        total = min(total_video, total_json) if total_video > 0 else total_json
        if total <= 0:
            print(f"[WARN] No frames to review for {jf.name}")
            cap.release()
            continue

        frames = frames[:total]

        # Build filtered list from current data (so edits affect what qualifies)
        filtered = build_filtered_indices(frames, MODE)
        if not filtered:
            print(f"[SKIP] {jf.name}: no frames match filter ({MODE})")
            cap.release()
            continue

        meta = data.get("review_meta", {})
        start_pos = 0
        if meta.get("mode") == MODE and "filtered_pos" in meta:
            start_pos = int(meta.get("filtered_pos", 0))
        start_pos = clamp(start_pos, 0, len(filtered) - 1)

        pos = start_pos
        cur = filtered[pos]

        selected_idx = 0
        last_cur = None

        print(f"\n[OPEN] {jf.name} ({video_path.name}) total_frames={total} filtered_frames={len(filtered)} mode={MODE}")
        print(f"  [START] filtered_pos {pos+1}/{len(filtered)} (frame {cur})")

        playing = True
        dirty = False
        changes = 0

        while True:
            ok, frame = get_frame(cap, cur)
            if not ok:
                break

            if last_cur is None or cur != last_cur:
                selected_idx = 0
                last_cur = cur

            rec = frames[cur]
            dets = rec.get("detections", []) or []

            if dets:
                selected_idx = clamp(selected_idx, 0, len(dets) - 1)
            else:
                selected_idx = 0

            shown = frame.copy()

            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d["bbox"]
                cid = int(d.get("class_id", -1))
                cname = d.get("class_name") or (CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else "unclassified")

                is_sel = (i == selected_idx)
                thickness = 4 if is_sel else 2
                color = (0, 255, 255) if is_sel else color_for_class(cid)

                cv.rectangle(shown, (x1, y1), (x2, y2), color, thickness)
                draw_label(shown, f"{cname} #{i}" + (" [SEL]" if is_sel else ""), x1, max(25, y1 - 8))

            draw_label(
                shown,
                f"{video_path.name} | mode={MODE} | filtered {pos+1}/{len(filtered)} | frame {cur+1}/{total} | play={playing} dirty={dirty} changes={changes} speed={speed:.2f}x",
                10, 30
            )
            if dets:
                draw_label(shown, f"selected bbox: #{selected_idx+1}/{len(dets)}", 10, 60)
            else:
                draw_label(shown, "no detections in this frame", 10, 60)

            cv.imshow(WINDOW_NAME, shown)

            # autosave
            if dirty and changes > 0 and (changes % AUTOSAVE_EVERY_CHANGES == 0):
                mark_progress(data, cur, pos, len(filtered), MODE)
                save_json(jf, data)
                dirty = False
                print(f"[AUTOSAVE] {jf.name} (changes={changes}, filtered_pos={pos}, frame={cur})")

            fps2 = fps if fps and fps > 1e-6 else 25.0
            delay = int(1000 / (fps2 * speed)) if playing else 0
            delay = max(1, delay) if playing else 0
            key = cv.waitKey(delay) & 0xFFFFFFFF

            # STOP EVERYTHING
            if key in (ord("x"), ord("X")):
                if dirty:
                    mark_progress(data, cur, pos, len(filtered), MODE)
                    save_json(jf, data)
                    print(f"[SAVE] {jf.name} (stop all)")
                stop_all = True
                break

            # next video
            if key in (ord("q"), 27):
                if dirty:
                    mark_progress(data, cur, pos, len(filtered), MODE)
                    save_json(jf, data)
                    print(f"[SAVE] {jf.name} (next video)")
                break

            # play/pause
            if key == ord(" "):
                playing = not playing
                continue

            # manual save
            if key in (ord("s"), ord("S")):
                mark_progress(data, cur, pos, len(filtered), MODE)
                save_json(jf, data)
                dirty = False
                print(f"[SAVE] {jf.name} (manual)")
                continue

            # TAB cycle selected
            if key == 9:
                if dets:
                    selected_idx = (selected_idx + 1) % len(dets)
                continue

            # stepping within filtered list
            if key in (ord("a"), ord("A")):
                playing = False
                pos = clamp(pos - 1, 0, len(filtered) - 1)
                cur = filtered[pos]
                continue
            if key in (ord("d"), ord("D")):
                playing = False
                pos = clamp(pos + 1, 0, len(filtered) - 1)
                cur = filtered[pos]
                continue
            if key in (ord("j"), ord("J")):
                playing = False
                pos = clamp(pos - 30, 0, len(filtered) - 1)
                cur = filtered[pos]
                continue
            if key in (ord("l"), ord("L")):
                playing = False
                pos = clamp(pos + 30, 0, len(filtered) - 1)
                cur = filtered[pos]
                continue

            # auto-advance while playing (within filtered frames)
            if playing and key == 0xFFFFFFFF:
                pos += 1
                if pos >= len(filtered):
                    break
                cur = filtered[pos]
                continue

            # relabel selected bbox
            if key in (ord("0"), ord("1"), ord("2")):
                if not dets:
                    continue
                cid = int(chr(key))
                cname = CLASS_NAMES[cid]
                dets[selected_idx]["class_id"] = cid
                dets[selected_idx]["class_name"] = cname
                dets[selected_idx]["checked"] = True
                rec["detections"] = dets
                rec["checked"] = True
                dirty = True
                changes += 1
                continue

            # delete selected bbox
            if key in (ord("y"), ord("Y")):
                if dets:
                    dets.pop(selected_idx)
                    if dets:
                        selected_idx = clamp(selected_idx, 0, len(dets) - 1)
                        rec["detections"] = dets
                    else:
                        selected_idx = 0
                        rec["detections"] = []
                    rec["checked"] = True
                    dirty = True
                    changes += 1
                continue

            # edit selected bbox (or add if none)
            if key in (ord("b"), ord("B")):
                playing = False
                xyxy = select_roi_scaled_xyxy(frame, max_w=1600, max_h=900)
                if xyxy is None:
                    continue

                if dets:
                    old_cid = int(dets[selected_idx].get("class_id", 0))
                    old_name = dets[selected_idx].get("class_name", CLASS_NAMES[old_cid])
                else:
                    old_cid, old_name = 0, CLASS_NAMES[0]

                edited = {"bbox": xyxy, "class_id": old_cid, "class_name": old_name, "checked": True}

                if dets:
                    dets[selected_idx] = edited
                else:
                    dets = [edited]
                    selected_idx = 0

                rec["detections"] = dets
                rec["checked"] = True
                dirty = True
                changes += 1
                continue

            # N: add bbox
            if key in (ord("n"), ord("N")):
                playing = False
                xyxy = select_roi_scaled_xyxy(frame, max_w=1600, max_h=900)
                if xyxy is None:
                    continue

                new_det = {"bbox": xyxy, "class_id": 0, "class_name": CLASS_NAMES[0], "checked": True}
                dets.append(new_det)

                rec["detections"] = dets
                rec["checked"] = True
                dirty = True
                changes += 1
                selected_idx = len(dets) - 1
                continue

        cap.release()

        if dirty:
            mark_progress(data, cur, pos, len(filtered), MODE)
            save_json(jf, data)
            print(f"[SAVE] {jf.name} (end of video)")

    cv.destroyAllWindows()
    print("[DONE] Finished filtered review.")


if __name__ == "__main__":
    main()
