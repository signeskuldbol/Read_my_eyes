import json
from pathlib import Path
import cv2 as cv

# NOTE: soome of the videos are fully checked others only partially. 
# 2_2 has a horse in the background obs!!!

# ---------------- CONFIG ----------------
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()
JSON_IN_DIR  = WORKSPACE_ROOT / "yolo_approach" / "yolo_labels_first_priority"      #"labels_yolo_predicted"
JSON_OUT_DIR = WORKSPACE_ROOT / "yolo_approach" / "labels_reviewed"
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]

WINDOW_NAME = (
    "Review (Space=play/pause, A/D step, J/L ±30, TAB select box, "
    "0/1/2 set class, B edit selected box, N add box, Y delete selected box, "
    "S save, Q next video, X stop all)"
)
FONT = cv.FONT_HERSHEY_SIMPLEX

AUTOSAVE_EVERY_CHANGES = 300
speed = 0.5  # 0.5 half speed, 1.0 normal, 2.0 double
# --------------------------------------
def select_roi_scaled_xyxy(frame, max_w=2500, max_h=1500):
    """
    Select ROI on a scaled-down view so it fits on screen.
    Returns bbox as [x1, y1, x2, y2] in ORIGINAL frame coordinates.
    """
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

    # Clamp to image bounds
    x1 = clamp(x1, 0, W - 1)
    x2 = clamp(x2, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    y2 = clamp(y2, 0, H - 1)

    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_label(img, text, x, y):
    cv.putText(img, text, (x, y), FONT, 0.7, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (x, y), FONT, 0.7, (255, 255, 255), 1, cv.LINE_AA)


def step_to(cap: cv.VideoCapture, frame_idx: int):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)


def get_frame(cap: cv.VideoCapture, frame_idx: int):
    step_to(cap, frame_idx)
    ok, fr = cap.read()
    return ok, fr


def load_for_resume(in_json: Path, out_json: Path) -> tuple[dict, Path]:
    """If reviewed exists, resume from it; else start from predicted."""
    if out_json.exists():
        return load_json(out_json), out_json
    return load_json(in_json), in_json


def mark_last_frame(data: dict, cur: int):
    data.setdefault("review_meta", {})
    data["review_meta"]["last_frame"] = int(cur)
    data["review_meta"]["reviewed"] = True


def main():
    stop_all = False
    json_files = sorted(JSON_IN_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in: {JSON_IN_DIR}")
        return

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    for jf in json_files:
        if stop_all:
            break

        out_json = JSON_OUT_DIR / jf.name.replace(".json", "_reviewed.json")

        # Resume: if reviewed exists, load it; otherwise load predicted
        data, loaded_from = load_for_resume(jf, out_json)

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

        # Start from last_frame if present
        cur = int(data.get("review_meta", {}).get("last_frame", 0))
        cur = clamp(cur, 0, total - 1)
        cap.set(cv.CAP_PROP_POS_FRAMES, cur)

        selected_idx = 0  # selection per-frame (reset when frame changes)
        last_cur = None

        print(f"\n[OPEN] {jf.name} ({video_path.name}) frames={total}")
        if loaded_from == out_json:
            print(f"  [RESUME] {out_json.name} starting at frame {cur}")
        else:
            print(f"  [NEW] {jf.name} starting at frame {cur}")

        playing = True
        dirty = False
        changes = 0

        while True:
            if playing:
                ok, frame = cap.read()
                if not ok:
                    break
                cur = int(cap.get(cv.CAP_PROP_POS_FRAMES)) - 1
            else:
                ok, frame = get_frame(cap, cur)
                if not ok:
                    break

            if cur >= total:
                break

            # if frame changed, reset selection if needed
            if last_cur is None or cur != last_cur:
                selected_idx = 0
                last_cur = cur

            rec = frames[cur]

            # Mark visited frame as checked
            if not rec.get("checked", False):
                rec["checked"] = True
                dirty = True
                changes += 1

            dets = rec.get("detections", [])
            # keep selected_idx valid
            if dets:
                selected_idx = clamp(selected_idx, 0, len(dets) - 1)
            else:
                selected_idx = 0

            shown = frame.copy()

            # draw detections (highlight selected)
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d["bbox"]
                cid = int(d.get("class_id", -1))
                cname = d.get("class_name") or (CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else "unclassified")

                is_sel = (i == selected_idx)
                thickness = 4 if is_sel else 2
                color = (255, 255, 0) if rec.get("checked", False) else (0, 255, 0)
                # make selected box a bit more obvious
                if is_sel:
                    color = (0, 255, 255)

                cv.rectangle(shown, (x1, y1), (x2, y2), color, thickness)
                draw_label(shown, f"{cname}  #{i}" + (" [SEL]" if is_sel else ""), x1, max(25, y1 - 8))

            draw_label(
                shown,
                f"{video_path.name} frame {cur+1}/{total} play={playing} dirty={dirty} changes={changes} speed={speed:.2f}x",
                10, 30
            )
            if dets:
                draw_label(shown, f"selected bbox: #{selected_idx+1}/{len(dets)}", 10, 60)
            else:
                draw_label(shown, "no detections in this frame", 10, 60)

            cv.imshow(WINDOW_NAME, shown)

            # autosave occasionally
            if dirty and changes > 0 and (changes % AUTOSAVE_EVERY_CHANGES == 0):
                mark_last_frame(data, cur)
                save_json(out_json, data)
                dirty = False
                print(f"[AUTOSAVE] {out_json.name} (changes={changes}, last_frame={cur})")

            fps2 = fps if fps and fps > 1e-6 else 25.0
            delay = int(1000 / (fps2 * speed)) if playing else 0
            delay = max(1, delay) if playing else 0
            key = cv.waitKey(delay) & 0xFFFFFFFF

            # STOP EVERYTHING (X)
            if key in (ord("x"), ord("X")):
                if dirty:
                    mark_last_frame(data, cur)
                    save_json(out_json, data)
                    print(f"[SAVE] {out_json.name} (stop all)")
                stop_all = True
                break

            # quit current video (q or esc) -> save and move to next
            if key in (ord("q"), 27):
                if dirty:
                    mark_last_frame(data, cur)
                    save_json(out_json, data)
                    print(f"[SAVE] {out_json.name} (next video)")
                break

            # play/pause
            if key == ord(" "):
                playing = not playing
                continue

            # manual save
            if key in (ord("s"), ord("S")):
                mark_last_frame(data, cur)
                save_json(out_json, data)
                dirty = False
                print(f"[SAVE] {out_json.name} (manual)")
                continue

            # TAB: cycle selected bbox
            if key == 9:  # TAB
                if dets:
                    selected_idx = (selected_idx + 1) % len(dets)
                continue

            # step controls
            if key in (ord("a"), ord("A")):
                playing = False
                cur = clamp(cur - 1, 0, total - 1)
                continue
            if key in (ord("d"), ord("D")):
                playing = False
                cur = clamp(cur + 1, 0, total - 1)
                continue
            if key in (ord("j"), ord("J")):
                playing = False
                cur = clamp(cur - 30, 0, total - 1)
                continue
            if key in (ord("l"), ord("L")):
                playing = False
                cur = clamp(cur + 30, 0, total - 1)
                continue

            # relabel (apply to selected bbox only)
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

            # Y: delete selected bbox
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

            # B: edit selected bbox (or add if none)
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

                edited = {
                    "bbox": xyxy,  # SAME FORMAT: [x1, y1, x2, y2]
                    "class_id": old_cid,
                    "class_name": old_name,
                    "checked": True
                }

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


            # N: add bbox (append)
            if key in (ord("n"), ord("N")):
                playing = False
                xyxy = select_roi_scaled_xyxy(frame, max_w=1600, max_h=900)
                if xyxy is None:
                    continue

                new_det = {
                    "bbox": xyxy,  # SAME FORMAT: [x1, y1, x2, y2]
                    "class_id": 0,
                    "class_name": CLASS_NAMES[0],
                    "checked": True
                }
                dets.append(new_det)

                rec["detections"] = dets
                rec["checked"] = True
                dirty = True
                changes += 1
                selected_idx = len(dets) - 1
                continue



        cap.release()

        if dirty:
            mark_last_frame(data, cur)
            save_json(out_json, data)
            print(f"[SAVE] {out_json.name} (end of video)")

    cv.destroyAllWindows()
    print("[DONE] Finished reviewing all JSONs.")


if __name__ == "__main__":
    main()
