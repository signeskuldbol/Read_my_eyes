import json, cv2, os, numpy as np
from pathlib import Path

# ------------------------------------------------------
Workspace_dir = Path(__file__).parent.parent.resolve()
print(f"Workspace directory: {Workspace_dir}")

DATASET_ROOT = Workspace_dir / "create_datasets" / "datasets" / "frames_sorted_no_pad_no_crop"
output_dir   = Workspace_dir / "yolo_approach" / "annotations"
output_dir.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]
FOLDER_TO_CLASS = {"background": "eye", "AU47": "eye_half_blink", "AU145": "eye_full_blink"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW = "Annotate Eyes.  z=undo  0=none  w=wrong  1/2/3=override  c=clear-override  n/Enter=next  q/Esc=quit"
MAX_SIZE = 1600  # pixels

def class_id(name): return CLASS_NAMES.index(name) if name in CLASS_NAMES else 0
def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---- Unicode-safe image read ----
def robust_imread(path: Path):
    if not path.exists() or not path.is_file():
        return None
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def iter_images(root: Path):
    for split in ["train", "val", "test"]:
        sdir = root / split
        if not sdir.exists(): 
            continue
        for folder in FOLDER_TO_CLASS.keys():
            d = sdir / folder
            if not d.exists(): 
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTS:
                    yield split, folder, p

def ensure_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            try:
                first = json.loads(f.readline())
                if first.get("type") == "dataset":
                    return
            except Exception:
                pass
    header = {
        "type": "dataset",
        "task": "detect",
        "name": "HorseEyes",
        "description": "Horse eye/blink dataset",
        "class_names": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "version": 0
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")

def append_image(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_done(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            if obj.get("type") == "image":
                done.add(Path(obj["file"]).resolve().as_posix())
    return done

def main():
    root = DATASET_ROOT.resolve()
    ndjson = output_dir / "eyes_merged_autolabel.ndjson"
    ensure_header(ndjson)
    done = load_done(ndjson)
    items = list(iter_images(root))
    if not items:
        print("No images found.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # --- State ---
    cur_boxes = []             # committed boxes [(x,y,w,h)] in ORIGINAL coords
    temp_box_abs = None        # live-drag box (x,y,w,h) in ORIGINAL coords
    none_flag = False
    wrong_flag = False
    current_override_class = None  # None = use folder default; else 0/1/2
    disp_scale = 1.0
    idx = -1
    start_pt = None
    drawing = False
    cur_img = None
    cur_W = 0
    cur_H = 0
    cur_path = None
    cur_split = ""
    cur_folder = ""

    def next_item():
        nonlocal idx, cur_boxes, none_flag, wrong_flag, temp_box_abs
        nonlocal cur_img, cur_W, cur_H, disp_scale, cur_path, cur_split, cur_folder
        cur_boxes.clear()
        temp_box_abs = None
        none_flag = False
        wrong_flag = False
        while True:
            idx += 1
            if idx >= len(items):
                return False
            cur_split, cur_folder, cur_path = items[idx]
            abs_p = cur_path.resolve().as_posix()
            if abs_p in done:
                continue
            img = robust_imread(cur_path)
            if img is None:
                append_image(ndjson, {
                    "type": "image",
                    "file": abs_p,
                    "width": 0,
                    "height": 0,
                    "split": cur_split,
                    "annotations": {"boxes": []},
                    "tags": ["read_failed"]
                })
                done.add(abs_p)
                continue
            cur_img = img
            cur_H, cur_W = img.shape[:2]
            long = max(cur_W, cur_H)
            disp_scale = MAX_SIZE / float(long) if long > MAX_SIZE else 1.0
            return True
        
        

    def save_current():
        if cur_path is None:
            return
        abs_p = cur_path.resolve().as_posix()

        # Determine class id: override wins, otherwise folder mapping
        if current_override_class is not None:
            cid = current_override_class
        else:
            cname = FOLDER_TO_CLASS.get(cur_folder, "eye")
            cid = class_id(cname)

        boxes = []
        if not none_flag:
            for (x, y, w, h) in cur_boxes:
                cx = (x + w / 2) / cur_W
                cy = (y + h / 2) / cur_H
                nw = w / cur_W
                nh = h / cur_H
                boxes.append([cid, clamp(cx, 0, 1), clamp(cy, 0, 1), clamp(nw, 0, 1), clamp(nh, 0, 1)])

        rec = {
            "type": "image",
            "file": abs_p,
            "width": cur_W,
            "height": cur_H,
            "split": cur_split,
            "annotations": {"boxes": boxes}
        }
        if wrong_flag:
            rec["tags"] = ["wrong_folder"]

        append_image(ndjson, rec)
        done.add(abs_p)

    def draw_overlay():
        disp = cv2.resize(cur_img, (int(cur_W * disp_scale), int(cur_H * disp_scale)))

        # Draw committed boxes (green)
        for (x, y, w, h) in cur_boxes:
            p1 = (int(x * disp_scale), int(y * disp_scale))
            p2 = (int((x + w) * disp_scale), int((y + h) * disp_scale))
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)

        # Draw live temp box (blue, thinner)
        if temp_box_abs is not None:
            tx, ty, tw, th = temp_box_abs
            p1 = (int(tx * disp_scale), int(ty * disp_scale))
            p2 = (int((tx + tw) * disp_scale), int((ty + th) * disp_scale))
            cv2.rectangle(disp, p1, p2, (255, 0, 0), 1)

        # Info overlay
        y = 25
        folder_class = FOLDER_TO_CLASS.get(cur_folder)
        info = [
            f"{cur_split}/{cur_folder}  folder-class: {folder_class}",
            f"{idx+1}/{len(items)}  boxes: {len(cur_boxes)}",
            "Drag=draw  z=undo  0=none  w=wrong  1/2/3=override  c=clear-override  n/Enter=next  q/Esc=quit",
        ]
        if current_override_class is not None:
            info.append(f"OVERRIDE ACTIVE → class_id={current_override_class} ({CLASS_NAMES[current_override_class]})")
        for t in info:
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        if none_flag:
            cv2.putText(disp, "NONE set (no boxes saved)", (10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA); y += 24
        if wrong_flag:
            cv2.putText(disp, "WRONG FOLDER", (10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return disp

    def on_mouse(event, x, y, flags, param):
        nonlocal start_pt, drawing, temp_box_abs
        if cur_img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            start_pt = (x, y)
            drawing = True
            temp_box_abs = None

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Live rubber-band preview (compute ORIGINAL coords)
            sx, sy = start_pt
            ex, ey = x, y
            ox0 = int(min(sx, ex) / disp_scale)
            oy0 = int(min(sy, ey) / disp_scale)
            ox1 = int(max(sx, ex) / disp_scale)
            oy1 = int(max(sy, ey) / disp_scale)
            ox0 = clamp(ox0, 0, cur_W - 1)
            oy0 = clamp(oy0, 0, cur_H - 1)
            ox1 = clamp(ox1, 0, cur_W - 1)
            oy1 = clamp(oy1, 0, cur_H - 1)
            w = max(0, ox1 - ox0)
            h = max(0, oy1 - oy0)
            temp_box_abs = (ox0, oy0, w, h) if w > 0 and h > 0 else None

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            sx, sy = start_pt
            ex, ey = x, y
            ox0 = int(min(sx, ex) / disp_scale)
            oy0 = int(min(sy, ey) / disp_scale)
            ox1 = int(max(sx, ex) / disp_scale)
            oy1 = int(max(sy, ey) / disp_scale)
            ox0 = clamp(ox0, 0, cur_W - 1)
            oy0 = clamp(oy0, 0, cur_H - 1)
            ox1 = clamp(ox1, 0, cur_W - 1)
            oy1 = clamp(oy1, 0, cur_H - 1)
            w = max(1, ox1 - ox0)
            h = max(1, oy1 - oy0)
            if w > 1 and h > 1:
                cur_boxes.append((ox0, oy0, w, h))
            drawing = False
            temp_box_abs = None

    cv2.setMouseCallback(WINDOW, on_mouse)

    if not next_item():
        print("Everything already annotated.")
        return

    while True:
        disp = draw_overlay()
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):   # quit
            break
        elif key in (13, ord('\r'), ord('\n'), ord('n')):  # save & next
            if drawing and temp_box_abs is not None:
                cur_boxes.append(temp_box_abs)
                drawing = False
                temp_box_abs = None
            save_current()
            if not next_item():
                print("All images annotated!")
                break
        elif key == ord('z') and cur_boxes:               # undo
            cur_boxes.pop()
        elif key == ord('0'):                             # none toggle
            none_flag = not none_flag
            if none_flag:
                cur_boxes.clear()
                temp_box_abs = None
        elif key == ord('w'):                             # wrong-folder toggle
            wrong_flag = not wrong_flag

        # ---- NEW: per-box class override hotkeys ----
        elif key == ord('1'):
            current_override_class = 0
            print("→ override class set to: eye (0)")
        elif key == ord('2'):
            current_override_class = 1
            print("→ override class set to: eye_half_blink (1)")
        elif key == ord('3'):
            current_override_class = 2
            print("→ override class set to: eye_full_blink (2)")
        elif key == ord('c'):  # clear override
            current_override_class = None
            print("→ override cleared; using folder default again")

    cv2.destroyAllWindows()
    print(f"Saved annotations to {ndjson}")

if __name__ == "__main__":
    main()
