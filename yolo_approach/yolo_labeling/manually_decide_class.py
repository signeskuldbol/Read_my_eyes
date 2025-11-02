import json, cv2, os, numpy as np
from pathlib import Path
from hashlib import md5

"""
This is for going through njson file and manually setting classes for all images. 
"""


# ================== CONFIG ==================
Workspace_dir = Path(__file__).parent.parent.resolve()
print(f"Workspace directory: {Workspace_dir}")

SRC_NDJSON = Workspace_dir / "yolo_approach" / "annotations" / "eyes_3_class.ndjson"
OUT_NDJSON = Workspace_dir / "yolo_approach" / "annotations" / "eyes_3_class_reviewed.ndjson"

# Filter to only one folder (exact name in the path): "background", "AU47", "AU145", or None for all
FILTER_FOLDER = None
FILTER_CLASS_ID = 1 # to check through one class to make it consistent

# Skip items that already have the 'reviewed' tag (so you can resume)
SKIP_REVIEWED = False

CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]
WINDOW = "Reviewer: 1/2/3=set class | ← back | →/n/Enter/r=mark reviewed & next | q=quit"
MAX_SIZE = 1400
# ===========================================

def robust_imread(path: Path):
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def load_ndjson(path: Path):
    header, records = None, []
    if not path.exists():
        raise FileNotFoundError(f"NDJSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            if obj.get("type") == "dataset":
                header = obj
            elif obj.get("type") == "image":
                records.append(obj)
    return header, records

def ensure_header_for_output(src_header):
    if src_header and src_header.get("type") == "dataset":
        hdr = dict(src_header)
        if not hdr.get("class_names"):
            hdr["class_names"] = {str(i): n for i, n in enumerate(CLASS_NAMES)}
        return hdr
    return {
        "type": "dataset",
        "task": "detect",
        "name": "Eyes (Reviewed)",
        "description": "Reviewed class labels",
        "class_names": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "version": 0,
    }

def path_contains_folder(p: Path, folder_name: str | None) -> bool:
    if not folder_name:
        return True
    return any(part.lower() == folder_name.lower() for part in p.parts)

def normalize_to_abs_boxes(rec, W, H):
    out = []
    for b in rec.get("annotations", {}).get("boxes", []):
        if len(b) < 5:
            continue
        _, cx, cy, w, h = b[:5]
        x = int((cx - w/2) * W)
        y = int((cy - h/2) * H)
        ww = int(w * W)
        hh = int(h * H)
        x = max(0, min(W-1, x))
        y = max(0, min(H-1, y))
        ww = max(1, min(W - x, ww))
        hh = max(1, min(H - y, hh))
        out.append((x, y, ww, hh))
    return out

def draw_overlay(img, boxes_abs, scale, idx, total, path_str, cur_class_id, reviewed_flag):
    disp = img
    H, W = disp.shape[:2]
    for (x, y, w, h) in boxes_abs:
        p1 = (int(x * scale), int(y * scale))
        p2 = (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
    y = 26
    badge = "REVIEWED" if reviewed_flag else "PENDING"
    info = [
        f"{idx+1}/{total}  [{badge}]",
        f"class: {CLASS_NAMES[cur_class_id]} ({cur_class_id})",
        Path(path_str).name,
        "1/2/3=set class | ← back | →/n/Enter/r=mark reviewed & next | q=quit"
    ]
    for t in info:
        cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 1, cv2.LINE_AA)
        y += 26
    return disp

def set_all_boxes_class(rec, class_id):
    boxes = rec.get("annotations", {}).get("boxes", [])
    new_boxes = []
    for b in boxes:
        if len(b) < 5:
            continue
        nb = list(b); nb[0] = int(class_id)
        new_boxes.append(nb)
    rec.setdefault("annotations", {})["boxes"] = new_boxes
    mark_reviewed(rec)

def mark_reviewed(rec):
    tags = set(rec.get("tags", []))
    if "auto_predicted" in tags:
        tags.discard("auto_predicted")
    tags.add("reviewed")
    rec["tags"] = sorted(list(tags))

def is_reviewed(rec):
    return "reviewed" in set(rec.get("tags", []))

def write_snapshot(out_path, header, all_records):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def merge_progress(src_records, out_records):
    """Return a list equal length to src_records with any edits from OUT applied (match by absolute file path)."""
    if not out_records:
        return src_records
    by_file = {}
    for r in out_records:
        if r.get("type") != "image":
            continue
        by_file[Path(r.get("file","")).resolve().as_posix()] = r
    merged = []
    for r in src_records:
        key = Path(r.get("file","")).resolve().as_posix()
        merged.append(by_file.get(key, r))
    return merged

def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Load source
    src_header, src_records = load_ndjson(SRC_NDJSON)
    header = ensure_header_for_output(src_header)

    # If OUT exists, merge prior progress so you can truly resume
    if OUT_NDJSON.exists():
        try:
            _, out_records = load_ndjson(OUT_NDJSON)
            src_records = merge_progress(src_records, out_records)
            print(f"Merged prior progress from {OUT_NDJSON}")
        except Exception as e:
            print(f"Could not merge from {OUT_NDJSON}: {e}")

        # Build the index list considering filters
    idxs = []
    for i, rec in enumerate(src_records):
        p = Path(rec.get("file", ""))
        if not path_contains_folder(p, FILTER_FOLDER):
            continue
        if SKIP_REVIEWED and is_reviewed(rec):
            continue

        # --- NEW: Filter by class id ---
        if FILTER_CLASS_ID is not None:
            boxes = rec.get("annotations", {}).get("boxes", [])
            if not boxes:
                continue
            first_class = int(boxes[0][0])
            if first_class != FILTER_CLASS_ID:
                continue
        # --------------------------------

        idxs.append(i)


    if not idxs:
        print("No items to review with current settings (maybe all are reviewed?).")
        return

    cur_k = 0

    while True:
        rec = src_records[idxs[cur_k]]
        file_path = Path(rec.get("file", ""))

        img = robust_imread(file_path)
        if img is None:
            disp = np.zeros((480, 640, 3), dtype=np.uint8)
            cur_class = int(rec.get("annotations", {}).get("boxes", [[0]])[0][0]) if rec.get("annotations", {}).get("boxes") else 0
            scale = 1.0
            boxes_abs = []
            cv2.putText(disp, f"FAILED TO READ: {file_path}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        else:
            H, W = img.shape[:2]
            long_side = max(W, H)
            scale = min(1.0, float(MAX_SIZE) / float(long_side))
            disp_img = cv2.resize(img, (int(W*scale), int(H*scale))) if scale < 1.0 else img.copy()
            boxes_abs = normalize_to_abs_boxes(rec, W, H)
            if rec.get("annotations", {}).get("boxes"):
                cur_class = int(rec["annotations"]["boxes"][0][0])
                if not (0 <= cur_class < len(CLASS_NAMES)):
                    cur_class = 0
            else:
                cur_class = 0
            disp = draw_overlay(disp_img, boxes_abs, scale, cur_k, len(idxs), str(file_path), cur_class, is_reviewed(rec))

        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(0) & 0xFFFFFFFF

        LEFT  = 2424832
        RIGHT = 2555904
        ENTER = 13

        if key in (ord('q'), 27):  # q or ESC
            break

        elif key in (ord('a'), LEFT):  # back
            cur_k = max(0, cur_k - 1)

        elif key in (ord('n'), RIGHT, ENTER, ord('r')):  # mark reviewed & next
            mark_reviewed(rec)
            write_snapshot(OUT_NDJSON, header, src_records)
            if cur_k < len(idxs) - 1:
                cur_k += 1
            else:
                print("Reached end.")
                break

        elif key == ord('1'):
            set_all_boxes_class(rec, 0)
            write_snapshot(OUT_NDJSON, header, src_records)
            if cur_k < len(idxs) - 1:
                cur_k += 1
            else:
                print("Reached end.")
                break

        elif key == ord('2'):
            set_all_boxes_class(rec, 1)
            write_snapshot(OUT_NDJSON, header, src_records)
            if cur_k < len(idxs) - 1:
                cur_k += 1
            else:
                print("Reached end.")
                break

        elif key == ord('3'):
            set_all_boxes_class(rec, 2)
            write_snapshot(OUT_NDJSON, header, src_records)
            if cur_k < len(idxs) - 1:
                cur_k += 1
            else:
                print("Reached end.")
                break

        # keep within bounds
        cur_k = max(0, min(cur_k, len(idxs) - 1))

    cv2.destroyAllWindows()
    # Final write just in case
    write_snapshot(OUT_NDJSON, header, src_records)
    print(f"Saved reviewed NDJSON to: {OUT_NDJSON}")

if __name__ == "__main__":
    main()
