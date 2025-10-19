# yolo_approach/auto_label_missing.py
# Prelabel unlabeled images with your trained YOLO model and merge into a new NDJSON.
import json, os
from pathlib import Path
from ultralytics import YOLO

# ----------------- EDIT THESE -----------------
# Your dataset root with train/val/test folders
Workspace_ROOT = Path(__file__).parent.parent.resolve()
DATASET_ROOT = Workspace_ROOT / "create_datasets" /"datasets" / "frames_sorted_no_pad_no_crop"

# Existing NDJSON that already contains your manual annotations
EXISTING_NDJSON = Workspace_ROOT / "yolo_approach" / "annotations" / "eyes.ndjson"

# The trained model weights (best.pt)
WEIGHTS = Workspace_ROOT / "yolo_approach" / "annotations" / "runs" / "horse_eyes" / "y12n_eye_v13" / "weights" / "best.pt"

# Output merged NDJSON (existing annotations + new auto labels for the rest)
OUT_NDJSON = EXISTING_NDJSON.with_name("eyes_merged_autolabel.ndjson")

# Model/prediction knobs
IMGSZ   = 768
CONF    = 0.15  # low-ish to favor recall (you will review)
IOU     = 0.6
MAXDET  = 20
# ----------------------------------------------

CLASS_NAMES = ["eye"]  # 1-class autolabel for now
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def iter_images(root: Path):
    for split in ["train", "val", "test"]:
        sdir = root / split
        if not sdir.exists():
            continue
        for p in sorted(sdir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                yield split, p

def load_done_set(ndjson: Path):
    done = set()
    header = None
    if not ndjson.exists():
        return header, done
    with ndjson.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except:
                continue
            if obj.get("type") == "dataset":
                header = obj
            elif obj.get("type") == "image":
                done.add(Path(obj["file"]).resolve().as_posix())
    return header, done

def write_header(out_path: Path, header_obj: dict | None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use existing header if present; otherwise create a minimal 1-class header
    if header_obj and header_obj.get("type") == "dataset":
        # Ensure class_names are at least {0:'eye'} for 1-class flow
        if not header_obj.get("class_names"):
            header_obj["class_names"] = {"0": "eye"}
        with out_path.open("w", encoding="utf-8") as w:
            w.write(json.dumps(header_obj, ensure_ascii=False) + "\n")
    else:
        header = {
            "type": "dataset",
            "task": "detect",
            "name": "HorseEyes",
            "description": "Merged manual + auto-labeled eye boxes",
            "class_names": {"0": "eye"},
            "version": 0
        }
        with out_path.open("w", encoding="utf-8") as w:
            w.write(json.dumps(header, ensure_ascii=False) + "\n")

def copy_existing_images(src_ndjson: Path, dst_ndjson: Path):
    if not src_ndjson.exists():
        return 0
    copied = 0
    with src_ndjson.open("r", encoding="utf-8") as f, dst_ndjson.open("a", encoding="utf-8") as w:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            if obj.get("type") == "image":
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                copied += 1
    return copied

def pred_boxes_for_image(model: YOLO, img_path: Path):
    # Returns list of [class_id, cx, cy, w, h] normalized to image size
    # We force class_id=0 (eye) for the autolabel stage
    res = model.predict(
        source=os.fspath(img_path),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAXDET,
        verbose=False
    )
    boxes_out = []
    if not res:
        return boxes_out, 0, 0
    r = res[0]
    # Get image size (Ultralytics stores in r.orig_shape as (H, W))
    H, W = (r.orig_shape if hasattr(r, "orig_shape") else (0, 0))
    if r.boxes is None or len(r.boxes) == 0 or H == 0 or W == 0:
        return boxes_out, W, H
    for b in r.boxes:
        # xywh normalized are available as b.xyxyn or b.xywhn; use xywhn if present
        if hasattr(b, "xywhn"):
            cx, cy, w, h = b.xywhn[0].tolist()
        else:
            # fall back: compute from xyxy in absolute, then normalize
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
        boxes_out.append([0, float(cx), float(cy), float(w), float(h)])
    return boxes_out, W, H

def main():
    model = YOLO(os.fspath(WEIGHTS))

    header, already_done = load_done_set(EXISTING_NDJSON)
    write_header(OUT_NDJSON, header)

    # 1) copy all existing annotated images into the merged file
    copied = copy_existing_images(EXISTING_NDJSON, OUT_NDJSON)
    print(f"[INFO] Copied {copied} existing annotated records.")

    # 2) iterate dataset; for images not in existing NDJSON, predict and append
    added = 0
    total_missing = 0
    with OUT_NDJSON.open("a", encoding="utf-8") as w:
        for split, img_path in iter_images(DATASET_ROOT):
            abs_p = img_path.resolve().as_posix()
            if abs_p in already_done:
                continue
            total_missing += 1
            boxes, W, H = pred_boxes_for_image(model, img_path)
            rec = {
                "type": "image",
                "file": abs_p,
                "width": W,
                "height": H,
                "split": split,
                "annotations": {"boxes": boxes}
            }
            # tag that these came from the model; handy in your annotator UI
            if not boxes:
                rec["tags"] = ["auto_no_detections"]
            else:
                rec["tags"] = ["auto_predicted"]
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1
            if added % 50 == 0:
                print(f"[INFO] Auto-labeled {added}/{total_missing} missing...")

    print(f"[DONE] Missing images found: {total_missing}")
    print(f"[DONE] Auto-labeled and appended: {added}")
    print(f"[OUT ] {OUT_NDJSON}")

if __name__ == "__main__":
    main()
