import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ===================== CONFIG =====================
# Base paths (you already have these in your snippet)
READ_MY_EYES_DIR = Path(__file__).parent.parent.resolve()
Base_path = Path(__file__).parent

# Model
MODEL_PATH = Base_path / "yolov12n_eye_detection.pt"

# Dataset I/O
INPUT_ROOT  = Base_path / "datasets" / "frames_sorted_no_pad_no_crop"
OUTPUT_ROOT = Base_path / "datasets" / "frames_sorted_no_pad_cropped"

# Detection params
IMGSZ = 768           # YOLO inference size (doesn't change saved crop size)
CONF_THRES = 0.25     # confidence threshold
IOU_THRES  = 0.5      # NMS IoU threshold

# Cropping
PADDING_FACTOR = 0.10  # 10% extra around the max side

# Image types
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ==================================================


def robust_imread(path: Path):
    """Read image from Unicode path safely."""
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def robust_imwrite(path: Path, img: np.ndarray) -> bool:
    """Write image to Unicode path safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
        path = path.with_suffix(ext)
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(os.fspath(path))
        return True
    except Exception:
        return False


def square_crop_with_padding(xyxy, img_w, img_h, pad_factor=0.10):
    """
    Given a bbox [x1,y1,x2,y2] in absolute pixels, make a square crop centered
    on the box center with side = max(w,h) * (1+pad_factor), clamped to image bounds.
    Returns (x0, y0, x1, y1) as ints.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    side = max(w, h) * (1.0 + pad_factor)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + int(round(side))
    y1 = y0 + int(round(side))

    # shift into bounds while keeping square
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > img_w:
        shift = x1 - img_w
        x0 = max(0, x0 - shift)
        x1 = img_w
    if y1 > img_h:
        shift = y1 - img_h
        y0 = max(0, y0 - shift)
        y1 = img_h

    # final clamp (defensive)
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(x0 + 1, min(x1, img_w))
    y1 = max(y0 + 1, min(y1, img_h))

    return x0, y0, x1, y1


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    # Load model once
    model = YOLO(str(MODEL_PATH))

    in_paths = list(iter_images(INPUT_ROOT))
    if not in_paths:
        print(f"[WARN] No images found under: {INPUT_ROOT}")
        return

    for in_path in tqdm(in_paths, desc="Cropping with YOLO"):
        img = robust_imread(in_path)
        rel = in_path.relative_to(INPUT_ROOT)
        out_path = OUTPUT_ROOT / rel

        if img is None:
            print(f"[WARN] Could not read image: {in_path}")
            # If can't read, skip writing
            continue

        H, W = img.shape[:2]

        # Run detection (single image)
        # Ultralytics returns a Results list; we take the first element
        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )
        res = results[0]

        # No detections?
        if res.boxes is None or len(res.boxes) == 0:
            # Save original image, warn
            _ = robust_imwrite(out_path, img)
            print(f"[WARN] No eye detected → kept original: {rel}")
            continue

        # Pick highest-confidence detection
        boxes = res.boxes
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
        best_idx = int(np.argmax(confs))
        xyxy = boxes.xyxy[best_idx].cpu().numpy().tolist()

        # Compute square crop with padding and clamp to bounds
        x0, y0, x1, y1 = square_crop_with_padding(
            xyxy, img_w=W, img_h=H, pad_factor=PADDING_FACTOR
        )

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            # Fallback: if something went weird, save original
            _ = robust_imwrite(out_path, img)
            print(f"[WARN] Empty crop (clamp issue) → kept original: {rel}")
            continue

        ok = robust_imwrite(out_path, crop)
        if not ok:
            print(f"[WARN] Failed to write crop: {out_path}")

    print(f"[DONE] Saved crops to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
