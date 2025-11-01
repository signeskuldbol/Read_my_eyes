import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch

# =========================
# Settings
# =========================
"""
This code takes the dataset videos and crops them to the eye using the trained yolo model.
It performs a two-pass approach:
1) Run YOLO eye detector on all frames to get eye centers and sizes (stored in a meta JSON).
2) Re-read the video, crop each frame around the detected eye center 
(using a global square size based on biggest detected eye size), and write the cropped video.

if no detections are found in a frame, it re-uses the last known center, or falls back to the frame center if none exist.
"""

Base_path = Path(__file__).parent

# Model
MODEL_PATH = Base_path / "yolov12n_eye_detection.pt"

# Dataset I/O 
INPUT_ROOT  = Base_path / "datasets" / "New"
OUTPUT_ROOT = Base_path / "datasets" / "New_cropped"
META_ROOT   = Base_path / "datasets" / "New_cropped_meta"  # where we store centers/sizes json
VIDEO_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
SKIP_EXISTING = True  # skip if output file already exists

# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Detection
CONF_THRES = 0.50  # confidence threshold
CLASS_NAME = None  # None = any class
BATCH = 8          # batch size for pass-1 inference

# Output clip size (use 224x224 for VideoMAE)
OUTPUT_SIZE = (224, 224)

# Fallbacks/guards
FALLBACK_BOX_FRAC = 1.0   # if no dets at all, crop to square of min(W,H)
MIN_SIDE_PX       = 64    # never smaller than this

# =========================
# Model & optional class filter
# =========================
model = YOLO(str(MODEL_PATH))
names = model.model.names if hasattr(model, "model") else model.names
class_id_filter = None
if CLASS_NAME is not None:
    if isinstance(names, dict):
        inv = {v: k for k, v in names.items()}
    else:
        inv = {n: i for i, n in enumerate(names)}
    if CLASS_NAME not in inv:
        raise SystemExit(f"Class '{CLASS_NAME}' not found. Available: {list(inv.keys())}")
    class_id_filter = inv[CLASS_NAME]

# =========================
# Helpers
# =========================
def clamp_center_to_bounds(cx, cy, side, W, H):
    half = side / 2.0
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    return cx, cy

def cxcywh_to_xyxy(cx, cy, w, h):
    return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

def parse_best_det(result, class_id_filter):
    """
    Robustly parse the highest-confidence detection.
    result.boxes.data shape: [N, 6] = x1,y1,x2,y2,conf,cls
    Returns (cx, cy, w, h) or (None, None, None, None) if no det.
    """
    if result.boxes is None or result.boxes.data is None or len(result.boxes.data) == 0:
        return None, None, None, None
    det = result.boxes.data.detach().cpu().numpy()  # (N,6)
    if class_id_filter is not None:
        det = det[det[:, 5] == float(class_id_filter)]
        if det.size == 0:
            return None, None, None, None
    best = det[np.argmax(det[:, 4])]
    x1, y1, x2, y2 = best[:4]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(cx), float(cy), float(w), float(h)

# =========================
# Process ONE video (two-pass)
# =========================
def process_video(in_path: Path, out_path: Path, meta_path: Path):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {in_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # ---------- PASS 1: detect on all frames (batched, no smoothing) ----------
    centers = []   # [(cx, cy)]
    sizes   = []   # [(w, h)]
    prev_cx = prev_cy = None
    max_side_seen = 0.0

    buf_frames = []
    pbar = tqdm(total=N_FRAMES, desc=f"[pass1] {in_path.name}", unit="f", leave=False) if N_FRAMES else None

    def consume_batch(batch_frames):
        nonlocal prev_cx, prev_cy, max_side_seen
        if not batch_frames:
            return
        results = model.predict(
            source=batch_frames,
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False
        )
        for r in results:
            det_cx, det_cy, det_w, det_h = parse_best_det(r, class_id_filter)

            # NO SMOOTHING:
            # If det exists -> use it (and update prev). Else -> reuse last center; fallback to frame center if never set.
            if det_cx is not None and det_cy is not None:
                cx, cy = det_cx, det_cy
                prev_cx, prev_cy = cx, cy
                sizes.append((det_w, det_h))
                max_side_seen = max(max_side_seen, max(det_w, det_h))
            else:
                if prev_cx is not None:
                    cx, cy = prev_cx, prev_cy
                else:
                    cx, cy = W / 2.0, H / 2.0
                sizes.append((0.0, 0.0))

            centers.append((cx, cy))

        if pbar:
            pbar.update(len(batch_frames))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf_frames.append(frame)
            if len(buf_frames) >= BATCH:
                consume_batch(buf_frames)
                buf_frames = []
        # flush remainder
        if buf_frames:
            consume_batch(buf_frames)
            buf_frames = []
    finally:
        if pbar:
            pbar.close()
        cap.release()

    # Determine global square side for this video
    if max_side_seen <= 0:
        global_side = float(np.clip(FALLBACK_BOX_FRAC * min(W, H), MIN_SIDE_PX, min(W, H)))
    else:
        global_side = float(np.clip(max_side_seen, MIN_SIDE_PX, min(W, H)))

    # Persist meta for traceability
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "video": str(in_path),
            "size": {"W": W, "H": H, "FPS": FPS, "frames": N_FRAMES},
            "global_side": global_side,
            "centers": centers,   # list of [cx, cy]
            "sizes": sizes        # list of [w, h] (0,0 if no det)
        }, f, ensure_ascii=False)

    # ---------- PASS 2: write cropped video ----------
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"[pass2] Could not reopen video {in_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, FPS, OUTPUT_SIZE)
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for write: {out_path}")

    f_idx = 0
    pbar2 = tqdm(total=N_FRAMES, desc=f"[pass2] {in_path.name}", unit="f", leave=False) if N_FRAMES else None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cx, cy = centers[f_idx] if f_idx < len(centers) else (W/2.0, H/2.0)

            # Clamp center so the global square stays inside
            cx, cy = clamp_center_to_bounds(cx, cy, global_side, W, H)

            # Compute and clip crop box
            x1, y1, x2, y2 = map(int, np.round(cxcywh_to_xyxy(cx, cy, global_side, global_side)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:  # ensure non-empty
                x2, y2 = min(W, x1 + 1), min(H, y1 + 1)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                # fallback: re-center to exact middle, re-clamp, retry once
                cx, cy = clamp_center_to_bounds(W/2.0, H/2.0, global_side, W, H)
                x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, global_side, global_side)
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError("Empty crop even after fallback clamping.")

            # Choose interpolation: AREA when shrinking, CUBIC when enlarging
            interp = cv2.INTER_AREA if (crop.shape[1] > OUTPUT_SIZE[0] or crop.shape[0] > OUTPUT_SIZE[1]) else cv2.INTER_CUBIC
            crop = cv2.resize(crop, OUTPUT_SIZE, interpolation=interp)
            out.write(crop)

            f_idx += 1
            if pbar2:
                pbar2.update(1)
    finally:
        cap.release()
        out.release()
        if pbar2:
            pbar2.close()

# =========================
# Batch over dataset (preserve structure)
# =========================
def main():
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_ROOT.mkdir(parents=True, exist_ok=True)

    files = [p for p in INPUT_ROOT.rglob("*") if p.suffix in VIDEO_EXTS]
    if not files:
        print(f"No videos found under: {INPUT_ROOT}")
        return

    for in_path in tqdm(files, desc="Total videos", unit="vid"):
        rel = in_path.relative_to(INPUT_ROOT)
        out_path  = (OUTPUT_ROOT / rel).with_suffix(".mp4")  # force mp4 extension
        meta_path = (META_ROOT / rel).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        if SKIP_EXISTING and out_path.exists():
            continue

        try:
            process_video(in_path, out_path, meta_path)
        except Exception as e:
            print(f"[WARN] Failed on {in_path}: {e}")

    print(f"All done.\n  Crops  → {OUTPUT_ROOT}\n  Meta   → {META_ROOT}")

if __name__ == "__main__":
    main()
