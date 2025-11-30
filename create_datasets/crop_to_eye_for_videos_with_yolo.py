import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch
from dataclasses import dataclass

# =========================
# Settings
# =========================
"""
Eye-region cropping for VideoMAE with stabilized bounding boxes.

Pipeline per video:
1) Run YOLO on all frames → per-frame eye bbox (or None) + confidence.
2) Replace all missing detections with the crop from the *closest* detected
   neighbour in time (tie → neighbour with higher confidence).
3) Clean and stabilize bbox sizes:
   - Remove spikes using global p5–p95.
   - Group w and h separately into segments where size stays within ±10%.
   - Merge short groups and assign each group a fixed size = p95 of that group.
   - Smooth sizes with EMA.
4) Smooth centers with EMA (bottom-anchored center from bbox).
5) For each frame: use a square crop with side = max(w, h), clamp to image,
   then resize to 224x224 for VideoMAE.
6) If no detections in the whole video: use a centered square crop for all frames.
"""

Base_path = Path(__file__).parent

# Model
MODEL_PATH = Base_path / "yolov12n_eye_detection.pt"

# Dataset I/O 
INPUT_ROOT  = Base_path / "datasets" / "NEW_split_new_way"  # input videos
OUTPUT_ROOT = Base_path / "datasets" / "NEW_split_cropped_new_way"
META_ROOT   = Base_path / "datasets" / "NEW_split_cropped_meta_new_way"  # where we store centers/sizes json
VIDEO_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
SKIP_EXISTING = True  # skip if output file already exists

# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Detection
CONF_THRES = 0.50  # confidence threshold
IOU_THRES  = 0.5
CLASS_NAME = None  # None = any class
BATCH      = 8     # batch size for pass-1 inference

# Output clip size (use 224x224 for VideoMAE)
OUTPUT_SIZE = (224, 224)

# Fallbacks/guards
FALLBACK_BOX_FRAC = 1.0   # if no dets at all, crop to square of min(W,H)
MIN_SIDE_PX       = 64    # never smaller than this

# Smoothing / grouping hyperparameters (similar to OF script)
CENTER_SMOOTHING = 0.75   # EMA factor for centers
BBOX_SMOOTHING   = 0.5    # EMA factor for (w, h)
GROUP_TOL        = 0.10   # ±10% tolerance for grouping sizes
GROUP_MIN_LEN    = 5      # minimum length of a size group

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


def clamp_xyxy(x1, y1, x2, y2, W, H):
    """Clamp bbox to image size and ensure >=1px width/height."""
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def smooth_centers(centers, alpha):
    """EMA smoothing for (cx, cy) centers. Handles None robustly."""
    smoothed = []
    prev = None
    for c in centers:
        if c is None:
            smoothed.append(None)
            prev = None
        else:
            cx, cy = c
            if prev is None:
                s_cx, s_cy = cx, cy
            else:
                p_cx, p_cy = prev
                s_cx = alpha * cx + (1.0 - alpha) * p_cx
                s_cy = alpha * cy + (1.0 - alpha) * p_cy
            smoothed.append((s_cx, s_cy))
            prev = (s_cx, s_cy)
    return smoothed


def replace_spikes_interp(values, low, high):
    """Replace outliers outside [low, high] by interpolating nearest valid neighbors."""
    vals = list(values)
    n = len(vals)

    def is_valid(x):
        return x is not None and low <= x <= high

    for i in range(n):
        if not is_valid(vals[i]):
            l = i - 1
            while l >= 0 and not is_valid(vals[l]):
                l -= 1
            r = i + 1
            while r < n and not is_valid(vals[r]):
                r += 1
            if l >= 0 and r < n:
                vals[i] = vals[l] + (vals[r] - vals[l]) * ((i - l) / (r - l))
            elif l >= 0:
                vals[i] = vals[l]
            elif r < n:
                vals[i] = vals[r]
            else:
                vals[i] = None
    return vals


def smooth_sizes(sizes, alpha):
    """EMA smoothing for per-frame bbox sizes (w, h). Keeps None."""
    smoothed = []
    prev = None
    for s in sizes:
        if s is None:
            smoothed.append(None)
            prev = None
        else:
            w, h = s
            if prev is None:
                sw, sh = w, h
            else:
                pw, ph = prev
                sw = alpha * w + (1 - alpha) * pw
                sh = alpha * h + (1 - alpha) * ph
            smoothed.append((sw, sh))
            prev = (sw, sh)
    return smoothed


@dataclass
class Group1D:
    start: int
    end: int
    mean: float
    p95: float


def _initial_groups_1d(vals, tol=0.10):
    """Build consecutive groups based on +/- tol of running mean."""
    groups = []
    n = len(vals)
    i = 0
    while i < n:
        start = i
        running_mean = float(vals[i])
        count = 1
        i += 1
        while i < n:
            x = float(vals[i])
            if abs(x - running_mean) / (running_mean + 1e-9) <= tol:
                count += 1
                running_mean += (x - running_mean) / count
                i += 1
            else:
                break
        end = i - 1
        groups.append((start, end))
    return groups


def _merge_short_groups_1d(groups, vals, min_len=5):
    """Merge groups shorter than min_len into nearest neighbor by mean distance."""
    def gmean(g):
        s, e = g
        return float(np.mean(vals[s:e+1]))

    merged = []
    i = 0
    while i < len(groups):
        g = groups[i]
        glen = g[1] - g[0] + 1

        if glen >= min_len:
            merged.append(g)
            i += 1
            continue

        m = gmean(g)
        prev_g = merged[-1] if merged else None
        next_g = groups[i+1] if i+1 < len(groups) else None

        if prev_g is None and next_g is None:
            merged.append(g)
            i += 1
            continue

        if prev_g is None:
            groups[i+1] = (g[0], next_g[1])
            i += 1
            continue

        if next_g is None:
            merged[-1] = (prev_g[0], g[1])
            i += 1
            continue

        prev_m = gmean(prev_g)
        next_m = gmean(next_g)

        if abs(m - prev_m) <= abs(m - next_m):
            merged[-1] = (prev_g[0], g[1])
        else:
            groups[i+1] = (g[0], next_g[1])
        i += 1

    return merged


def _finalize_groups_1d(groups, vals):
    out = []
    for (s, e) in groups:
        seg = np.asarray(vals[s:e+1], dtype=np.float32)
        out.append(Group1D(
            start=s,
            end=e,
            mean=float(np.mean(seg)),
            p95=float(np.percentile(seg, 95))
        ))
    return out


def compute_grouped_sizes_separate(ws, hs, tol=0.10, min_len=5):
    """
    Group w and h separately based on running-mean tolerance.
    Short groups merge into nearest neighbor by mean distance.
    Returns full-length per-frame sequences for w and h.
    """
    ws = np.asarray(ws, dtype=np.float32)
    hs = np.asarray(hs, dtype=np.float32)
    n = len(ws)

    # --- W groups ---
    w0 = _initial_groups_1d(ws, tol=tol)
    w1 = _merge_short_groups_1d(w0, ws, min_len=min_len)
    w_groups = _finalize_groups_1d(w1, ws)

    clean_w = np.zeros(n, dtype=np.float32)
    for g in w_groups:
        clean_w[g.start:g.end+1] = g.p95

    # --- H groups ---
    h0 = _initial_groups_1d(hs, tol=tol)
    h1 = _merge_short_groups_1d(h0, hs, min_len=min_len)
    h_groups = _finalize_groups_1d(h1, hs)

    clean_h = np.zeros(n, dtype=np.float32)
    for g in h_groups:
        clean_h[g.start:g.end+1] = g.p95

    return clean_w.tolist(), clean_h.tolist(), w_groups, h_groups


def fill_missing_with_nearest_neighbor(raw_centers, raw_sizes, raw_confs):
    """
    Replace None detections by copying (center, size, conf) from the
    temporally closest detected frame (tie → higher confidence).
    """
    n = len(raw_centers)
    centers = list(raw_centers)
    sizes   = list(raw_sizes)
    confs   = list(raw_confs)

    det_indices = [i for i, c in enumerate(centers) if c is not None]
    if not det_indices:
        # no detections at all; caller will handle fallback
        return centers, sizes, confs

    # nearest detection to the left
    left = [-1] * n
    last = -1
    for i in range(n):
        left[i] = last
        if centers[i] is not None:
            last = i

    # nearest detection to the right
    right = [-1] * n
    last = -1
    for i in range(n - 1, -1, -1):
        right[i] = last
        if centers[i] is not None:
            last = i

    for i in range(n):
        if centers[i] is not None:
            continue

        li = left[i]
        ri = right[i]

        if li == -1 and ri == -1:
            # should not happen if we have any detections
            continue

        if li == -1:
            j = ri
        elif ri == -1:
            j = li
        else:
            dl = i - li
            dr = ri - i
            if dl < dr:
                j = li
            elif dr < dl:
                j = ri
            else:
                # equal distance → choose higher confidence
                conf_l = confs[li] if confs[li] is not None else 0.0
                conf_r = confs[ri] if confs[ri] is not None else 0.0
                j = li if conf_l >= conf_r else ri

        centers[i] = centers[j]
        sizes[i]   = sizes[j]
        confs[i]   = confs[j]

    return centers, sizes, confs


# =========================
# Process ONE video (two-pass, with smoothing)
# =========================
def process_video(in_path: Path, out_path: Path, meta_path: Path):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {in_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # ---------- PASS 1: detect on all frames ----------
    raw_centers = []   # [(cx, cy)] or None
    raw_sizes   = []   # [(w, h)] or None
    raw_confs   = []   # [conf] or None
    det_ws, det_hs = [], []

    buf_frames = []
    pbar = tqdm(total=N_FRAMES, desc=f"[pass1] {in_path.name}", unit="f", leave=False) if N_FRAMES else None

    def consume_batch(batch_frames):
        if not batch_frames:
            return
        results = model.predict(
            source=batch_frames,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=DEVICE,
            verbose=False
        )
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.detach().cpu().numpy()

                # optional class filter
                if class_id_filter is not None:
                    cls = boxes.cls.detach().cpu().numpy()
                    mask = cls == float(class_id_filter)
                    if not mask.any():
                        raw_centers.append(None)
                        raw_sizes.append(None)
                        raw_confs.append(None)
                        continue
                    confs_masked = confs.copy()
                    confs_masked[~mask] = -1e9
                    idx = int(confs_masked.argmax().item())
                else:
                    idx = int(confs.argmax().item())

                xyxy = boxes.xyxy[idx].detach().cpu().numpy()
                x1, y1, x2, y2 = clamp_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H)

                w = float(x2 - x1)
                h = float(y2 - y1)

                # bottom-anchored center (like OF script)
                cx = float((x1 + x2) / 2.0)
                cy = float(y2 - 0.5 * h)

                conf = float(confs[idx])

                raw_centers.append((cx, cy))
                raw_sizes.append((w, h))
                raw_confs.append(conf)
                det_ws.append(w)
                det_hs.append(h)
            else:
                raw_centers.append(None)
                raw_sizes.append(None)
                raw_confs.append(None)

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
        if buf_frames:
            consume_batch(buf_frames)
            buf_frames = []
    finally:
        if pbar:
            pbar.close()
        cap.release()

    num_frames = len(raw_centers)

    # ---------- Size / center logic ----------
    if len(det_ws) == 0 or len(det_hs) == 0:
        # no detections at all → fallback to center crop of full frame
        global_side = float(np.clip(FALLBACK_BOX_FRAC * min(W, H),
                                    MIN_SIDE_PX, min(W, H)))
        final_centers = [(W / 2.0, H / 2.0)] * num_frames
        final_sides   = [global_side] * num_frames
        grouped_info = None
    else:
        # 1) FIRST: replace all None detections with nearest neighbor
        raw_centers, raw_sizes, raw_confs = fill_missing_with_nearest_neighbor(
            raw_centers, raw_sizes, raw_confs
        )
        # now every frame has (center, size)

        # 2) Percentile bounds from true detections
        p5_w  = float(np.percentile(det_ws, 5))
        p95_w = float(np.percentile(det_ws, 95))
        p5_h  = float(np.percentile(det_hs, 5))
        p95_h = float(np.percentile(det_hs, 95))

        raw_ws = [s[0] for s in raw_sizes]
        raw_hs = [s[1] for s in raw_sizes]

        # 3) Spike removal (p5–p95)
        clean_ws = replace_spikes_interp(raw_ws, p5_w, p95_w)
        clean_hs = replace_spikes_interp(raw_hs, p5_h, p95_h)

        v_ws = np.array(clean_ws, dtype=np.float32)
        v_hs = np.array(clean_hs, dtype=np.float32)

        # 4) Group sizes and assign group p95
        g_ws, g_hs, w_groups, h_groups = compute_grouped_sizes_separate(
            v_ws, v_hs, tol=GROUP_TOL, min_len=GROUP_MIN_LEN
        )

        grouped_sizes = [(w, h) for w, h in zip(g_ws, g_hs)]

        # 5) EMA smoothing
        smooth_sizes_list   = smooth_sizes(grouped_sizes, BBOX_SMOOTHING)
        smooth_centers_list = smooth_centers(raw_centers, CENTER_SMOOTHING)

        # 6) Final per-frame centers and square sides (side = max(w, h))
        final_centers = []
        final_sides   = []

        median_side = max(
            float(np.median(v_ws)),
            float(np.median(v_hs)),
            float(MIN_SIDE_PX),
        )
        median_side = float(np.clip(median_side, MIN_SIDE_PX, min(W, H)))

        for c, s in zip(smooth_centers_list, smooth_sizes_list):
            # after fill, both c and s should be valid
            if c is None or s is None:
                cx, cy = W / 2.0, H / 2.0
                side = median_side
            else:
                cx, cy = c
                w, h   = s
                side   = float(max(w, h))
                side   = float(np.clip(side, MIN_SIDE_PX, min(W, H)))

            final_centers.append((cx, cy))
            final_sides.append(side)

        grouped_info = {
            "p5_w": p5_w,
            "p95_w": p95_w,
            "p5_h": p5_h,
            "p95_h": p95_h,
            "w_groups": [g.__dict__ for g in w_groups],
            "h_groups": [g.__dict__ for g in h_groups],
        }

        global_side = float(np.clip(max(final_sides), MIN_SIDE_PX, min(W, H)))

    # ---------- Save meta ----------
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "video": str(in_path),
            "size": {"W": W, "H": H, "FPS": FPS, "frames": N_FRAMES},
            "global_side": global_side,
            "centers": final_centers,   # [ (cx, cy) ]
            "sides":   final_sides,     # [ square side ]
            "grouped_info": grouped_info,
        }, f, ensure_ascii=False)

    # ---------- PASS 2: crop and write video ----------
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
            if not ok or f_idx >= len(final_centers):
                break

            cx, cy = final_centers[f_idx]
            side   = final_sides[f_idx]

            # Clamp center so the square stays inside
            cx, cy = clamp_center_to_bounds(cx, cy, side, W, H)

            # Compute and clip crop box
            x1, y1, x2, y2 = map(int, np.round(cxcywh_to_xyxy(cx, cy, side, side)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                x2, y2 = min(W, x1 + 1), min(H, y1 + 1)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                # fallback: re-center to exact middle, re-clamp, retry once
                cx, cy = clamp_center_to_bounds(W/2.0, H/2.0, side, W, H)
                x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, side, side)
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError("Empty crop even after fallback clamping.")

            # Resize to 224x224 for VideoMAE
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
# Batch over dataset
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
