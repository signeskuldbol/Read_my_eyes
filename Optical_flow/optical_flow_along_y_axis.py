import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass


# it is still sensitive to movements of the horse head, so a stable head pose is recommended
"""
Eye-blink detection using optical flow with stabilized bounding boxes.

This system detects an eye region using YOLO, then computes dense optical flow
inside the cropped window. The stability is achieved through a multi-step
cleanup + grouping pipeline applied to the bounding-box sizes, and exponential
smoothing applied to the detection centers.

MAIN IDEAS
----------
1. YOLO detects an eye bounding box per frame. Missing detections give None.

2. Outlier removal:
   Width and height values outside the global p5-p95 percentile range are treated
   as spikes and replaced by neighbor interpolation.

3. Independent grouping for width and height:
   Frames are grouped into segments where width (or height) changes no more than
   ±10% from the group's running mean. This forms stable pieces of consistent size.

4. Short groups (<5 frames) are treated as noise and merged into the nearest
   neighbor group based on which neighbor mean is closer.

5. Each final group receives a fixed size equal to its p95 value. This makes
   sizes perfectly stable within each segment.

6. Smooth transitions between groups:
   After grouping, an EMA is applied to w and h so size changes happen gradually
   instead of jumping.

7. Bottom-anchored center (IMPORTANT):
   The vertical center is computed from the *bottom* edge:
       cy = y2 - 0.5*h
   because the top edge often jitters downward during blinks.
   This center is computed FIRST, then smoothed with EMA.

8. Center smoothing:
   The detection centers (cx, cy) are smoothed with EMA:
       smoothed = alpha * new + (1 - alpha) * old
   - alpha close to 1.0 → follow detection more (less smoothing)
   - alpha close to 0.0 → follow history more (more smoothing)

9. Optical flow with fixed-size crops:
   Farneback requires consecutive inputs to have identical shape.
   Therefore, each crop is resized to a fixed resolution (FLOW_W_FIX, FLOW_H_FIX)
   before optical flow. We only reset flow when detection/crop is invalid.

10. Vertical-only flow scoring (NEW):
    We measure blink activity only along the y-axis:
        score = mean(|dy|)
    This ignores horizontal motion and removes a lot of head-movement noise.
"""

# ---------- Inference settings ----------
DET_CONF = 0.25
DET_IOU  = 0.5
FONT = cv.FONT_HERSHEY_SIMPLEX

# Optical flow threshold for "blink" / action
FLOW_BLINK_THR = 1

# Center smoothing factor (0 = use prior, 1 = fully follow new detection)
CENTER_SMOOTHING = 0.75

# BBox size EMA smoothing (0 = prior, 1 = fully follow new size)
BBOX_SMOOTHING = 0.5

# ---------------- Paths ----------------
Workspace_Path = Path(__file__).parent.parent.parent.resolve()
detection_eye_model_path = Workspace_Path / "Read_my_eyes" / "create_datasets" / "yolov12n_eye_detection.pt"

videos_dir   = Workspace_Path / "Read_my_eyes" / "create_datasets" / "original_videos_annotations" / "videos"
outputs_dir  = Workspace_Path / "Read_my_eyes" / "Optical_flow" / "outputs_optical_flow" / f"outputs_optical_flow_along_y_axis_{FLOW_BLINK_THR}_{CENTER_SMOOTHING}_{BBOX_SMOOTHING}"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


def compute_flow(frame1, frame2):
    """Compute dense optical flow between two frames using Farneback."""
    if frame1.ndim == 3 and frame1.shape[2] == 3:
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if frame2.ndim == 3 and frame2.shape[2] == 3:
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    flow = cv.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=5,
        iterations=3,
        poly_n=10,
        poly_sigma=1.2,
        flags=0
    )
    return flow


def mean_vertical_flow(flow):
    """
    NEW: Compute mean vertical flow magnitude only.
    flow[..., 1] is dy. We use absolute dy so up/down both count.
    """
    dy = flow[..., 1]
    return float(np.mean(np.abs(dy)))


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


def nice_label(img, text, org, color, scale=0.7, thickness=2):
    """Draw text label with filled background."""
    x, y = org
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    pad = 4
    cv.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + baseline), (0, 0, 0), -1)
    cv.putText(img, text, (x + pad, y - pad), FONT, scale, color, thickness, cv.LINE_AA)


def make_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv.VideoWriter:
    """Create VideoWriter with fallback codecs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_candidates = ["mp4v", "avc1", "H264", "XVID"]
    for cc in fourcc_candidates:
        fourcc = cv.VideoWriter_fourcc(*cc)
        w = cv.VideoWriter(str(path), fourcc, fps if fps and fps > 1e-3 else 25.0, size)
        if w.isOpened():
            print(f"[OK] Writing to {path.name} with codec {cc}")
            return w
    raise RuntimeError(f"Could not open writer: {path}")


def smooth_centers(centers, alpha):
    """EMA smoothing for (cx, cy) centers."""
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
    vals = values.copy()
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
    """EMA smoothing for per-frame bbox sizes."""
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


# =========================
#  1D grouping helpers
# =========================

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
            merged.append(g); i += 1; continue

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


def process_one_video(detection_model: YOLO, video_path: Path):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    W  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0

    out_path = outputs_dir / f"{video_path.stem}_optical_flow.mp4"
    print(f"Processing {video_path.name} → {out_path.name}  ({W}x{H} @ {fps:.2f} fps)")

    # --------- PASS 1 ---------
    frames  = []
    centers = []
    sizes   = []

    det_ws, det_hs = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        det_results = detection_model.predict(
            source=frame, conf=DET_CONF, iou=DET_IOU, save=False, verbose=False
        )
        det = det_results[0]
        boxes = getattr(det, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            idx   = int(confs.argmax().item())

            xyxy = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = clamp_xyxy(
                xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H
            )

            w = x2 - x1
            h = y2 - y1

            cx = (x1 + x2) / 2.0
            cy = y2 - 0.5 * h  # bottom-anchored center

            centers.append((cx, cy))
            sizes.append((w, h))

            det_ws.append(w)
            det_hs.append(h)
        else:
            centers.append(None)
            sizes.append(None)

    cap.release()

    print(f"centers collected: {len(centers)}, sizes collected: {len(sizes)}")

    if len(det_ws) == 0 or len(det_hs) == 0:
        print("[INFO] No detections found in this video. Saving full-frame output.")
        writer = make_video_writer(out_path, fps, (W, H))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return

    # --------- Percentiles ---------
    p5_w  = float(np.percentile(det_ws, 5))
    p95_w = float(np.percentile(det_ws, 95))
    p5_h  = float(np.percentile(det_hs, 5))
    p95_h = float(np.percentile(det_hs, 95))

    print(f"[INFO] Width bounds p5-p95:  {p5_w:.1f} - {p95_w:.1f}")
    print(f"[INFO] Height bounds p5-p95: {p5_h:.1f} - {p95_h:.1f}")

    raw_ws = [s[0] if s is not None else None for s in sizes]
    raw_hs = [s[1] if s is not None else None for s in sizes]

    clean_ws = replace_spikes_interp(raw_ws, p5_w, p95_w)
    clean_hs = replace_spikes_interp(raw_hs, p5_h, p95_h)

    valid_mask = [(w is not None and h is not None) for w, h in zip(clean_ws, clean_hs)]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]

    if len(valid_indices) == 0:
        print("[INFO] No valid sizes after cleaning. Saving full-frame output.")
        writer = make_video_writer(out_path, fps, (W, H))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return

    v_ws = np.array([clean_ws[i] for i in valid_indices], dtype=np.float32)
    v_hs = np.array([clean_hs[i] for i in valid_indices], dtype=np.float32)

    g_ws, g_hs, w_groups, h_groups = compute_grouped_sizes_separate(v_ws, v_hs)

    grouped_ws_full = [None] * len(frames)
    grouped_hs_full = [None] * len(frames)

    for k, i in enumerate(valid_indices):
        grouped_ws_full[i] = g_ws[k]
        grouped_hs_full[i] = g_hs[k]

    clean_sizes = [
        (w, h) if (w is not None and h is not None) else None
        for w, h in zip(grouped_ws_full, grouped_hs_full)
    ]

    smooth_sizes_list = smooth_sizes(clean_sizes, BBOX_SMOOTHING)
    smooth_centers_list = smooth_centers(centers, CENTER_SMOOTHING)

    writer = make_video_writer(out_path, fps, (W, H))

    # --------- FIXED FLOW SIZE ---------
    FLOW_W_FIX = max(2, int(round(p95_w)))
    FLOW_H_FIX = max(2, int(round(p95_h)))

    # --------- PASS 2 (VERTICAL-ONLY FLOW) ---------
    prev_crop_gray = None

    for idx, (frame, center, size) in enumerate(zip(frames, smooth_centers_list, smooth_sizes_list)):
        annotated = frame.copy()
        action = 0
        mean_mag = 0.0

        if center is None or size is None:
            prev_crop_gray = None

        if center is not None and size is not None:
            cx, cy = center
            bw, bh = size

            half_w = int(bw // 2)
            half_h = int(bh // 2)

            x1 = int(cx - half_w)
            y1 = int(cy - half_h)
            x2 = x1 + int(bw)
            y2 = y1 + int(bh)
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)

            eye_crop = frame[y1:y2, x1:x2]
            if eye_crop.size > 0:
                eye_gray = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)

                # resize to fixed size for flow
                eye_gray_fix = cv.resize(
                    eye_gray, (FLOW_W_FIX, FLOW_H_FIX),
                    interpolation=cv.INTER_LINEAR
                )

                if prev_crop_gray is not None:
                    flow = compute_flow(prev_crop_gray, eye_gray_fix)

                    # --- NEW: vertical-only score ---
                    mean_mag = mean_vertical_flow(flow)

                    action = 1 if mean_mag >= FLOW_BLINK_THR else 0

                prev_crop_gray = eye_gray_fix
            else:
                prev_crop_gray = None

            if action == 1:
                box_color = (0, 255, 0)
                status_text = "BLINK"
            else:
                box_color = (0, 255, 255)
                status_text = "NONE"

            cv.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"{status_text} | mean|dy|: {mean_mag:.2f}"
            label_y = max(y1 - 10, 20)
            nice_label(annotated, label_text, (x1, label_y), box_color)

            cv.putText(annotated, label_text, (10, 30), FONT, 0.8, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(annotated, label_text, (10, 30), FONT, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        else:
            info_txt = "No eye detection/size"
            cv.putText(annotated, info_txt, (10, 30), FONT, 0.8, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(annotated, info_txt, (10, 30), FONT, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        writer.write(annotated)

    writer.release()
    print("Saved:", out_path)


def main():
    detection_model = YOLO(str(detection_eye_model_path))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in videos_dir.iterdir()
        if p.suffix in VIDEO_EXTS and p.is_file()
    ]
    if not video_files:
        print(f"[WARN] No videos found in: {videos_dir}")
        return

    for vid in sorted(video_files):
        out_path = outputs_dir / f"{vid.stem}_annotated_of.mp4"
        if out_path.exists():
            print(f"[SKIP] Output already exists for {vid.name}, skipping.")
            continue

        try:
            process_one_video(detection_model, vid)
        except Exception as e:
            print(f"[ERROR] Failed on {vid.name}: {e}")


if __name__ == "__main__":
    main()
