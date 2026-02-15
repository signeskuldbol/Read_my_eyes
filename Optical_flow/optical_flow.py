import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix



# it is still sensitive to movements of the horse head, so a stable head pose is recommended
"""
Optical-flow blink detection (binary) + evaluation against time-based ground truth.

Goal
----
Detect whether a horse is blinking in each video frame (0/1), using:
  1) a YOLO detector to localize the eye region, and
  2) dense optical flow inside the eye crop to measure motion consistent with a blink.

Why stabilization is needed
---------------------------
The YOLO eye box typically jitters from frame to frame (small shifts in position/size),
especially when the horse head moves. Dense optical flow is *very* sensitive to this:
if the crop changes shape or jumps around, the motion score can spike even without a blink.

Therefore, we stabilize the eye region across time before computing optical flow.

Pipeline overview (per video)
-----------------------------
A) Eye localization + ROI stabilization
   - Run YOLO per frame to get one or more candidate eye boxes.
   - If multiple boxes exist:
       * first valid frame: choose the highest confidence box
       * later frames: choose the box whose center is closest to the previously used center
     (this keeps tracking consistent and avoids jumping between detections)

   - Convert the chosen box to (center, width, height) and stabilize:

   1. Outlier removal (size spikes):
      Compute global size percentiles (p5-p95) from valid detections.
      Width/height values outside this range are treated as spikes and replaced
      by interpolation from neighboring valid frames.

   2. Piecewise-constant grouping (size stabilization):
      Width and height are handled independently.
      Frames are grouped into segments where size changes are within ±10% of the segment mean.
      Very short segments (default <5 frames) are treated as noise and merged into neighbors.

   3. Within-segment fixed size:
      Each final segment is assigned a single “stable” size value (p95 of that segment),
      so the ROI size does not fluctuate within the segment.

   4. Smooth transitions between segments:
      Apply exponential moving average (EMA) to width/height so changes between segments
      happen gradually instead of jumping.

   5. Stable center point:
      Compute the vertical center using a *bottom anchor*:
          cy = y2 - 0.5*h
      This reduces sensitivity to the top eyelid moving down during blinks.
      Then apply EMA smoothing to (cx, cy) to reduce jitter.

B) Optical flow scoring (per frame)
   - Crop the stabilized ROI from each frame.
   - Resize each crop to a fixed resolution (FLOW_W_FIX, FLOW_H_FIX) so consecutive crops
     always have the same shape (required by Farneback optical flow).
   - Compute dense Farneback optical flow between consecutive crops.
   - Convert flow to a scalar “blink activity” magnitude per frame:
       * If USE_VERTICAL_FLOW=True: score = mean(|dy|)
         (reduces sensitivity to sideways head motion)
       * Else: score = mean(sqrt(dx² + dy²))

C) Thresholding + padding (binary blink prediction)
   - Convert magnitude score to a per-frame binary prediction:
         above = (magnitude >= threshold)

   - Then apply padding to make events more stable/realistic:
       * PADDING_FRAMES_BEFORE extends the blink a few frames earlier
       * PADDING_FRAMES_AFTER  extends the blink a few frames later
     This helps when the magnitude briefly dips below threshold mid-blink.

Threshold selection (per video)
-------------------------------
The threshold is computed from the distribution of magnitudes within the video
(using only frames where flow was computed). You can choose one of three modes:

  1) THR_MODE="iqr"
     Robust baseline + spread:
        threshold = median + K_IQR * (p75 - p25)
     (usually the most stable across videos, because it is less affected by outliers)

  2) THR_MODE="range_pct"
     Fraction of the min-max range:
        threshold = min + RANGE_PCT * (max - min)
     (simple, but can be sensitive to extreme spikes/outliers)

  3) THR_MODE="baseline_plus"
     Baseline with a fixed offset:
        threshold = median + BASELINE_ADD
     (useful if your videos have consistent magnitude scaling)

If too few valid flow frames exist, the code falls back to FLOW_BLINK_THR_FALLBACK.

Outputs
-------
For each video:
  - Annotated video with ROI box + magnitude + threshold + blink label
  - Magnitude plot over time with GT / prediction overlap shading
  - Timeline plot (GT / Pred / Perf)
  - Row-normalized 2x2 confusion matrix
  - Per-video precision / recall / F1 / accuracy

Ground truth parsing
--------------------
Only AU47 and AU145 (optionally with L/R suffix) are used as blink labels.
All GT events are converted into per-frame binary labels:
  0 = no blink
  1 = blink
Overlaps remain blink (still binary).
"""

# =========================
# CONFIG
# =========================
DET_CONF = 0.25
DET_IOU = 0.1
FONT = cv.FONT_HERSHEY_SIMPLEX

USE_VERTICAL_FLOW = False

# fallback if threshold can't be computed
FLOW_BLINK_THR_FALLBACK = 0.8 if USE_VERTICAL_FLOW else 1.2

# PAdding
PADDING_FRAMES_AFTER = 6  # extend blink after last above-threshold frame
PADDING_FRAMES_BEFORE = 2 # extend blink before first above-threshold frame 

# Smoothing
CENTER_SMOOTHING = 0.75
BBOX_SMOOTHING = 0.5

SHOW = False

# =========================
# THRESHOLD OPTIONS
# =========================
THR_MODE = "iqr"          # "iqr" | "range_pct" | "baseline_plus"

# --- mode: iqr --- use baseline + spread * k
K_IQR = 2.8               # higher -> fewer detections
MIN_IQR = 1e-6

# --- mode: range_pct --- use range 
RANGE_PCT = 0.45         # 0.50 => halfway between min and max

# --- mode: baseline_plus ---
BASELINE_ADD = 0.80       # thr = median + this constant

if THR_MODE == "iqr":
    mode_variable = f"K{K_IQR*10}"
elif THR_MODE == "range_pct":
    mode_variable = f"R{int(RANGE_PCT*100)}"
elif THR_MODE == "baseline_plus":
    mode_variable = f"B{int(BASELINE_ADD*100)}"

# =========================
# PATHS
# =========================
Workspace_Path = Path(__file__).parent.parent.parent.resolve()
detection_eye_model_path = (
    Workspace_Path
    / "yolo_models"
    / "v1_yolo_half_100_epochs"
    / "y12n_3class_final_all"
    / "weights"
    / "yolo_best_half_E_100.pt"
)
GT_annotations_dir = (
    Workspace_Path
    / "Read_my_eyes"
    / "create_datasets"
    / "original_videos_annotations"
    / "JSONAnnotations"
)
ANNOTATIONS_JSON = GT_annotations_dir / "annotations.json"
videos_dir = (
    Workspace_Path
    / "Read_my_eyes"
    / "create_datasets"
    / "original_videos_annotations"
    / "videos"
)

outputs_dir = (
    Workspace_Path
    / "Read_my_eyes"
    / "Optical_flow"
    / "outputs_optical_flow"
    / f"thrMode_{THR_MODE}_{mode_variable}"
)
outputs_dir.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


# =========================
# TIME + GT PARSING
# =========================
def time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalize_code(code: str) -> Optional[str]:
    code = (code or "").strip().upper()
    m = re.match(r"^(AU47|AU145)([LR])?$", code)
    return m.group(1) if m else None


TARGETS = {"AU47", "AU145"}


def get_video_fps_and_frames(video_path: Path) -> Tuple[float, int]:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or nframes <= 0:
        raise RuntimeError(f"Bad fps/frames for {video_path}: fps={fps}, nframes={nframes}")
    return fps, nframes


def gt_events_to_frame_labels_binary(events: List[dict], fps: float, nframes: int) -> np.ndarray:
    y = np.zeros(nframes, dtype=np.uint8)

    for e in events:
        key = normalize_code(e.get("Code", ""))
        if not key or key not in TARGETS:
            continue

        st, et = e.get("Start time"), e.get("End time")
        if not st or not et:
            continue

        try:
            start_s = time_to_seconds(st)
            end_s = time_to_seconds(et)
        except Exception:
            continue

        if end_s <= start_s:
            continue

        s_idx = max(0, int(np.floor(start_s * fps)))
        e_idx = min(nframes, int(np.ceil(end_s * fps)))
        if e_idx <= s_idx:
            continue

        y[s_idx:e_idx] = 1

    return y


# =========================
# METRICS + PLOTTING (binary)
# =========================
def plot_magnitude_with_gt_and_pred(
    video_name: str,
    fps: float,
    mags: np.ndarray,
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    threshold: float,
    out_path: Path,
):
    mags = np.asarray(mags, dtype=np.float32)
    gt_binary = np.asarray(gt_binary, dtype=np.uint8)
    pred_binary = np.asarray(pred_binary, dtype=np.uint8)

    n = min(len(mags), len(gt_binary), len(pred_binary))
    mags = mags[:n]
    gt = (gt_binary[:n] != 0)
    pr = (pred_binary[:n] != 0)

    t = np.arange(n, dtype=np.float32) / float(fps)

    overlap = gt & pr
    gt_only = gt & ~pr
    pr_only = pr & ~gt

    def mask_to_spans(mask: np.ndarray) -> List[Tuple[float, float]]:
        spans = []
        if not np.any(mask):
            return spans
        idx = np.where(mask)[0]
        s = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                spans.append((s / fps, (prev + 1) / fps))
                s = i
                prev = i
        spans.append((s / fps, (prev + 1) / fps))
        return spans

    fig, ax = plt.subplots(figsize=(14, 4), dpi=200)
    ax.plot(t, mags, linewidth=1.2, label="Flow magnitude")

    ax.axhline(
        y=float(threshold),
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )

    for a, b in mask_to_spans(gt_only):
        ax.axvspan(a, b, color="purple", alpha=0.25, label="_nolegend_")
    for a, b in mask_to_spans(pr_only):
        ax.axvspan(a, b, color="red", alpha=0.25, label="_nolegend_")
    for a, b in mask_to_spans(overlap):
        ax.axvspan(a, b, color="green", alpha=0.30, label="_nolegend_")

    from matplotlib.patches import Patch
    span_legend = [
        Patch(facecolor="purple", alpha=0.25, label="GT blink only"),
        Patch(facecolor="red", alpha=0.25, label="OF blink only"),
        Patch(facecolor="green", alpha=0.30, label="Overlap (GT & OF)"),
    ]

    ax.set_title(f"{video_name} — Flow Magnitude vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + span_legend, labels + [p.get_label() for p in span_legend], loc="upper right", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def precision_recall_f1_binary(gt: np.ndarray, pr: np.ndarray) -> Dict[str, float]:
    gt = (gt.astype(np.uint8) != 0)
    pr = (pr.astype(np.uint8) != 0)
    tp = int(np.sum(gt & pr))
    fp = int(np.sum(~gt & pr))
    fn = int(np.sum(gt & ~pr))
    tn = int(np.sum(~gt & ~pr))
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def save_confusion_matrix_binary(
    labels: List[int],
    preds: List[int],
    output_path: Path,
    title: str,
    class_labels: Tuple[str, str] = ("No blink", "Blink"),
):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            pct = f"{cm_norm[i, j] * 100:0.1f}%"
            cnt = int(cm[i, j])
            ax.text(
                j, i, f"{pct}\n({cnt})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10, fontweight="bold"
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


# =========================
# TIMELINE PLOTTING (binary, 3 rows)
# =========================
def mask_to_segments(mask: np.ndarray, fps: float) -> List[Tuple[float, float]]:
    segs = []
    if not np.any(mask):
        return segs
    idx = np.where(mask)[0]
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((s / fps, (prev + 1 - s) / fps))
            s = i
            prev = i
    segs.append((s / fps, (prev + 1 - s) / fps))
    return segs


def plot_video_timeline_binary_3rows(video_name: str, fps: float, gt: np.ndarray, pr: np.ndarray, out_path: Path):
    total_dur = len(gt) / fps

    gt_b = (gt != 0)
    pr_b = (pr != 0)

    blink_any = gt_b | pr_b
    match = (gt_b == pr_b) & blink_any
    mismatch = (gt_b != pr_b) & blink_any

    gt_segs = mask_to_segments(gt_b, fps)
    pr_segs = mask_to_segments(pr_b, fps)
    ok_segs = mask_to_segments(match, fps)
    bad_segs = mask_to_segments(mismatch, fps)

    fig, ax = plt.subplots(figsize=(14, 3.0))
    ax.set_title(video_name)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, total_dur)

    y_gt, y_pr, y_pf = 2, 1, 0
    h = 0.6

    ax.broken_barh(gt_segs, (y_gt - h / 2, h), facecolors="tab:blue", label="Blink (GT)")
    ax.broken_barh(pr_segs, (y_pr - h / 2, h), facecolors="tab:orange", label="Blink (Pred)")
    ax.broken_barh(ok_segs, (y_pf - h / 2, h), facecolors="tab:green", label="Match")
    ax.broken_barh(bad_segs, (y_pf - h / 2, h), facecolors="tab:red", label="Mismatch")

    ax.set_yticks([y_gt, y_pr, y_pf])
    ax.set_yticklabels(["GT", "Pred", "Perf"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if SHOW:
        plt.show()
    plt.close(fig)


# =========================
# OPTICAL FLOW + STABILISED CROP HELPERS
# =========================
def compute_flow(gray1, gray2):
    return cv.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=5, iterations=3,
        poly_n=10, poly_sigma=1.2, flags=0
    )


def mean_vertical_flow_mag(flow) -> float:
    return float(np.mean(np.abs(flow[..., 1])))


def mean_full_flow_mag(flow) -> float:
    dx = flow[..., 0]
    dy = flow[..., 1]
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def mean_signed_dy(flow) -> float:
    return float(np.mean(flow[..., 1]))


def clamp_xyxy(x1, y1, x2, y2, W, H):
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
    x, y = org
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    pad = 4
    cv.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + baseline), (0, 0, 0), -1)
    cv.putText(img, text, (x + pad, y - pad), FONT, scale, color, thickness, cv.LINE_AA)


def make_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv.VideoWriter:
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
# BOX SELECTION (closest to last used center)
# =========================
def pick_box_first_conf_then_closest(
    xyxys: np.ndarray,
    confs: np.ndarray,
    prev_center: Optional[Tuple[float, float]],
) -> int:
    if xyxys.shape[0] == 0:
        raise ValueError("No boxes")
    if prev_center is None:
        return int(np.argmax(confs))
    pcx, pcy = prev_center
    centers = (xyxys[:, 0:2] + xyxys[:, 2:4]) * 0.5
    dx = centers[:, 0] - pcx
    dy = centers[:, 1] - pcy
    d2 = dx * dx + dy * dy
    return int(np.argmin(d2))


# =========================
# 1D GROUPING HELPERS
# =========================
@dataclass
class Group1D:
    start: int
    end: int
    mean: float
    p95: float


def _initial_groups_1d(vals, tol=0.10):
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
    def gmean(g):
        s, e = g
        return float(np.mean(vals[s : e + 1]))

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
        next_g = groups[i + 1] if i + 1 < len(groups) else None

        if prev_g is None and next_g is None:
            merged.append(g)
            i += 1
            continue

        if prev_g is None:
            groups[i + 1] = (g[0], next_g[1])
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
            groups[i + 1] = (g[0], next_g[1])
        i += 1

    return merged


def _finalize_groups_1d(groups, vals):
    out = []
    for (s, e) in groups:
        seg = np.asarray(vals[s : e + 1], dtype=np.float32)
        out.append(Group1D(start=s, end=e, mean=float(np.mean(seg)), p95=float(np.percentile(seg, 95))))
    return out


def compute_grouped_sizes_separate(ws, hs, tol=0.10, min_len=5):
    ws = np.asarray(ws, dtype=np.float32)
    hs = np.asarray(hs, dtype=np.float32)
    n = len(ws)

    w0 = _initial_groups_1d(ws, tol=tol)
    w1 = _merge_short_groups_1d(w0, ws, min_len=min_len)
    w_groups = _finalize_groups_1d(w1, ws)

    clean_w = np.zeros(n, dtype=np.float32)
    for g in w_groups:
        clean_w[g.start : g.end + 1] = g.p95

    h0 = _initial_groups_1d(hs, tol=tol)
    h1 = _merge_short_groups_1d(h0, hs, min_len=min_len)
    h_groups = _finalize_groups_1d(h1, hs)

    clean_h = np.zeros(n, dtype=np.float32)
    for g in h_groups:
        clean_h[g.start : g.end + 1] = g.p95

    return clean_w.tolist(), clean_h.tolist()


# =========================
# ROBUST THRESHOLD + POST-PROCESSING
# =========================
def compute_threshold(mags: np.ndarray, mode: str = THR_MODE) -> float:
    """
    Compute per-video threshold from mags using THR_MODE.

    mags: float array, may contain many zeros (frames where flow wasn't computed)
    We compute stats only on mags>0.
    """
    mags = np.asarray(mags, dtype=np.float32)
    valid = mags[mags > 1e-12]

    # fallback if not enough valid flow
    if valid.size < 10:
        return float(FLOW_BLINK_THR_FALLBACK)  # your old fixed threshold as fallback

    if mode == "iqr":
        p25 = float(np.percentile(valid, 25))
        p50 = float(np.percentile(valid, 50))
        p75 = float(np.percentile(valid, 75))
        iqr = max(p75 - p25, float(MIN_IQR))
        return float(p50 + K_IQR * iqr)

    elif mode == "range_pct":
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        rng = vmax - vmin
        if rng <= 1e-12:
            return float("inf")  # basically "no spikes" => no blinks
        return float(vmin + RANGE_PCT * rng)

    elif mode == "baseline_plus":
        baseline = float(np.median(valid))
        return float(baseline + float(BASELINE_ADD))

    else:
        raise ValueError(f"Unknown THR_MODE={mode!r}. Use 'iqr', 'range_pct', or 'baseline_plus'.")



def apply_padding(mask: np.ndarray, pad_after: int, pad_before: int) -> np.ndarray:
    """
    For every True frame, extend True forward by pad_after frames and backward by pad_before frames.
     - If pad_after <= 0, no forward padding.
     - If pad_before <= 0, no backward padding.
    """
    if pad_after <= 0 and pad_before <= 0:
        return mask.astype(np.uint8)
    n = len(mask)
    out = mask.copy()
    idx = np.where(mask)[0]
    for i in idx:
        j = min(n, i + pad_after + 1)
        out[i:j] = True
        if pad_before > 0:
            k = max(0, i - pad_before)
            out[k:i] = True
    return out.astype(np.uint8)

# =========================
# PROCESS ONE VIDEO
# =========================
def process_one_video_optflow_binary(
    detection_model: YOLO,
    video_path: Path,
    out_video_path: Path,
) -> Tuple[np.ndarray, float, int, np.ndarray, float]:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv.CAP_PROP_FPS) or 25.0)

    frames = []
    centers = []
    sizes = []
    det_ws, det_hs = [], []

    prev_center: Optional[Tuple[float, float]] = None

    # PASS 1: frames + detect eye box per frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        det_results = detection_model.predict(source=frame, conf=DET_CONF, iou=DET_IOU, save=False, verbose=False)
        r = det_results[0]
        boxes = getattr(r, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy().astype(np.float32)
            xyxys = boxes.xyxy.cpu().numpy().astype(np.float32)

            pick = pick_box_first_conf_then_closest(xyxys, confs, prev_center)
            xyxy = xyxys[pick]

            x1, y1, x2, y2 = clamp_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H)
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = y2 - 0.5 * h  # bottom-anchored center

            prev_center = (cx, cy)
            centers.append((cx, cy))
            sizes.append((w, h))
            det_ws.append(w)
            det_hs.append(h)
        else:
            centers.append(None)
            sizes.append(None)
            prev_center = None

    cap.release()
    nframes = len(frames)
    if nframes == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    if len(det_ws) == 0 or len(det_hs) == 0:
        writer = make_video_writer(out_video_path, fps, (W, H))
        for f in frames:
            writer.write(f)
        writer.release()
        mags_np = np.zeros(nframes, dtype=np.float32)
        thr = float(FLOW_BLINK_THR_FALLBACK)
        pred = np.zeros(nframes, dtype=np.uint8)
        return pred, fps, nframes, mags_np, thr

    # size cleaning + grouping
    p5_w = float(np.percentile(det_ws, 5))
    p95_w = float(np.percentile(det_ws, 95))
    p5_h = float(np.percentile(det_hs, 5))
    p95_h = float(np.percentile(det_hs, 95))

    raw_ws = [s[0] if s is not None else None for s in sizes]
    raw_hs = [s[1] if s is not None else None for s in sizes]
    clean_ws = replace_spikes_interp(raw_ws, p5_w, p95_w)
    clean_hs = replace_spikes_interp(raw_hs, p5_h, p95_h)

    valid_mask = [(w is not None and h is not None) for w, h in zip(clean_ws, clean_hs)]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]

    if len(valid_indices) == 0:
        writer = make_video_writer(out_video_path, fps, (W, H))
        for f in frames:
            writer.write(f)
        writer.release()
        mags_np = np.zeros(nframes, dtype=np.float32)
        thr = float(FLOW_BLINK_THR_FALLBACK)
        pred = np.zeros(nframes, dtype=np.uint8)
        return pred, fps, nframes, mags_np, thr

    v_ws = np.array([clean_ws[i] for i in valid_indices], dtype=np.float32)
    v_hs = np.array([clean_hs[i] for i in valid_indices], dtype=np.float32)
    g_ws, g_hs = compute_grouped_sizes_separate(v_ws, v_hs)

    grouped_ws_full = [None] * nframes
    grouped_hs_full = [None] * nframes
    for k, i in enumerate(valid_indices):
        grouped_ws_full[i] = g_ws[k]
        grouped_hs_full[i] = g_hs[k]

    clean_sizes = [(w, h) if (w is not None and h is not None) else None for w, h in zip(grouped_ws_full, grouped_hs_full)]
    smooth_sizes_list = smooth_sizes(clean_sizes, BBOX_SMOOTHING)
    smooth_centers_list = smooth_centers(centers, CENTER_SMOOTHING)

    FLOW_W_FIX = max(2, int(round(p95_w)))
    FLOW_H_FIX = max(2, int(round(p95_h)))

    mags = np.zeros(nframes, dtype=np.float32)
    dys = np.zeros(nframes, dtype=np.float32)
    prev_crop_gray = None

    for idx, (frame, center, size) in enumerate(zip(frames, smooth_centers_list, smooth_sizes_list)):
        if center is None or size is None:
            prev_crop_gray = None
            continue

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
        if eye_crop.size <= 0:
            prev_crop_gray = None
            continue

        eye_gray = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)
        eye_gray_fix = cv.resize(eye_gray, (FLOW_W_FIX, FLOW_H_FIX), interpolation=cv.INTER_LINEAR)

        if prev_crop_gray is not None:
            flow = compute_flow(prev_crop_gray, eye_gray_fix)
            mags[idx] = mean_vertical_flow_mag(flow) if USE_VERTICAL_FLOW else mean_full_flow_mag(flow)
            dys[idx] = mean_signed_dy(flow)

        prev_crop_gray = eye_gray_fix

    thr_video = compute_threshold(mags, mode=THR_MODE)

    # --- threshold-only prediction ---
    above = mags >= thr_video
    pred = apply_padding(above, pad_after=PADDING_FRAMES_AFTER, pad_before=PADDING_FRAMES_BEFORE)

    # annotated video
    writer = make_video_writer(out_video_path, fps, (W, H))
    for idx, (frame, center, size) in enumerate(zip(frames, smooth_centers_list, smooth_sizes_list)):
        annotated = frame.copy()
        mag = float(mags[idx])
        dy = float(dys[idx])
        is_blink = int(pred[idx]) == 1

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

            if is_blink:
                box_color = (0, 255, 0)
                status = "BLINK"
            else:
                box_color = (0, 255, 255)
                status = "NONE"

            cv.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"{status} | mag:{mag:.2f} thr:{thr_video:.2f} dy:{dy:+.3f}"
            label_y = max(y1 - 10, 20)
            nice_label(annotated, label_text, (x1, label_y), box_color)

        writer.write(annotated)

    writer.release()
    print(f"[saved] {out_video_path.name} | thr={thr_video:.3f}")

    return pred, fps, nframes, mags, float(thr_video)


# =========================
# MAIN
# =========================
def main():
    if not ANNOTATIONS_JSON.exists():
        raise FileNotFoundError(f"Missing: {ANNOTATIONS_JSON}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")
    if not detection_eye_model_path.exists():
        raise FileNotFoundError(f"Missing eye model: {detection_eye_model_path}")

    data = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must map video_name -> list of events.")

    detection_model = YOLO(str(detection_eye_model_path))

    print(f"[INFO] Videos: {videos_dir}")
    print(f"[INFO] Out: {outputs_dir}")
    print(f"[INFO] USE_VERTICAL_FLOW={USE_VERTICAL_FLOW}")
    print(f"[INFO] Threshold: median + {K_IQR}*IQR | padding_after={PADDING_FRAMES_AFTER} ")

    summary = {}
    all_gt_frames = []
    all_pr_frames = []

    for video_name, events in data.items():
        if not isinstance(events, list):
            continue

        video_path = videos_dir / video_name
        if not video_path.exists():
            alt = videos_dir / Path(video_name).name
            if alt.exists():
                video_path = alt
            else:
                print(f"[skip] video not found: {video_name}")
                continue

        fps, nframes = get_video_fps_and_frames(video_path)
        gt = gt_events_to_frame_labels_binary(events, fps, nframes)

        out_video = outputs_dir / "annotated_videos" / f"{video_path.stem}_annotated_optflow.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)

        pr, _, _, mags, thr_video = process_one_video_optflow_binary(
            detection_model=detection_model,
            video_path=video_path,
            out_video_path=out_video,
        )

        n = min(len(gt), len(pr))
        gt2 = gt[:n].astype(np.uint8)
        pr2 = pr[:n].astype(np.uint8)

        stats = precision_recall_f1_binary(gt2, pr2)

        out_timeline = outputs_dir / "timelines" / f"{video_path.stem}_timeline_GT_Pred_Perf.png"
        plot_video_timeline_binary_3rows(video_path.name, fps, gt2, pr2, out_timeline)

        out_cm = outputs_dir / "confusion_matrices" / f"{video_path.stem}_cm_2x2_row_norm.png"
        save_confusion_matrix_binary(
            labels=gt2.tolist(),
            preds=pr2.tolist(),
            output_path=out_cm,
            title=f"{video_path.name} (thr={thr_video:.3f})",
        )

        out_mag_plot = outputs_dir / "magnitude_plots" / f"{video_path.stem}_magnitude_with_GT.png"
        plot_magnitude_with_gt_and_pred(
            video_name=video_path.name,
            fps=fps,
            mags=mags,
            gt_binary=gt2,
            pred_binary=pr2,
            threshold=thr_video,
            out_path=out_mag_plot,
        )

        summary[video_path.name] = {
            "fps": fps,
            "nframes_eval": int(n),
            "duration_s": float(n / fps),
            "settings": {
                "det_conf": DET_CONF,
                "det_iou": DET_IOU,
                "use_vertical_flow": USE_VERTICAL_FLOW,
                "thr_mode": f"median+{K_IQR}*IQR",
                "thr_value": float(thr_video),
                "padding_frames_after": int(PADDING_FRAMES_AFTER),
                "center_smoothing": CENTER_SMOOTHING,
                "bbox_smoothing": BBOX_SMOOTHING,
                "eye_model": str(detection_eye_model_path),
            },
            "frame_level_metrics": stats,
            "timeline_png": str(out_timeline),
            "cm_2x2_png": str(out_cm),
            "annotated_video": str(out_video),
            "magnitude_plot_png": str(out_mag_plot),
        }

        all_gt_frames.append(gt2)
        all_pr_frames.append(pr2)

        print(
            f"[done] {video_path.name} | thr={thr_video:.3f} | "
            f"P/R/F1: {stats['precision']:.3f}/{stats['recall']:.3f}/{stats['f1']:.3f}"
        )

    if all_gt_frames:
        gt_all = np.concatenate(all_gt_frames, axis=0)
        pr_all = np.concatenate(all_pr_frames, axis=0)
        overall = precision_recall_f1_binary(gt_all, pr_all)

        out_cm_all = outputs_dir / "confusion_matrices" / "ALL_videos_cm_2x2_row_norm.png"
        save_confusion_matrix_binary(
            labels=gt_all.tolist(),
            preds=pr_all.tolist(),
            output_path=out_cm_all,
            title=f"ALL videos (thr=per-video median+{K_IQR}*IQR)",
        )

        summary["_OVERALL_"] = {
            "frame_level_metrics": overall,
            "cm_2x2_png": str(out_cm_all),
            "nframes_total": int(len(gt_all)),
        }

        print("\n[OVERALL]")
        print(f"  Precision: {overall['precision']:.3f}")
        print(f"  Recall:    {overall['recall']:.3f}")
        print(f"  F1:        {overall['f1']:.3f}")
        print(f"  Accuracy:  {overall['accuracy']:.3f}")

    out_json = outputs_dir / "summary_metrics_optflow_BINARY.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
