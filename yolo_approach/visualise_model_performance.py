import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.colors import LogNorm

"""
This script visualise YOLO performance on videos with GT annotations, producing:
- Per-video timeline plots showing GT, Pred, and Perf (match/mismatch)
- Per-video confusion matrices (3x3: None/Half/Full)
- Summary JSON with metrics and settings for each video

Model rules:
- No overlap allowed: if 2+ boxes overlap in any way, only the most confident is kept (class-agnostic)
- Max 2 bbx are kept (most confident ones). done after overlap bbx are removed. 
- All coherent blink frames are seen as the same label, with FULL dominating HALF if both exist in the same episode.
- Class mapping: YOLO classes mapped to NONE/HALF/FULL as configured below, everything else => NONE
"""

# ---------------- CONFIG ----------------
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()

INFO_VIDEOS_DIR = WORKSPACE_ROOT / "create_datasets" / "original_videos_annotations"
ANNOTATIONS_JSON = INFO_VIDEOS_DIR / "JSONAnnotations" / "annotations.json"
VIDEOS_DIR = INFO_VIDEOS_DIR / "videos"

OUT_DIR = WORKSPACE_ROOT / "yolo_approach" / "visualisation_videos_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- EDIT THIS ---
WEIGHTS = WORKSPACE_ROOT.parent.resolve() / "yolo_half_100_epochs" /"runs"/ "blink_detection" / "y12n_3class_final_all" / "weights" / "yolo_best_half_E_100.pt"

# YOLO infer settings
CONF_THRES = 0.25
IOU_FOR_MODEL_NMS = 0.1
IMG_SIZE = 896

# Never allow more than this many boxes per frame after all filtering
MAX_BOXES_PER_FRAME = 2

# Toggle: apply "connected episode FULL dominates" to PREDICTIONS only
PROMOTE_CONNECTED_PREDS = True
CONNECTED_GAP_FRAMES = 0  

# --- EDIT THESE TO MATCH YOUR YOLO CLASSES ---
YOLO_HALF_CLS = 1
YOLO_FULL_CLS = 2

SHOW = False


# ---------------- Label encoding ----------------
LBL_NONE = 0
LBL_HALF = 1
LBL_FULL = 2


# ---------------- HELPERS: time + GT parsing ----------------
def time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def normalize_code(code: str) -> str | None:
    code = (code or "").strip().upper()
    m = re.match(r"^(AU47|AU145)([LR])?$", code)
    return m.group(1) if m else None

TARGETS = {"AU47": LBL_HALF, "AU145": LBL_FULL}

def get_video_fps_and_frames(video_path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or nframes <= 0:
        raise RuntimeError(f"Bad fps/frames for {video_path}: fps={fps}, nframes={nframes}")
    return float(fps), int(nframes)

def gt_events_to_frame_labels(events: List[dict], fps: float, nframes: int) -> np.ndarray:
    """
    GT is NOT post-processed (no episode promotion).
    If GT annotation overlaps exist, FULL overwrites HALF in those frames.
    """
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

        lbl = TARGETS[key]
        if lbl == LBL_FULL:
            y[s_idx:e_idx] = LBL_FULL
        else:
            y[s_idx:e_idx] = np.maximum(y[s_idx:e_idx], LBL_HALF)

    return y


# ---------------- YOLO mapping ----------------
def yolo_cls_to_label(cls_idx: int) -> int:
    if cls_idx == YOLO_HALF_CLS:
        return LBL_HALF
    if cls_idx == YOLO_FULL_CLS:
        return LBL_FULL
    return LBL_NONE  # eyes + everything else


# ---------------- Overlap suppression + cap ----------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def keep_only_nonoverlapping_most_conf(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    max_boxes: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1 (your rule): class-agnostic suppression:
      - sort by confidence desc
      - keep a box only if it has IoU == 0 with all kept boxes
    Step 2 (new): cap total kept boxes to max_boxes (default=2),
      keeping the max_boxes most confident among the kept set.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, confs, clss

    order = np.argsort(-confs)
    kept = []
    for idx in order:
        b = boxes_xyxy[idx]
        if all(iou_xyxy(b, boxes_xyxy[k]) == 0.0 for k in kept):
            kept.append(idx)

    if not kept:
        return boxes_xyxy[:0], confs[:0], clss[:0]

    kept = np.array(kept, dtype=int)

    # CAP to max_boxes by confidence (kept is already in descending conf order, but re-apply to be safe)
    if len(kept) > max_boxes:
        kept_order = np.argsort(-confs[kept])
        kept = kept[kept_order[:max_boxes]]

    return boxes_xyxy[kept], confs[kept], clss[kept]


# ---------------- Connected episode promotion (pred only) ----------------
def promote_connected_episodes(labels: np.ndarray, gap_frames: int = 1) -> np.ndarray:
    n = len(labels)
    out = labels.copy()

    blink = (labels != LBL_NONE)
    if not np.any(blink):
        return out

    idxs = np.where(blink)[0]
    start = idxs[0]
    prev = idxs[0]
    episodes = []
    for i in idxs[1:]:
        if i - prev <= gap_frames + 1:
            prev = i
        else:
            episodes.append((start, prev))
            start = i
            prev = i
    episodes.append((start, prev))

    for s, e in episodes:
        s2 = max(0, s - gap_frames)
        e2 = min(n - 1, e + gap_frames)
        segment = out[s2:e2 + 1]
        if np.any(segment == LBL_FULL):
            out[s2:e2 + 1] = LBL_FULL
        else:
            out[s2:e2 + 1] = LBL_HALF

    return out


# ---------------- YOLO → per-frame labels ----------------
def yolo_video_to_frame_labels(model: YOLO, video_path: Path, nframes: int) -> np.ndarray:
    pred = np.zeros(nframes, dtype=np.uint8)

    results = model.predict(
        source=str(video_path),
        stream=True,
        conf=CONF_THRES,
        iou=IOU_FOR_MODEL_NMS,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    for fi, r in enumerate(results):
        if fi >= nframes:
            break
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        boxes, confs, clss = keep_only_nonoverlapping_most_conf(
            boxes, confs, clss, max_boxes=MAX_BOXES_PER_FRAME
        )

        if len(clss) == 0:
            continue

        mapped = np.array([yolo_cls_to_label(int(c)) for c in clss], dtype=np.uint8)

        # If multiple remaining disjoint boxes: FULL dominates; else HALF; else NONE
        if np.any(mapped == LBL_FULL):
            pred[fi] = LBL_FULL
        elif np.any(mapped == LBL_HALF):
            pred[fi] = LBL_HALF
        else:
            pred[fi] = LBL_NONE

    return pred


# ---------------- Metrics ----------------
def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0

def compute_confusion_3x3(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gt, pr):
        cm[int(g), int(p)] += 1
    return cm

def per_class_precision_recall_from_cm(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    labels = ["none", "half", "full"]
    out = {}
    precisions, recalls = [], []
    for c, name in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        out[name] = {"precision": prec, "recall": rec}
        precisions.append(prec)
        recalls.append(rec)
    out["macro"] = {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls))}
    return out

def blink_detection_precision_recall(gt: np.ndarray, pr: np.ndarray) -> Dict[str, float]:
    gt_b = (gt != LBL_NONE)
    pr_b = (pr != LBL_NONE)
    tp = int(np.sum(gt_b & pr_b))
    fp = int(np.sum(~gt_b & pr_b))
    fn = int(np.sum(gt_b & ~pr_b))
    return {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

def plot_confusion_matrix_3x3(
    cm: np.ndarray,
    out_path_counts: Path,
    out_path_norm: Path,
    title_base: str,
    labels=("None", "Half", "Full"),
):
    """
    Saves
      Row-normalized (%), so you can read performance per GT class
    """

    cm = np.asarray(cm, dtype=float)

    # ----------  Row-normalized (percent) ----------
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use("seaborn-v0_8-white")
    im = ax.imshow(cm_row, vmin=0, vmax=1, cmap="Reds")
    ax.set_title(f"{title_base}\nRow-normalized (GT → %)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)

    for i in range(3):
        for j in range(3):
            pct = cm_row[i, j] * 100.0
            cnt = int(cm[i, j])

            # better contrast rule
            txt_color = "white" if cm_row[i, j] > 0.5 else "black"

            ax.text(j, i,
                    f"{pct:5.1f}%\n({cnt:,})",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=9,
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within GT row")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path_norm, dpi=250)
    plt.close(fig)


# ---------------- Timeline plotting ----------------
def labels_to_segments(labels: np.ndarray, fps: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    half, full = [], []
    n = len(labels)

    def flush(run_lbl, s, e):
        if run_lbl == LBL_HALF:
            half.append((s / fps, (e - s) / fps))
        elif run_lbl == LBL_FULL:
            full.append((s / fps, (e - s) / fps))

    run_lbl = labels[0]
    run_s = 0
    for i in range(1, n):
        if labels[i] != run_lbl:
            flush(run_lbl, run_s, i)
            run_lbl = labels[i]
            run_s = i
    flush(run_lbl, run_s, n)
    return half, full

def perf_segments(gt: np.ndarray, pr: np.ndarray, fps: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    blink_any = (gt != LBL_NONE) | (pr != LBL_NONE)
    match = (gt == pr) & blink_any
    mismatch = (gt != pr) & blink_any

    def mask_to_segments(mask: np.ndarray):
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

    return mask_to_segments(match), mask_to_segments(mismatch)

def plot_video_timeline_3rows(video_name: str, fps: float, gt: np.ndarray, pr: np.ndarray, out_path: Path):
    total_dur = len(gt) / fps

    gt_half, gt_full = labels_to_segments(gt, fps)
    pr_half, pr_full = labels_to_segments(pr, fps)
    ok_segs, bad_segs = perf_segments(gt, pr, fps)

    fig, ax = plt.subplots(figsize=(14, 3.0))
    ax.set_title(video_name)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, total_dur)

    y_gt, y_pr, y_pf = 2, 1, 0
    h = 0.6

    ax.broken_barh(gt_half, (y_gt - h/2, h), facecolors="tab:orange", label="Half")
    ax.broken_barh(gt_full, (y_gt - h/2, h), facecolors="tab:blue",   label="Full")

    ax.broken_barh(pr_half, (y_pr - h/2, h), facecolors="tab:orange")
    ax.broken_barh(pr_full, (y_pr - h/2, h), facecolors="tab:blue")

    ax.broken_barh(ok_segs,  (y_pf - h/2, h), facecolors="tab:green", label="Match")
    ax.broken_barh(bad_segs, (y_pf - h/2, h), facecolors="tab:red",   label="Mismatch")

    ax.set_yticks([y_gt, y_pr, y_pf])
    ax.set_yticklabels(["GT", "Pred", "Perf"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if SHOW:
        plt.show()
    plt.close(fig)


# ---------------- MAIN ----------------
def main():
    if not ANNOTATIONS_JSON.exists():
        raise FileNotFoundError(f"Missing: {ANNOTATIONS_JSON}")
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Missing: {VIDEOS_DIR}")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")

    data = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must map video_name -> list of events.")

    model = YOLO(str(WEIGHTS))
    print(f"[INFO] model.names: {getattr(model, 'names', None)}")
    print(f"[INFO] HALF_CLS={YOLO_HALF_CLS}, FULL_CLS={YOLO_FULL_CLS}, MAX_BOXES_PER_FRAME={MAX_BOXES_PER_FRAME}")
    print(f"[INFO] Videos dir: {VIDEOS_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")

    summary = {}

    for video_name, events in data.items():
        if not isinstance(events, list):
            continue

        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            alt = VIDEOS_DIR / Path(video_name).name
            if alt.exists():
                video_path = alt
            else:
                print(f"[skip] video not found: {video_name}")
                continue

        fps, nframes = get_video_fps_and_frames(video_path)

        gt = gt_events_to_frame_labels(events, fps, nframes)  # GT untouched
        pr = yolo_video_to_frame_labels(model, video_path, nframes)

        if PROMOTE_CONNECTED_PREDS:
            pr2 = promote_connected_episodes(pr, gap_frames=CONNECTED_GAP_FRAMES)
        else:
            pr2 = pr

        cm3 = compute_confusion_3x3(gt, pr2)
        pr_stats = per_class_precision_recall_from_cm(cm3)
        blink_stats = blink_detection_precision_recall(gt, pr2)

        out_timeline = OUT_DIR / f"{video_path.stem}_timeline_GT_Pred_Perf.png"
        plot_video_timeline_3rows(video_path.name, fps, gt, pr2, out_timeline)

        out_cm_counts = OUT_DIR / f"{video_path.stem}_confusion_counts_log.png"
        out_cm_norm   = OUT_DIR / f"{video_path.stem}_confusion_row_norm.png"

        plot_confusion_matrix_3x3(
            cm3,
            out_path_counts=out_cm_counts,
            out_path_norm=out_cm_norm,
            title_base=f"{video_path.name}",
        )

        summary[video_path.name] = {
            "fps": fps,
            "nframes": nframes,
            "duration_s": nframes / fps,
            "settings": {
                "conf_thres": CONF_THRES,
                "model_iou": IOU_FOR_MODEL_NMS,
                "imgsz": IMG_SIZE,
                "max_boxes_per_frame": MAX_BOXES_PER_FRAME,
                "promote_connected_preds": PROMOTE_CONNECTED_PREDS,
                "connected_gap_frames": CONNECTED_GAP_FRAMES,
                "yolo_half_cls": YOLO_HALF_CLS,
                "yolo_full_cls": YOLO_FULL_CLS,
                "weights": str(WEIGHTS),
            },
            "confusion_3x3_rows_GT_cols_Pred": cm3.tolist(),
            "per_class_precision_recall": pr_stats,
            "blink_detection_precision_recall": blink_stats,
            "timeline_png": str(out_timeline),
            "confusion_counts_png": str(out_cm_counts),
            "confusion_row_norm_png": str(out_cm_norm),

        }

        print(f"[done] {video_path.name}")
        print(f"       blink P/R: {blink_stats['precision']:.3f} / {blink_stats['recall']:.3f}")
        print(f"       macro P/R: {pr_stats['macro']['precision']:.3f} / {pr_stats['macro']['recall']:.3f}")

    out_json = OUT_DIR / "summary_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_json}")

if __name__ == "__main__":
    main()
