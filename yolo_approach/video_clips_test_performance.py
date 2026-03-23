import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

"""
This script visualise video-based YOLO performance on video clips, producing:
- Per-video confusion matrices (3x3: None/Half/Full)
- Summary JSON with metrics and settings for each video 
- Overall performance metrics across the test set.
- Overall confusion matrix across the test set.

Model rules:
- if only one frame is predicted as "half-blink" in an episode, the whole episode is labeled as "half-blink"
- if at least one frame is predicted as "full-blink" in an episode, the whole episode is labeled as "full-blink"
- full blink dominates half-blink, meaning that if there is a mix of half and full predictions in an episode, the episode is labeled as full-blink.

Video clips distribution in the test set:
- full-blink: 54 clips
- half-blink: 61 clips
- none: 310 clips
"""

# ---------------- CONFIG ----------------
CONF_THRES = 0.25
IOU_FOR_MODEL_NMS = 0.1
IMG_SIZE = 1120

MAX_BOXES_PER_FRAME = 2

# --- YOLO CLASSES ---
YOLO_HALF_CLS = 1
YOLO_FULL_CLS = 2

# ---------------- Label encoding ----------------
LBL_NONE = 0
LBL_HALF = 1
LBL_FULL = 2

LABEL_TO_NAME = {
    LBL_NONE: "none",
    LBL_HALF: "half",
    LBL_FULL: "full",
}

NAME_TO_LABEL = {
    "none": LBL_NONE,
    "half": LBL_HALF,
    "full": LBL_FULL,
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}

#  ---------------- PATHS ----------------
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
VIDEOS_DIR = WORKSPACE_ROOT / "create_datasets" / "datasets" / "test_clips"
OUT_DIR = WORKSPACE_ROOT / "yolo_approach" / "test_results" / "test_result_video_based"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS = WORKSPACE_ROOT.parent.resolve() / "Read_my_eyes" / "yolo_approach" / "blink_detector_Ep20.pt"

# ---------------- Code ----------------
@dataclass
class VideoResult:
    video_path: str
    folder_name: str
    ground_truth_label: int
    ground_truth_name: str
    predicted_label: int
    predicted_name: str
    num_frames: int
    detected_half_frames: int
    detected_full_frames: int
    correct: bool


def normalize_folder_label(folder_name: str) -> int:
    key = folder_name.strip().lower()
    if key not in NAME_TO_LABEL:
        raise ValueError(
            f"Unknown folder label '{folder_name}'. "
            f"Expected one of: {list(NAME_TO_LABEL.keys())}"
        )
    return NAME_TO_LABEL[key]


def get_video_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )


# ---------------- Overlap suppression + cap ----------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

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
    max_boxes: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1: class-agnostic suppression
      - sort by confidence descending
      - keep a box only if it has IoU == 0 with all kept boxes
    Step 2: cap total kept boxes to max_boxes, keeping most confident
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

    if len(kept) > max_boxes:
        kept_order = np.argsort(-confs[kept])
        kept = kept[kept_order[:max_boxes]]

    return boxes_xyxy[kept], confs[kept], clss[kept]


def infer_video_label(model: YOLO, video_path: Path) -> Tuple[int, int, int, int]:
    """
    Stream-based inference aligned with frame-based preprocessing.

    Returns:
        predicted_video_label, num_frames, detected_half_frames, detected_full_frames
    """
    detected_half_frames = 0
    detected_full_frames = 0
    num_frames = 0

    results = model.predict(
        source=str(video_path),
        stream=True,
        conf=CONF_THRES,
        iou=IOU_FOR_MODEL_NMS,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    for r in results:
        num_frames += 1

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

        best_half = 0.0
        best_full = 0.0

        for c, cf in zip(clss, confs):
            if int(c) == YOLO_HALF_CLS:
                best_half = max(best_half, float(cf))
            elif int(c) == YOLO_FULL_CLS:
                best_full = max(best_full, float(cf))

        if best_full > 0:
            detected_full_frames += 1
        elif best_half > 0:
            detected_half_frames += 1

    if num_frames == 0:
        raise RuntimeError(f"No readable frames found in video: {video_path}")

    if detected_full_frames > 0:
        pred_label = LBL_FULL
    elif detected_half_frames > 0:
        pred_label = LBL_HALF
    else:
        pred_label = LBL_NONE

    return pred_label, num_frames, detected_half_frames, detected_full_frames


# ---------------- Metrics ----------------
def compute_confusion_3x3(y_true: List[int], y_pred: List[int], num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def blink_any_precision_recall_f1_video(video_results: List[VideoResult]) -> Dict[str, float]:
    gt_b = np.array([vr.ground_truth_label != LBL_NONE for vr in video_results], dtype=bool)
    pr_b = np.array([vr.predicted_label != LBL_NONE for vr in video_results], dtype=bool)

    tp = int(np.sum(gt_b & pr_b))
    tn = int(np.sum(~gt_b & ~pr_b))
    fp = int(np.sum(~gt_b & pr_b))
    fn = int(np.sum(gt_b & ~pr_b))

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = f1_from_pr(prec, rec)
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def plot_confusion_matrix_3x3(
    cm: np.ndarray,
    out_path_norm: Path,
    title_base: str,
    labels=("None", "Half", "Full"),
):
    """
    Saves row-normalized (%) confusion matrix, so you can read performance per GT class.
    """
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use("seaborn-v0_8-white")
    im = ax.imshow(cm_row, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"{title_base}\nRow-normalized (GT → %)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)

    for i in range(3):
        for j in range(3):
            pct = cm_row[i, j] * 100.0
            cnt = int(cm[i, j])
            txt_color = "white" if cm_row[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{pct:5.1f}%\n({cnt:,})",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within GT row")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path_norm, dpi=250)
    plt.close(fig)

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def per_class_precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    labels = ["none", "half", "full"]
    out = {}

    for c, name in enumerate(labels):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())

        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = f1_from_pr(prec, rec)

        out[name] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return out


def compute_metrics_from_confusion(cm: np.ndarray) -> Dict:
    total = int(cm.sum())
    accuracy = safe_div(np.trace(cm), total)

    per_class = per_class_precision_recall_f1_from_cm(cm)

    macro_precision = float(np.mean([per_class[k]["precision"] for k in ["none", "half", "full"]]))
    macro_recall = float(np.mean([per_class[k]["recall"] for k in ["none", "half", "full"]]))
    macro_f1 = float(np.mean([per_class[k]["f1"] for k in ["none", "half", "full"]]))

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "num_samples": total,
    }


def evaluate_subset(video_results: List[VideoResult]) -> Dict:
    y_true = [vr.ground_truth_label for vr in video_results]
    y_pred = [vr.predicted_label for vr in video_results]

    cm = compute_confusion_3x3(y_true, y_pred, num_classes=3)
    metrics = compute_metrics_from_confusion(cm)

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "label_order": [LABEL_TO_NAME[0], LABEL_TO_NAME[1], LABEL_TO_NAME[2]],
            "matrix": cm.tolist(),
        },
        "cm_raw": cm,
        "num_videos": len(video_results),
    }


def main():
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {VIDEOS_DIR}")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))

    all_video_results: List[VideoResult] = []
    folder_results: Dict[str, List[VideoResult]] = {}
    skipped_videos: List[Dict[str, str]] = []
    corrupted_videos: List[Dict[str, str]] = []

    gt_folders = [p for p in VIDEOS_DIR.iterdir() if p.is_dir()]
    gt_folders = sorted(gt_folders, key=lambda p: p.name.lower())

    if not gt_folders:
        raise RuntimeError(f"No subfolders found in dataset directory: {VIDEOS_DIR}")

    print(f"[INFO] model.names: {getattr(model, 'names', None)}")
    print(f"[INFO] HALF_CLS={YOLO_HALF_CLS}, FULL_CLS={YOLO_FULL_CLS}, MAX_BOXES_PER_FRAME={MAX_BOXES_PER_FRAME}")
    print(f"[INFO] Videos dir: {VIDEOS_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")

    for folder in gt_folders:
        gt_label = normalize_folder_label(folder.name)
        videos = get_video_files(folder)

        print(f"\nProcessing folder: {folder.name} ({len(videos)} videos)")
        folder_results[folder.name] = []

        for video_path in videos:
            print(f"  -> {video_path.name}")

            try:
                pred_label, num_frames, detected_half_frames, detected_full_frames = infer_video_label(
                    model, video_path
                )

                result = VideoResult(
                    video_path=video_path.name,
                    folder_name=folder.name,
                    ground_truth_label=gt_label,
                    ground_truth_name=LABEL_TO_NAME[gt_label],
                    predicted_label=pred_label,
                    predicted_name=LABEL_TO_NAME[pred_label],
                    num_frames=num_frames,
                    detected_half_frames=detected_half_frames,
                    detected_full_frames=detected_full_frames,
                    correct=(gt_label == pred_label),
                )

                folder_results[folder.name].append(result)
                all_video_results.append(result)

            except Exception as e:
                error_msg = str(e)
                print(f"  [skip] {video_path.name}: {error_msg}")

                skipped_entry = {
                    "video_path": str(video_path),
                    "folder_name": folder.name,
                    "error": error_msg,
                }
                skipped_videos.append(skipped_entry)

                corrupted_videos.append({
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "folder_name": folder.name,
                    "error": error_msg,
                })

    per_folder_summary = {}
    for folder_name, results in folder_results.items():
        subset_eval = evaluate_subset(results) if results else {
            "metrics": {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "per_class": {
                    "none": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                    "half": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                    "full": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                },
                "num_samples": 0,
            },
            "confusion_matrix": {
                "label_order": [LABEL_TO_NAME[0], LABEL_TO_NAME[1], LABEL_TO_NAME[2]],
                "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            },
            "num_videos": 0,
        }

        per_folder_summary[folder_name] = {
            "ground_truth_folder_label": folder_name,
            "num_videos": subset_eval["num_videos"],
            "metrics": subset_eval["metrics"],
            "videos": [asdict(r) for r in results],
        }

    overall_eval = evaluate_subset(all_video_results) if all_video_results else {
        "metrics": {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {
                "none": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                "half": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                "full": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            },
            "num_samples": 0,
        },
        "confusion_matrix": {
            "label_order": [LABEL_TO_NAME[0], LABEL_TO_NAME[1], LABEL_TO_NAME[2]],
            "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        },
        "num_videos": 0,
    }

    overall_blink_vs_none = (
        blink_any_precision_recall_f1_video(all_video_results)
        if all_video_results
        else {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    )

    if all_video_results:
        overall_cm = overall_eval["cm_raw"]
        out_cm = OUT_DIR / "_OVERALL_confusion_matrix.png"
        plot_confusion_matrix_3x3(
            overall_cm,
            out_cm,
            title_base="Overall video-level confusion matrix",
        )

    summary = {
        "settings": {
            "videos_dir": str(VIDEOS_DIR),
            "weights": str(WEIGHTS),
            "conf_threshold": CONF_THRES,
            "iou_for_model_nms": IOU_FOR_MODEL_NMS,
            "img_size": IMG_SIZE,
            "max_boxes_per_frame": MAX_BOXES_PER_FRAME,
            "postprocessing": {
                "stream_based_inference": True,
                "class_agnostic_overlap_suppression": True,
                "keep_only_nonoverlapping_most_conf": True,
                "max_boxes_after_filtering": MAX_BOXES_PER_FRAME,
            },
            "video_prediction_rule": {
                "full_blink_dominates": True,
                "half_blink_if_any_half_and_no_full": True,
                "none_if_no_half_or_full_detected": True,
            },
            "yolo_class_mapping": {
                "1": "half",
                "2": "full",
            },
            "evaluation_label_order": [LABEL_TO_NAME[0], LABEL_TO_NAME[1], LABEL_TO_NAME[2]],
        },
        "dataset_summary": {
            "total_processed_videos": len(all_video_results),
            "folders_found": [f.name for f in gt_folders],
        },
        "per_folder": per_folder_summary,
        "overall": {
            "num_videos": overall_eval["num_videos"],
            "metrics": overall_eval["metrics"],
            "blink_vs_none": overall_blink_vs_none,
        },
    }

    out_json = OUT_DIR / "video_based_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved report to: {out_json}")

    if corrupted_videos:
        for i, item in enumerate(corrupted_videos, start=1):
            print(f"{i}. {item['video_name']}  | folder={item['folder_name']}")
            print(f"   path: {item['video_path']}")
            print(f"   error: {item['error']}")


if __name__ == "__main__":
    main()