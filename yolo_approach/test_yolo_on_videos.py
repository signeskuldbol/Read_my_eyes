from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

"""
Script for exporting annotated prediction videos for a YOLO model on a folder of videos,
with the SAME per-frame box filtering rules as the evaluation script (but WITHOUT any
"connected episode" overwrite logic).

Filtering rules (per frame):
1) No overlap allowed: if 2+ boxes overlap (any positive-area intersection),
   only the most confident is kept (class-agnostic).
2) After overlap filtering, keep at most 2 boxes total (most confident ones).
3) Draw colors:
   - blue for class 0 (eye)
   - green for class 1 (half_blink)
   - red for class 2 (full_blink)
"""

# ---------------- overlap filtering ----------------
def boxes_intersect_xyxy(a, b) -> bool:
    """Return True if two xyxy boxes overlap with positive area (any intersection)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return (ix2 > ix1) and (iy2 > iy1)

def filter_boxes_no_overlap_then_cap(xyxy: np.ndarray, conf: np.ndarray, max_boxes: int = 2) -> np.ndarray:
    """
    Step 1: Sort by confidence (desc) and keep a box only if it does NOT intersect any kept box.
    Step 2: Cap to max_boxes (most confident) AFTER overlap filtering.
    Returns indices into the original arrays.
    """
    if xyxy.size == 0:
        return np.array([], dtype=int)

    order = np.argsort(-conf)  # high -> low
    keep = []

    # Step 1: strict intersection removal
    for idx in order:
        b = xyxy[idx]
        if not any(boxes_intersect_xyxy(b, xyxy[k]) for k in keep):
            keep.append(idx)

    if not keep:
        return np.array([], dtype=int)

    keep = np.array(keep, dtype=int)

    # Step 2: cap AFTER overlap filtering
    if keep.size > max_boxes:
        # keep[] are already in descending conf order due to the greedy loop, but do explicit sort for safety
        keep_sorted = keep[np.argsort(-conf[keep])]
        keep = keep_sorted[:max_boxes]

    return keep


# ---------------- drawing ----------------
def draw_boxes(img, xyxy, conf, cls, names):
    out = img.copy()
    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        cl_int = int(cl)
        if cl_int == 0:
            color = (255, 0, 0)  # Blue for class 0 (eye)
        elif cl_int == 1:
            color = (0, 255, 0)  # Green for class 1 (half_blink)
        elif cl_int == 2:
            color = (0, 0, 255)  # Red for class 2 (full_blink)
        else:
            color = (255, 255, 255)  # White for any unexpected class

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names[cl_int]} {c:.2f}" if names and cl_int in names else f"{cl_int} {c:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


# ---------------- main ----------------
def run_folder(
    model_path: str,
    videos_dir: str,
    out_dir: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.1,   # used internally by Ultralytics; our strict filtering is applied afterward
    max_boxes_per_frame: int = 2,
):
    model = YOLO(model_path)
    names = getattr(model, "names", None)

    videos_dir = Path(videos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    video_paths = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in video_exts])
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {videos_dir}")

    for video_path in video_paths:
        print(f"\nProcessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  !! Could not open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = out_dir / f"{video_path.stem}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        # Stream predictions
        results = model.predict(
            source=str(video_path),
            stream=True,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )

        for r in results:
            frame = r.orig_img  # BGR uint8

            if r.boxes is not None and len(r.boxes) > 0:
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                cls = b.cls.cpu().numpy()

                keep_idx = filter_boxes_no_overlap_then_cap(xyxy, conf, max_boxes=max_boxes_per_frame)

                if keep_idx.size > 0:
                    frame = draw_boxes(frame, xyxy[keep_idx], conf[keep_idx], cls[keep_idx], names)

            writer.write(frame)

        writer.release()
        cap.release()
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
    INFO_VIDEOS_DIR = WORKSPACE_ROOT / "create_datasets" / "original_videos_annotations"
    videos_dir = INFO_VIDEOS_DIR / "videos"
    Workspace_Path = Path(__file__).parent.parent.parent.resolve()
    model_path = Workspace_Path / "yolo_models" / "v2_eyes_halved" / "weights" / "best_v2_not_finished_E_129.pt"

    out_dir = WORKSPACE_ROOT / "yolo_approach" / "pred_videos_v2"

    run_folder(
        model_path=model_path,
        videos_dir=videos_dir,
        out_dir=out_dir,
        conf_thres=0.25,
        iou_thres=0.1,
        max_boxes_per_frame=2,
    )
