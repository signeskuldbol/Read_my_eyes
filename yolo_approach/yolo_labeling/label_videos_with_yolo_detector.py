# export_detector_preds_to_json.py
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

"""
For each video under DATASET_ROOT:
- run YOLO detector on every frame to get eye bbox(es)
- optionally crop each bbox and run YOLO classification model to get class label
- save one JSON per video
- each frame tagged auto_predicted=true

Detections stored as:
  bbox (xyxy pixels) + class_id + class_name
(no confidence stored)
"""

# ----------------- CONFIG -----------------
Workspace_ROOT = Path(__file__).parent.parent.parent.resolve()
print(f"[INFO] Workspace root: {Workspace_ROOT}")
DATASET_ROOT = Workspace_ROOT / "create_datasets" / "datasets" / "full_without_avoid"

WEIGHTS_DET = Workspace_ROOT / "create_datasets" / "yolov12n_eye_detection.pt"
WEIGHTS_CLS = Workspace_ROOT / "yolo_approach" / "cls_best.pt"

OUTPUT_DIR = Workspace_ROOT / "yolo_approach" / "labels_yolo_predicted"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m2ts"}

# Detector settings
DET_CONF = 0.25
DET_IOU = 0.5
DET_IMGSZ = 768
DEVICE = None

# ---- Toggle classifier on/off ----
USE_CLASSIFIER = False  # <-- set False to label everything as "eye"

# Classifier settings
CLS_IMGSZ = 224
CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]
NAMES = {  # classifier internal order differs
    0: "eye",
    1: "eye_full_blink",
    2: "eye_half_blink"
}

# Optional: keep only top-k detections per frame (None = keep all)
TOPK = None
# -----------------------------------------


def collect_done_outputs(output_dir: Path) -> set[str]:
    """Returns a set of output JSON filenames that already exist."""
    return {p.name for p in output_dir.glob("*.json")}


def list_videos(root: Path):
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def safe_out_name(video_path: Path) -> str:
    rel = video_path.relative_to(DATASET_ROOT)
    return "__".join(rel.with_suffix("").parts) + ".json"


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


def classify_crop(cls_model: YOLO | None, crop: np.ndarray):
    """
    Returns (class_id, class_name). No confidence.
    If USE_CLASSIFIER is False, always returns (0, "eye").
    """
    if not USE_CLASSIFIER:
        return 0, "eye"

    if cls_model is None:
        # should never happen, but keep it safe
        return 0, "eye"

    res = cls_model.predict(source=crop, imgsz=CLS_IMGSZ, device=DEVICE, verbose=False)[0]

    if getattr(res, "probs", None) is not None:
        cid = int(res.probs.top1)
        cname = NAMES.get(cid, str(cid))
        return cid, cname

    # fallback for older ultralytics
    try:
        probs = res.probs.data.cpu().numpy().ravel()
        cid = int(np.argmax(probs))
        cname = NAMES.get(cid, str(cid))
        return cid, cname
    except Exception:
        return 0, "eye"


def process_video(det_model: YOLO, cls_model: YOLO | None, video_path: Path, out_json: Path):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open: {video_path}")
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)

    meta = {
        "video": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "width": W,
        "height": H,
        "detector_weights": str(WEIGHTS_DET),
        "use_classifier": bool(USE_CLASSIFIER),
        "classifier_weights": str(WEIGHTS_CLS) if USE_CLASSIFIER else None,
        "bbox_format": "xyxy pixels [x1,y1,x2,y2]",
        "det_conf": DET_CONF,
        "det_iou": DET_IOU,
        "det_imgsz": DET_IMGSZ,
        "cls_imgsz": CLS_IMGSZ if USE_CLASSIFIER else None,
        "class_names": CLASS_NAMES if USE_CLASSIFIER else ["eye"],
    }

    frames = []
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det_res = det_model.predict(
            source=frame,
            conf=DET_CONF,
            iou=DET_IOU,
            imgsz=DET_IMGSZ,
            device=DEVICE,
            verbose=False
        )[0]

        dets = []
        if getattr(det_res, "boxes", None) is not None and len(det_res.boxes) > 0:
            xyxy_all = det_res.boxes.xyxy.cpu().numpy()
            conf_all = det_res.boxes.conf.cpu().numpy()  # only for sorting/topk
            idxs = list(range(len(xyxy_all)))
            idxs.sort(key=lambda i: float(conf_all[i]), reverse=True)
            if TOPK is not None:
                idxs = idxs[: int(TOPK)]

            for i in idxs:
                x1, y1, x2, y2 = xyxy_all[i]
                x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)

                # If not using classifier, we don't even need to crop
                if USE_CLASSIFIER:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    cid, cname = classify_crop(cls_model, crop)
                else:
                    cid, cname = 0, "eye"

                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cid,
                    "class_name": cname
                })

        frames.append({
            "frame_index": fidx,
            "time_sec": (fidx / fps) if fps > 1e-6 else None,
            "auto_predicted": True,
            "detections": dets
        })

        if fidx % 500 == 0 and fidx > 0:
            print(f"  ... {video_path.name}: {fidx}/{total_frames if total_frames else '?'} frames")

        fidx += 1

    cap.release()

    payload = {"meta": meta, "frames": frames}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_json.name} ({len(frames)} frames)")


def main():
    if not WEIGHTS_DET.exists():
        raise FileNotFoundError(f"Missing detector weights: {WEIGHTS_DET}")
    if USE_CLASSIFIER and (not WEIGHTS_CLS.exists()):
        raise FileNotFoundError(f"Missing classifier weights: {WEIGHTS_CLS}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset root: {DATASET_ROOT}")

    det_model = YOLO(str(WEIGHTS_DET))

    cls_model = None
    if USE_CLASSIFIER:
        cls_model = YOLO(str(WEIGHTS_CLS))

        # ---- attach classification transforms (fixes: no attribute 'transforms') ----
        imgsz = CLS_IMGSZ
        try:
            from ultralytics.data.augment import classify_transforms
        except Exception:
            from ultralytics.data.utils import classify_transforms

        cls_model.model.transforms = classify_transforms(imgsz)
        cls_model.overrides = getattr(cls_model, "overrides", {})
        cls_model.overrides["imgsz"] = imgsz

    # sanity check
    if getattr(det_model, "task", None) == "classify":
        raise RuntimeError(f"Detector model is classification: {WEIGHTS_DET}")

    videos = list_videos(DATASET_ROOT)
    if not videos:
        print(f"[ERROR] No videos found under: {DATASET_ROOT}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done_outputs = collect_done_outputs(OUTPUT_DIR)
    print(f"[INFO] Found {len(videos)} videos.")
    print(f"[INFO] OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print(f"[INFO] Existing JSONs in OUTPUT_DIR: {len(done_outputs)}")
    print(f"[INFO] Remaining to process: {max(0, len(videos) - len(done_outputs))}")
    print(f"[INFO] USE_CLASSIFIER = {USE_CLASSIFIER}")

    for i, v in enumerate(videos, 1):
        out_name = safe_out_name(v)
        out_json = OUTPUT_DIR / out_name

        if out_name in done_outputs:
            print(f"[SKIP] Already done: {out_name}")
            continue

        print(f"\n[{i}/{len(videos)}] {v}")
        try:
            process_video(det_model, cls_model, v, out_json)
            done_outputs.add(out_name)
        except Exception as e:
            print(f"[ERROR] Failed on {v.name}: {e}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
