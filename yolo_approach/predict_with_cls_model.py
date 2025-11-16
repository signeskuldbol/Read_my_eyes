import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# ---------------- Paths ----------------
Workspace_Path = Path(__file__).parent.parent.parent.resolve()
detection_eye_model_path = Workspace_Path / "Read_my_eyes" / "create_datasets" / "yolov12n_eye_detection.pt"
cls_model_path           = Workspace_Path / "runs" / "classification_blink" / "blink_classifier_yolo11l3" / "weights" / "best.pt"

# Folder with input videos and output folder
videos_dir   = Workspace_Path / "Read_my_eyes" / "create_datasets" / "original_videos_annotations" / "videos" 
outputs_dir  = Workspace_Path / "Read_my_eyes" / "yolo_approach" / "outputs1"

# ---------- Inference settings ----------
DET_CONF = 0.25
DET_IOU  = 0.5     # NMS IoU
CLS_CONF_PRINT_THRESHOLD = 0.0
FONT = cv.FONT_HERSHEY_SIMPLEX
action = None
class_list = []
CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]
# -----------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def nice_label(img, text, org, color, scale=0.7, thickness=2):
    x, y = org
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    pad = 4
    cv.rectangle(img, (x, y - th - 2*pad), (x + tw + 2*pad, y + baseline), (0, 0, 0), -1)
    cv.putText(img, text, (x + pad, y - pad), FONT, scale, color, thickness, cv.LINE_AA)

def make_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_candidates = ["mp4v", "avc1", "H264", "XVID"]
    for cc in fourcc_candidates:
        fourcc = cv.VideoWriter_fourcc(*cc)
        w = cv.VideoWriter(str(path), fourcc, fps if fps and fps > 1e-3 else 25.0, size)
        if w.isOpened():
            print(f"[OK] Writing to {path.name} with codec {cc}")
            return w
    raise RuntimeError(f"Could not open writer: {path}")

def process_one_video(detection_model: YOLO, cls_model: YOLO, video_path: Path):
    cap = cv.VideoCapture(str(video_path))
    class_list = []
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    W  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0

    out_path = outputs_dir / f"{video_path.stem}_annotated.mp4"
    writer = make_video_writer(out_path, fps, (W, H))
    print(f"Processing {video_path.name} → {out_path.name}  ({W}x{H} @ {fps:.2f} fps)")

    fidx = 0
    t0 = time.time()

    # Colors per class (match CLASS_NAMES)
    color_map = {
        "eye": (0, 255, 0),
        "eye_half_blink": (0, 255, 255),
        "eye_full_blink": (0, 0, 255),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        det_results = detection_model.predict(source=frame, conf=DET_CONF, iou=DET_IOU, save=False, verbose=False)
        annotated = frame.copy()

        det = det_results[0]
        boxes = getattr(det, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = clamp_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H)

                eye_crop = frame[y1:y2, x1:x2]
                if eye_crop.size == 0:
                    continue

                # Classification with imgsz to force proper transforms
                cls_results = cls_model.predict(source=eye_crop, imgsz=224, save=False, verbose=False)
                cls_res = cls_results[0]
                names = getattr(cls_res, "names", None) or {i: n for i, n in enumerate(CLASS_NAMES)}

                # --- Class selection (no blink override) ---
                chosen_id, chosen_conf = 0, 0.0
                if hasattr(cls_res, "probs") and cls_res.probs is not None:
                    # Standard top-1 prediction
                    chosen_id = int(cls_res.probs.top1)
                    chosen_conf = float(cls_res.probs.top1conf)
                else:
                    # Fallback for older versions
                    try:
                        probs = cls_res.probs.data.cpu().numpy().ravel()
                        chosen_id = int(np.argmax(probs))
                        chosen_conf = float(probs[chosen_id])
                    except Exception:
                        chosen_id, chosen_conf = 0, 0.0


                cls_label = names.get(chosen_id, str(chosen_id))
                class_list.append(cls_label)
                conf_txt  = f"{chosen_conf:.2f}"

                color = color_map.get(cls_label, (255, 255, 255))
                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_text = f"{cls_label} {conf_txt}" if chosen_conf >= CLS_CONF_PRINT_THRESHOLD else cls_label
                nice_label(annotated, label_text, (x1, y1 - 6), color, scale=0.7, thickness=2)


        #cv.putText(annotated, f"Action~{action:.1f}", (10, 30), FONT, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        #cv.putText(annotated, f"Action~{action:.1f}", (10, 30), FONT, 0.7, (255, 255, 255), 1, cv.LINE_AA)

        writer.write(annotated)

    cap.release()
    writer.release()
    print("Saved:", out_path)

    # ---------- Play saved result at half speed ----------
    print("Playing annotated video at half speed. Press 'q' to quit.")
    print(f"[INFO] Class predictions in this video: {set(class_list)}")
    play = cv.VideoCapture(str(out_path))
    delay_ms = int(2000 / max(1e-6, fps))  # double the normal delay => half speed
    while True:
        ok, fr = play.read()
        if not ok:
            break
        cv.imshow("Eye State Detection (Annotated)", fr)
        if cv.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    play.release()
    cv.destroyAllWindows()

def main():
    # Load models once
    detection_model = YOLO(str(detection_eye_model_path))
    cls_model       = YOLO(str(cls_model_path))
    try:
        # Newer Ultralytics
        from ultralytics.data.augment import classify_transforms
    except Exception:
        # Fallback if module path moved in your version
        from ultralytics.data.utils import classify_transforms

    imgsz = 224
    # Attach transforms so the predictor won't look for a missing attribute
    cls_model.model.transforms = classify_transforms(imgsz)

    # Also set a default imgsz override so predictors build the same size consistently
    cls_model.overrides = getattr(cls_model, "overrides", {})
    cls_model.overrides["imgsz"] = imgsz

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos (non-recursive). For recursive: use videos_dir.rglob("*")
    video_files = [p for p in videos_dir.iterdir() if p.suffix in VIDEO_EXTS and p.is_file()]
    if not video_files:
        print(f"[WARN] No videos found in: {videos_dir}")
        return

    for vid in sorted(video_files):
        if vid in outputs_dir.iterdir():
            print(f"[SKIP] Output already exists for {vid.name}, skipping.")
            continue
        try:
            process_one_video(detection_model, cls_model, vid)
        except Exception as e:
            print(f"[ERROR] Failed on {vid.name}: {e}")
        print(f"[INFO] Class predictions in this video: {set(class_list)}")

if __name__ == "__main__":
    main()
