import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

""" Optical flow based eye blink detection.

Very sensitive to movements and do not detect slow blinks. 

uses a fixed size bbox around the eye based on max detected bbox size in the video,
and computes dense optical flow inside that bbox between consecutive frames. the centre is smoothed over
time to reduce jitter.

Thresholds needs finetuning and may not generalize well across different videos.

"""



# ---------------- Paths ----------------
Workspace_Path = Path(__file__).parent.parent.parent.resolve()
detection_eye_model_path = Workspace_Path / "Read_my_eyes" / "create_datasets" / "yolov12n_eye_detection.pt"

# Folders with input videos and output folder
videos_dir   = Workspace_Path / "Read_my_eyes" / "create_datasets" / "original_videos_annotations" / "videos"
outputs_dir  = Workspace_Path / "Read_my_eyes" / "Optical_flow" / "outputs_optical_flow"

# ---------- Inference settings ----------
DET_CONF = 0.25
DET_IOU  = 0.5
FONT = cv.FONT_HERSHEY_SIMPLEX

# Optical flow threshold for "blink" / action 
FLOW_HALF_BLINK_THR = 1
FLOW_BLINK_THR = 1.75

# Center smoothing factor (0 = no smoothing, 1 = fully follow new detection)
# Typical useful range: 0.5–0.9
CENTER_SMOOTHING = 0.7

# Playback slowdown factor (2.0 = half speed, 4.0 = quarter speed)
PLAYBACK_SLOWDOWN = 2.0

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


def compute_flow(frame1, frame2):
    """
    Compute dense optical flow between two frames using Farneback.

    This function accepts either BGR images (H, W, 3) or grayscale (H, W).
    If input is BGR, it converts to grayscale internally.
    """
    # Handle frame1: BGR or grayscale
    if frame1.ndim == 3 and frame1.shape[2] == 3:
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    # Handle frame2: BGR or grayscale
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


def clamp_xyxy(x1, y1, x2, y2, W, H):
    """
    Clamp a bounding box (x1, y1, x2, y2) to image size (W, H)
    and ensure it has at least 1 pixel width/height.
    """
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
    """
    Draw a text label with a filled background rectangle for better visibility.
    """
    x, y = org
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    pad = 4
    cv.rectangle(
        img,
        (x, y - th - 2 * pad),
        (x + tw + 2 * pad, y + baseline),
        (0, 0, 0),
        -1
    )
    cv.putText(
        img,
        text,
        (x + pad, y - pad),
        FONT,
        scale,
        color,
        thickness,
        cv.LINE_AA
    )


def make_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv.VideoWriter:
    """
    Create a cv.VideoWriter for the given path, fps, and frame size.
    Tries a few codecs until one works.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_candidates = ["mp4v", "avc1", "H264", "XVID"]
    for cc in fourcc_candidates:
        fourcc = cv.VideoWriter_fourcc(*cc)
        w = cv.VideoWriter(
            str(path),
            fourcc,
            fps if fps and fps > 1e-3 else 25.0,
            size
        )
        if w.isOpened():
            print(f"[OK] Writing to {path.name} with codec {cc}")
            return w
    raise RuntimeError(f"Could not open writer: {path}")


def smooth_centers(centers, alpha):
    """
    Apply exponential smoothing to a list of (cx, cy) centers.

    centers: list of (cx, cy) or None
    alpha: smoothing factor in [0, 1], where:
        - 0.0 = keep old detection
        - 1.0 = use new detection
    Returns: list of smoothed centers (cx, cy) or None.
    """
    smoothed = []
    prev = None

    for c in centers:
        if c is None:
            # If detection is missing, break smoothing chain and output None
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


def process_one_video(detection_model: YOLO, video_path: Path):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    W  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0  # original frame rate

    out_path = outputs_dir / f"{video_path.stem}_annotated_of.mp4"
    print(f"Processing {video_path.name} → {out_path.name}  ({W}x{H} @ {fps:.2f} fps)")

    # --------- PASS 1: read all frames + find max bbox size ---------
    frames  = []       # list of raw frames (BGR)
    centers = []       # detection center (cx, cy) per frame (None if no detection)
    max_w, max_h = 0, 0

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
            # Choose one box per frame – the most confident one
            confs = boxes.conf.cpu().numpy()
            idx   = int(confs.argmax().item())

            xyxy = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = clamp_xyxy(
                xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H
            )
            w = x2 - x1
            h = y2 - y1
            max_w = max(max_w, w)
            max_h = max(max_h, h)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))
        else:
            centers.append(None)

    cap.release()

    # Case 1: no detections at all in the entire video → save full-frame video as-is
    if max_w == 0 or max_h == 0:
        print("[INFO] No detections found in this video. Saving full-frame output.")
        writer = make_video_writer(out_path, fps, (W, H))
        for frame in frames:
            writer.write(frame)
        writer.release()
        print("Saved full-frame video:", out_path)
        return

    print(f"[INFO] Max bbox size over video: {max_w}x{max_h}")

    # Smooth the detected centers to reduce jitter
    smooth_centers_list = smooth_centers(centers, CENTER_SMOOTHING)

    # Output video = original full-frame size
    writer = make_video_writer(out_path, fps, (W, H))

    # --------- PASS 2: optical flow inside fixed bbox, visualized on full frame ---------
    prev_crop_gray = None
    fidx = 0

    for frame, center in zip(frames, smooth_centers_list):
        fidx += 1
        annotated = frame.copy()

        action = 0
        mean_mag = 0.0

        if center is not None:
            cx, cy = center

            # Use fixed bbox size around the (smoothed) center for the whole video
            half_w = max_w // 2
            half_h = max_h // 2

            x1 = int(cx - half_w)
            y1 = int(cy - half_h)
            x2 = x1 + max_w
            y2 = y1 + max_h
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)

            eye_crop = frame[y1:y2, x1:x2]
            if eye_crop.size > 0:
                eye_gray = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)

                # Optical flow between previous crop and current crop
                if prev_crop_gray is not None and eye_gray.shape == prev_crop_gray.shape:
                    flow = compute_flow(prev_crop_gray, eye_gray)
                    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

                    mean_mag = float(mag.mean())

                    # Blink / action detection based on threshold
                    if mean_mag >= FLOW_BLINK_THR:
                        action = 2
                    elif mean_mag >= FLOW_HALF_BLINK_THR:
                        action = 1
                    else:
                        action = 0

                # Update reference crop for next frame
                prev_crop_gray = eye_gray
            else:
                # If cropping failed, reset reference
                prev_crop_gray = None

            # Choose bbox color based on blink vs no blink
            if action == 2:
                box_color = (0, 0, 255)   # red
                status_text = "FULL BLINK"
            elif action == 1:
                box_color = (0, 255, 0)   # green
                status_text = "HALF BLINK"
            else:
                box_color = (0, 255, 255) # yellow
                status_text = "NO BLINK"

            # Draw bbox on the original frame
            cv.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Label for the bbox (status + mean flow)
            label_text = f"{status_text} | mean: {mean_mag:.2f}"
            # Place label slightly above the bbox (clamped to stay on-screen)
            label_y = max(y1 - 10, 20)
            nice_label(
                annotated,
                label_text,
                (x1, label_y),
                box_color,
                scale=0.7,
                thickness=2
            )

            # Also show the same info in the top-left corner for readability
            cv.putText(
                annotated,
                label_text,
                (10, 30),
                FONT,
                0.8,
                (0, 0, 0),
                3,
                cv.LINE_AA
            )
            cv.putText(
                annotated,
                label_text,
                (10, 30),
                FONT,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

        else:
            # No detection in this frame: reset reference and say so
            prev_crop_gray = None
            info_txt = "No eye detection"
            cv.putText(
                annotated,
                info_txt,
                (10, 30),
                FONT,
                0.8,
                (0, 0, 0),
                3,
                cv.LINE_AA
            )
            cv.putText(
                annotated,
                info_txt,
                (10, 30),
                FONT,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

        writer.write(annotated)

    writer.release()
    print("Saved:", out_path)


def main():
    # Load detection model once
    detection_model = YOLO(str(detection_eye_model_path))

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos
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
