import json
import random
from pathlib import Path
import cv2 as cv

"""
Takes the annotations created using "check_yolo_detections.py" and converts them to yolo detector format,
they are all saved in a single folder structure with images and labels. Then, leave-one-video-out cross-validation 
folds are created,where validation videos come only from the MAIN reviewed JSONs, while training includes all other 
MAIN videos plus all EXTRA videos (blink classes only).

NOTE: In the blink only videos only frames with at least one blink box are included. 
As only blink frames are checked!!
"""


# ---------------- CONFIG ----------------
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()

# MAIN reviewed JSONs (eligible for cross-val validation)
MAIN_JSON_DIR = WORKSPACE_ROOT / "yolo_approach" / "labels_reviewed"

# EXTRA JSONs (train-only; ONLY keep blink classes 1/2 from these)
EXTRA_BLINK_ONLY_JSON_DIR = WORKSPACE_ROOT / "yolo_approach" / "labels_extra_blink_only"
# If you don't want extra: set to None
# EXTRA_BLINK_ONLY_JSON_DIR = None

# Output dataset root
OUT_ROOT = WORKSPACE_ROOT / "yolo_approach" / "dataset"

CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]

# Keep classes per source:
MAIN_KEEP_CLASS_IDS = None      # None = keep all 3 classes
EXTRA_KEEP_CLASS_IDS = {1, 2}   # keep only blink classes from extra folder

RANDOM_SEED = 42

IMG_EXT = ".jpg"
JPG_QUALITY = 95
# ---------------------------------------


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def xyxy_to_yolo_norm(x1, y1, x2, y2, W, H):
    x1 = clamp(float(x1), 0.0, W - 1.0)
    x2 = clamp(float(x2), 0.0, W - 1.0)
    y1 = clamp(float(y1), 0.0, H - 1.0)
    y2 = clamp(float(y2), 0.0, H - 1.0)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cx / W, cy / H, bw / W, bh / H)


def save_image(img_path: Path, frame):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if IMG_EXT.lower() == ".jpg":
        cv.imwrite(str(img_path), frame, [int(cv.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    else:
        cv.imwrite(str(img_path), frame)


def write_fold_yaml(fold_dir: Path):
    """
    Ultralytics can use train/val .txt lists (each line is a full image path).
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = fold_dir / "dataset.yaml"
    content = "train: train.txt\nval: val.txt\n\nnames:\n"
    for i, n in enumerate(CLASS_NAMES):
        content += f"  {i}: {n}\n"
    yaml_path.write_text(content, encoding="utf-8")


def write_txt_list(path: Path, image_paths):
    path.write_text("\n".join(str(p) for p in image_paths) + ("\n" if image_paths else ""), encoding="utf-8")


def collect_checked_frames(json_dir: Path, keep_class_ids, tag: str, require_kept_box: bool = False):
    """
    Returns:
      per_video: dict video_key -> dict frame_idx -> list[detections_kept]
      video_paths: dict video_key -> Path(video)

    Only includes rec["checked"] == True frames.

    If require_kept_box=True, a checked frame is included ONLY if, after filtering,
    there is at least one kept detection (e.g., at least one blink box).
    """
    per_video = {}
    video_paths = {}

    if json_dir is None:
        return per_video, video_paths

    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSONs found in {json_dir} ({tag})")
        return per_video, video_paths

    for jf in json_files:
        data = load_json(jf)

        video_path = Path(data["meta"]["video"])
        if not video_path.exists():
            print(f"[WARN] Missing video for {jf.name}: {video_path}")
            continue

        frames = data.get("frames", [])
        if not frames:
            continue

        video_key = video_path.as_posix()
        video_paths[video_key] = video_path

        for idx, rec in enumerate(frames):
            if rec.get("checked", False) is not True:
                continue

            dets = rec.get("detections", [])
            kept = []
            for d in dets:
                cid = int(d.get("class_id", -1))
                if cid < 0 or cid >= len(CLASS_NAMES):
                    continue
                if keep_class_ids is not None and cid not in keep_class_ids:
                    continue
                kept.append(d)

            # NEW: only keep this frame if it has at least one kept box
            if require_kept_box and len(kept) == 0:
                continue

            per_video.setdefault(video_key, {})[idx] = kept

    print(f"[INFO] Collected {tag}: videos={len(per_video)}")
    return per_video, video_paths


def export_all_images_and_labels(per_video_all, video_paths_all, out_images_all, out_labels_all):
    """
    Export each (video, frame) once. Labels written to labels_all with same stem.
    Returns:
      image_map: dict (video_key, frame_idx) -> Path(image_path)
    """
    out_images_all.mkdir(parents=True, exist_ok=True)
    out_labels_all.mkdir(parents=True, exist_ok=True)

    image_map = {}
    total = 0
    empty_labels = 0

    for video_key, frame_dict in per_video_all.items():
        video_path = video_paths_all[video_key]
        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            continue

        for frame_idx in sorted(frame_dict.keys()):
            cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {frame_idx} from {video_path.name}")
                continue

            H, W = frame.shape[:2]
            dets = frame_dict[frame_idx]

            yolo_lines = []
            for d in dets:
                cid = int(d.get("class_id", -1))
                x1, y1, x2, y2 = d["bbox"]
                cx, cy, bw, bh = xyxy_to_yolo_norm(x1, y1, x2, y2, W, H)
                if bw <= 0 or bh <= 0:
                    continue
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            stem = f"{video_path.stem}_f{frame_idx:06d}"
            img_path = out_images_all / f"{stem}{IMG_EXT}"
            lab_path = out_labels_all / f"{stem}.txt"

            save_image(img_path, frame)
            lab_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

            if not yolo_lines:
                empty_labels += 1

            image_map[(video_key, frame_idx)] = img_path
            total += 1

        cap.release()

    print(f"[OK] Exported total images: {total}")
    print(f"[INFO] Images with EMPTY label files (negatives): {empty_labels}")
    return image_map


def main():
    random.seed(RANDOM_SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_images_all = OUT_ROOT / "images"
    out_labels_all = OUT_ROOT / "labels"
    folds_dir = OUT_ROOT / "folds"

    # 1) Collect data per source with different class-keeping rules
    per_main, paths_main = collect_checked_frames(
        MAIN_JSON_DIR, MAIN_KEEP_CLASS_IDS, tag="MAIN(all-classes)", require_kept_box=False
    )

    per_extra, paths_extra = collect_checked_frames(
        EXTRA_BLINK_ONLY_JSON_DIR, EXTRA_KEEP_CLASS_IDS, tag="EXTRA(train-only blink)", require_kept_box=True
    )


    # 2) Merge into one export set (so we export frames once)
    per_all = {}
    paths_all = {}

    # Merge MAIN
    for vk, frames_dict in per_main.items():
        per_all[vk] = dict(frames_dict)
    paths_all.update(paths_main)

    # Merge EXTRA (may overlap same video; if so, we need to merge frame keys carefully)
    for vk, frames_dict in per_extra.items():
        if vk not in per_all:
            per_all[vk] = {}
        # If a frame exists in both, we keep MAIN version (all classes),
        # because that's richer. Extra is mainly for additional frames.
        for frame_idx, dets in frames_dict.items():
            if frame_idx not in per_all[vk]:
                per_all[vk][frame_idx] = dets
    paths_all.update(paths_extra)

    # 3) Export images+labels once
    image_map = export_all_images_and_labels(per_all, paths_all, out_images_all, out_labels_all)

    # 4) Create leave-one-video-out folds using MAIN videos only for validation membership
    main_videos = sorted(per_main.keys())
    random.shuffle(main_videos)
    k = len(main_videos)

    print(f"[INFO] Creating {k} folds (leave-one-video-out). Validation videos come ONLY from MAIN.")

    for i, val_video in enumerate(main_videos, start=1):
        fold_name = f"fold_{i:02d}"
        fold_dir = folds_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_images = []
        val_images = []

        # val: all MAIN frames of val_video
        for frame_idx in sorted(per_main[val_video].keys()):
            p = image_map.get((val_video, frame_idx))
            if p is not None:
                val_images.append(p)

        # train: MAIN frames from all other MAIN videos
        for vk in main_videos:
            if vk == val_video:
                continue
            for frame_idx in sorted(per_main[vk].keys()):
                p = image_map.get((vk, frame_idx))
                if p is not None:
                    train_images.append(p)

        # plus EXTRA frames (train-only)
        for vk, frames_dict in per_extra.items():
            for frame_idx in sorted(frames_dict.keys()):
                p = image_map.get((vk, frame_idx))
                if p is not None:
                    train_images.append(p)

        # Write fold files
        write_txt_list(fold_dir / "train.txt", train_images)
        write_txt_list(fold_dir / "val.txt", val_images)
        write_fold_yaml(fold_dir)

        print(
            f"[{fold_name}] train={len(train_images)} val={len(val_images)} "
            f"val_video={Path(val_video).name} extra_in_train={sum(len(v) for v in per_extra.values())}"
        )

    print("\n[DONE]")
    print(f"  Root:   {OUT_ROOT}")
    print(f"  Images: {out_images_all}")
    print(f"  Labels: {out_labels_all}")
    print(f"  Folds:  {folds_dir}")

if __name__ == "__main__":
    main()
