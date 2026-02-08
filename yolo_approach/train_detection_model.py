# ===================== CONFIG =====================
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path

"""
Current Distrubution in data:
=== YOLO INSTANCE COUNTS ===
eye             : 19494
eye_half_blink  : 3693
eye_full_blink  : 833

=== IMAGE STATS ===
Total images:        23248
Images with no boxes:677
Images with boxes:   22571
"""
# ===================== CONFIG =====================
USE_CROSS_VALIDATION = True     # True = train ALL folds automatically, False = final train on all data

# Training hyperparams

if USE_CROSS_VALIDATION:
    EPOCHS = 100 # decrese if training continues after performance stops improving. 
    PATIENCE = 20  # early stopping patience (epochs with no improvement after best epoch)
    IMGSZ = 640  # smaller for faster training on folds
else:
    EPOCHS = 100 # decrese if training continues after performance stops improving. 
    PATIENCE = 10  # not used without validation, but set it anyway
    IMGSZ = 896  # larger for final train on all data
BATCH = 24 # adjust based on GPU memory.
DEVICE = 0 # 0 = first GPU,
WORKERS = 8 # lower if error (linux run can be higher than windows)

# Augmentations
FLIPLR = 0.5 # horizontal flip probability (not all images are perfectly centered, so this can help generalization.
MIXUP = 0.1 # good for to avoid overfitting to easy examples! 

TRANSLATE = 0.1 # avoids learning positional bias. 0.1 = up to 10% of size shift in x and y direction.
DEGREES = 3 # small rotation can help generalization, but too much can make it unrealistic. 
SCALE = 0.2 # learn different sizes of eyes.
# HSV is already applied by default in YOLOv12, so we can skip it here.

JOB_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()
print(f"[INFO] JOB_WORKSPACE_ROOT: {JOB_WORKSPACE_ROOT}")
PROJECT = JOB_WORKSPACE_ROOT / "yolo" / "runs" / "blink_detection" # where to save training runs (models, logs, etc.)
BASE_NAME = "y12n_3class"
MODEL_WEIGHTS = "yolo12n.pt"
# ==================================================


def list_folds(folds_root: Path) -> list[Path]:
    """Return sorted fold directories like fold_01, fold_02, ..."""
    if not folds_root.exists():
        raise SystemExit(f"[ERROR] folds folder not found: {folds_root}")

    folds = [p for p in folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    folds = sorted(folds, key=lambda p: p.name)
    if not folds:
        raise SystemExit(f"[ERROR] No fold_* directories found in: {folds_root}")
    return folds


def train_one(model: YOLO, data_yaml: Path, run_name: str):
    if not data_yaml.exists():
        raise SystemExit(f"[ERROR] data.yaml not found: {data_yaml}")

    print(f"\n[TRAIN] {run_name}")
    print(f"       data = {data_yaml}")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE, # early stopping
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=run_name,

        # augmentations
        fliplr=FLIPLR,
        mixup=MIXUP,
        translate=TRANSLATE,
        degrees=DEGREES,
        scale=SCALE,
    )


def main():
    READ_EYE_DIR = Path(__file__).parent.parent.resolve()
    print(f"[INFO] READ_EYE workspace root: {READ_EYE_DIR}")

    dataset_root = READ_EYE_DIR / "yolo_approach" / "dataset"
    folds_root = dataset_root / "folds"
    final_data_yaml = dataset_root / "dataset_all.yaml"

    SETTINGS.update({"datasets_dir": str(dataset_root)})

    if USE_CROSS_VALIDATION:
        folds = list_folds(folds_root)
        print(f"[INFO] Mode: CROSS-VALIDATION | folds={len(folds)} | root={folds_root}")

        for fold_dir in folds:
            fold_id = fold_dir.name.replace("fold_", "")  # "01", "02", ...
            data_yaml = fold_dir / "dataset.yaml"
            run_name = f"{BASE_NAME}_fold{fold_id}"

            # fresh model each fold (recommended; folds are independent)
            model = YOLO(MODEL_WEIGHTS)
            train_one(model, data_yaml, run_name)

        print("\n[DONE] Finished training all folds.")

    else:
        print("[INFO] Mode: FINAL TRAIN (all data)")
        if not final_data_yaml.exists():
            raise SystemExit(
                f"[ERROR] Missing {final_data_yaml}\n"
                f"Create it with:\n"
                f"  train: images_all\n"
                f"  names:\n"
                f"    0: eye\n"
                f"    1: eye_half_blink\n"
                f"    2: eye_full_blink\n"
            )

        model = YOLO(MODEL_WEIGHTS)
        train_one(model, final_data_yaml, f"{BASE_NAME}_final_all")
        print("\n[DONE] Finished final training.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
