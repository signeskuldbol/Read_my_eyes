# ===================== CONFIG =====================
from xml.parsers.expat import model
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path

# ===================== CONFIG =====================
USE_CROSS_VALIDATION = False     # True = train ALL folds automatically, False = final train on all data

# Training hyperparams
if USE_CROSS_VALIDATION:
    EPOCHS = 80 # decrese if training continues after performance stops improving. 
    PATIENCE = 20  # early stopping patience (epochs with no improvement after best epoch)
    IMGSZ = 896  
else:
    EPOCHS = 20 # decrese if training continues after performance stops improving. 
    PATIENCE = 10  # not used without validation, but set it anyway
    IMGSZ = 896  
BATCH = 12 # adjust based on GPU memory.
DEVICE = 0 # 0 = first GPU,
WORKERS = 3 # lower if error (linux run can be higher than windows)

# Augmentations
FLIPLR = 0.5 # horizontal flip probability (not all images are perfectly centered, so this can help generalization.
MIXUP = 0 # good for to avoid overfitting to easy examples! 

TRANSLATE = 0.1 # avoids learning positional bias. 0.1 = up to 10% of size shift in x and y direction.
DEGREES = 3 # small rotation can help generalization, but too much can make it unrealistic. 
SCALE = 0.2 # learn different sizes of eyes.
# HSV is already applied by default in YOLOv12, so we can skip it here.

JOB_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()
PROJECT = JOB_WORKSPACE_ROOT / "yolo_models" / "v2_eyes_halved_try_2_Ep_20" # where to save training runs (models, logs, etc.)
BASE_NAME = "eye_detection_v2_halved" # base name for training runs; fold number or "final_all" will be appended to this.
MODEL_WEIGHTS = "yolo12n.pt"
READ_EYE_DIR = Path(__file__).parent.parent.resolve()
dataset_root = READ_EYE_DIR / "yolo_approach" / "dataset_v2_eye_frames_downsampled"
folds_root = dataset_root / "folds"
final_data_yaml = dataset_root / "dataset.yaml"

RESUME = False  # set False if you want a fresh run
LAST_PT = PROJECT / f"{BASE_NAME}_final_all" / "weights" / "last.pt"  # adjust run folder if needed
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


def train_one(model: YOLO, data_yaml: Path, run_name: str, resume=False):
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
        save_json=True, # save training results to json file for easy parsing later
        plots=True, # save training curves (loss, metrics) as images
        workers=WORKERS,
        project=PROJECT,
        name=run_name,
        val=USE_CROSS_VALIDATION,
        resume=RESUME,
        
        # augmentations
        fliplr=FLIPLR,
        mixup=MIXUP,
        translate=TRANSLATE,
        degrees=DEGREES,
        scale=SCALE,
    )


def main():
    SETTINGS.update({"datasets_dir": str(dataset_root)})

    if USE_CROSS_VALIDATION:
        folds = [Path('C:/Users/signe/Job/Read_my_eyes/yolo_approach/dataset_v2_eye_frames_downsampled/folds/fold_06')] #list_folds(folds_root)
        print(f"[INFO] Mode: CROSS-VALIDATION | folds={len(folds)} | root={folds_root}")

        for fold_dir in folds:
            fold_id = fold_dir.name.replace("fold_", "")  # "01", "02", ...
            data_yaml = fold_dir / "dataset.yaml"
            run_name = f"{BASE_NAME}_fold{fold_id}"

            # fresh model each fold (recommended; folds are independent)
            model = YOLO(str(LAST_PT)) if RESUME else YOLO(MODEL_WEIGHTS)
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
        model = YOLO(str(LAST_PT)) if RESUME else YOLO(MODEL_WEIGHTS)
        train_one(model, final_data_yaml, f"{BASE_NAME}_final_all", resume=RESUME)
        print("\n[DONE] Finished final training.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
