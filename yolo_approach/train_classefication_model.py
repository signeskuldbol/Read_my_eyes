# train_model_cls.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path
from weighted_sampler_cls import WeightedClassificationTrainer


"""Train a YOLO classification model using a custom WeightedClassificationTrainer that oversamples minority classes during training.
We also do augmentation via mixup and horizontal flips.
Assumes dataset is organized as: train/ val/ test/ subfolders for each class
"""

# ---- Paths ----
JOB_DIR = Path(__file__).parent.parent.resolve()
DATA_ROOT = JOB_DIR / "create_datasets" / "datasets" / "classified_frames_cropped_no_crop_removed_split"  # expects train/ val/ test/
# ----------------

def main():
    # Optional: set datasets_dir (not required, but harmless)
    SETTINGS.update({'datasets_dir': str(DATA_ROOT)})

    model = YOLO("yolo11l-cls.pt")

    results = model.train(
        data=str(DATA_ROOT),   # path to folder with train/ val/ (and optionally test/)
        epochs=80,
        imgsz=224,             # common for cls; you can try 256/320 if you like
        batch=32,              # classification benefits from a bit larger batch if VRAM allows
        device=0,
        workers=2,
        project="runs/classification_blink",
        name="blink_classifier_yolo11l",
        trainer=WeightedClassificationTrainer,  
        mixup=0.1, # blends two random images and their labels together.
        fliplr=0.5, # horizontal flip with 50% probability
        )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # safe on Windows
    main()

