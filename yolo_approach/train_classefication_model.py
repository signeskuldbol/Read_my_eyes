# train_model_cls.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path

# ---- Paths ----
JOB_DIR = Path(__file__).parent.parent.resolve()
DATA_ROOT = JOB_DIR / "create_datasets" / "datasets" / "cls_data"  # expects train/ val/ test/
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
        project="runs/cls_horse_eyes",
        name="eyes_cls_v1",
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # safe on Windows
    main()

