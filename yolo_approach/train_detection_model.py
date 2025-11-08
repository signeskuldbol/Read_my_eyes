# train_model.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path
from weighted_sampler_cls import WeightedClassificationTrainer

# ---- Paths ----
JOB_DIR = Path(__file__).parent.parent.parent.resolve()  
DATA_YAML = JOB_DIR / "yolo_data" / "horse_eyes" / "data.yaml"
HYP_YAML  = JOB_DIR / "yolo_approach" / "hyp_imbalance.yaml"  
# ----------------------


def main():
    SETTINGS.update({'datasets_dir': str(JOB_DIR / "yolo_data" / "horse_eyes")})

    model = YOLO("yolo12n.pt")  
    results = model.train(
        data=DATA_YAML,
        single_cls=False,          
        epochs=80,
        imgsz=768,
        batch=16,
        device=0,                 
        workers=2,                
        project="runs/horse_eyes",
        name="y12n_3_class",
        trainer=WeightedClassificationTrainer,  # <-- tell YOLO to use our trainer
        # (optional) extra augmentation still helps:
        mixup=0.1,
        cutmix=0.2,
        fliplr=0.5,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # safe on Windows
    main()
