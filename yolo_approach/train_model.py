# train_model.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

DATA_YAML = r"C:\_ultra_data\horse_eyes\data.yaml"

def main():
    # Keep Ultralytics conversions off OneDrive/Unicode paths (already done, but safe)
    SETTINGS.update({'datasets_dir': r'C:\_ultra_data'})

    model = YOLO("yolo12n.pt")  
    results = model.train(
        data=DATA_YAML,
        single_cls=True,          # 1 class = eye
        epochs=80,
        imgsz=768,
        batch=16,
        device=0,                 
        workers=2,                
        project="runs/horse_eyes",
        name="y12n_eye_v1",
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # safe on Windows
    main()
