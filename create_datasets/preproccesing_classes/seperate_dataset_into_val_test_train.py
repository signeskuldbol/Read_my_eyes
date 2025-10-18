import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility (optional)
random.seed(42)

# ---- CONFIG ----
Workspace_DIR = Path(__file__).parent  
DATASET_DIR = Workspace_DIR / "datasets" / "unpadded_dataset"
OUTPUT_DIR = Workspace_DIR / "datasets" / "sorted_no_pad_no_crop"
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
# ----------------

def split_dataset(dataset_dir: Path, output_dir: Path, ratios: dict):
    classes = [d for d in dataset_dir.iterdir()
        if d.is_dir() and d.name not in {"train", "val", "test"}]

    for cls in classes:
        files = list(cls.glob("*"))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * ratios["train"])
        n_val = int(n_total * ratios["val"])
        n_test = n_total - n_train - n_val

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            split_dir = output_dir / split_name / cls.name
            split_dir.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                shutil.copy(f, split_dir / f.name)

        print(f"Processed class '{cls.name}': "
              f"{n_train} train, {n_val} val, {n_test} test")

if __name__ == "__main__":
    split_dataset(DATASET_DIR, OUTPUT_DIR, SPLIT_RATIOS)
    print("\n Dataset split complete!")
