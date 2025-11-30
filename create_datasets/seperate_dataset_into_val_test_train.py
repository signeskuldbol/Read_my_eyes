# seperate_dataset_into_val_test_train.py
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ---- CONFIG ----
Workspace_DIR = Path(__file__).parent.parent.resolve()
print("Workspace_DIR:", Workspace_DIR)

DATASET_DIR = Workspace_DIR / "create_datasets" / "datasets" / "New"
OUTPUT_DIR  = Workspace_DIR / "create_datasets" / "datasets" / "New_split_new_way"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}  # must sum to 1.0
VIDEO_EXTS   = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
SEED         = 42  # reproducibility
# --------------

def assert_ratios_ok(ratios: dict):
    s = sum(ratios.values())
    assert abs(s - 1.0) < 1e-6, f"SPLIT_RATIOS must sum to 1.0, got {s}"

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix in VIDEO_EXTS

def find_leaf_class_dirs(root: Path):
    """
    Return directories that actually contain video files (leaf 'class' dirs).
    Example:
      .../dataset_binary_cropped/Actions/AU47/*.mp4  -> class dir = AU47
      .../dataset_binary_cropped/Background/*.mp4    -> class dir = Background
    """
    leaf_dirs = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            leaf_dirs.add(p.parent)
    return sorted(leaf_dirs)

def split_indices(n: int, ratios: dict, rng: random.Random):
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    # ensure all assigned
    n_test  = n - n_train - n_val
    idxs = list(range(n))
    rng.shuffle(idxs)
    return (
        idxs[:n_train],
        idxs[n_train:n_train + n_val],
        idxs[n_train + n_val:],
    )

def main():
    assert_ratios_ok(SPLIT_RATIOS)
    rng = random.Random(SEED)

    # 1) Find all leaf class dirs (dirs that actually contain videos)
    class_dirs = find_leaf_class_dirs(DATASET_DIR)
    if not class_dirs:
        print(f"No video files with {VIDEO_EXTS} under: {DATASET_DIR}")
        return

    totals = {"train": 0, "val": 0, "test": 0}

    for cls_dir in class_dirs:
        # Class name = leaf folder name
        cls_name = cls_dir.name

        # 2) Collect files in this class dir (non-recursive; change to rglob if needed)
        files = [p for p in cls_dir.glob("*") if is_video(p)]
        if not files:
            continue

        # 3) Split indices
        train_idx, val_idx, test_idx = split_indices(len(files), SPLIT_RATIOS, rng)
        splits = {
            "train": [files[i] for i in train_idx],
            "val":   [files[i] for i in val_idx],
            "test":  [files[i] for i in test_idx],
        }

        # 4) Copy preserving class name (flat inside class)
        for split_name, split_files in splits.items():
            out_cls_dir = OUTPUT_DIR / split_name / cls_name
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                dst = out_cls_dir / src.name
                shutil.copy2(src, dst)
                totals[split_name] += 1

        print(f"Processed class '{cls_name}': "
              f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    print("\nDataset split complete!")
    print(f"  Train: {totals['train']}")
    print(f"  Val:   {totals['val']}")
    print(f"  Test:  {totals['test']}")
    print(f"Output at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
