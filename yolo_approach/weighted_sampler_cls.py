# weighted_sampler_cls.py
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.data.build import build_dataloader
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
import numpy as np

"""
Trainer that oversamples minority classes for TRAIN via WeightedRandomSampler.
Inspired by https://y-t-g.github.io/tutorials/yolo-class-balancing/
Works on Ultralytics classification, keeps val/test unweighted, and adds a no-op
`reset()` to loaders so Ultralytics' trainer doesn't crash.
"""

def _extract_targets(dataset):
    """Return 1D numpy array of class indices for each sample; handle many layouts."""
    # 1) Direct attributes commonly used
    for attr in ("targets", "labels", "y", "cls"):
        if hasattr(dataset, attr):
            arr = getattr(dataset, attr)
            if arr is not None and len(arr):
                return np.asarray(arr, dtype=np.int64)

    # 2) torchvision-like containers on the dataset itself
    for attr in ("samples", "imgs"):
        if hasattr(dataset, attr):
            samples = getattr(dataset, attr)
            if samples is not None and len(samples):
                def get_label(s):
                    if isinstance(s, dict):
                        for k in ("label", "class", "target", "cls"):
                            if k in s:
                                return int(s[k])
                    if isinstance(s, (list, tuple)):
                        # take the last int-like entry
                        for v in reversed(s):
                            if isinstance(v, (int, np.integer)):
                                return int(v)
                    raise ValueError("Unrecognized sample format")
                return np.asarray([get_label(s) for s in samples], dtype=np.int64)

    # 3) Some wrappers keep the real dataset under .dataset
    if hasattr(dataset, "dataset") and dataset.dataset is not None:
        return _extract_targets(dataset.dataset)

    raise RuntimeError(
        "Could not access class targets from dataset. "
        "Checked: targets/labels/y/cls, samples/imgs, and nested .dataset.*"
    )

class WeightedClassificationTrainer(ClassificationTrainer):
    """Trainer that oversamples minority classes for TRAIN via WeightedRandomSampler."""
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        dataset = self.build_dataset(dataset_path, mode=mode)

        # ---- Validation / Test: use Ultralytics helper (note: 'batch', not 'batch_size') ----
        if mode != "train":
            loader = build_dataloader(
                dataset=dataset,
                batch=batch_size,          # Ultralytics uses 'batch' here
                workers=self.args.workers,
                rank=rank,
                shuffle=False,
                drop_last=self.args.compile,
            )
            # Add a no-op reset so Ultralytics' trainer can call loader.reset()
            setattr(loader, "reset", lambda: None)
            return loader

        # ---- Train: build weighted sampler + plain PyTorch DataLoader ----
        targets = _extract_targets(dataset)
        if targets.size == 0:
            raise RuntimeError("Dataset has zero samples.")

        num_classes = int(targets.max()) + 1
        class_counts = np.bincount(targets, minlength=num_classes)
        class_counts[class_counts == 0] = 1  # avoid /0

        # Inverse-frequency weighting (aggressive). Soften by **0.5 if needed.
        class_weights = class_counts.sum() / class_counts
        # class_weights = (class_counts.sum() / class_counts) ** 0.5  # softer option

        sample_weights = torch.as_tensor(class_weights[targets], dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # ~1 epoch worth of samples
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # sampler controls sampling
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=self.args.compile,
            persistent_workers=bool(self.args.workers > 0),
        )
        # Add a no-op reset so Ultralytics' trainer can call loader.reset()
        setattr(loader, "reset", lambda: None)
        return loader
