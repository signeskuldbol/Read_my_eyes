import os
import tarfile
from pathlib import Path
import numpy as np, torch
from PIL import Image

import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix, classification_report


from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop

import evaluate

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)
import torch.nn as nn
from torch.optim import AdamW

import evaluate


# -----------------------
# Config 
# -----------------------
model_ckpt = "MCG-NJU/videomae-base" 
WORKSPACE_PATH = Path(__file__).parent.parent # read_my_eyes/
output_dir = WORKSPACE_PATH / "Video_MAE" / "VideoMAE_binary_output"


dataset_root = WORKSPACE_PATH/ "create_datasets"/ "datasets"/ "dataset_binary_cropped_split"


#### hyperparameters ####
num_frames_to_sample = 16 # how many frames per video clip
sample_rate = 1 #see each frame (blinks are fast)
batch_size_train = 8 # TODO increase if VRAM. # Videos per GPU step
batch_size_eval = batch_size_train
num_epochs_train = 30 # TODO If training is noisy/unstable (up). If epochs get too slow (down). 
warm_up_ratio = 0.1 # gradually start training. good with pretrained models
logging_steps = 30 # how often to print loss. low value if you want to closely monitor training
fp16_bool = True  # store and compute your tensors using 16-bit floating-points. save memory, faster
gradient_acc_steps = 4  #TODO
metric_for_best_model = "accuracy" # "loss" also possible
save_strategy = "no" # "steps", "no" also possible
input_resolution = 224
N_dont_freeze_last = 8 # how many of the last blocks to keep trainable 
alpha = 1.5  # raise to emphasize minority classes more

# --- Optimizer learning rates ---
# Step size for weight updates. lower if loss oscillates or overfits. high if loss plateaus or slow
lr_encoder_rest = 1e-5        # first layers. keep low (is ignored if layers frozen)
lr_encoder_last = 2e-5        # last N_last_layers layers by default (can be higher as those layers are more task-specific)
lr_classifier_head = 5e-4     # classifier head, set high as it's trained from scratch
weight_decay_encoder = 0.05   # controls weight like regularization. it discourages weights from growing too large by adding a penalty to the loss function. typical range 0.01–0.1
weight_decay_head = 0.0       # no weight decay on head. we dont want to regularize it
fraction_last = 0             #TODO # last % of layers get lr_encoder_last, rest get lr_encoder_rest
#########################

# -----------------------
# Collect files & classes
# -----------------------
def glob_mp4(split):
    return list((dataset_root / split).glob("*/*.mp4"))

train_files = glob_mp4("train")
val_files   = glob_mp4("val")
test_files  = glob_mp4("test")

all_files = train_files + val_files + test_files
print(f"Total videos: {len(all_files)}")

# class labels are the parent folder names of each video
class_labels = sorted({p.parent.name for p in all_files})
label2id = {c: i for i, c in enumerate(class_labels)}
id2label = {i: c for c, i in label2id.items()}
print(f"Unique classes: {class_labels}.")


# -----------------------
# Add class weights 
# -----------------------
from collections import Counter
train_label_ids = [label2id[p.parent.name] for p in train_files]
counts = Counter(train_label_ids)

num_classes = len(class_labels)
total = sum(counts.values())


counts_arr = torch.tensor([max(counts.get(i, 0), 1) for i in range(num_classes)], dtype=torch.float32)

raw = (total / counts_arr).pow(alpha)   # bigger for smaller classes
class_weights = raw / raw.mean()        # normalize to mean ≈ 1
print("Class weights (alpha):", class_weights.tolist())

# -----------------------
# Processor & transforms
# -----------------------

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)

# Match VideoMAE's expected spatial size
if "shortest_edge" in image_processor.size:
    size_hw = image_processor.size["shortest_edge"]
    resize_to = (size_hw, size_hw)
else:
    resize_to = (image_processor.size["height"], image_processor.size["width"])

spatial_transforms = Compose([
    Resize(256, interpolation=Image.BICUBIC),
    CenterCrop(resize_to[0]),
])

def sample_indices(total_frames, num_frames, rate, random_clip=False):
    """
    Returns indices of length `num_frames`, spaced by `rate`.
    If video too short, fall back to linspace over available frames.
    """
    required = (num_frames - 1) * rate + 1
    if total_frames >= required:
        if random_clip:
            # choose a random start so that start + required <= total_frames
            max_start = total_frames - required
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        else:
            start = 0
        idx = start + np.arange(0, num_frames * rate, rate)
        return idx.astype(np.int64)
    else:
        # fallback: uniform sampling across available frames
        return np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)


def load_and_preprocess(video_path, random_clip):
    """
    Loads frames with Decord, applies spatial transforms, then uses HF processor.
    Returns dict with 'pixel_values' (T,C,H,W) tensor and label int.
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")

    idx = sample_indices(total, num_frames_to_sample, sample_rate, random_clip)
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8

    # to PIL, spatial transforms, then processor
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames = [spatial_transforms(f) for f in pil_frames]

    # processor returns dict with 'pixel_values' of shape (1, T, C, H, W)
    inputs = image_processor(pil_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]  # (T, C, H, W)
    return pixel_values


class UCFClipDataset(Dataset):
    def __init__(self, files, label2id, split):
        self.files = files
        self.label2id = label2id
        self.random_clip = (split == "train")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label_name = path.parent.name
        label = self.label2id[label_name]
        pixel_values = load_and_preprocess(path, random_clip=self.random_clip)
        # Return dict; Trainer's data_collator will stack these
        return {
            "pixel_values": pixel_values,  # (T, C, H, W)
            "labels": torch.tensor(label, dtype=torch.long),
        }


train_dataset = UCFClipDataset(train_files, label2id, split="train")
val_dataset   = UCFClipDataset(val_files,   label2id, split="val")
test_dataset  = UCFClipDataset(test_files,  label2id, split="test")

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# -----------------------
# Model
# -----------------------
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    num_labels=len(class_labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# -----------------------
# Freeze backbone (encoder), train only the classifier head
# -----------------------
# Unfreeze last N_dont_freeze_last blocks
num_layers = len(list(model.videomae.encoder.layer))  
for name, p in model.named_parameters():
    if "classifier" in name:
        p.requires_grad = True
    elif "videomae.encoder.layer." in name:
        idx = int(name.split("videomae.encoder.layer.")[1].split(".")[0])
        p.requires_grad = (idx >= num_layers - N_dont_freeze_last)
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params (last {N_dont_freeze_last} blocks + head): {trainable:,} / {total:,}")


# -----------------------
# Trainer setup and training
# -----------------------
# metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=preds, references=eval_pred.label_ids)

def data_collator(examples):
    # examples: list of dicts with pixel_values:(T,C,H,W), labels:int
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)  # (B,T,C,H,W)
    labels = torch.tensor([int(ex["labels"]) for ex in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


# -----------------------
# TrainingArguments 
# -----------------------
common_kwargs = dict(
    output_dir=str(output_dir),
    remove_unused_columns=False,
    per_device_train_batch_size=batch_size_train,      # keep 1 for 8GB GPU
    per_device_eval_batch_size=batch_size_eval,
    num_train_epochs=num_epochs_train,
    warmup_ratio=warm_up_ratio,
    logging_steps=logging_steps,
    report_to=[],                                # disable TensorBoard/W&B
    fp16=fp16_bool,                                   # lower memory usage
    gradient_accumulation_steps=gradient_acc_steps,               # simulate larger batch
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=True if metric_for_best_model != "loss" else False,
    save_strategy=save_strategy,
)

args = TrainingArguments(
    output_dir=str(output_dir),
    remove_unused_columns=False,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size= batch_size_eval,
    num_train_epochs=num_epochs_train,
    warmup_ratio=warm_up_ratio,
    logging_steps=logging_steps,
    report_to=[],
    fp16=fp16_bool,
    gradient_accumulation_steps=gradient_acc_steps,
    load_best_model_at_end=False,  # not supported without eval loop
    save_strategy=save_strategy,
    save_total_limit=2,
)


# -----------------------
# Trainer initialization + optimizer with layer-wise lr decay
# -----------------------
def make_optimizer(m):
    head, last, rest = [], [], []

    
    num_layers = len(list(m.videomae.encoder.layer))
    N_last_layers = max(1, int(round(num_layers * fraction_last)))  
    split_idx = num_layers - N_last_layers                          # boundary index

    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n:
            head.append(p)
        elif "videomae.encoder.layer." in n:
            idx = int(n.split("videomae.encoder.layer.")[1].split(".")[0])
            (last if idx >= split_idx else rest).append(p)
        else:
            rest.append(p)

    print(f"Optimizer groups → rest: {len(rest)} params, last: {len(last)} params, head: {len(head)} params")
    return AdamW([
        {"params": rest, "lr": lr_encoder_rest, "weight_decay": weight_decay_encoder},
        {"params": last, "lr": lr_encoder_last, "weight_decay": weight_decay_encoder},
        {"params": head, "lr": lr_classifier_head, "weight_decay": weight_decay_head},
    ])

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    
    def create_optimizer(self):
        if self.optimizer is None:
            print(">>> Creating custom optimizer with layer-wise LRs ...")
            self.optimizer = make_optimizer(self.model)

    
    def create_scheduler(self, num_training_steps, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

# build the trainer 
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=image_processor,  
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------
# Training and evaluation
# -----------------------
print(" Starting training ...")
train_results = trainer.train()

print(" Training complete. Evaluating on validation set ...")
val_metrics = trainer.evaluate(val_dataset)
print("Validation metrics:", val_metrics)

print(" Evaluating on test set ...")
test_metrics = trainer.evaluate(test_dataset)
print("Test metrics:", test_metrics)


# -----------------------
# Save and see results (metrics)
# -----------------------
# Save the trained weights & config
save_dir = output_dir / f"model_final_Layers_trained{N_dont_freeze_last}alpha{alpha}_2_classes"
save_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(save_dir))             # saves model + tokenizer/processor state
image_processor.save_pretrained(str(save_dir))

# ---- quick single-video prediction ----
test_path = test_files[0]  # or Path("path/to/your/video.mp4")
frames = load_and_preprocess(test_path, random_clip=False)  # (T,C,H,W)
with torch.no_grad():
    logits = model(pixel_values=frames.unsqueeze(0).to(model.device)).logits
pred = logits.argmax(-1).item()
print("Predicted:", id2label[pred], " | True:", test_path.parent.name)


def predict_dataset(ds):
    all_preds, all_labels = [], []
    for i in range(len(ds)):
        ex = ds[i]
        pv = ex["pixel_values"].unsqueeze(0).to(model.device)
        with torch.no_grad():
            pred = model(pixel_values=pv).logits.argmax(-1).item()
        all_preds.append(pred); all_labels.append(int(ex["labels"]))
    return np.array(all_preds), np.array(all_labels)

preds, labels = predict_dataset(test_dataset)
print(classification_report(labels, preds, target_names=class_labels))



# -----------------------
# Coloured confusion matrix 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Build confusion matrices with a consistent label order
cm = confusion_matrix(labels, preds, labels=list(range(len(class_labels))))
with np.errstate(invalid="ignore", divide="ignore"):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # replace NaNs if a class has 0 support

# Save path (same parent as your "final" folder)
cm_path = (output_dir / f"CM__Layers_trained{N_dont_freeze_last}_weights_{alpha}_2_classes.png")
cm_path.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")  # colored heatmap
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Axis ticks & labels
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45, ha="right")
ax.set_yticklabels(class_labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix (row-normalized). Layers_trained:{N_dont_freeze_last} alpha:{alpha} ")

# Annotate cells with "percent (count)"
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        pct = f"{cm_norm[i, j]*100:0.0f}%"
        cnt = int(cm[i, j])
        ax.text(j, i, f"{pct}\n({cnt})",
                ha="center", va="center",
                color="black" if cm_norm[i, j] < 0.6 else "white",
                fontsize=8, fontweight="bold")

fig.tight_layout()
fig.savefig(cm_path, bbox_inches="tight")   
print(f"Saved confusion matrix to: {cm_path}")

#save hyperparameters used in training
hyperparams_path = output_dir / f"hyperparameters_CM_Layers_trained{N_dont_freeze_last}_alpha_{alpha}.txt"
hyperparams_path.parent.mkdir(parents=True, exist_ok=True)
with open(hyperparams_path, "w") as f:
    f.write(f"Training Hyperparameters for CM_Layers_trained_{N_dont_freeze_last}_alpha_{alpha}\n")
    f.write(f"trained on {dataset_root} dataset\n")
    f.write(f"num_frames_to_sample: {num_frames_to_sample}\n")
    f.write(f"sample_rate: {sample_rate}\n")
    f.write(f"batch_size_train: {batch_size_train}\n")
    f.write(f"batch_size_eval: {batch_size_eval}\n")
    f.write(f"num_epochs_train: {num_epochs_train}\n")
    f.write(f"warm_up_ratio: {warm_up_ratio}\n")
    f.write(f"logging_steps: {logging_steps}\n")
    f.write(f"fp16_bool: {fp16_bool}\n")
    f.write(f"gradient_acc_steps: {gradient_acc_steps}\n")
    f.write(f"metric_for_best_model: {metric_for_best_model}\n")
    f.write(f"save_strategy: {save_strategy}\n")
    f.write(f"input_resolution: {input_resolution}\n")
    f.write(f"N_dont_freeze_last: {N_dont_freeze_last}\n")
    f.write(f"alpha (class weight exponent): {alpha}\n")
    f.write(f"Learning Rates:\n")
    f.write(f"  lr_encoder_rest: {lr_encoder_rest}\n")
    f.write(f"  lr_encoder_last: {lr_encoder_last}\n")
    f.write(f"  lr_classifier_head: {lr_classifier_head}\n")
    f.write(f"Weight Decays:\n")
    f.write(f"  weight_decay_encoder: {weight_decay_encoder}\n")
    f.write(f"  weight_decay_head: {weight_decay_head}\n")
    f.write(f"fraction_last: {fraction_last}\n")
print(f"Saved training hyperparameters to: {hyperparams_path}")

plt.show()
