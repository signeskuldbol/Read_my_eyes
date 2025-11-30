# run_videomae_on_folder.py

import csv
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# -----------------------
# CONFIG – ADJUST THESE
# -----------------------
WORKSPACE_PATH = Path(__file__).parent.parent  # "Read_my_eyes" root

# folder where Trainer saved the fine-tuned model
MODEL_DIR = WORKSPACE_PATH / "Video_MAE" / "VideoMAE_binary_output_new_crop_method" / "model_final_Layers_trained8alpha1.5_2_classes"

# folder with videos you want to classify
INPUT_VIDEOS_DIR = WORKSPACE_PATH / "create_datasets" / "datasets" / "NEW_split_cropped_new_way" / "test"

# where to save CSV with predictions
OUTPUT_CSV = WORKSPACE_PATH / "Video_MAE" / "outputs" / f"results_{INPUT_VIDEOS_DIR.name}.csv"

# video sampling settings (should match training)
NUM_FRAMES_TO_SAMPLE = 16
SAMPLE_RATE = 1
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

# -----------------------
# Load processor & model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# get spatial size from processor (same as training)
if "shortest_edge" in image_processor.size:
    size_hw = image_processor.size["shortest_edge"]
    resize_to = (size_hw, size_hw)
else:
    resize_to = (image_processor.size["height"], image_processor.size["width"])

spatial_transforms = Compose([
    Resize(256, interpolation=Image.BICUBIC),
    CenterCrop(resize_to[0]),
])

id2label = model.config.id2label  # comes from fine-tuned config


# -----------------------
# Frame sampling & loading
# -----------------------
def sample_indices(total_frames, num_frames, rate):
    """Same logic as training: fixed-length clip, uniform fallback."""
    required = (num_frames - 1) * rate + 1
    if total_frames >= required:
        start = 0
        idx = start + np.arange(0, num_frames * rate, rate)
        return idx.astype(np.int64)
    else:
        return np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)


def load_and_preprocess(video_path):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")

    idx = sample_indices(total, NUM_FRAMES_TO_SAMPLE, SAMPLE_RATE)
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,C) uint8

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames = [spatial_transforms(f) for f in pil_frames]

    inputs = image_processor(pil_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]  # (T,C,H,W)
    return pixel_values


# -----------------------
# Inference loop
# -----------------------
def iter_videos(root: Path):
    for p in root.rglob("*"):
        if p.suffix in VIDEO_EXTS and p.is_file():
            yield p


def main():
    INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(INPUT_VIDEOS_DIR))
    if not videos:
        print(f"No videos found under {INPUT_VIDEOS_DIR}")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "true_label", "pred_label", "pred_id", "confidence", "correct"])

        for vid in sorted(videos):
            try:
                frames = load_and_preprocess(vid)  # (T,C,H,W)
                with torch.no_grad():
                    logits = model(pixel_values=frames.unsqueeze(0).to(device)).logits  # (1,num_labels)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    pred_id = int(probs.argmax())
                    # Robust lookup for both int-key and str-key configs
                    if isinstance(id2label, dict):
                        if pred_id in id2label:
                            pred_label = id2label[pred_id]
                        elif str(pred_id) in id2label:
                            pred_label = id2label[str(pred_id)]
                        else:
                            pred_label = str(pred_id)  # fallback: just use the id as string
                    else:
                        pred_label = id2label[pred_id]

                    conf = float(probs[pred_id])

            except Exception as e:
                print(f"[ERROR] {vid}: {e}")
                continue

            print(f"{vid}: {pred_label} (id={pred_id}, p={conf:.3f})")
            # Ground truth based on parent folder name
            true_label = vid.parent.name

            # Is prediction correct?
            is_correct = (pred_label == true_label)

            writer.writerow([
                str(vid),
                true_label,
                pred_label,
                pred_id,
                f"{conf:.6f}",
                is_correct
            ])


    print(f"\nSaved predictions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
