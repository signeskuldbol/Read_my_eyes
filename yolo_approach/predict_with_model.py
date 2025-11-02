import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch

# =========================
# Settings
# =========================
"""
This code takes a video as input and uses a YOLO model to predict if the video contains any actions: eye closed, eye half-blink, or eye open.
OBS: the video should already be cropped to the eye region using the crop_to_eye_with_yolo_eye_detector.py script.
"""

Base_path = Path(__file__).parent

# Model
MODEL_PATH_classify = Base_path / "yolov12n_eye_classify_3_class.pt"

# Dataset I/O 
INPUT_ROOT  = Base_path / "datasets" / "New"
VIDEO_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =========================
# Model & optional class filter
# =========================
model = YOLO(str(MODEL_PATH_classify))
CLASS_NAMES = ["eye", "eye_half_blink", "eye_full_blink"]
# =========================
# Process video 
# =========================
