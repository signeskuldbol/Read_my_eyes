from pathlib import Path
import cv2, os





#configs 
unpadded_videos_dir = Path(__file__).parent.parent.resolve()  / "create_datasets" / "datasets" / "unpadded_dataset"
output_labels_dir = Path(__file__).parent.parent.resolve()  / "yolo_approach" / "labels"


# iterate through images and label them for yolo classification
