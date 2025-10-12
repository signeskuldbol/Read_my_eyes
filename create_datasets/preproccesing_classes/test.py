# my_processing_pipeline.py
from pathlib import Path
from split_videos_into_new_with_T_frames import SplitConfig, VideoSplitter

# Define your paths and parameters
Base_path = Path(__file__).parent
input_folder = Base_path / "datasets" / "final_no_padding"
output_folder = Base_path / "datasets" / "T_cropped_no_pad"
T = 6  # number of frames per clip

# Create config
config = SplitConfig(
    input_dir=input_folder,
    output_dir=output_folder,
    T=T
)

# Create and run the splitter
splitter = VideoSplitter(config)
splitter.process_all()
