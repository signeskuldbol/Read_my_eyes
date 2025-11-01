# This script extracts blink-related AU clips from the original data.
import json
import os
import subprocess
from pathlib import Path
import sys
import random

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from Read_my_eyes.configs import CREATE_DATASETS_FOLDER_DIR

FFMPEG = r"C:/ffmpeg/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"


"""
# if you want to make clips of specific actions use this:

AU = [
    "AU143",   # blink
    "AU47",    # half blink
    "AU145",   # full blink
]

# for background use this: 

# if you want to make sure the clips of interest 
# does not contain some specefik movement use this:
AU_avoid = [
    "AU143", "AU143L", "AU143R",   # blink
    "AU47",  "AU47L",  "AU47R",    # half blink
    "AU145", "AU145L", "AU145R",   # full blink
    ]

AU_background = ["AD51", "AD38", "EAD104", "VC70", "AD53", "AD1", "AD58"]
"""

"""
This code creates video clips for specific action units (AUs) from a set of original videos 
and their annotations. It reads the annotations from a JSON file, extracts clips corresponding
to the specified AUs, and saves them to an output directory. The clips can be padded with random
time before and after the action to provide context. Additionally, it can avoid creating clips 
that contain specific unwanted AUs. This is usefull when making background clips.

For background clips a duration of 16 frames is used.
for action clips a random padding is used to avoid having actions placed at the middle always.
"""




# The AU, EAD, etc of interest
AU = ["AD51", "AD38", "EAD104", "VC70", "AD53", "AD1", "AD58"] #TODO: change to desired AUs for action clips

AU_avoid = [
    "AU143", "AU143L", "AU143R",   # blink
    "AU47",  "AU47L",  "AU47R",    # half blink
    "AU145", "AU145L", "AU145R",   # full blink
    ] #TODO: comment out if not needed
# obs: go to TODO and set to "background" or au as needed to control naming

desired_number_frames = 16  # for background clips

# Base paths
videos_path = CREATE_DATASETS_FOLDER_DIR / "original_videos_annotations" / "videos"
annotations_file_path = CREATE_DATASETS_FOLDER_DIR / "original_videos_annotations" / "JSONAnnotations" / "annotations.json"

name_output = "background"  #"background" or  "action" #TODO: change


FPS = 25  
Frame_time = ( 1.0 / FPS ) # used for padding 

def parse_time_to_seconds(t: str) -> float:
    # t like "HH:MM:SS.mmm"
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

# format times for ffmpeg
def fmt(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    t_int = int(t)
    hh = t_int // 3600
    mm = (t_int % 3600) // 60
    ss = t_int % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def check_for_overlap(
                            data: dict,                
                            video_name: str,
                            clip_start_time: str,
                            clip_duration_s: float,
                            forbidden_aus: list[str],
                            *,
                            exact_match: bool = False,
                            safety_pad_s: float = 0.5 # extra padding to avoid edge overlaps
                        ) -> bool:
    if video_name not in data:
        return False

    cs = parse_time_to_seconds(clip_start_time)
    ce = cs + float(clip_duration_s)

    for row in data[video_name]:
        code = str(row.get("Code", "")).strip()
        if exact_match:
            if code not in forbidden_aus:
                continue
        else:
            if not any(fau in code for fau in forbidden_aus):
                continue

        st = row.get("Start time")
        dur = row.get("Duration (s)")
        
        if not st or dur is None:
            continue

        try:
            rs = parse_time_to_seconds(str(st))
            rd = float(dur)
        except Exception:
            continue

        fs = max(0.0, rs - safety_pad_s)
        fe = rs + rd + safety_pad_s

        if max(cs, fs) < min(ce, fe):
            return True

    return False



# Find list of videos. make name be the same as in Json file
list_video_names = [ f for f in os.listdir(videos_path) if f.lower().endswith(".mp4")]



with open(annotations_file_path, "r", encoding="utf-8") as json_file:
    video_data = json.load(json_file)

i = 0
for au in AU:
    for video in list_video_names:
        # Skip videos that have no annotations in JSON
        if video not in video_data:
            print(f"[WARN] {video} not in annotations.json")
            continue

        for action_unit in video_data[video]:
            code = action_unit.get("Code", "")
            if au in code: # if you want to save all of au143, AU47, and AU145 to the same folder replace = with in and set AU = [AU143, AU47, AU145] then you also get Left and right versions 
                
                # Times & duration
                start_str = action_unit.get("Start time")
                dur_val = action_unit.get("Duration (s)")

                if not start_str or dur_val is None:
                    continue

                S = parse_time_to_seconds(start_str)
                D = float(dur_val)


                # OBS: this is only to get start time as it is garanteed to get 16 frames when using FFMPEG
                
                # compute how much padding/cropping is needed for 16 frames
                number_of_frames = max(0, int(D * FPS))
                delta = desired_number_frames - number_of_frames  # + => too short, - => too long
                shift_frames = -delta / 2.0  # negative delta -> positive shift (move start forward)
                new_start = S + (shift_frames * Frame_time)
                new_start = max(0.0, new_start)
                new_duration = desired_number_frames * Frame_time

                action_start = fmt(new_start)
                duration_str = f"{new_duration:.3f}"

                try:
                    if AU_avoid is not None:
                        if check_for_overlap(video_data,
                                            video_name=video,
                                            clip_start_time=action_start,
                                            clip_duration_s=new_duration,
                                            forbidden_aus=AU_avoid,
                                            exact_match=False,
                                            safety_pad_s=Frame_time * 2): # add safety so no overlap happens
                            continue
                except NameError:
                    pass   # AU_avoid was never defined


                input_file_path = videos_path / video

                # Output folder per AU
                output_dir = CREATE_DATASETS_FOLDER_DIR / "datasets" / "New" / "background"    # TODO au when action else "background"
                output_dir.mkdir(parents=True, exist_ok=True)

                out_path = output_dir / f"{code}_{Path(video).stem.replace('_Video', '')}_{name_output}_{i}.mp4"

                # Note: stream copy is fast but not frame-accurate.
                # For more accurate cuts, replace "-c", "copy" with codec re-encode options.
                cmd = [
                    FFMPEG, "-y",
                    "-i", str(input_file_path),     
                    "-ss", action_start,            
                    "-t", duration_str,
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-c:a", "copy",
                    str(out_path),
                    ]
                
                subprocess.run(cmd, check=False)
                i += 1
