# This script extracts blink-related AU clips from the original data.
import json
import os
import math
import subprocess
from pathlib import Path
import sys
# add parent folder to Python path so configs.py is found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Read_my_eyes.configs import CREATE_DATASETS_FOLDER_DIR

FFMPEG = r"C:/ffmpeg/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"


"""
AU = [
    "AU143", "AU143L", "AU143R",   # blink
    "AU47",  "AU47L",  "AU47R",    # half blink
    "AU145", "AU145L", "AU145R",   # full blink
]

AU = [
    "AU143",   # blink
    "AU47",    # half blink
    "AU145",   # full blink
]

# if you want to make sure the clips of interest 
# does not contain some specefik movement use this:
AU_avoid = [
    "AU143", "AU143L", "AU143R",   # blink
    "AU47",  "AU47L",  "AU47R",    # half blink
    "AU145", "AU145L", "AU145R",   # full blink
    ]

"""
# The AU, EAD, etc of interest
AU = ["AD51", "AD38", "EAD104", "VC70", "AD53", "AD1", "AD58"]

AU_avoid = [
    "AU143", "AU143L", "AU143R",   # blink
    "AU47",  "AU47L",  "AU47R",    # half blink
    "AU145", "AU145L", "AU145R",   # full blink
    ]

name_output = "background"  #"background" or  "action"  



# Base paths
base_path = CREATE_DATASETS_FOLDER_DIR
videos_path = base_path / "original_videos_annotations" / "videos"
annotations_file_path = base_path / "original_videos_annotations" / "JSONAnnotations" / "annotations.json"
FPS = 25  
Frame_time = 2 * ( 1.0 / FPS ) # used for padding 

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
                            safety_pad_s: float = 0.0
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

                # one-frame padding
                new_start = max(0.0, S - Frame_time)
                new_duration = D + 2.0 * Frame_time # 2*Frame_time to account for start padding

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
                                            safety_pad_s=Frame_time):
                            continue
                except NameError:
                    pass   # AU_avoid was never defined


                input_file_path = videos_path / video

                # Output folder per AU
                output_dir = base_path / "datasets" / "AUs" / "background"    # au when action else "background"
                output_dir.mkdir(parents=True, exist_ok=True)

                out_path = output_dir / f"{Path(video).stem.replace('_Video', '')}_{name_output}_{code}_{i}.mp4"

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
