# video_splitter.py
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

"""
Use this script to split videos into clips of exactly T frames each.
Videos with fewer than T frames will have frames repeated as evenly as possible to reach T frames.
Videos with more than T frames will be split into multiple overlapping clips of T frames each,
ensuring full coverage of the original video.
The output clips will be saved in a specified output directory,
"""


@dataclass
class SplitConfig:
    input_dir: Path
    output_dir: Path
    T: int
    # Recognized video extensions (lowercase)
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError("T must be a positive integer (frames per clip).")
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class VideoSplitter:
    def __init__(self, config: SplitConfig) -> None:
        self.cfg = config

    # ----------------------------
    # Public API
    # ----------------------------
    def process_all(self) -> None:
        """Process all videos under input_dir recursively."""
        videos = sorted([
            p for p in self.cfg.input_dir.rglob("*")
            if p.suffix.lower() in self.cfg.extensions and p.is_file()
        ])
        if not videos:
            print(f"[INFO] No videos found in {self.cfg.input_dir}")
            return

        for vpath in videos:
            rel = vpath.relative_to(self.cfg.input_dir)
            print(f"[INFO] Processing: {rel}")
            try:
                self._process_single_video(vpath)
            except Exception as e:
                print(f"[WARN] Skipping {rel} due to error: {e}")

    # ----------------------------
    # Internals
    # ----------------------------
    def _process_single_video(self, vpath: Path) -> None:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {vpath}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps        = cap.get(cv2.CAP_PROP_FPS)
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size       = (width, height)

        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video reports zero frames: {vpath}")
        if fps is None or fps <= 0:
            # Fallback to a safe default if metadata is missing
            fps = 25.0

        T = self.cfg.T
        if frame_count >= T:
            windows = self._generate_windows_cover_all(frame_count, T)  # list of (start,end) inclusive, 0-based
            for clip_idx, (s, e) in enumerate(windows, start=1):
                indices = list(range(s, e + 1))  # exactly T indices by construction
                self._write_clip(cap, indices, size, fps, vpath, clip_idx)
        else:
            # Need to repeat frames to reach T, evenly distributed
            indices = self._expand_indices_even(frame_count, T)          # length T
            self._write_clip(cap, indices, size, fps, vpath, clip_idx=1)

        cap.release()

    def _select_fourcc(self, ext_lower: str) -> int:
        """Choose a reasonable FourCC for common containers."""
        if ext_lower in (".mp4", ".m4v", ".mov", ".mkv"):
            return cv2.VideoWriter_fourcc(*"mp4v")
        if ext_lower == ".avi":
            return cv2.VideoWriter_fourcc(*"XVID")
        # Fallback
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _write_clip(self, cap: cv2.VideoCapture, indices: List[int],
                    size: Tuple[int, int], fps: float, src_path: Path, clip_idx: int) -> None:
        # --- Build matching output path (preserve subfolders) ---
        rel_path = src_path.relative_to(self.cfg.input_dir)
        out_subdir = self.cfg.output_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Keep original name + add _i before extension
        stem = src_path.stem
        ext = src_path.suffix  # keep original extension
        out_name = f"{stem}_{clip_idx}{ext}"
        out_path = out_subdir / out_name

        # --- Create video writer ---
        fourcc = self._select_fourcc(ext.lower())
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for: {out_path}")

        # --- Write frames ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in indices:
            # Safety clamp
            idx = int(max(0, min(idx, total_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Try one more time at clamped index
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok2, frame = cap.read()
                if not ok2 or frame is None:
                    writer.release()
                    raise RuntimeError(f"Failed to read frame {idx} from {src_path}")

            if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)

        writer.release()
        print(f"[OK] {out_path.relative_to(self.cfg.output_dir)} ({len(indices)} frames)")

    # ----------------------------
    # Windowing logic (N >= T)
    # ----------------------------
    def _generate_windows_cover_all(self, N: int, T: int) -> List[Tuple[int, int]]:
        """
        For N >= T:
        Create K = ceil(N / T) windows of length T that cover all frames [0..N-1].
        Distribute the stride integers as evenly as possible so coverage spans the whole video.

        Examples (0-based):
          N=11, T=3 -> K=4, strides [3,3,2] -> starts [0,3,6,8]
                        windows [0..2], [3..5], [6..8], [8..10]
          N=11, T=5 -> K=3, strides [3,3] -> starts [0,3,6]
                        windows [0..4], [3..7], [6..10]
        """
        K = math.ceil(N / T)
        if K == 1:
            # one window: clamp to [0..N-1] if N<T this branch won't be used
            return [(0, min(T, N) - 1)]

        S = N - T        # total spread to distribute over (K-1) gaps
        gaps = K - 1
        base = S // gaps
        rem  = S % gaps
        # First 'rem' gaps get (base+1), remaining get 'base'
        strides = [base + 1] * rem + [base] * (gaps - rem)

        starts = [0]
        for s in strides:
            starts.append(starts[-1] + s)

        windows: List[Tuple[int, int]] = []
        for st in starts:
            en = st + T - 1
            if en >= N:
                en = N - 1
                st = en - T + 1
            windows.append((st, en))
        return windows

    # ----------------------------
    # Padding logic (N < T)
    # ----------------------------
    def _expand_indices_even(self, N: int, T: int) -> List[int]:
        """
        Returns a length-T list of frame indices in [0..N-1] that
        repeats frames as evenly as possible.

        Example spirit for N=4, T=6: 0,1,1,2,3,3 (1-based: 1,2,2,3,4,4)
        """
        if N <= 0:
            raise ValueError("Video reports zero frames.")
        if N == 1:
            return [0] * T

        extra = T - N
        reps = [1] * N
        if extra > 0:
            # Distribute extra repeats at approximately even positions (avoid front-only bias)
            pos = np.linspace(1, N - 1, extra, endpoint=True)
            pos = np.rint(pos).astype(int)
            for p in pos:
                reps[p] += 1

        indices: List[int] = []
        for i, r in enumerate(reps):
            indices.extend([i] * r)

        # Safety: exact length T
        if len(indices) > T:
            indices = indices[:T]
        elif len(indices) < T:
            indices.extend([N - 1] * (T - len(indices)))
        return indices


# ----------------------------
# Optional: simple example usage guarded by __main__
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("T", type=int)
    args = parser.parse_args()

    cfg = SplitConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        T=args.T,
    )
    VideoSplitter(cfg).process_all()