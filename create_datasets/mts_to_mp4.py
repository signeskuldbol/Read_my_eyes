import subprocess
from pathlib import Path

# -------- CONFIG --------
Workspace_ROOT = Path(__file__).parent.parent.parent.parent
print(Workspace_ROOT)

VIDEO_ROOT = Workspace_ROOT / "create_datasets" / "datasets" / "full_without_avoid" / "pain" # <-- folder to search in
OUT_ROOT   = Workspace_ROOT / "create_datasets" / "datasets" / "full_without_avoid" / "pain_new"  # <-- where converted videos go

CRF = 18                 # quality (18 = visually near-lossless)
PRESET = "veryfast"      # speed vs compression
# ------------------------


def convert_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-i", str(src),
        "-vsync", "0",
        "-fflags", "+genpts",
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mts_files = sorted(VIDEO_ROOT.rglob("*.mts"))
    if not mts_files:
        print(f"[INFO] No .mts files found under: {VIDEO_ROOT}")
        return

    print(f"[INFO] Found {len(mts_files)} .mts files")

    ok = 0
    fail = 0

    for src in mts_files:
        rel = src.relative_to(VIDEO_ROOT)
        dst = (OUT_ROOT / rel).with_suffix(".mp4")

        if dst.exists():
            print(f"[SKIP] {dst}")
            continue

        print(f"[CONVERT] {src} -> {dst}")
        try:
            convert_one(src, dst)
            ok += 1
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed: {src}")
            fail += 1

    print(f"[DONE] Converted={ok}  Failed={fail}  Output={OUT_ROOT}")


if __name__ == "__main__":
    main()
