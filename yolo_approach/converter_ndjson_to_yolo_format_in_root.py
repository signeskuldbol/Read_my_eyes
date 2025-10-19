# ndjson_to_yolo_dataset.py
import json, os, shutil
from pathlib import Path

# ---- EDIT THESE 2 ----
NDJSON = Path(r"C:\Users\Signe Møller\OneDrive\Skrivebord\Job\Read_my_eyes\yolo_approach\annotations\eyes_copy.ndjson")
OUT    = Path(r"C:\_ultra_data\horse_eyes")  # ASCII, not on OneDrive
# ----------------------

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # hardlink if possible (fast, no extra disk); falls back to copy
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = {}
    images = []

    with NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") == "dataset":
                # read class names if present
                cn = o.get("class_names", {})
                # normalize keys to int
                names = {int(k): v for k, v in cn.items()} if cn else {}
            elif o.get("type") == "image":
                images.append(o)

    if not names:
        # default single class if header missing/empty
        names = {0: "eye"}

    # counters
    n_train = n_val = 0

    for rec in images:
        split = rec.get("split", "train")
        src = Path(rec["file"])
        if not src.exists():
            print(f"[WARN] missing file: {src}")
            continue

        # copy image
        dst_img = OUT / "images" / split / src.name
        safe_link_or_copy(src, dst_img)

        # write YOLO txt labels
        boxes = rec.get("annotations", {}).get("boxes", [])
        lines = []
        for b in boxes:
            if len(b) < 5:
                continue
            cls, cx, cy, w, h = b[:5]
            cls = int(cls)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dst_lbl = OUT / "labels" / split / (src.stem + ".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        (dst_lbl).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if split == "train": n_train += 1
        elif split == "val": n_val += 1

    # write data.yaml
    yaml = [
        f'path: {OUT.as_posix()}',
        'train: images/train',
        'val: images/val',
        'names:'
    ] + [f'  {i}: {n}' for i, n in sorted(names.items())]

    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"[OK] Built YOLO dataset at: {OUT}")
    print(f"      train images: {n_train}")
    print(f"      val images:   {n_val}")
    print(f"      yaml:         {(OUT/'data.yaml')}")
    # quick sanity: list a few files
    ti = list((OUT / "images" / "train").glob("*.*"))[:3]
    vi = list((OUT / "images" / "val").glob("*.*"))[:3]
    print("[sample train]", [p.name for p in ti])
    print("[sample val]  ", [p.name for p in vi])

if __name__ == "__main__":
    main()
