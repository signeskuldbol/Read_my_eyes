# ndjson_to_yolo_dataset.py
import json, os, shutil
from pathlib import Path

# ---- Paths ----
READ_MY_EYES_DIR = Path(__file__).parent.parent.resolve()  # project root (works on C: or D:)
NDJSON = READ_MY_EYES_DIR / "yolo_approach" / "annotations" / "eyes_reviewed.ndjson"
OUT    = READ_MY_EYES_DIR.parent / "yolo_data" / "horse_eyes"
# ----------------------

# Behavior toggles
OVERWRITE_EXISTING = True  # set True to overwrite dest files atomically if they exist

def _samefile(a: Path, b: Path) -> bool:
    """Robust samefile check for Windows (works with hardlinks)."""
    try:
        return os.path.samefile(a, b)
    except Exception:
        # Fall back: compare resolved paths; not perfect across hardlinks but avoids crashes
        try:
            return a.resolve(strict=False) == b.resolve(strict=False)
        except Exception:
            return False

def safe_link_or_copy(src: Path, dst: Path):
    """
    Prefer hardlink for speed/space; fall back to copy.
    - If dst exists:
        - If same file → skip
        - Else overwrite only if OVERWRITE_EXISTING=True (atomic replace), otherwise skip
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if _samefile(src, dst):
            return "skip-same"
        if not OVERWRITE_EXISTING:
            return "skip-exists"
        # atomic overwrite
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return "copy-replace"

    # Try hardlink; if not possible, copy
    try:
        os.link(src, dst)
        return "link"
    except FileExistsError:
        # created by a race; treat as done
        return "skip-exists"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"

def main():
    if not NDJSON.exists():
        raise FileNotFoundError(
            f"NDJSON not found:\n  {NDJSON}\n"
            f"Check your drive/path. Current project root:\n  {READ_MY_EYES_DIR}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    names = {}
    images = []

    # Read NDJSON
    with NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("type")
            if t == "dataset":
                cn = o.get("class_names", {}) or {}
                names = {int(k): v for k, v in cn.items()} if cn else {}
            elif t == "image":
                images.append(o)

    if not names:
        names = {0: "eye"}  # default if header missing

    # counters
    n_train = n_val = 0
    c_stats = {"link":0, "copy":0, "copy-replace":0, "skip-exists":0, "skip-same":0}

    for rec in images:
        split = rec.get("split", "train")
        src = Path(rec["file"])
        if not src.exists():
            print(f"[WARN] missing file: {src}")
            continue

        # copy image
        dst_img = OUT / "images" / split / src.name
        how = safe_link_or_copy(src, dst_img)
        c_stats[how] = c_stats.get(how, 0) + 1

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
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

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

    # summary
    print(f"[OK] Built YOLO dataset at: {OUT}")
    print(f"      train images: {n_train}")
    print(f"      val images:   {n_val}")
    print(f"      yaml:         {(OUT/'data.yaml')}")
    print(f"      ops: {c_stats}")

    # quick sanity: list a few files
    ti = list((OUT / "images" / "train").glob("*.*"))[:3]
    vi = list((OUT / "images" / "val").glob("*.*"))[:3]
    print("[sample train]", [p.name for p in ti])
    print("[sample val]  ", [p.name for p in vi])

if __name__ == "__main__":
    main()
