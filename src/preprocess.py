"""
preprocess.py — M1: Data preprocessing for Cats vs Dogs classification.

Steps:
  1. Scan raw dataset folders (PetImages/Cat, PetImages/Dog)
  2. Remove corrupted / unreadable images (this dataset famously has a few)
  3. Resize every image to 224x224 RGB
  4. Split into train/val/test = 80/10/10
  5. Save to data/processed/{train,val,test}/{cat,dog}/

Data augmentation (random flips/rotations) is applied at TRAINING time in
train.py via torchvision transforms — the standard practice — so the files
saved here are clean, deterministic 224x224 RGB images.

Usage:
    python src/preprocess.py --raw-dir data/raw/PetImages --out-dir data/processed
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

IMG_SIZE = (224, 224)
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
CLASSES = {"Cat": "cat", "Dog": "dog"}   # raw folder name -> processed folder name
SEED = 42


def is_valid_image(path: Path) -> bool:
    """Return True if the file is a readable, non-corrupted image.

    The Kaggle Cats/Dogs dataset contains a handful of truncated or
    non-image files; training would crash on them, so we filter here.
    """
    try:
        with Image.open(path) as img:
            img.verify()                      # checks integrity, cheap
        with Image.open(path) as img:         # verify() invalidates the handle
            img.convert("RGB")                # must be convertible to RGB
        return True
    except Exception:
        return False


def resize_and_save(src: Path, dst: Path) -> None:
    """Resize one image to 224x224 RGB and save as JPEG."""
    with Image.open(src) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=95)


def split_files(files: list, splits: dict = SPLITS, seed: int = SEED) -> dict:
    """Shuffle file list deterministically and split into train/val/test.

    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    files = sorted(files)                     # deterministic base order
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def preprocess(raw_dir: Path, out_dir: Path, limit_per_class: int | None = None) -> dict:
    """Run the full preprocessing pipeline. Returns per-split file counts."""
    stats = {"corrupted_removed": 0}
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for raw_cls, out_cls in CLASSES.items():
        cls_dir = raw_dir / raw_cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}\n"
                f"Point --raw-dir at the folder that CONTAINS 'Cat' and 'Dog'."
            )

        candidates = [p for p in cls_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if limit_per_class:
            candidates = sorted(candidates)[:limit_per_class]

        valid = []
        for p in candidates:
            if is_valid_image(p):
                valid.append(p)
            else:
                stats["corrupted_removed"] += 1
                print(f"  [skip] corrupted image: {p}")

        for split, files in split_files(valid).items():
            for src in files:
                dst = out_dir / split / out_cls / f"{src.stem}.jpg"
                resize_and_save(src, dst)
            key = f"{split}/{out_cls}"
            stats[key] = len(files)
            print(f"  {key:<12}: {len(files)} images")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/PetImages"),
                        help="Folder containing Cat/ and Dog/ subfolders")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Optional cap per class for quick experiments")
    args = parser.parse_args()

    print(f"Preprocessing {args.raw_dir} -> {args.out_dir}")
    stats = preprocess(args.raw_dir, args.out_dir, args.limit_per_class)
    print("\nDone. Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
