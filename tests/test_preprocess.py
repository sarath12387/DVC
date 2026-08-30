"""
test_preprocess.py — M3: Unit tests for data preprocessing functions.

These tests generate tiny synthetic images on the fly, so they run in CI
without the real Kaggle dataset.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import IMG_SIZE, is_valid_image, resize_and_save, split_files


def _make_image(path: Path, size=(64, 48), color=(255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


# ---------------------------------------------------------------- is_valid_image
def test_valid_image_is_accepted(tmp_path):
    img = _make_image(tmp_path / "ok.jpg")
    assert is_valid_image(img) is True


def test_corrupted_image_is_rejected(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is definitely not a jpeg")
    assert is_valid_image(bad) is False


def test_truncated_image_is_rejected(tmp_path):
    """A real image cut off halfway (the classic corruption in this dataset)."""
    ok = _make_image(tmp_path / "full.jpg")
    data = ok.read_bytes()
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(data[: len(data) // 2])
    assert is_valid_image(truncated) is False


# ---------------------------------------------------------------- resize_and_save
def test_resize_produces_224_rgb(tmp_path):
    src = _make_image(tmp_path / "src.jpg", size=(500, 300))
    dst = tmp_path / "out" / "dst.jpg"
    resize_and_save(src, dst)

    assert dst.exists()
    with Image.open(dst) as img:
        assert img.size == IMG_SIZE          # exactly 224x224
        assert img.mode == "RGB"             # exactly 3 channels


def test_resize_handles_grayscale_input(tmp_path):
    src = tmp_path / "gray.jpg"
    Image.new("L", (100, 100), 128).save(src)  # 1-channel image
    dst = tmp_path / "gray_out.jpg"
    resize_and_save(src, dst)
    with Image.open(dst) as img:
        assert img.mode == "RGB"             # converted to 3 channels


# ---------------------------------------------------------------- split_files
def test_split_ratios_are_80_10_10():
    files = [f"img_{i}.jpg" for i in range(1000)]
    splits = split_files(files)

    assert len(splits["train"]) == 800
    assert len(splits["val"]) == 100
    assert len(splits["test"]) == 100


def test_split_has_no_overlap_and_no_loss():
    files = [f"img_{i}.jpg" for i in range(101)]  # non-divisible count
    splits = split_files(files)
    all_out = splits["train"] + splits["val"] + splits["test"]

    assert len(all_out) == len(files)             # nothing dropped
    assert len(set(all_out)) == len(files)        # nothing duplicated


def test_split_is_deterministic():
    files = [f"img_{i}.jpg" for i in range(50)]
    assert split_files(files) == split_files(files)
