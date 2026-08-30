"""
performance_tracking.py — M5: Model performance tracking post-deployment.

Sends a batch of labeled test images to the DEPLOYED service, compares the
predictions against the true labels (folder names), and writes an accuracy
report. This simulates collecting "real requests with true labels".

Usage (after `docker compose up -d`):
    python scripts/performance_tracking.py \
        --url http://localhost:8000 \
        --test-dir data/processed/test \
        --n-per-class 15
"""

import argparse
import csv
import random
from pathlib import Path

import requests

CLASSES = ["cat", "dog"]


def sample_test_images(test_dir: Path, n_per_class: int) -> list:
    """Pick n images per class from data/processed/test/{cat,dog}."""
    samples = []
    rng = random.Random(42)
    for cls in CLASSES:
        files = sorted((test_dir / cls).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No test images in {test_dir / cls} — "
                                    "run preprocess.py first.")
        for path in rng.sample(files, min(n_per_class, len(files))):
            samples.append((path, cls))
    rng.shuffle(samples)
    return samples


def query_service(base_url: str, image_path: Path) -> str:
    with open(image_path, "rb") as fh:
        resp = requests.post(
            f"{base_url}/predict",
            files={"file": (image_path.name, fh, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["label"]


def main(base_url: str, test_dir: Path, n_per_class: int, report: Path) -> None:
    samples = sample_test_images(test_dir, n_per_class)
    rows, correct = [], 0

    print(f"Sending {len(samples)} labeled requests to {base_url} ...")
    for path, true_label in samples:
        pred = query_service(base_url, path)
        match = pred == true_label
        correct += match
        rows.append({"file": path.name, "true": true_label,
                     "predicted": pred, "correct": match})
        print(f"  {path.name:<28} true={true_label:<4} pred={pred:<4} "
              f"{'OK' if match else 'WRONG'}")

    accuracy = correct / len(rows)
    print(f"\nPost-deployment accuracy: {correct}/{len(rows)} = {accuracy:.1%}")

    with open(report, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "true", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report written to {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-deployment performance tracking")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--n-per-class", type=int, default=15)
    parser.add_argument("--report", type=Path, default=Path("performance_report.csv"))
    args = parser.parse_args()
    main(args.url, args.test_dir, args.n_per_class, args.report)
