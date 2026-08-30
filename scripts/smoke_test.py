"""
smoke_test.py — M4: Post-deploy smoke test.

Calls the deployed service's /health endpoint and makes one real prediction
call with a generated test image. Exits non-zero on any failure, which makes
the CI/CD pipeline fail (the exact behavior M4 requires).

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
"""

import argparse
import io
import sys

import requests
from PIL import Image


def check_health(base_url: str) -> None:
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("status") == "ok", f"Unexpected health body: {body}"
    assert body.get("model_loaded") is True, "Service is up but model NOT loaded"
    print(f"[PASS] /health -> {body}")


def check_prediction(base_url: str) -> None:
    # Generate a small in-memory test image (no dataset needed on the runner)
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (150, 120, 90)).save(buf, "JPEG")
    buf.seek(0)

    resp = requests.post(
        f"{base_url}/predict",
        files={"file": ("smoke.jpg", buf, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    assert body.get("label") in {"cat", "dog"}, f"Bad label: {body}"
    probs = body.get("probabilities", {})
    assert abs(sum(probs.values()) - 1.0) < 0.05, f"Bad probabilities: {probs}"
    print(f"[PASS] /predict -> {body}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-deploy smoke test")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        check_health(args.url)
        check_prediction(args.url)
    except Exception as exc:
        print(f"[FAIL] Smoke test failed: {exc}")
        sys.exit(1)

    print("Smoke test passed.")
