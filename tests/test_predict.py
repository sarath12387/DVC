"""
test_predict.py — M3: Unit tests for model utility / inference functions.

A tiny untrained SimpleCNN is created and saved by a fixture, so these tests
run in CI without the real trained model — we test the inference CONTRACT
(shapes, probability axioms, output schema), not model accuracy.
"""

import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import load_model, predict_image, preprocess_image
from train import SimpleCNN


@pytest.fixture(scope="module")
def model_checkpoint(tmp_path_factory):
    """Save a small untrained model in the same format train.py uses."""
    path = tmp_path_factory.mktemp("models") / "model.pt"
    model = SimpleCNN(num_classes=2)
    torch.save({"state_dict": model.state_dict(), "classes": ["cat", "dog"]},
               path)
    return path


@pytest.fixture()
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (120, 90, 60)).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------- preprocess_image
def test_preprocess_image_shape(sample_image_bytes):
    tensor = preprocess_image(sample_image_bytes)
    assert tensor.shape == (1, 3, 224, 224)   # batch, channels, H, W
    assert tensor.dtype == torch.float32


def test_preprocess_rejects_non_image_bytes():
    with pytest.raises(Exception):
        preprocess_image(b"not an image at all")


# ---------------------------------------------------------------- load_model
def test_load_model_returns_eval_model_and_classes(model_checkpoint):
    model, classes = load_model(model_checkpoint)
    assert classes == ["cat", "dog"]
    assert model.training is False            # must be in eval mode


# ---------------------------------------------------------------- predict_image
def test_predict_output_schema(model_checkpoint, sample_image_bytes):
    model, classes = load_model(model_checkpoint)
    result = predict_image(model, classes, sample_image_bytes)

    assert set(result.keys()) == {"label", "probabilities"}
    assert result["label"] in classes
    assert set(result["probabilities"].keys()) == set(classes)


def test_predict_probabilities_are_valid(model_checkpoint, sample_image_bytes):
    model, classes = load_model(model_checkpoint)
    probs = predict_image(model, classes, sample_image_bytes)["probabilities"]

    for p in probs.values():
        assert 0.0 <= p <= 1.0                       # each prob in [0, 1]
    assert abs(sum(probs.values()) - 1.0) < 1e-2     # probs sum to ~1


def test_predict_label_matches_argmax(model_checkpoint, sample_image_bytes):
    model, classes = load_model(model_checkpoint)
    result = predict_image(model, classes, sample_image_bytes)
    argmax_label = max(result["probabilities"], key=result["probabilities"].get)
    assert result["label"] == argmax_label
