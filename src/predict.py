"""
predict.py — Inference utility: load the trained model, predict on one image.

Used by the FastAPI service (app.py) and by the unit tests.

CLI usage:
    python src/predict.py --image some_cat.jpg --model models/model.pt
"""

import argparse
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import SimpleCNN  # same package; Docker sets PYTHONPATH=/app/src

IMG_SIZE = (224, 224)

_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: Path, device: str = "cpu"):
    """Load the serialized checkpoint. Returns (model, class_names)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, classes


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Bytes -> normalized (1, 3, 224, 224) tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0)


def predict_image(model, classes, image_bytes: bytes, device: str = "cpu") -> dict:
    """Run inference on raw image bytes.

    Returns {"label": str, "probabilities": {class: float, ...}}
    """
    x = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    probs = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    label = max(probs, key=probs.get)
    return {"label": label, "probabilities": probs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cat vs dog for one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/model.pt"))
    args = parser.parse_args()

    model, classes = load_model(args.model)
    result = predict_image(model, classes, args.image.read_bytes())
    print(result)
