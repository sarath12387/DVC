"""
train.py — M1: Baseline CNN training with MLflow experiment tracking.

- Simple 3-block CNN for 224x224 RGB binary classification (cat vs dog)
- Data augmentation on the training set (random flip / rotation)
- Logs to MLflow: parameters, per-epoch metrics, loss curves,
  confusion matrix, and the serialized model (models/model.pt)

Usage:
    python src/train.py --data-dir data/processed --epochs 5 --lr 0.001
    mlflow ui   # then open http://localhost:5000 to inspect runs
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless environments (CI/Docker)
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["cat", "dog"]


class SimpleCNN(nn.Module):
    """Baseline CNN: 3 conv blocks -> global pool -> dense head."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def make_loaders(data_dir: Path, batch_size: int):
    """Build train/val/test dataloaders. Augmentation only on train."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])

    loaders = {}
    for split, tf in [("train", train_tf), ("val", eval_tf), ("test", eval_tf)]:
        ds = datasets.ImageFolder(data_dir / split, transform=tf)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"), num_workers=2)
    return loaders


def evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy, y_true, y_pred) on a dataloader."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return loss_sum / total, correct / total, np.array(y_true), np.array(y_pred)


def train(data_dir: Path, epochs: int, lr: float, batch_size: int,
          model_out: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(data_dir, batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("cats-vs-dogs")
    with mlflow.start_run():
        # ---- log parameters ----
        mlflow.log_params({
            "model": "SimpleCNN", "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "optimizer": "Adam",
            "augmentation": "hflip+rot15", "img_size": 224, "device": str(device),
        })

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            # ---- train one epoch ----
            model.train()
            running, seen = 0.0, 0
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * y.size(0)
                seen += y.size(0)
            train_loss = running / seen

            val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_acc": val_acc}, step=epoch)
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # ---- final test evaluation ----
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, loaders["test"], criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")

        # ---- artifact 1: loss curves ----
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(history["val_acc"], label="val_acc", color="green")
        ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig("loss_curves.png"); plt.close(fig)
        mlflow.log_artifact("loss_curves.png")

        # ---- artifact 2: confusion matrix ----
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, colorbar=False)
        fig.tight_layout()
        fig.savefig("confusion_matrix.png"); plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png")

        # ---- artifact 3: serialized model ----
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES},
                   model_out)
        mlflow.log_artifact(str(model_out))
        print(f"Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-out", type=Path, default=Path("models/model.pt"))
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.lr, args.batch_size, args.model_out)
