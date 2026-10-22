#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务 3：使用 PyTorch 的多层感知机（MLP）识别 MNIST。

网络结构：
    784 -> 512 -> 256 -> 10

训练技巧：
    - BatchNorm：稳定隐藏层分布，加快收敛；
    - Dropout：随机丢弃部分神经元，降低过拟合；
    - weight_decay：L2 正则化；
    - 验证集早停式保存最佳模型；
    - 测试集不参与梯度更新和模型选择；每个 epoch 记录测试指标用于绘图。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "outputs" / "task3_dl" / "mnist"


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def ensure_mnist_raw_files(data_dir: Path) -> Path:
    """用 torchvision 下载 MNIST，原始数据仍由 pandas/PIL 读取。"""
    raw_dir = data_dir / "MNIST" / "raw"
    required = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    if not all((raw_dir / name).exists() for name in required):
        datasets.MNIST(root=str(data_dir), train=True, download=True)
        datasets.MNIST(root=str(data_dir), train=False, download=True)
    return raw_dir


def read_idx_images(path: Path) -> np.ndarray:
    """读取 MNIST 的 idx 图像文件，返回 [N,28,28] 的 uint8 数组。"""
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"不是有效的 MNIST 图像文件：{path}")
        images = np.frombuffer(handle.read(), dtype=np.uint8)
        return images.reshape(count, rows, columns)


def read_idx_labels(path: Path) -> pd.DataFrame:
    """读取 MNIST 的 idx 标签文件，用 pandas 保存标签。"""
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"不是有效的 MNIST 标签文件：{path}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    return pd.DataFrame({"label": labels[:count].astype(np.int64)})


class PandasMNISTDataset(torch.utils.data.Dataset):
    """
    自定义 Dataset：
    - 标签来自 pandas DataFrame；
    - 图像通过 PIL.Image 读取；
    - 最后交给 PyTorch 的 transform 转为 Tensor。
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: pd.Series,
        transform: transforms.Compose,
    ) -> None:
        self.images = images
        self.labels = labels.to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pil_image = Image.fromarray(self.images[index], mode="L")
        image_tensor = self.transform(pil_image)
        return image_tensor, int(self.labels[index])


def build_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    raw_dir = ensure_mnist_raw_files(data_dir)
    train_images = read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")["label"]
    test_images = read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")["label"]

    # 用 pandas 打乱并划分训练集/验证集，同时保留划分清单，便于复现实验。
    split_table = pd.DataFrame(
        {
            "index": np.arange(len(train_labels), dtype=np.int64),
            "label": train_labels.to_numpy(dtype=np.int64),
        }
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_size = 5000
    validation_table = split_table.iloc[:validation_size].copy()
    train_table = split_table.iloc[validation_size:].copy()
    validation_table["split"] = "validation"
    train_table["split"] = "train"
    pd.concat([train_table, validation_table]).to_csv(
        data_dir / "mnist_split_manifest.csv",
        index=False,
    )

    train_indices = train_table["index"].to_numpy(dtype=np.int64)
    validation_indices = validation_table["index"].to_numpy(dtype=np.int64)
    train_set = PandasMNISTDataset(
        train_images[train_indices],
        train_labels.iloc[train_indices].reset_index(drop=True),
        transform,
    )
    validation_set = PandasMNISTDataset(
        train_images[validation_indices],
        train_labels.iloc[validation_indices].reset_index(drop=True),
        transform,
    )
    test_set = PandasMNISTDataset(
        test_images,
        test_labels.reset_index(drop=True),
        transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, test_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss) * labels.size(0)
        total_samples += labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum())
        if collect_predictions:
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    result: dict[str, object] = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": correct / max(1, total_samples),
    }
    if collect_predictions:
        result["predictions"] = torch.cat(all_predictions)
        result["labels"] = torch.cat(all_labels)
    return result


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [item["train_acc"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_acc"] for item in history], label="validation")
    axes[1].plot(epochs, [item["test_acc"] for item in history], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted label",
        ylabel="True label",
        title="MNIST MLP confusion matrix",
    )
    threshold = matrix.max() / 2.0
    for row in range(10):
        for col in range(10):
            axis.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, col] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
) -> None:
    count = min(18, images.size(0))
    figure, axes = plt.subplots(3, 6, figsize=(12, 6))
    for index in range(count):
        axis = axes[index // 6, index % 6]
        axis.imshow(images[index].squeeze(0).cpu(), cmap="gray")
        axis.set_title(
            f"true={int(labels[index])}  pred={int(predictions[index])}",
            fontsize=9,
        )
        axis.axis("off")
    for index in range(count, 18):
        axes[index // 6, index % 6].axis("off")
    figure.suptitle("Sample predictions on MNIST test set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch MLP 识别 MNIST")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, test_loader = build_loaders(
        args.data_dir.resolve(),
        args.batch_size,
        args.seed,
    )
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"device={device}  train={len(train_loader.dataset)}  "
          f"validation={len(validation_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += batch_size

        train_loss = running_loss / seen
        train_accuracy = correct / seen
        validation = evaluate(model, validation_loader, criterion, device)
        val_loss = float(validation["loss"])
        val_accuracy = float(validation["accuracy"])
        test_during_training = evaluate(model, test_loader, criterion, device)
        test_epoch_loss = float(test_during_training["loss"])
        test_epoch_accuracy = float(test_during_training["accuracy"])
        epoch_seconds = time.time() - epoch_start
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "test_loss": test_epoch_loss,
                "test_acc": test_epoch_accuracy,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_accuracy:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  "
            f"test_acc={test_epoch_accuracy:.4f}  "
            f"time={epoch_seconds:.1f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_result = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_predictions=True,
    )
    test_predictions = test_result.pop("predictions")
    test_labels = test_result.pop("labels")
    test_accuracy = float(test_result["accuracy"])
    test_loss = float(test_result["loss"])

    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(
        confusion,
        (test_labels.numpy(), test_predictions.numpy()),
        1,
    )
    per_class_accuracy = {
        str(label): float(confusion[label].sum() and confusion[label, label] / confusion[label].sum())
        for label in range(10)
    }

    model_path = output / "mnist_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "784-512-256-10",
            "test_accuracy": test_accuracy,
            "epoch": best_epoch,
        },
        model_path,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "device": str(device),
        "architecture": "784 -> 512 -> 256 -> 10",
        "activation": "ReLU",
        "regularization": {
            "batch_norm": True,
            "dropout": 0.2,
            "weight_decay_l2": args.weight_decay,
        },
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "train_size": len(train_loader.dataset),
        "validation_size": len(validation_loader.dataset),
        "test_size": len(test_loader.dataset),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy_curve": [
            item["test_acc"] for item in history
        ],
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "history": history,
        "training_seconds": round(time.time() - start_time, 3),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_training_curves(history, output / "training_curves.png")
    plot_confusion_matrix(confusion, output / "confusion_matrix.png")

    sample_images, sample_labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        sample_predictions = model(sample_images.to(device)).argmax(dim=1).cpu()
    plot_predictions(
        sample_images,
        sample_labels,
        sample_predictions,
        output / "sample_predictions.png",
    )

    print("=" * 78)
    print(f"最佳 epoch：{best_epoch}")
    print(f"测试集 loss：{test_loss:.6f}")
    print(f"测试集准确率：{test_accuracy * 100:.2f}%")
    print(f"模型参数数量：{parameter_count}")
    print(f"模型文件：{model_path}")
    print(f"结果目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
