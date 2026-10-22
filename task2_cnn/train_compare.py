#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务 2：卷积神经网络图像分类与结构改进对比。

数据目录需为 ImageFolder 格式：
    data_dir/
      train/class_a, train/class_b
      val/class_a, val/class_b
      test/class_a, test/class_b

基线：普通卷积 CNN。
改进：深度可分离卷积 + SE 通道注意力 + 残差连接 + 强数据增强 + 标签平滑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_DATA = ROOT / "datasets" / "classification"
DEFAULT_OUTPUT = ROOT / "outputs" / "task2_cnn"


class BaselineCNN(nn.Module):
    """普通卷积基线：局部卷积 + 最大池化 + 全局平均池化。"""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class DepthwiseSeparableBlock(nn.Module):
    """深度可分离卷积 + SE 注意力 + 残差连接。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        x = self.se(x)
        if self.use_residual:
            x = x + residual
        return self.act(x)


class ImprovedCNN(nn.Module):
    """
    改进模型：
    - 深度可分离卷积降低参数量和计算量；
    - SE 通道注意力增强重要特征；
    - 残差连接改善梯度传播；
    - 强数据增强和标签平滑提高泛化能力。
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 192, stride=2),
            DepthwiseSeparableBlock(192, 192),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def build_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    strong_augmentation: bool,
    seed: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if strong_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.65, 1.0),
                    ratio=(0.8, 1.25),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(data_dir / "train", train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", eval_transform)
    generator = torch.Generator().manual_seed(seed)
    return (
        {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            ),
            "val": DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0),
            "test": DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0),
        },
        train_dataset.classes,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect: bool = False,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    predictions_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += float(loss) * labels.size(0)
        pred = logits.argmax(1)
        correct += int((pred == labels).sum())
        count += labels.size(0)
        if collect:
            predictions_all.append(pred.cpu())
            labels_all.append(labels.cpu())
    result: dict[str, object] = {
        "loss": loss_sum / max(1, count),
        "accuracy": correct / max(1, count),
    }
    if collect:
        result["predictions"] = torch.cat(predictions_all)
        result["labels"] = torch.cat(labels_all)
    return result


def train_one_model(
    model_name: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_accuracy_at_best_loss = -1.0
    best_epoch = 0
    best_state = None
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.size(0)
        scheduler.step()
        train_loss = loss_sum / count
        train_accuracy = correct / count
        val = evaluate(model, loaders["val"], criterion, device)
        test = evaluate(model, loaders["test"], criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": float(val["loss"]),
                "val_acc": float(val["accuracy"]),
                "test_loss": float(test["loss"]),
                "test_acc": float(test["accuracy"]),
            }
        )
        print(
            f"{model_name} epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_acc={float(val['accuracy']):.4f} "
            f"test_acc={float(test['accuracy']):.4f}"
        )
        if float(val["loss"]) < best_val_loss:
            best_val_loss = float(val["loss"])
            best_accuracy_at_best_loss = float(val["accuracy"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_result = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        collect=True,
    )
    predictions = test_result.pop("predictions")
    labels = test_result.pop("labels")
    num_classes = len(loaders["test"].dataset.classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels.numpy(), predictions.numpy()), 1)
    per_class_accuracy = {
        loaders["test"].dataset.classes[index]: float(
            confusion[index, index] / max(1, confusion[index].sum())
        )
        for index in range(num_classes)
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "classes": loaders["test"].dataset.classes,
            "test_accuracy": float(test_result["accuracy"]),
        },
        model_path,
    )
    metrics = {
        "model": model_name,
        "parameters": parameter_count,
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_accuracy_at_best_loss,
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "history": history,
        "training_seconds": round(time.time() - start, 3),
    }
    (output_dir / f"{model_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "model": model,
        "predictions": predictions,
        "labels": labels,
    }


def plot_curves(
    baseline_history: list[dict[str, float]],
    improved_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    epochs = [item["epoch"] for item in baseline_history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(epochs, [i["train_loss"] for i in baseline_history], label="baseline")
    axes[0, 0].plot(epochs, [i["train_loss"] for i in improved_history], label="improved")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, [i["test_loss"] for i in baseline_history], label="baseline")
    axes[0, 1].plot(epochs, [i["test_loss"] for i in improved_history], label="improved")
    axes[0, 1].set_title("Test Loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, [i["train_acc"] for i in baseline_history], label="baseline")
    axes[1, 0].plot(epochs, [i["train_acc"] for i in improved_history], label="improved")
    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, [i["test_acc"] for i in baseline_history], label="baseline")
    axes[1, 1].plot(epochs, [i["test_acc"] for i in improved_history], label="improved")
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(
    confusion: np.ndarray,
    classes: list[str],
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes)
    axis.set_yticklabels(classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = confusion.max() / 2
    for row in range(len(classes)):
        for col in range(len(classes)):
            axis.text(
                col,
                row,
                str(int(confusion[row, col])),
                ha="center",
                va="center",
                color="white" if confusion[row, col] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(
    dataset: datasets.ImageFolder,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    classes: list[str],
    output_path: Path,
    count: int = 16,
) -> None:
    # ImageFolder 顺序与 DataLoader 的 shuffle=False 顺序一致。
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    for axis, index in zip(axes.flat, indices):
        image, _ = dataset[index]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(image)
        true_name = classes[int(labels[index])]
        pred_name = classes[int(predictions[index])]
        axis.set_title(
            f"true={true_name}\npred={pred_name}",
            color="green" if true_name == pred_name else "red",
            fontsize=9,
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def benchmark_fps(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    runs: int = 100,
) -> float:
    model.eval()
    input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(10):
        model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return runs / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN 图像分类与改进对比")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_loaders, classes = build_loaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        strong_augmentation=False,
        seed=args.seed,
    )
    improved_loaders, improved_classes = build_loaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        strong_augmentation=True,
        seed=args.seed,
    )
    assert classes == improved_classes
    print(f"device={device} classes={classes}")
    print(
        f"train={len(baseline_loaders['train'].dataset)} "
        f"val={len(baseline_loaders['val'].dataset)} "
        f"test={len(baseline_loaders['test'].dataset)}"
    )

    baseline_model = BaselineCNN(len(classes)).to(device)
    improved_model = ImprovedCNN(len(classes)).to(device)
    baseline_result = train_one_model(
        "baseline_cnn",
        baseline_model,
        baseline_loaders,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        output,
    )
    improved_result = train_one_model(
        "improved_cnn",
        improved_model,
        improved_loaders,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        output,
    )
    baseline_metrics = baseline_result["metrics"]
    improved_metrics = improved_result["metrics"]

    plot_curves(
        baseline_metrics["history"],
        improved_metrics["history"],
        output / "training_curves.png",
    )
    plot_confusion(
        np.asarray(baseline_metrics["confusion_matrix"]),
        classes,
        "Baseline CNN Confusion Matrix",
        output / "baseline_confusion_matrix.png",
    )
    plot_confusion(
        np.asarray(improved_metrics["confusion_matrix"]),
        classes,
        "Improved CNN Confusion Matrix",
        output / "improved_confusion_matrix.png",
    )
    plot_predictions(
        improved_loaders["test"].dataset,
        improved_result["predictions"],
        improved_result["labels"],
        classes,
        output / "improved_predictions.png",
    )

    baseline_fps = benchmark_fps(baseline_model, args.image_size, device)
    improved_fps = benchmark_fps(improved_model, args.image_size, device)
    comparison = {
        "classes": classes,
        "image_size": args.image_size,
        "device": str(device),
        "baseline": {
            **baseline_metrics,
            "inference_fps_batch1": baseline_fps,
        },
        "improved": {
            **improved_metrics,
            "inference_fps_batch1": improved_fps,
        },
        "improvement": {
            "accuracy_delta": improved_metrics["test_accuracy"]
            - baseline_metrics["test_accuracy"],
            "parameter_ratio": improved_metrics["parameters"]
            / baseline_metrics["parameters"],
            "fps_ratio": improved_fps / baseline_fps,
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    labels = ["Baseline", "Improved"]
    accuracy = [
        baseline_metrics["test_accuracy"],
        improved_metrics["test_accuracy"],
    ]
    bars = axis.bar(labels, accuracy, color=["#4c72b0", "#dd8452"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Baseline vs improved CNN")
    for bar, value in zip(bars, accuracy):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value * 100:.2f}%",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("=" * 72)
    print(f"baseline test acc: {baseline_metrics['test_accuracy']:.4f}")
    print(f"improved test acc: {improved_metrics['test_accuracy']:.4f}")
    print(f"baseline params: {baseline_metrics['parameters']}")
    print(f"improved params: {improved_metrics['parameters']}")
    print(f"baseline FPS: {baseline_fps:.2f}")
    print(f"improved FPS: {improved_fps:.2f}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
