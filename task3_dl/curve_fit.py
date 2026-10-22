#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务 2：使用 PyTorch 自动求导和梯度下降拟合二元函数。

真实函数：
    y = 2*x1^2 + 1.5*x1*x2 + 3*sin(x1) + 0.5*cos(2*x2)
        + exp(-0.5*x1) + 0.8*x2 + 2026 + noise

待拟合形式：
    y = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2)
        + exp(a5*x1) + a6*x2 + b

默认在 x1,x2 ∈ [-3,3] 上均匀采样，噪声服从 [-3,3] 均匀分布。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
import torch


DEFAULT_OUTPUT = ROOT / "outputs" / "task3_dl" / "curve_fit"
PARAM_NAMES = ("a1", "a2", "a3", "a4", "a5", "a6", "b")
TRUE_PARAMS = (2.0, 1.5, 3.0, 0.5, -0.5, 0.8, 2026.0)


def true_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """题目给定的无噪声真实函数。"""
    return (
        2.0 * x1.square()
        + 1.5 * x1 * x2
        + 3.0 * torch.sin(x1)
        + 0.5 * torch.cos(2.0 * x2)
        + torch.exp(-0.5 * x1)
        + 0.8 * x2
        + 2026.0
    )


def fitted_function(
    parameters: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """
    参数化模型。

    a5 被限制在 [-3,3]，避免优化初期出现 exp(很大的正数) 导致溢出。
    收敛后 a5 约为 -0.5，这个限制不会影响最终结果。
    """
    a1, a2, a3, a4, a5, a6, b = parameters.unbind()
    a5_safe = torch.clamp(a5, -3.0, 3.0)
    return (
        a1 * x1.square()
        + a2 * x1 * x2
        + a3 * torch.sin(x1)
        + a4 * torch.cos(2.0 * x2)
        + torch.exp(a5_safe * x1)
        + a6 * x2
        + b
    )


def sample_data(
    count: int,
    seed: int,
    domain: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """采样 x1、x2，添加 [-3,3] 均匀噪声，返回 x1、x2、带噪声目标 y。"""
    generator = torch.Generator().manual_seed(seed)
    x1 = torch.rand(count, generator=generator, dtype=dtype) * (2 * domain) - domain
    x2 = torch.rand(count, generator=generator, dtype=dtype) * (2 * domain) - domain
    noise = torch.rand(count, generator=generator, dtype=dtype) * 6.0 - 3.0
    y = true_function(x1, x2) + noise
    return x1, x2, y


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """决定系数 R² = 1 - SS_res / SS_tot。"""
    residual = torch.sum((y_true - y_pred).square())
    total = torch.sum((y_true - y_true.mean()).square())
    return float(1.0 - residual / total)


def evaluate(
    parameters: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        prediction = fitted_function(parameters, x1, x2)
        mse = float(torch.mean((prediction - y).square()))
        mae = float(torch.mean(torch.abs(prediction - y)))
        r2 = r2_score(y, prediction)
    return {"mse": mse, "mae": mae, "r2": r2}


def plot_surfaces(
    parameters: torch.Tensor,
    output_path: Path,
    domain: float,
    resolution: int = 70,
) -> None:
    grid = np.linspace(-domain, domain, resolution)
    grid_x1, grid_x2 = np.meshgrid(grid, grid)
    x1 = torch.from_numpy(grid_x1).to(torch.float64)
    x2 = torch.from_numpy(grid_x2).to(torch.float64)

    with torch.no_grad():
        true_surface = true_function(x1, x2).numpy()
        fitted_surface = fitted_function(parameters, x1, x2).numpy()

    z_min = min(float(true_surface.min()), float(fitted_surface.min()))
    z_max = max(float(true_surface.max()), float(fitted_surface.max()))

    figure = plt.figure(figsize=(15, 6))
    for index, (surface, title) in enumerate(
        (
            (true_surface, "Ground-truth function (without noise)"),
            (fitted_surface, "PyTorch fitted function"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        plot = axis.plot_surface(
            grid_x1,
            grid_x2,
            surface,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("y")
        axis.set_title(title)
        axis.set_zlim(z_min, z_max)
        axis.view_init(elev=28, azim=-58)
        figure.colorbar(plot, ax=axis, shrink=0.62, pad=0.08)

    figure.suptitle(
        "Function fitting with PyTorch autograd and gradient descent",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(
    parameters: torch.Tensor,
    history: list[dict[str, float]],
    x_test1: torch.Tensor,
    x_test2: torch.Tensor,
    y_test: torch.Tensor,
    output_path: Path,
) -> None:
    steps = [item["step"] for item in history]
    losses = [item["loss"] for item in history]

    with torch.no_grad():
        prediction = fitted_function(parameters, x_test1, x_test2)
        x1_line = torch.linspace(-3, 3, 300, dtype=torch.float64)
        x2_line = torch.zeros_like(x1_line)
        true_line = true_function(x1_line, x2_line)
        fitted_line = fitted_function(parameters, x1_line, x2_line)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(steps, losses, color="#1f77b4", linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("gradient step")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    sample_count = min(1800, y_test.numel())
    axes[1].scatter(
        y_test[:sample_count].numpy(),
        prediction[:sample_count].numpy(),
        s=8,
        alpha=0.35,
        color="#2ca02c",
    )
    values = torch.cat(
        [y_test[:sample_count], prediction[:sample_count]]
    ).numpy()
    low, high = float(values.min()), float(values.max())
    axes[1].plot([low, high], [low, high], "r--", linewidth=1.5, label="ideal")
    axes[1].set_xlabel("noisy target y")
    axes[1].set_ylabel("prediction")
    axes[1].set_title("Prediction on test data")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        x1_line.numpy(),
        true_line.numpy(),
        label="true (x2=0)",
        linewidth=2.2,
    )
    axes[2].plot(
        x1_line.numpy(),
        fitted_line.numpy(),
        "--",
        label="fitted (x2=0)",
        linewidth=2.0,
    )
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("y")
    axes[2].set_title("Cross-section at x2 = 0")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch 自动求导拟合二元函数")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--domain", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float64

    train_x1, train_x2, train_y = sample_data(
        args.train_size,
        args.seed,
        args.domain,
        dtype,
    )
    test_x1, test_x2, test_y = sample_data(
        args.test_size,
        args.seed + 1,
        args.domain,
        dtype,
    )
    clean_test_y = true_function(test_x1, test_x2)

    # b 初始化为目标均值，避免从 0 靠近 2026 时浪费大量迭代。
    parameters = torch.tensor(
        [1.5, 1.0, 2.0, 0.0, -0.4, 0.5, float(train_y.mean())],
        dtype=dtype,
        requires_grad=True,
    )
    # 这里使用最基础的批量梯度下降，而不是 Adam，便于观察梯度更新本身。
    optimizer = torch.optim.SGD([parameters], lr=args.lr)

    history: list[dict[str, float]] = []
    start_time = time.time()
    for step in range(args.steps + 1):
        prediction = fitted_function(parameters, train_x1, train_x2)
        loss = torch.mean((prediction - train_y).square())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                }
            )
        if step % 2000 == 0:
            current = parameters.detach().cpu().numpy()
            print(
                f"step={step:5d} loss={float(loss.detach()):.6f} "
                f"params={np.round(current, 4)}"
            )

    elapsed = time.time() - start_time
    train_metrics = evaluate(parameters, train_x1, train_x2, train_y)
    noisy_test_metrics = evaluate(parameters, test_x1, test_x2, test_y)
    clean_test_metrics = evaluate(parameters, test_x1, test_x2, clean_test_y)

    fitted = parameters.detach().cpu().numpy()
    true_array = np.asarray(TRUE_PARAMS, dtype=np.float64)
    absolute_error = np.abs(fitted - true_array)
    result = {
        "formula": (
            "y = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2) "
            "+ exp(a5*x1) + a6*x2 + b"
        ),
        "sampling": {
            "x1_range": [-args.domain, args.domain],
            "x2_range": [-args.domain, args.domain],
            "noise_range": [-3, 3],
            "train_samples": args.train_size,
            "test_samples": args.test_size,
        },
        "optimizer": {
            "name": "SGD",
            "learning_rate": args.lr,
            "steps": args.steps,
            "loss": "MSE",
        },
        "true_params": dict(zip(PARAM_NAMES, true_array.tolist())),
        "fitted_params": dict(zip(PARAM_NAMES, fitted.tolist())),
        "absolute_error": dict(zip(PARAM_NAMES, absolute_error.tolist())),
        "train": train_metrics,
        "test_noisy": noisy_test_metrics,
        "test_clean_function": clean_test_metrics,
        "elapsed_seconds": round(elapsed, 3),
    }
    (output / "params.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(history)

    plot_surfaces(parameters.detach(), output / "true_vs_fitted_surface.png", args.domain)
    plot_diagnostics(
        parameters.detach(),
        history,
        test_x1,
        test_x2,
        test_y,
        output / "loss_and_diagnostics.png",
    )

    print("=" * 78)
    print("拟合参数：")
    for name, true_value, fitted_value, error in zip(
        PARAM_NAMES,
        true_array,
        fitted,
        absolute_error,
    ):
        print(
            f"  {name:>2s}: true={true_value:10.6f}  "
            f"fitted={fitted_value:10.6f}  |error|={error:.6f}"
        )
    print(f"测试集 R2（含噪声）：{noisy_test_metrics['r2']:.8f}")
    print(f"测试集 R2（对无噪声真函数）：{clean_test_metrics['r2']:.8f}")
    print(f"测试集 MSE：{noisy_test_metrics['mse']:.6f}")
    print(f"结果目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
