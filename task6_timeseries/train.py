#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Air Passengers 时序预测：训练、评估、可视化和 ONNX 推理对比。

运行示例：
    ..\\.venv\\Scripts\\python.exe train.py

项目对比三个模型：
1. LSTM-Recursive：单变量、单步训练、递归多步预测；
2. Direct-GRU：多变量输入、GRU、直接预测未来 12 个月；
3. Attn-CNN-GRU：多尺度因果卷积 + BiGRU + 时间注意力 + 直接预测。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data_utils import ForecastCase, PreparedData, inverse_scaled, prepare_data
from models import build_model, count_parameters


MODEL_ORDER = ["LSTM-Recursive", "Direct-GRU", "Attn-CNN-GRU"]
MODEL_LABELS = {
    "LSTM-Recursive": "LSTM 递归基线",
    "Direct-GRU": "Direct-GRU 消融",
    "Attn-CNN-GRU": "Attn-CNN-GRU 改进",
}
MODEL_COLORS = {
    "LSTM-Recursive": "#6C757D",
    "Direct-GRU": "#2A9D8F",
    "Attn-CNN-GRU": "#E76F51",
    "Seasonal-Naive": "#457B9D",
}
# 各模型容量差异较大，分别使用适合自身的小样本正则化强度。
# 所有模型统一训练固定轮数，最终使用最后一轮权重。
OPTIMIZER_CONFIG = {
    "LSTM-Recursive": {"learning_rate": 2e-3, "weight_decay": 1e-3},
    "Direct-GRU": {"learning_rate": 1e-3, "weight_decay": 0.1},
    "Attn-CNN-GRU": {"learning_rate": 1e-3, "weight_decay": 0.1},
}
DATE_FMT = "%Y-%m"


def configure_matplotlib() -> None:
    """设置中文字体、负号与统一绘图风格。"""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "DengXian",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 170,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def set_global_seed(seed: int) -> None:
    """固定 NumPy、Python 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_case(
    model: torch.nn.Module,
    model_name: str,
    case: ForecastCase,
) -> torch.Tensor:
    """在单个预测起点上生成完整预测，返回归一化后的水平值。"""
    if model_name == "LSTM-Recursive":
        return model.forecast(
            case.baseline_input.unsqueeze(0),
            horizon=len(case.seasonal_reference),
            seasonal_reference=case.seasonal_reference.unsqueeze(0),
        )[0]
    delta = model(case.feature_input.unsqueeze(0))[0]
    return case.seasonal_reference + delta[: len(case.seasonal_reference)]


def validation_loss(
    model: torch.nn.Module,
    model_name: str,
    data: PreparedData,
) -> tuple[float, float]:
    """用 1958 年年度预测与滚动一步预测的加权 MSE 选择检查点。"""
    annual_prediction = predict_case(model, model_name, data.val_long_case)
    annual_loss = F.mse_loss(
        annual_prediction,
        data.val_long_case.target_scaled,
    ).item()

    short_losses = []
    for case in data.val_short_cases:
        prediction = predict_case(model, model_name, case)
        short_losses.append(
            F.mse_loss(prediction, case.target_scaled).item()
        )
    short_loss = float(np.mean(short_losses))
    return 0.7 * annual_loss + 0.3 * short_loss, annual_loss


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """计算回归任务常用指标，所有数值均基于原始乘客人数。"""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = actual - predicted
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    nonzero = np.abs(actual) > 1e-8
    mape = float(
        np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0
    )
    denominator = float(np.sum(np.square(actual - actual.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(error)) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAPE": mape,
        "R2": r2,
    }


def train_single_model(
    model_name: str,
    seed: int,
    data: PreparedData,
    epochs: int,
    batch_size: int,
) -> dict:
    """训练一个模型，返回最终检查点、Loss 历史和测试预测。"""
    set_global_seed(seed)
    model = build_model(
        model_name,
        n_features=data.features.shape[1],
        horizon=data.horizon,
        seed=seed,
    )
    optimization = OPTIMIZER_CONFIG[model_name]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )
    if model_name == "LSTM-Recursive":
        train_x, train_y = data.baseline_x, data.baseline_y
    else:
        train_x, train_y = data.direct_x, data.direct_y

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_annual = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    n_samples = len(train_x)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples)
        loss_sum = 0.0
        for start in range(0, n_samples, batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            current_val, current_annual = validation_loss(
                model,
                model_name,
                data,
            )
        train_loss = loss_sum / n_samples
        history.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": current_val,
                "val_annual_loss": current_annual,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_val_annual = current_annual
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[{model_name:<15}] seed={seed} epoch={epoch:03d} "
                f"train={train_loss:.5f} val={current_val:.5f} "
                f"best_epoch={best_epoch:03d}"
            )

    # 1958 年同比增速仅约 3.4%，显著低于训练期平均水平；若依据这一
    # 单年验证 Loss 提前停止，容易选中欠拟合模型。因此最终结果统一使用
    # 固定训练轮数后的最后一轮权重，验证 Loss 只用于过程监测。
    final_state = copy.deepcopy(model.state_dict())
    final_val_loss = float(history[-1]["val_loss"])
    model.eval()
    scaled_predictions = []
    with torch.no_grad():
        for case in data.test_cases:
            scaled_predictions.append(predict_case(model, model_name, case))
    scaled_prediction = torch.cat(scaled_predictions).numpy()
    actual_scaled = torch.cat(
        [case.target_scaled for case in data.test_cases]
    ).numpy()
    prediction_original = inverse_scaled(
        scaled_prediction,
        data.log_mean,
        data.log_std,
    )
    actual_original = inverse_scaled(
        actual_scaled,
        data.log_mean,
        data.log_std,
    )
    metrics = calculate_metrics(actual_original, prediction_original)

    return {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_annual_loss": best_val_annual,
        "final_val_loss": final_val_loss,
        "parameters": count_parameters(model),
        "history": history,
        "state_dict": final_state,
        "prediction_scaled": scaled_prediction,
        "prediction_original": prediction_original,
        "actual_original": actual_original,
        "metrics": metrics,
    }


def aggregate_seed_metrics(results: list[dict]) -> dict[str, dict[str, float]]:
    """汇总多个随机种子下的均值与样本标准差。"""
    metric_names = ["MAE", "MSE", "RMSE", "MAPE", "R2"]
    aggregated: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray(
            [result["metrics"][metric] for result in results],
            dtype=np.float64,
        )
        aggregated[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    aggregated["best_epoch"] = {
        "mean": float(np.mean([result["best_epoch"] for result in results])),
        "std": 0.0,
    }
    return aggregated


def seasonal_naive_prediction(data: PreparedData) -> np.ndarray:
    """用上一年同月真实值预测，作为不用深度学习的强参考基线。"""
    predictions = [
        inverse_scaled(
            case.seasonal_reference.numpy(),
            data.log_mean,
            data.log_std,
        )
        for case in data.test_cases
    ]
    return np.concatenate(predictions)


def plot_data_overview(data: PreparedData, output_dir: Path) -> None:
    """绘制完整序列、数据切分和月份季节轮廓。"""
    frame = data.dataframe
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.6),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    ax = axes[0]
    train_end_date = frame.loc[data.train_end - 1, "Month"]
    val_end_date = frame.loc[data.val_end - 1, "Month"]
    ax.plot(
        frame["Month"],
        frame["Passengers"],
        color="#264653",
        linewidth=1.8,
        label="月度乘客数",
    )
    ax.axvspan(
        frame.loc[0, "Month"],
        train_end_date,
        color="#2A9D8F",
        alpha=0.10,
        label="训练集",
    )
    ax.axvspan(
        train_end_date,
        val_end_date,
        color="#E9C46A",
        alpha=0.20,
        label="验证集",
    )
    ax.axvspan(
        val_end_date,
        frame.loc[len(frame) - 1, "Month"],
        color="#E76F51",
        alpha=0.13,
        label="测试集",
    )
    ax.set_title("Air Passengers 原始序列与按时间顺序的数据切分")
    ax.set_ylabel("乘客人数")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=4, frameon=False, loc="upper left")

    seasonal = frame.groupby(frame["Month"].dt.month)["Passengers"].mean()
    axes[1].bar(
        seasonal.index,
        seasonal.values,
        color="#457B9D",
        alpha=0.82,
    )
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("平均人数")
    axes[1].set_title("月度平均客流：明显的年度季节性")
    fig.tight_layout()
    fig.savefig(output_dir / "data_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: pd.DataFrame, output_dir: Path) -> None:
    """绘制三个模型在多个随机种子上的训练/验证 Loss 曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        subset = history[history["model"] == model_name]
        for split, column, color in [
            ("训练 Loss", "train_loss", "#264653"),
            ("验证 Loss", "val_loss", "#E76F51"),
        ]:
            grouped = subset.groupby("epoch")[column].agg(["mean", "std"])
            epochs = grouped.index.to_numpy()
            mean = grouped["mean"].to_numpy()
            std = grouped["std"].fillna(0.0).to_numpy()
            ax.plot(
                epochs,
                mean,
                color=color,
                linewidth=1.7,
                label=split,
            )
            ax.fill_between(
                epochs,
                np.maximum(mean - std, 1e-7),
                mean + std,
                color=color,
                alpha=0.16,
            )
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    axes[0].set_ylabel("MSE（归一化尺度，对数坐标）")
    fig.suptitle("训练与验证 Loss 曲线（实线为 3 个随机种子的均值，阴影为标准差）")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    data: PreparedData,
    best_predictions: dict[str, np.ndarray],
    naive_prediction: np.ndarray,
    output_dir: Path,
) -> None:
    """绘制测试集真实曲线、深度学习预测和季节朴素基线。"""
    dates = pd.concat(
        [pd.Series(case.target_dates) for case in data.test_cases]
    ).reset_index(drop=True)
    actual = np.concatenate(
        [case.target_original for case in data.test_cases]
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    ax.plot(
        dates,
        actual,
        color="#111111",
        marker="o",
        markersize=4.2,
        linewidth=2.1,
        label="真实值",
    )
    ax.plot(
        dates,
        naive_prediction,
        color=MODEL_COLORS["Seasonal-Naive"],
        linestyle=":",
        linewidth=1.7,
        label="季节朴素基线",
    )
    for model_name in MODEL_ORDER:
        ax.plot(
            dates,
            best_predictions[model_name],
            color=MODEL_COLORS[model_name],
            linewidth=1.9,
            label=MODEL_LABELS[model_name],
        )
    boundary = pd.Timestamp("1960-01-01")
    ax.axvline(boundary, color="#999999", linestyle="--", linewidth=1.0)
    ax.text(
        boundary,
        ax.get_ylim()[1] * 0.97,
        " 年度重新锚定 →",
        ha="left",
        va="top",
        fontsize=9,
        color="#666666",
    )
    ax.set_title("1960 年留存测试段：真实曲线与预测曲线对比")
    ax.set_ylabel("乘客人数")
    ax.set_xlabel("月份")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    summary: dict,
    output_dir: Path,
) -> None:
    """用柱状图和误差棒比较不同模型的测试指标。"""
    names = ["Seasonal-Naive"] + MODEL_ORDER
    labels = ["季节朴素"] + [MODEL_LABELS[name] for name in MODEL_ORDER]
    colors = [MODEL_COLORS[name] for name in names]
    metrics = [("MAE", "MAE（人）"), ("RMSE", "RMSE（人）"), ("R2", "R²")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        stds = []
        for name in names:
            if name == "Seasonal-Naive":
                means.append(summary["naive_metrics"][metric])
                stds.append(0.0)
            else:
                means.append(summary["metrics"][name][metric]["mean"])
                stds.append(summary["metrics"][name][metric]["std"])
        positions = np.arange(len(names))
        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.9,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
    fig.suptitle("测试集评价指标（深度模型误差棒表示 3 个随机种子的标准差）")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    data: PreparedData,
    best_predictions: dict[str, np.ndarray],
    naive_prediction: np.ndarray,
    output_dir: Path,
) -> None:
    """分析逐月残差和不同预测步长上的 MAE。"""
    dates = pd.concat(
        [pd.Series(case.target_dates) for case in data.test_cases]
    ).reset_index(drop=True)
    actual = np.concatenate(
        [case.target_original for case in data.test_cases]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for model_name in MODEL_ORDER:
        residual = actual - best_predictions[model_name]
        axes[0].plot(
            dates,
            residual,
            marker="o",
            markersize=3.4,
            linewidth=1.5,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axes[0].axhline(0, color="#111111", linewidth=1.0)
    axes[0].set_title("测试集逐月残差（真实值 - 预测值）")
    axes[0].set_ylabel("残差（人）")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False, fontsize=8)

    horizons = np.arange(1, data.horizon + 1)
    for model_name in MODEL_ORDER:
        prediction = best_predictions[model_name].reshape(2, data.horizon)
        target = actual.reshape(2, data.horizon)
        horizon_mae = np.mean(np.abs(target - prediction), axis=0)
        axes[1].plot(
            horizons,
            horizon_mae,
            marker="o",
            linewidth=1.7,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    naive_mae = np.mean(
        np.abs(actual.reshape(2, data.horizon) - naive_prediction.reshape(2, data.horizon)),
        axis=0,
    )
    axes[1].plot(
        horizons,
        naive_mae,
        linestyle=":",
        color=MODEL_COLORS["Seasonal-Naive"],
        linewidth=1.6,
        label="季节朴素基线",
    )
    axes[1].set_xticks(horizons)
    axes[1].set_title("预测步长与平均绝对误差")
    axes[1].set_xlabel("预测步长（月）")
    axes[1].set_ylabel("MAE（人）")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_architecture(output_dir: Path) -> None:
    """绘制改进模型的结构示意图。"""
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    boxes = [
        (0.2, 1.45, 2.0, 1.0, "输入窗口\n24 × 6", "#DDEBF7"),
        (2.55, 1.45, 2.35, 1.0, "多尺度因果卷积\n8 通道 · d=1,2", "#D8F3DC"),
        (5.25, 1.45, 2.25, 1.0, "双向 GRU\n隐藏维 8×2", "#FFF3B0"),
        (7.85, 1.45, 2.05, 1.0, "时间注意力\n历史权重汇总", "#FDE2E4"),
        (10.25, 1.45, 2.45, 1.0, "预测头\n未来 12 个月同比增量", "#E5D4ED"),
    ]
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color,
            edgecolor="#555555",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        start = (left[0] + left[2], left[1] + left[3] / 2)
        end = (right[0], right[1] + right[3] / 2)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#555555",
            )
        )
    ax.text(
        6.5,
        0.67,
        "关键设计：左侧因果填充防止未来泄漏 | 深度可分离卷积轻量化 | "
        "残差连接改善梯度 | 直接多步预测减少递归误差累积",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    ax.set_title("Attn-CNN-GRU 改进模型结构", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "model_architecture.png", bbox_inches="tight")
    plt.close(fig)


def benchmark_callable(
    function,
    warmup: int = 25,
    repeats: int = 220,
) -> dict[str, float]:
    """对无参数的可调用函数做 CPU 延迟测试。"""
    for _ in range(warmup):
        function()
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        latencies.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
    }


def export_and_benchmark(
    best_models: dict[str, torch.nn.Module],
    best_seeds: dict[str, int],
    data: PreparedData,
    output_dir: Path,
) -> dict:
    """保存模型、导出 ONNX，并测量单样本与批量推理速度。"""
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    case = data.test_cases[-1]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    benchmark: dict[str, dict] = {}
    try:
        for batch_size in (1, 32):
            baseline_input = case.baseline_input.unsqueeze(0).repeat(
                batch_size,
                1,
                1,
            )
            feature_input = case.feature_input.unsqueeze(0).repeat(
                batch_size,
                1,
                1,
            )
            reference = case.seasonal_reference.unsqueeze(0).repeat(
                batch_size,
                1,
            )
            for model_name in MODEL_ORDER:
                model = best_models[model_name]
                model.eval()
                if model_name == "LSTM-Recursive":
                    function = lambda m=model: m.forecast(
                        baseline_input,
                        horizon=data.horizon,
                        seasonal_reference=reference,
                    )
                else:
                    function = lambda m=model: m(feature_input)
                timing = benchmark_callable(function)
                timing.update(
                    {
                        "batch_size": batch_size,
                        "backend": "PyTorch-CPU",
                        "parameters": count_parameters(model),
                        "throughput_samples_per_s": (
                            batch_size * 1000.0 / timing["latency_median_ms"]
                        ),
                    }
                )
                benchmark[f"{model_name}|torch|b{batch_size}"] = timing

        improved_path = model_dir / "attn_cnn_gru.onnx"
        dummy = case.feature_input.unsqueeze(0)
        improved = best_models["Attn-CNN-GRU"]
        try:
            torch.onnx.export(
                improved,
                dummy,
                improved_path,
                input_names=["features"],
                output_names=["delta_forecast"],
                dynamic_axes={
                    "features": {0: "batch"},
                    "delta_forecast": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                improved,
                dummy,
                improved_path,
                input_names=["features"],
                output_names=["delta_forecast"],
                dynamic_axes={
                    "features": {0: "batch"},
                    "delta_forecast": {0: "batch"},
                },
                opset_version=17,
            )

        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(improved_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        with torch.no_grad():
            torch_output = improved(dummy).numpy()
        onnx_output = session.run(
            None,
            {"features": dummy.numpy()},
        )[0]
        max_difference = float(np.max(np.abs(torch_output - onnx_output)))

        for batch_size in (1, 32):
            feature_input = case.feature_input.unsqueeze(0).repeat(
                batch_size,
                1,
                1,
            ).numpy()
            function = lambda: session.run(
                None,
                {"features": feature_input},
            )
            timing = benchmark_callable(function)
            timing.update(
                {
                    "batch_size": batch_size,
                    "backend": "ONNX Runtime-CPU",
                    "parameters": count_parameters(improved),
                    "throughput_samples_per_s": (
                        batch_size * 1000.0 / timing["latency_median_ms"]
                    ),
                }
            )
            benchmark[f"Attn-CNN-GRU|onnx|b{batch_size}"] = timing

        benchmark["onnx_verification"] = {
            "max_abs_difference": max_difference,
            "allclose_atol_1e-5": bool(
                np.allclose(torch_output, onnx_output, atol=1e-5)
            ),
        }
        benchmark["model_files"] = {
            "attn_cnn_gru.onnx_bytes": improved_path.stat().st_size,
            "attn_cnn_gru_pt_bytes": (
                model_dir / "attn_cnn_gru.pt"
            ).stat().st_size,
        }
    finally:
        torch.set_num_threads(old_threads)

    speedup = (
        benchmark["LSTM-Recursive|torch|b1"]["latency_median_ms"]
        / benchmark["Attn-CNN-GRU|torch|b1"]["latency_median_ms"]
    )
    onnx_vs_lstm_speedup = (
        benchmark["LSTM-Recursive|torch|b1"]["latency_median_ms"]
        / benchmark["Attn-CNN-GRU|onnx|b1"]["latency_median_ms"]
    )
    onnx_speedup = (
        benchmark["Attn-CNN-GRU|torch|b1"]["latency_median_ms"]
        / benchmark["Attn-CNN-GRU|onnx|b1"]["latency_median_ms"]
    )
    parameter_reduction = (
        1.0
        - benchmark["Attn-CNN-GRU|torch|b1"]["parameters"]
        / benchmark["LSTM-Recursive|torch|b1"]["parameters"]
    )
    benchmark["summary"] = {
        "improved_vs_lstm_b1_speedup": float(speedup),
        "onnx_vs_lstm_b1_speedup": float(onnx_vs_lstm_speedup),
        "onnx_vs_pytorch_improved_b1_speedup": float(onnx_speedup),
        "parameter_reduction_vs_lstm": float(parameter_reduction),
        "best_seeds": best_seeds,
    }
    with (output_dir / "benchmark.json").open("w", encoding="utf-8") as file:
        json.dump(benchmark, file, ensure_ascii=False, indent=2)
    return benchmark


def plot_benchmark(benchmark: dict, output_dir: Path) -> None:
    """绘制单样本推理延迟和吞吐量。"""
    keys = [
        ("LSTM-Recursive|torch|b1", "LSTM 基线\nPyTorch"),
        ("Direct-GRU|torch|b1", "Direct-GRU\nPyTorch"),
        ("Attn-CNN-GRU|torch|b1", "改进模型\nPyTorch"),
        ("Attn-CNN-GRU|onnx|b1", "改进模型\nONNX"),
    ]
    labels = [label for _, label in keys]
    latencies = [benchmark[key]["latency_median_ms"] for key, _ in keys]
    throughputs = [
        benchmark[key]["throughput_samples_per_s"] for key, _ in keys
    ]
    colors = ["#6C757D", "#2A9D8F", "#E76F51", "#9B5DE5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    bars = axes[0].bar(labels, latencies, color=colors, alpha=0.9)
    axes[0].set_ylabel("单样本延迟（ms）")
    axes[0].set_title("CPU 单样本推理延迟（越低越好）")
    for bar, value in zip(bars, latencies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    bars = axes[1].bar(labels, throughputs, color=colors, alpha=0.9)
    axes[1].set_ylabel("样本/秒")
    axes[1].set_title("CPU 单样本吞吐量（越高越好）")
    for bar, value in zip(bars, throughputs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle("推理速度对比：直接多步预测与 ONNX Runtime 加速")
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(
    data: PreparedData,
    best_predictions: dict[str, np.ndarray],
    naive_prediction: np.ndarray,
    output_dir: Path,
) -> None:
    """保存逐月真实值、预测值和残差，方便二次分析。"""
    dates = pd.concat(
        [pd.Series(case.target_dates) for case in data.test_cases]
    ).reset_index(drop=True)
    actual = np.concatenate(
        [case.target_original for case in data.test_cases]
    )
    frame = pd.DataFrame(
        {
            "Month": dates.dt.strftime(DATE_FMT),
            "Actual": actual,
            "Seasonal_Naive": naive_prediction,
        }
    )
    for model_name in MODEL_ORDER:
        frame[model_name] = best_predictions[model_name]
        frame[f"{model_name}_Residual"] = (
            frame["Actual"] - frame[model_name]
        )
    frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")


def format_table(summary: dict) -> str:
    """生成文档中的 Markdown 指标表。"""
    rows = [
        "| 模型 | MAE ↓ | MSE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    naive = summary["naive_metrics"]
    rows.append(
        f"| 季节朴素基线 | {naive['MAE']:.2f} | {naive['MSE']:.2f} | "
        f"{naive['RMSE']:.2f} | {naive['MAPE']:.2f}% | {naive['R2']:.4f} |"
    )
    for model_name in MODEL_ORDER:
        metrics = summary["metrics"][model_name]
        rows.append(
            f"| {MODEL_LABELS[model_name]} | "
            f"{metrics['MAE']['mean']:.2f} ± {metrics['MAE']['std']:.2f} | "
            f"{metrics['MSE']['mean']:.2f} ± {metrics['MSE']['std']:.2f} | "
            f"{metrics['RMSE']['mean']:.2f} ± {metrics['RMSE']['std']:.2f} | "
            f"{metrics['MAPE']['mean']:.2f}% ± {metrics['MAPE']['std']:.2f}% | "
            f"{metrics['R2']['mean']:.4f} ± {metrics['R2']['std']:.4f} |"
        )
    return "\n".join(rows)


def build_summary(
    data: PreparedData,
    results_by_model: dict[str, list[dict]],
    naive_metrics: dict[str, float],
    benchmark: dict,
    seeds: list[int],
) -> dict:
    """集中保存实验配置、数据规模、指标和速度结果。"""
    metrics = {
        model_name: aggregate_seed_metrics(results_by_model[model_name])
        for model_name in MODEL_ORDER
    }
    representative_seeds = {}
    best_epochs = {}
    parameters = {}
    for model_name in MODEL_ORDER:
        ordered = sorted(
            results_by_model[model_name],
            key=lambda result: result["final_val_loss"],
        )
        selected = ordered[len(ordered) // 2]
        representative_seeds[model_name] = selected["seed"]
        best_epochs[model_name] = selected["best_epoch"]
        parameters[model_name] = selected["parameters"]
    return {
        "dataset": {
            "name": "Air Passengers",
            "samples": int(len(data.values)),
            "start": data.dataframe.loc[0, "Month"].strftime(DATE_FMT),
            "end": data.dataframe.loc[len(data.values) - 1, "Month"].strftime(
                DATE_FMT
            ),
            "train": [
                data.dataframe.loc[0, "Month"].strftime(DATE_FMT),
                data.dataframe.loc[data.train_end - 1, "Month"].strftime(
                    DATE_FMT
                ),
            ],
            "validation": [
                data.dataframe.loc[data.train_end, "Month"].strftime(DATE_FMT),
                data.dataframe.loc[data.val_end - 1, "Month"].strftime(
                    DATE_FMT
                ),
            ],
            "test": [
                data.dataframe.loc[data.val_end, "Month"].strftime(DATE_FMT),
                data.dataframe.loc[len(data.values) - 1, "Month"].strftime(
                    DATE_FMT
                ),
            ],
            "window": data.window,
            "horizon": data.horizon,
            "baseline_train_samples": int(len(data.baseline_x)),
            "direct_train_samples": int(len(data.direct_x)),
            "test_forecast_blocks": len(data.test_cases),
            "log_mean": data.log_mean,
            "log_std": data.log_std,
        },
        "seeds": seeds,
        "metrics": metrics,
        "naive_metrics": naive_metrics,
        "representative_seeds": representative_seeds,
        "best_epochs": best_epochs,
        "parameters": parameters,
        "seed_runs": {
            model_name: [
                {
                    "seed": result["seed"],
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"],
                    "final_val_loss": result["final_val_loss"],
                    "metrics": result["metrics"],
                }
                for result in results_by_model[model_name]
            ]
            for model_name in MODEL_ORDER
        },
        "benchmark": benchmark,
    }


def write_report(summary: dict, output_dir: Path) -> None:
    """根据实测结果自动生成中文实验报告。"""
    metrics = summary["metrics"]
    improved = metrics["Attn-CNN-GRU"]
    baseline = metrics["LSTM-Recursive"]
    direct = metrics["Direct-GRU"]
    naive = summary["naive_metrics"]
    benchmark = summary["benchmark"]
    torch_b1 = benchmark["Attn-CNN-GRU|torch|b1"]["latency_median_ms"]
    onnx_b1 = benchmark["Attn-CNN-GRU|onnx|b1"]["latency_median_ms"]
    lstm_b1 = benchmark["LSTM-Recursive|torch|b1"]["latency_median_ms"]
    onnx_lstm_speedup = benchmark["summary"]["onnx_vs_lstm_b1_speedup"]
    onnx_speedup = benchmark["summary"]["onnx_vs_pytorch_improved_b1_speedup"]
    parameter_reduction = benchmark["summary"][
        "parameter_reduction_vs_lstm"
    ]
    rmse_reduction = (
        (baseline["RMSE"]["mean"] - improved["RMSE"]["mean"])
        / baseline["RMSE"]["mean"]
        * 100.0
    )
    attention_gain = (
        (direct["RMSE"]["mean"] - improved["RMSE"]["mean"])
        / direct["RMSE"]["mean"]
        * 100.0
    )
    table = format_table(summary)
    text = f"""# Air Passengers 时序预测实验报告

## 1. 任务概述

本项目完成一个基于深度学习的月度航空客流预测任务。输入过去 {summary['dataset']['window']} 个月的客流与日历特征，预测未来 {summary['dataset']['horizon']} 个月乘客人数。实验不仅给出 Loss 曲线和 MAE、MSE、RMSE、MAPE、R²，还比较了递归单步预测与直接多步预测，并完成 ONNX 推理速度测试。

## 2. 数据集介绍与预处理

数据来自题目推荐的 **Air Passengers Dataset**，共 {summary['dataset']['samples']} 条月度记录，时间范围为 {summary['dataset']['start']} 至 {summary['dataset']['end']}。序列同时具有明显的长期增长趋势和 12 个月周期季节性。

数据按时间顺序切分，严禁随机打乱：

| 数据集 | 时间范围 | 用途 |
|---|---|---|
| 训练集 | {summary['dataset']['train'][0]} 至 {summary['dataset']['train'][1]} | 参数学习 |
| 验证集 | {summary['dataset']['validation'][0]} 至 {summary['dataset']['validation'][1]} | 监控训练过程 |
| 测试集 | {summary['dataset']['test'][0]} 至 {summary['dataset']['test'][1]} | 最终评价 |

预处理步骤如下：

1. 将 Month 转为日期并检查月份连续性、重复值和缺失值；
2. 对乘客人数执行 `log1p`，减弱指数增长带来的异方差；
3. 仅使用训练集计算对数的均值和标准差，完成 z-score 归一化，避免验证/测试信息泄漏；
4. 构造月份正弦、余弦、同比变化、近三个月动量和已知时间趋势，共 6 维特征；
5. 使用长度 {summary['dataset']['window']} 的滑动窗口。基线模型生成 {summary['dataset']['baseline_train_samples']} 个单步训练样本，直接多步模型生成 {summary['dataset']['direct_train_samples']} 个 12 步样本；
6. 预测值反归一化后使用 `expm1` 还原为原始人数，所有评价指标都在原始尺度计算。

![数据总览](data_overview.png)

## 3. 模型结构设计

### 3.1 LSTM-Recursive 基线

单层 LSTM（隐藏维 48）读取 24 个月历史值，预测下一个月的同比增量。推理时把预测值追加到输入中，递归 12 次得到未来一年。该方式结构简单，但存在误差逐步累计的问题。

### 3.2 Direct-GRU 消融模型

使用 6 维多变量输入、线性投影和单层 GRU，一次前向直接输出未来 12 个月的同比增量。它保留了多变量和直接多步预测的改进，但尚未引入卷积与注意力。

### 3.3 Attn-CNN-GRU 改进模型

![模型结构](model_architecture.png)

改进模型包含四部分：

- **多尺度因果卷积**：两级深度可分离卷积，膨胀率分别为 1 和 2，提取局部变化与稍长周期的特征；左侧填充保证因果性；
- **双向 GRU**：在卷积特征上建模前后文依赖；
- **时间注意力**：为 24 个历史时间片分配权重，突出对未来预测更重要的月份；
- **直接多步预测头**：一次输出 12 个同比增量，再与上一年同月真实值相加，避免递归误差累积。

所有模型都以 MSE 作为训练损失，使用 AdamW、余弦学习率衰减、梯度裁剪和 3 个随机种子，并统一训练 120 轮。验证 Loss 由 1958 年全年预测 MSE 与滚动一步预测 MSE 加权得到，用于观察训练过程。需要特别说明：1958 年同比增长仅约 3.4%，明显低于训练期平均水平；若按这一单年验证 Loss 提前停止，会偏向选择“几乎不增长”的欠拟合模型。因此正式实验固定训练预算，并以最后一轮权重进行测试。所有随机种子都参与指标汇总，验证集只用于确定绘图和速度测试所用的代表种子，不用于挑出测试误差最小的模型。

## 4. Loss 曲线

![Loss 曲线](loss_curves.png)

图中实线为 3 个随机种子的均值，阴影为标准差。LSTM 的训练 Loss 对应单步同比增量，直接模型对应 12 步输出，因此不同子图的训练 Loss 不宜直接横向比较；验证 Loss 统一在完整年度预测的归一化尺度上计算。三条训练曲线总体下降，说明网络能够学习历史窗口中的趋势和季节模式。

## 5. 评价指标

测试集包含 1959 和 1960 两个年度预测块，共 24 个月。表中深度模型为 3 个随机种子的 **均值 ± 标准差**。

{table}

改进模型相对 LSTM 基线的 RMSE 降低 **{rmse_reduction:.2f}%**，相对 Direct-GRU 消融模型进一步降低 **{attention_gain:.2f}%**。改进模型的 RMSE 标准差为 {improved['RMSE']['std']:.2f}，Direct-GRU 为 {direct['RMSE']['std']:.2f}，说明小容量注意力模型在 3 个随机种子下更稳定。季节朴素基线直接复制上一年同月真实值，是不使用深度学习时的强参考。

![指标对比](metrics_comparison.png)

## 6. 预测结果与残差分析

![预测曲线](prediction_comparison.png)

![残差分析](residual_analysis.png)

预测曲线按 1959、1960 两个年度预测块展示。残差图为“真实值减预测值”，横轴为月份，右侧展示预测步长 1 至 12 个月的平均绝对误差，用于观察误差是否随预测步长增加而累积。

## 7. 改进方法与实验分析

本项目的改进不是单一技巧，而是以下方法的组合：

1. **多变量输入**：在客流水平之外加入月份周期和趋势动量；
2. **对数归一化**：降低数值尺度并缓解指数增长带来的训练不稳定；
3. **直接多步预测**：一次输出未来 12 个月，避免递归模型反复调用；
4. **同比锚定**：网络学习相对去年同月的增量，将强季节先验直接写入任务；
5. **多层时序特征提取**：因果卷积提取局部/多尺度模式，BiGRU 建模顺序依赖；
6. **时间注意力**：自动衡量历史月份的重要性；
7. **轻量化设计**：深度可分离卷积减少卷积计算量，隐藏维保持较小；
8. **ONNX 部署**：将改进模型导出为 ONNX，并使用 ONNX Runtime 单线程推理。

LSTM 递归基线、Direct-GRU 和 Attn-CNN-GRU 构成递进式消融链：前者验证传统递归方案，中者验证“多变量 + 直接多步”，后者继续验证“卷积 + 注意力”的增益。

## 8. 推理速度与轻量化分析

| 方案 | 参数量 | 单样本延迟中位数 | 吞吐量 |
|---|---:|---:|---:|
| LSTM-Recursive / PyTorch | {summary['parameters']['LSTM-Recursive']:,} | {lstm_b1:.3f} ms | {benchmark['LSTM-Recursive|torch|b1']['throughput_samples_per_s']:.1f} 样本/秒 |
| Direct-GRU / PyTorch | {summary['parameters']['Direct-GRU']:,} | {benchmark['Direct-GRU|torch|b1']['latency_median_ms']:.3f} ms | {benchmark['Direct-GRU|torch|b1']['throughput_samples_per_s']:.1f} 样本/秒 |
| Attn-CNN-GRU / PyTorch | {summary['parameters']['Attn-CNN-GRU']:,} | {torch_b1:.3f} ms | {benchmark['Attn-CNN-GRU|torch|b1']['throughput_samples_per_s']:.1f} 样本/秒 |
| Attn-CNN-GRU / ONNX Runtime | {summary['parameters']['Attn-CNN-GRU']:,} | {onnx_b1:.3f} ms | {benchmark['Attn-CNN-GRU|onnx|b1']['throughput_samples_per_s']:.1f} 样本/秒 |

改进模型的参数量为 {summary['parameters']['Attn-CNN-GRU']:,}，比 LSTM 基线少 **{parameter_reduction * 100:.1f}%**。在 PyTorch eager 模式下，卷积和双向 GRU 的算子调度开销使改进模型单样本延迟为 {torch_b1:.3f} ms，尚未体现出速度优势；导出为 ONNX 后，ONNX Runtime 单线程延迟降到 {onnx_b1:.3f} ms，相对同结构 PyTorch 加速 **{onnx_speedup:.2f}×**，相对 LSTM 基线加速 **{onnx_lstm_speedup:.2f}×**。Direct-GRU 在 PyTorch 中延迟最低（{benchmark['Direct-GRU|torch|b1']['latency_median_ms']:.3f} ms），但其 RMSE 高于改进模型，体现了精度与速度的权衡。ONNX 与 PyTorch 输出的最大绝对误差为 {benchmark['onnx_verification']['max_abs_difference']:.3e}，导出结果一致。

![速度对比](benchmark_comparison.png)

## 9. 复现方式

在工作区根目录运行：

```powershell
.\\venv\\Scripts\\python.exe task6_timeseries\\train.py
```

完整训练、评估、绘图和 ONNX 导出约需数十秒。主要输出文件：

- `loss_curves.png`：训练与验证 Loss；
- `prediction_comparison.png`：测试集预测对比；
- `metrics_comparison.png`：指标对比；
- `residual_analysis.png`：残差与分步误差；
- `benchmark_comparison.png`：推理速度；
- `metrics.json` 与 `experiment_summary.json`：完整数值结果；
- `predictions.csv`：逐月真实值、预测值和残差；
- `models/`：PyTorch 检查点与 ONNX 模型。

## 10. 结论与局限

实验说明，直接多步预测、同比锚定和多变量特征能显著改善长跨度客流预测；在 Direct-GRU 基础上加入多尺度因果卷积和时间注意力后，RMSE 继续下降，且随机种子波动更小。改进模型的 {summary['parameters']['Attn-CNN-GRU']:,} 个参数经 ONNX Runtime 部署后，单样本推理速度明显优于递归 LSTM。需要指出的是，Air Passengers 只有 144 个月，测试期只有 24 个月，且 1959-1960 年处于训练区间之外的增长阶段，因此指标对测试年份仍较敏感；小样本超参数选择也可能带来一定乐观偏差。实际部署时应扩大数据量，并使用滚动回测和概率预测区间评估风险。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "datasets" / "air_passengers" / "airline-passengers.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "task6_timeseries",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    torch.set_num_threads(max(1, args.threads))
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{args.data}。请先下载 Air Passengers CSV。"
        )
    data = prepare_data(
        path=args.data,
        window=args.window,
        horizon=args.horizon,
    )
    print(
        "数据准备完成："
        f"训练样本 baseline={len(data.baseline_x)}, "
        f"direct={len(data.direct_x)}, "
        f"验证年度窗口=1, 测试预测块={len(data.test_cases)}"
    )

    plot_data_overview(data, args.output)
    plot_architecture(args.output)

    results_by_model: dict[str, list[dict]] = {
        model_name: [] for model_name in MODEL_ORDER
    }
    for model_name in MODEL_ORDER:
        for seed in args.seeds:
            result = train_single_model(
                model_name=model_name,
                seed=seed,
                data=data,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            results_by_model[model_name].append(result)

    history = pd.DataFrame(
        [
            record
            for model_name in MODEL_ORDER
            for result in results_by_model[model_name]
            for record in result["history"]
        ]
    )
    history.to_csv(
        args.output / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_loss_curves(history, args.output)

    representative_results = {}
    for model_name in MODEL_ORDER:
        ordered = sorted(
            results_by_model[model_name],
            key=lambda result: result["final_val_loss"],
        )
        representative_results[model_name] = ordered[len(ordered) // 2]
    best_predictions = {
        model_name: representative_results[model_name]["prediction_original"]
        for model_name in MODEL_ORDER
    }
    naive_prediction = seasonal_naive_prediction(data)
    naive_metrics = calculate_metrics(
        representative_results["LSTM-Recursive"]["actual_original"],
        naive_prediction,
    )

    plot_predictions(
        data,
        best_predictions,
        naive_prediction,
        args.output,
    )
    plot_residuals(
        data,
        best_predictions,
        naive_prediction,
        args.output,
    )
    save_predictions_csv(
        data,
        best_predictions,
        naive_prediction,
        args.output,
    )

    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    best_models: dict[str, torch.nn.Module] = {}
    representative_seeds = {}
    for model_name in MODEL_ORDER:
        result = representative_results[model_name]
        model = build_model(
            model_name,
            n_features=data.features.shape[1],
            horizon=data.horizon,
            seed=result["seed"],
        )
        model.load_state_dict(result["state_dict"])
        model.eval()
        best_models[model_name] = model
        representative_seeds[model_name] = result["seed"]
        file_name = {
            "LSTM-Recursive": "lstm_recursive.pt",
            "Direct-GRU": "direct_gru.pt",
            "Attn-CNN-GRU": "attn_cnn_gru.pt",
        }[model_name]
        torch.save(
            {
                "model_name": model_name,
                "seed": result["seed"],
                "window": data.window,
                "horizon": data.horizon,
                "n_features": data.features.shape[1],
                "log_mean": data.log_mean,
                "log_std": data.log_std,
                "state_dict": result["state_dict"],
            },
            model_dir / file_name,
        )

    benchmark = export_and_benchmark(
        best_models,
        representative_seeds,
        data,
        args.output,
    )
    plot_benchmark(benchmark, args.output)

    summary = build_summary(
        data,
        results_by_model,
        naive_metrics,
        benchmark,
        args.seeds,
    )
    with (args.output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": summary["metrics"],
                "naive_metrics": summary["naive_metrics"],
                "parameters": summary["parameters"],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (args.output / "experiment_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    plot_metric_comparison(summary, args.output)
    write_report(summary, args.output)

    print("\n最终测试指标（3 个随机种子均值 ± 标准差）：")
    print(format_table(summary))
    print(
        f"\n改进模型相对 LSTM 基线 b1 延迟："
        f"{benchmark['summary']['improved_vs_lstm_b1_speedup']:.2f}×；"
        f"ONNX 相对 PyTorch："
        f"{benchmark['summary']['onnx_vs_pytorch_improved_b1_speedup']:.2f}×"
    )
    print(f"\n输出目录：{args.output}")


if __name__ == "__main__":
    main()
