#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取两个任务的 JSON 结果，生成可直接查看的实验报告。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "task3_dl"


def main() -> int:
    curve_path = OUTPUT / "curve_fit" / "params.json"
    mnist_path = OUTPUT / "mnist" / "metrics.json"
    if not curve_path.exists() or not mnist_path.exists():
        raise FileNotFoundError(
            "请先运行 curve_fit.py 和 train_mnist.py，再生成报告"
        )

    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    mnist = json.loads(mnist_path.read_text(encoding="utf-8"))
    concept_text = (ROOT / "task3_dl" / "concepts.md").read_text(encoding="utf-8")
    # 去掉概念文档的一级标题，并把内部标题降一级，嵌入到题一之下。
    concept_text = concept_text.split("\n", 1)[1].strip()
    concept_text = concept_text.replace("\n### ", "\n#### ")
    concept_text = concept_text.replace("\n## ", "\n### ")
    if concept_text.startswith("## "):
        concept_text = "### " + concept_text[3:]

    lines = [
        "# 深度学习作业提交文档：概念、函数拟合与 MNIST MLP",
        "",
        "## 题一：深度学习概念理解",
        "",
        concept_text,
        "",
        "---",
        "",
        "## 题二：PyTorch 梯度下降拟合二元函数",
        "",
        "拟合模型：",
        "",
        "```text",
        curve["formula"],
        "```",
        "",
        f"- 采样范围：x1, x2 ∈ {curve['sampling']['x1_range']}",
        f"- 噪声：均匀分布 {curve['sampling']['noise_range']}",
        f"- 训练样本：{curve['sampling']['train_samples']}",
        f"- 优化器：{curve['optimizer']['name']}，学习率 "
        f"{curve['optimizer']['learning_rate']}，迭代 {curve['optimizer']['steps']} 次",
        "",
        "| 参数 | 真实值 | 拟合值 | 绝对误差 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("a1", "a2", "a3", "a4", "a5", "a6", "b"):
        lines.append(
            f"| {name} | {curve['true_params'][name]:.6f} | "
            f"{curve['fitted_params'][name]:.6f} | "
            f"{curve['absolute_error'][name]:.6f} |"
        )
    lines += [
        "",
        f"- 测试集 MSE：**{curve['test_noisy']['mse']:.6f}**",
        f"- 测试集 R²（含噪声）：**{curve['test_noisy']['r2']:.8f}**",
        f"- 测试集 R²（对无噪声真函数）："
        f"**{curve['test_clean_function']['r2']:.8f}**",
        "",
        "原函数曲面与拟合曲面：",
        "",
        "![true vs fitted](curve_fit/true_vs_fitted_surface.png)",
        "",
        "损失曲线、预测散点和截面：",
        "",
        "![diagnostics](curve_fit/loss_and_diagnostics.png)",
        "",
        "---",
        "",
        "## 题三：MNIST 多层感知机",
        "",
        f"- 网络结构：`{mnist['architecture']}`",
        f"- 激活函数：{mnist['activation']}",
        f"- BatchNorm：{mnist['regularization']['batch_norm']}",
        f"- Dropout：{mnist['regularization']['dropout']}",
        f"- L2 权重衰减：{mnist['regularization']['weight_decay_l2']}",
        f"- 参数数量：{mnist['parameter_count']}",
        f"- 训练 / 验证 / 测试：{mnist['train_size']} / "
        f"{mnist['validation_size']} / {mnist['test_size']}",
        f"- 最佳 epoch：{mnist['best_epoch']}",
        f"- 最佳验证准确率：**{mnist['best_validation_accuracy'] * 100:.2f}%**",
        f"- 测试集准确率：**{mnist['test_accuracy'] * 100:.2f}%**",
        f"- 测试集 loss：{mnist['test_loss']:.6f}",
        f"- 训练耗时：{mnist['training_seconds']:.1f} 秒",
        "",
        "训练曲线（损失、训练/验证/测试准确率）：",
        "",
        "![training curves](mnist/training_curves.png)",
        "",
        "混淆矩阵：",
        "",
        "![confusion matrix](mnist/confusion_matrix.png)",
        "",
        "测试样本预测示例：",
        "",
        "![sample predictions](mnist/sample_predictions.png)",
        "",
        "## 结论与调参思路",
        "",
        "1. 曲线拟合中，自动求导负责计算损失对 7 个参数的梯度，"
        "SGD 按 `parameter -= lr * gradient` 更新参数。",
        "2. 由于噪声存在，拟合参数不会完全等于真值；增加样本数、"
        "降低学习率或延长训练可以减小参数误差。",
        "3. MNIST 中 BatchNorm 和 Dropout 分别改善优化稳定性和泛化能力；"
        "验证集用于选择最佳模型，测试曲线用于观察泛化趋势，"
        "最终准确率以最佳验证模型在测试集上的评估为准。",
        "4. 若出现过拟合，可增大 Dropout、增大 L2 权重衰减、"
        "减少隐藏层宽度或使用数据增强；若欠拟合，可增加宽度/深度或延长训练。",
    ]
    report_path = OUTPUT / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
