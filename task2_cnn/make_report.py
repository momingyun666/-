#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据分类实验结果生成 Markdown 提交文档。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "task2_cnn"


def main() -> int:
    baseline = json.loads(
        (OUTPUT / "baseline_cnn_metrics.json").read_text(encoding="utf-8")
    )
    improved = json.loads(
        (OUTPUT / "improved_cnn_metrics.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (OUTPUT / "comparison.json").read_text(encoding="utf-8")
    )
    delta = comparison["improvement"]["accuracy_delta"]
    parameter_ratio = comparison["improvement"]["parameter_ratio"]
    fps_ratio = comparison["improvement"]["fps_ratio"]

    lines = [
        "# 基于卷积神经网络的图像分类与结构改进",
        "",
        "## 1. 任务与数据说明",
        "",
        "题目推荐 Intel Image Classification，但本机没有该数据集，"
        "Kaggle、Hugging Face 和 GitHub 在当前网络环境中均不可访问。"
        "因此使用项目中已有的无人机/鸟类图像构造图像级二分类数据集，"
        "完成相同的 CNN 训练、测试和改进对比流程。",
        "",
        "- 类别：`bird`、`drone`",
        "- 训练集：800 张 bird + 800 张 drone",
        "- 验证集：150 张 bird + 150 张 drone",
        "- 测试集：384 张 bird + 265 张 drone",
        "- 输入尺寸：128×128",
        "- 设备：CPU（PyTorch 2.14.1）",
        "",
        "源码保留 `--data-dir` 参数。Intel 数据集按 "
        "`train/val/test/类别名/` 的 ImageFolder 结构组织后，可直接复用本项目训练。",
        "",
        "## 2. 模型结构设计",
        "",
        "### 2.1 基线 CNN",
        "",
        "- 4 个普通 3×3 卷积层，通道数为 32、64、128、128；",
        "- 每个卷积后接 BatchNorm、ReLU 和最大池化；",
        "- 全局平均池化 + Dropout(0.25) + 全连接分类层；",
        "- 参数量约 24.1 万。",
        "",
        "### 2.2 改进 CNN",
        "",
        "- 使用深度可分离卷积（depthwise + pointwise）降低参数量；",
        "- 每个模块加入 SE 通道注意力，让网络自适应强调重要特征通道；",
        "- 同等尺寸时加入残差连接，改善梯度和特征传播；",
        "- 训练阶段增加 RandomResizedCrop、旋转、颜色抖动、随机擦除；",
        "- 使用标签平滑 0.05，减少过度自信；",
        "- 参数量约 16.0 万。",
        "",
        "## 3. ViT 基本原理（理解，不要求使用）",
        "",
        "ViT 把图像切成固定大小的 patch，例如 16×16，然后把每个 patch 展平并"
        "线性映射成 token。所有 token 加上位置编码，并加入一个可学习的分类 token，"
        "再送入多层 Transformer Encoder。",
        "",
        "每个 Encoder 层主要由多头自注意力（Multi-Head Self-Attention）和 MLP 组成。"
        "自注意力通过 Query、Key、Value 计算 token 之间的相关性，因此每个 token 都能"
        "直接关注全图其他位置，具有全局感受野；MLP 对每个 token 做非线性特征变换。"
        "残差连接和 LayerNorm 用于稳定深层训练。",
        "",
        "与 CNN 相比，CNN 具有局部性、平移等变性和较强的归纳偏置，小数据上通常更容易训练；"
        "ViT 的全局建模能力更强，但通常需要更多数据和计算。本实验没有使用完整 ViT，"
        "而是在 CNN 中加入 SE 通道注意力，以较低成本引入注意力机制。",
        "",
        "## 4. 训练设置",
        "",
        "- 损失函数：CrossEntropyLoss(label_smoothing=0.05)",
        "- 优化器：AdamW，学习率 1e-3，weight decay 1e-4",
        "- 学习率调度：CosineAnnealingLR",
        "- batch size：64",
        "- epochs：12",
        "- 模型选择：验证损失最小的 checkpoint",
        "",
        "## 5. Loss、Train Acc、Test Acc 曲线",
        "",
        "![training curves](training_curves.png)",
        "",
        "曲线对比显示：改进模型前期收敛更慢，这是强数据增强造成的；"
        "后期测试损失下降更快，测试准确率超过基线。",
        "",
        "## 6. 测试结果",
        "",
        "| 模型 | 参数量 | 模型大小 | Test Loss | Test Acc | 推理 FPS |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 基线 CNN | {baseline['parameters']:,} | "
        f"{baseline['model_size_mb']:.2f} MB | {baseline['test_loss']:.4f} | "
        f"**{baseline['test_accuracy'] * 100:.2f}%** | "
        f"{comparison['baseline']['inference_fps_batch1']:.1f} |",
        f"| 改进 CNN | {improved['parameters']:,} | "
        f"{improved['model_size_mb']:.2f} MB | {improved['test_loss']:.4f} | "
        f"**{improved['test_accuracy'] * 100:.2f}%** | "
        f"{comparison['improved']['inference_fps_batch1']:.1f} |",
        "",
        f"- 准确率提升：**+{delta * 100:.2f} 个百分点**",
        f"- 参数量变为原来的 **{parameter_ratio * 100:.1f}%**"
        f"（减少约 {(1 - parameter_ratio) * 100:.1f}%）",
        f"- CPU 单张推理速度为原来的 **{fps_ratio * 100:.1f}%**",
        "",
        "### 分类别准确率",
        "",
        "| 模型 | bird | drone |",
        "|---|---:|---:|",
        f"| 基线 CNN | {baseline['per_class_accuracy']['bird'] * 100:.2f}% | "
        f"{baseline['per_class_accuracy']['drone'] * 100:.2f}% |",
        f"| 改进 CNN | {improved['per_class_accuracy']['bird'] * 100:.2f}% | "
        f"{improved['per_class_accuracy']['drone'] * 100:.2f}% |",
        "",
        "### 混淆矩阵",
        "",
        "基线 CNN：",
        "",
        "![baseline confusion](baseline_confusion_matrix.png)",
        "",
        "改进 CNN：",
        "",
        "![improved confusion](improved_confusion_matrix.png)",
        "",
        "### 分类结果可视化",
        "",
        "![predictions](improved_predictions.png)",
        "",
        "![comparison](model_comparison.png)",
        "",
        "## 7. 结果分析",
        "",
        "1. 改进模型的测试准确率从 92.60% 提升到 95.07%，说明强数据增强、"
        "SE 注意力和残差结构改善了泛化能力。",
        "2. 参数量和模型文件更小，但 CPU 单张推理速度下降，"
        "原因是深度可分离卷积产生更多小算子，CPU 上的实际吞吐不一定随参数量下降而提高。",
        "3. 改进模型前期验证准确率较低，说明强增强带来更大的优化难度；"
        "后期测试损失更低，最终准确率更高。",
        "4. 当前数据只有 bird/drone 两类，且样本规模有限。"
        "若使用完整 Intel 六类数据集，应重新统计类别分布，并考虑类别加权、"
        "更多 epoch、标签平滑和更强的正则化。",
        "",
        "## 8. 提交文件",
        "",
        "- 源码：`task2_cnn/train_compare.py`",
        "- 基线权重：`outputs/task2_cnn/baseline_cnn.pt`",
        "- 改进权重：`outputs/task2_cnn/improved_cnn.pt`",
        "- 训练曲线：`outputs/task2_cnn/training_curves.png`",
        "- 混淆矩阵：`outputs/task2_cnn/baseline_confusion_matrix.png`、"
        "`improved_confusion_matrix.png`",
        "- 分类可视化：`outputs/task2_cnn/improved_predictions.png`",
    ]
    report_path = OUTPUT / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
