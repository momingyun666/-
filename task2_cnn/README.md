# 基于 CNN 的图像分类与改进对比

## 数据

当前使用本地无人机/鸟类图像构造 ImageFolder 数据：

```text
datasets/classification/
├── train/bird, train/drone
├── val/bird, val/drone
└── test/bird, test/drone
```

若拿到 Intel Image Classification，可整理成同样结构，直接运行：

```powershell
.\.venv\Scripts\python.exe task2_cnn\train_compare.py --data-dir datasets/intel
```

## 训练与对比

```powershell
.\.venv\Scripts\python.exe task2_cnn\train_compare.py
.\.venv\Scripts\python.exe task2_cnn\make_report.py
```

## 主要输出

- `training_curves.png`：基线/改进模型的 Train Loss、Test Loss、Train Acc、Test Acc
- `baseline_confusion_matrix.png`
- `improved_confusion_matrix.png`
- `improved_predictions.png`
- `model_comparison.png`
- `comparison.json`
- `report.md`

## 当前结果

- 基线 CNN：Test Acc 92.60%，241,442 参数
- 改进 CNN：Test Acc 95.07%，160,002 参数
- 改进模型参数量减少约 33.7%，但 CPU 单张推理 FPS 下降

Intel 数据集在当前网络环境不可下载，因此本实验使用本地 bird/drone 图像完成相同的训练与改进流程；报告中已明确注明数据来源差异。
