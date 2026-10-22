# Task 3：PyTorch 自动求导、函数拟合与 MNIST MLP

本目录包含三个部分：

1. [`concepts.md`](concepts.md)：深度学习基础概念说明；
2. [`curve_fit.py`](curve_fit.py)：使用 PyTorch 自动求导和 SGD 拟合二元函数；
3. [`train_mnist.py`](train_mnist.py)：用 MLP 训练并评估 MNIST。

运行结束后，`outputs/task3_dl/report.md` 是包含题一、题二、题三的完整提交文档。

## 环境

当前项目 `.venv` 已安装以下 CPU 版本依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install matplotlib
```

## 运行

```powershell
# 1. 曲线拟合，生成参数、R²、曲面图和诊断图
.\.venv\Scripts\python.exe task3_dl\curve_fit.py

# 2. MNIST MLP 训练和测试
.\.venv\Scripts\python.exe task3_dl\train_mnist.py

# 3. 汇总生成 outputs/task3_dl/report.md
.\.venv\Scripts\python.exe task3_dl\make_report.py
```

## 曲线拟合设置

- `x1, x2` 默认都在 `[-3, 3]` 上均匀采样；
- 噪声为 `[-3, 3]` 上的均匀分布；
- 使用 20000 个训练样本、4000 个测试样本；
- 使用 PyTorch 自动求导和基础 SGD，默认学习率 `1e-3`，迭代 12000 次；
- 输出每个参数的拟合值、绝对误差、MSE 和 R²。

若题目对 `x1, x2` 的采样范围另有要求，可通过 `--domain` 修改，例如：

```powershell
.\.venv\Scripts\python.exe task3_dl\curve_fit.py --domain 5
```

## MNIST 设置

- 训练集 60000 张，其中 55000 张用于训练、5000 张用于验证；
- 测试集 10000 张；
- 标签和训练/验证集划分由 `pandas` 管理，图片由 `PIL.Image` 读取；
- 输入 784 维，隐藏层 512、256，输出 10 类；
- 交叉熵损失、Adam、batch size 128、默认训练 5 个 epoch；
- 使用 BatchNorm、Dropout 和 L2 正则化。

训练图同时绘制训练/验证损失、训练/验证/测试准确率；另外输出混淆矩阵和测试样本预测图。

输出包括：

```text
outputs/task3_dl/
├── curve_fit/
│   ├── params.json
│   ├── loss_history.csv
│   ├── true_vs_fitted_surface.png
│   └── loss_and_diagnostics.png
└── mnist/
    ├── mnist_mlp.pt
    ├── metrics.json
    ├── training_curves.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

`data/mnist_split_manifest.csv` 是 pandas 生成的训练/验证集索引清单。
