# 时序模型作业：Air Passengers 月度客流预测

本项目使用 PyTorch 完成 Air Passengers 时间序列预测，包含：

- LSTM 递归单步预测基线；
- Direct-GRU 多变量直接多步消融模型；
- Attn-CNN-GRU 改进模型（因果卷积、BiGRU、时间注意力）；
- 训练/验证 Loss 曲线；
- MAE、MSE、RMSE、MAPE、R² 多随机种子评价；
- 真实曲线与预测曲线对比、残差分析；
- ONNX Runtime 推理加速实验。

## 运行

在工作区根目录执行：

```powershell
.\.venv\Scripts\python.exe task6_timeseries\train.py
```

脚本会读取 `datasets/air_passengers/airline-passengers.csv`，并把图片、模型、
指标和完整中文报告写入 `outputs/task6_timeseries/`。

完整实验默认训练 3 个模型 × 3 个随机种子 × 120 个 epoch，在 CPU 上通常
数十秒完成。可用 `--epochs`、`--seeds` 等参数调整。

## 主要输出

| 文件 | 内容 |
|---|---|
| `data_overview.png` | 原始序列、数据切分和季节轮廓 |
| `loss_curves.png` | 三个模型的训练/验证 Loss |
| `prediction_comparison.png` | 测试集真实值与预测值对比 |
| `metrics_comparison.png` | MAE、RMSE、R² 对比 |
| `residual_analysis.png` | 逐月残差和分预测步长误差 |
| `benchmark_comparison.png` | PyTorch/ONNX 推理速度 |
| `metrics.json` | 指标和参数量 |
| `experiment_summary.json` | 完整实验记录 |
| `predictions.csv` | 逐月预测结果与残差 |
| `report.md` | 可直接提交的中文实验报告 |

## 防泄漏说明

数据严格按时间先后切分。归一化参数只在训练集上计算；测试期每个年度预测块
只使用预测起点及以前的数据。同比锚点使用上一年同月的真实观测值，在预测起点
时已经可见。
