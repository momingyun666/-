# YOLO 无人机识别与模型量化

完整实验文档见随任务提交的非代码材料包中的 `report.md`。

## 环境

当前虚拟环境已安装：

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics onnx onnxruntime lap pandas
```

## 流水线

如数据集压缩包不在默认位置，可先设置环境变量：

```powershell
$env:YOLO_DATA_ZIP = "D:\datasets\drone and bird.v2i.yolov11.zip"
```

```powershell
# 1. 解压数据、生成平衡子集与 data.yaml
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py prepare

# 2. 三张示例图片推理（官方 YOLO 权重不可用时回退到 COCO 预训练检测器）
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py pretrained

# 3. 训练（从零或从指定权重继续训练）
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py train --epochs 10

# 4. 测试集评估、混淆矩阵和准确率指标
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py evaluate --weights runs\detect\runs_task4\adapt_10e\weights\best.pt

# 5. 无人机视频追踪
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py video --weights runs\detect\runs_task4\adapt_10e\weights\best.pt

# 6. ONNX 导出、INT8 动态量化和速度对比
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py export --weights runs\detect\runs_task4\adapt_10e\weights\best.pt

# 7. 生成 Markdown 报告
.\.venv\Scripts\python.exe task4_yolo\run_pipeline.py report
```

## 数据说明

- 原始数据：3893 张训练图、649 张验证图、196 张测试图。
- 原始 test 划分只有 `bird`，没有 `drone`，不能用于评估无人机。
- 因此从训练集重新划分平衡验证集，并将原始 valid 作为测试集。
- CPU 训练子集：800 张无人机 + 800 张鸟；验证集 150 + 150。
- 为了适配近景视频，又用 COCO 预训练检测器为视频帧生成约 100 张近景伪标签，做 10 轮域适应微调。

## 主要输出

```text
outputs/task4_yolo/
├── report.md
├── dataset_summary.json
├── pretrained_images/
├── evaluation/
│   ├── training_curves.png
│   ├── accuracy_summary.png
│   ├── confusion_matrix_conf25.png
│   └── metrics.json
├── video/
│   ├── example_tracked.mp4
│   ├── video_keyframe.jpg
│   └── video_metrics.json
└── models/
    ├── drone_yolo11n_fp32.onnx
    ├── drone_yolo11n_int8.onnx
    └── benchmark.json
```

## 说明

- 题目提到的 `drone.pt` 没有出现在当前素材中，GitHub 也无法访问，因此 YOLO 使用 `yolo11n.yaml` 从零训练，再做视频域适应微调。
- 示例 COCO 图片使用可访问的 torchvision SSDLite 预训练检测器推理，不能冒充官方 YOLO 权重结果。
- INT8 动态量化显著减小模型体积，但在本机 CPU 上不一定加速；ONNX FP32 反而是最快的部署格式。
