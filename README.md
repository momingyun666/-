# OpenCV + Pillow 计算机视觉基础作业

本项目完成了题目中的 5 个任务，所有输入素材位于 `assets/`，运行结果位于 `outputs/`。

PyTorch 的函数拟合与 MNIST MLP 作业见 [`task3_dl/README.md`](task3_dl/README.md)。

YOLO 无人机识别与 ONNX/INT8 量化作业见 [`task4_yolo/README.md`](task4_yolo/README.md)。

CNN 图像分类与结构改进作业见 [`task2_cnn/README.md`](task2_cnn/README.md)。

Air Passengers 时序预测、注意力模型与 ONNX 加速作业见
[`task6_timeseries/README.md`](task6_timeseries/README.md)。

## 环境

建议使用 Python 3.10+。当前工作区已经创建 `.venv`，直接运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

也可以按需安装依赖：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> 这里使用 `opencv-contrib-python`，因为 OpenCV 4.12 的精简构建可能不包含 CSRT/KCF 追踪器。

## 运行命令

```powershell
# 运行全部任务
.\.venv\Scripts\python.exe main.py

# 只运行第 3、4、5 题
.\.venv\Scripts\python.exe main.py --tasks 3 4 5

# 无桌面环境默认不弹窗；如果需要显示窗口
.\.venv\Scripts\python.exe main.py --tasks 1 2 --show

# 如果 example4.mp4 的初始框需要调整
.\.venv\Scripts\python.exe main.py --tasks 5 --roi4 1370,790,330,290
```

## 任务说明

### 1. 读取图片与通道分离

- 用 `cv2.imread` 读取图片，用 `PIL.Image.open` 读取同一张图片；
- OpenCV 默认是 BGR，Pillow 默认是 RGB，代码中通过 `cv2.cvtColor` 明确转换；
- 用 `cv2.split` 分离 B、G、R 三个通道；
- 输出灰度通道和单色通道，并生成拼图 `outputs/task1/channels_montage.jpg`。

### 2. 几何变换

- 缩放：`cv2.resize`，缩小图像时使用 `INTER_AREA`；
- 旋转：`cv2.getRotationMatrix2D` + `cv2.warpAffine`，扩大画布避免裁切；
- 平移：仿射矩阵最后两列控制 x、y 方向位移；
- 翻转：`cv2.flip(image, 1)` 水平翻转，`cv2.flip(image, 0)` 垂直翻转。

### 3. 轮廓检测

1. 在 HSV 空间分割出天空，取反得到飞机候选区域；
2. 形态学闭运算填补孔洞，保留最大连通域，去掉云层噪声；
3. 用 GrabCut 细化主体边界；
4. 用 `cv2.findContours` 提取外轮廓，取面积最大的轮廓作为飞机主体；
5. 用 `cv2.contourArea` 计算面积，用 `cv2.arcLength(contour, True)` 计算闭合周长。

结果保存在 `outputs/task3/contour_stats.json`、`contour_main.jpg`、`plane_mask.png` 中。

### 4. 运动目标检测

对 `example2.mp4` 的每一帧同时计算：

- **帧差法**：当前灰度帧与上一帧做 `absdiff`，阈值化后做形态学处理。它不使用背景模型，主要检测运动边缘；
- **MOG2**：`cv2.createBackgroundSubtractorMOG2` 在线学习背景，能够保留更完整的车辆区域。

两路结果并排写入 `outputs/task4/motion_comparison.mp4`，统计结果写在 `motion_metrics.json`。

### 5. 小球追踪

- 使用 OpenCV 的 **CSRT** 追踪器；
- `example3.mp4` 通过红色 HSV 分割自动获得初始框，并每 30 帧用检测结果校正一次；
- `example4.mp4` 第一帧白球位于右下角，默认初始框为 `(1370, 790, 330, 290)`，可通过 `--roi4` 修改；追踪过程中每 10 帧用白球 HSV 检测器校正，检测到追踪框尺寸异常时也会立即重新初始化；
- 输出包含边界框、中心点和运动轨迹的视频，以及逐帧中心坐标 CSV。

提高追踪准确度可以从以下方面入手：

1. 用检测器自动初始化，减少手工框选误差；
2. 追踪器与检测器互相校正，偏差过大时重新初始化；
3. 用 Kalman 滤波或运动模型预测遮挡时的位置；
4. 对高帧率视频降采样，减小相邻帧间位移；
5. 稳定光照，必要时做白平衡、对比度增强和多尺度检测。

## 输出目录

```text
outputs/
├── task1/   # 原图、通道图、通道统计
├── task2/   # 缩放、旋转、平移、翻转
├── task3/   # 轮廓图、掩膜、面积和周长
├── task4/   # 对比视频、关键帧、统计指标
├── task5/   # 两个追踪视频、轨迹 CSV、统计指标
└── report.md
```
