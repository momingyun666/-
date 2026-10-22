#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算机视觉基础作业：OpenCV + Pillow

运行方式：
    python main.py
    python main.py --tasks 3 4 5
    python main.py --show

所有结果保存在 outputs/ 下。图片和视频素材放在 assets/ 下。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ROOT / "assets"
DEFAULT_OUTPUT = ROOT / "outputs"


def log(message: str) -> None:
    """输出带时间戳的进度，便于观察长视频处理进度。"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    return path


def prepare_output(root: Path, task: int) -> Path:
    out_dir = root / f"task{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def put_label(
    image: np.ndarray,
    text: str,
    org: tuple[int, int] = (12, 30),
    scale: float = 0.7,
) -> None:
    """给图像添加英文标签。OpenCV 的 putText 不支持中文字符。"""
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_grid(
    tiles: list[tuple[str, np.ndarray]],
    tile_width: int,
    cols: int = 2,
    margin: int = 12,
) -> np.ndarray:
    """把多张图排成带标签的网格，便于一次性查看结果。"""
    rendered: list[np.ndarray] = []
    for title, image in tiles:
        tile = resize_to_width(image, tile_width)
        canvas = np.full((tile.shape[0] + 42, tile.shape[1], 3), 245, np.uint8)
        canvas[42:] = tile
        put_label(canvas, title, (10, 29), 0.62)
        rendered.append(canvas)

    cell_h = max(item.shape[0] for item in rendered)
    rows = math.ceil(len(rendered) / cols)
    grid = np.full(
        (rows * cell_h + (rows + 1) * margin, cols * tile_width + (cols + 1) * margin, 3),
        245,
        np.uint8,
    )
    for index, item in enumerate(rendered):
        row, col = divmod(index, cols)
        x = margin + col * (tile_width + margin)
        y = margin + row * (cell_h + margin)
        grid[y : y + item.shape[0], x : x + item.shape[1]] = item
    return grid


def show_images(images: Iterable[tuple[str, np.ndarray]], enabled: bool) -> None:
    """可选地调用 OpenCV 窗口显示图片；服务器/无桌面环境建议关闭。"""
    if not enabled:
        return
    for title, image in images:
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task1_channels(assets: Path, output: Path, show: bool) -> dict[str, Any]:
    """
    Task 1：用 OpenCV 和 Pillow 读取图片，并分离 B/G/R 三个通道。

    OpenCV 默认按 BGR 读图，Pillow 默认按 RGB 读图。这里同时使用两个库，
    并通过 cvtColor 明确两种颜色顺序的转换关系。
    """
    image_path = require_file(assets / "example1.jpg")
    out_dir = prepare_output(output, 1)
    channels_dir = out_dir / "channels"
    channels_dir.mkdir(exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")

    pil_image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(pil_image)

    cv2.imwrite(str(out_dir / "original_opencv.jpg"), bgr)
    pil_image.save(out_dir / "original_pillow.jpg", quality=95)

    # OpenCV 的 split 返回 B、G、R；灰度通道保留亮度，彩色通道只保留单色分量。
    blue, green, red = cv2.split(bgr)
    zeros = np.zeros_like(blue)
    colored = {
        "B": cv2.merge([blue, zeros, zeros]),
        "G": cv2.merge([zeros, green, zeros]),
        "R": cv2.merge([zeros, zeros, red]),
    }
    gray_channels = {"B": blue, "G": green, "R": red}

    stats: dict[str, Any] = {
        "shape_bgr": list(bgr.shape),
        "shape_rgb": list(rgb.shape),
        "dtype": str(bgr.dtype),
        "opencv_first_pixel_bgr": bgr[0, 0].tolist(),
        "pillow_first_pixel_rgb": rgb[0, 0].tolist(),
        "channels": {},
    }

    for name in ("B", "G", "R"):
        gray = gray_channels[name]
        cv2.imwrite(str(channels_dir / f"{name}_gray.png"), gray)
        cv2.imwrite(str(channels_dir / f"{name}_color.png"), colored[name])
        grey_values = gray.astype(np.float32)
        stats["channels"][name] = {
            "min": int(gray.min()),
            "max": int(gray.max()),
            "mean": round(float(grey_values.mean()), 3),
            "std": round(float(grey_values.std()), 3),
        }

    # 第一行放原图，下面依次展示 B/G/R 的灰度通道与单色通道。
    gray_previews = {
        name: cv2.cvtColor(gray_channels[name], cv2.COLOR_GRAY2BGR)
        for name in ("B", "G", "R")
    }
    channel_preview = make_grid(
        [
            ("Original (BGR)", bgr),
            ("Blue channel", colored["B"]),
            ("Green channel", colored["G"]),
            ("Red channel", colored["R"]),
            ("B gray", gray_previews["B"]),
            ("G gray", gray_previews["G"]),
            ("R gray", gray_previews["R"]),
        ],
        tile_width=320,
        cols=3,
    )
    cv2.imwrite(str(out_dir / "channels_montage.jpg"), channel_preview)
    write_json(out_dir / "channel_stats.json", stats)

    show_images(
        [
            ("Original", bgr),
            ("Blue", colored["B"]),
            ("Green", colored["G"]),
            ("Red", colored["R"]),
        ],
        show,
    )
    log(f"Task 1 完成：图片尺寸 {bgr.shape[1]}x{bgr.shape[0]}，通道统计已保存")
    return stats


def task2_transforms(assets: Path, output: Path, show: bool) -> dict[str, Any]:
    """
    Task 2：缩放、旋转、平移、翻转。

    旋转矩阵由 getRotationMatrix2D 生成；该函数围绕“旋转中心”旋转。
    为了让旋转后的图像不被裁切，这里扩大画布，并把中心平移到新画布中心。
    """
    image_path = require_file(assets / "example1.jpg")
    out_dir = prepare_output(output, 2)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")
    height, width = image.shape[:2]

    # 1) 缩放：按比例缩放，使用 INTER_AREA 在缩小图像时减少锯齿。
    scaled = cv2.resize(
        image,
        None,
        fx=0.6,
        fy=0.6,
        interpolation=cv2.INTER_AREA,
    )

    # 2) 旋转：围绕原图中心旋转 30°，并扩大输出画布避免四个角被裁掉。
    angle = 30.0
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_value = abs(rotation[0, 0])
    sin_value = abs(rotation[0, 1])
    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)
    rotation[0, 2] += new_width / 2.0 - center[0]
    rotation[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # 3) 平移：矩阵最后两列分别表示 x、y 方向的位移。
    tx, ty = 120, 60
    translation = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(
        image,
        translation,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # 4) 翻转：flipCode=1 水平翻转，0 垂直翻转，-1 同时水平与垂直翻转。
    flipped_horizontal = cv2.flip(image, 1)
    flipped_vertical = cv2.flip(image, 0)

    results = {
        "scaled": scaled,
        "rotated": rotated,
        "translated": translated,
        "flipped_horizontal": flipped_horizontal,
        "flipped_vertical": flipped_vertical,
    }
    for name, result in results.items():
        cv2.imwrite(str(out_dir / f"{name}.jpg"), result)

    montage = make_grid(
        [
            ("Original", image),
            ("Scaled 0.6x", scaled),
            ("Rotated 30 deg", rotated),
            ("Translated (+120, +60)", translated),
            ("Flip horizontal", flipped_horizontal),
            ("Flip vertical", flipped_vertical),
        ],
        tile_width=360,
        cols=2,
    )
    cv2.imwrite(str(out_dir / "transforms_montage.jpg"), montage)

    stats = {
        "original_size": [width, height],
        "scaled_size": [int(scaled.shape[1]), int(scaled.shape[0])],
        "rotation_angle_deg": angle,
        "rotated_canvas_size": [new_width, new_height],
        "translation_px": [tx, ty],
        "flip_codes": {"horizontal": 1, "vertical": 0},
    }
    write_json(out_dir / "transform_stats.json", stats)

    show_images(
        [
            ("Scaled", scaled),
            ("Rotated", rotated),
            ("Translated", translated),
            ("Flip horizontal", flipped_horizontal),
            ("Flip vertical", flipped_vertical),
        ],
        show,
    )
    log("Task 2 完成：缩放、旋转、平移、水平/垂直翻转结果已保存")
    return stats


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    return result


def largest_component_mask(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == index, 255, 0).astype(np.uint8)


def task3_contours(assets: Path, output: Path, show: bool) -> dict[str, Any]:
    """
    Task 3：检测 example1.jpg 中飞机的主体轮廓。

    思路：
    1. 天空在 HSV 中具有稳定的蓝色范围，先将“不是天空”的像素作为前景；
    2. 用形态学闭运算填补机身内部的小孔，保留最大连通域，去掉云层等噪声；
    3. 用 GrabCut 在最大连通域附近细化边界；
    4. findContours 提取外轮廓，取面积最大的轮廓作为飞机主体；
    5. contourArea 计算面积，arcLength 计算闭合轮廓周长。
    """
    image_path = require_file(assets / "example1.jpg")
    out_dir = prepare_output(output, 3)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    # 该图的天空是中等饱和度、较高亮度的蓝色。蓝色垂尾的亮度也较低，
    # 因此把 V 的阈值提高，避免把蓝色尾翼误删。
    sky = (
        (hue >= 85)
        & (hue <= 130)
        & (saturation >= 45)
        & (value >= 140)
    ).astype(np.uint8) * 255
    foreground = cv2.bitwise_not(sky)
    foreground = clean_binary_mask(foreground, 7)
    main_mask = largest_component_mask(foreground)
    main_mask = cv2.morphologyEx(
        main_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=3,
    )

    x, y, w, h = cv2.boundingRect(main_mask)
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # GrabCut：确定前景用 GC_FGD，可能前景用 GC_PR_FGD，矩形外默认是背景。
    grabcut_mask = np.full(image.shape[:2], cv2.GC_BGD, np.uint8)
    grabcut_mask[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = (
        cv2.GC_PR_FGD
    )
    grabcut_mask[main_mask > 0] = cv2.GC_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image,
        grabcut_mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    refined = clean_binary_mask(refined, 5)

    contours, _ = cv2.findContours(
        refined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError("未检测到有效轮廓，请检查分割阈值")

    main_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main_contour))
    perimeter = float(cv2.arcLength(main_contour, closed=True))
    bx, by, bw, bh = cv2.boundingRect(main_contour)

    # 过滤非常小的噪声轮廓，再统计“全部有效轮廓”的面积和周长。
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]
    all_area = float(sum(cv2.contourArea(c) for c in valid_contours))
    all_perimeter = float(
        sum(cv2.arcLength(c, closed=True) for c in valid_contours)
    )

    main_overlay = image.copy()
    cv2.drawContours(main_overlay, [main_contour], -1, (0, 0, 255), 3)
    cv2.rectangle(main_overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    put_label(
        main_overlay,
        f"Area={area:.1f}px^2  Perimeter={perimeter:.1f}px",
        (16, 36),
        0.75,
    )

    all_overlay = image.copy()
    cv2.drawContours(all_overlay, valid_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(all_overlay, [main_contour], -1, (0, 0, 255), 3)

    filled = np.zeros_like(image)
    cv2.drawContours(filled, [main_contour], -1, (40, 180, 255), -1)
    mask_view = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(out_dir / "contour_main.jpg"), main_overlay)
    cv2.imwrite(str(out_dir / "contour_all.jpg"), all_overlay)
    cv2.imwrite(str(out_dir / "plane_mask.png"), refined)
    cv2.imwrite(str(out_dir / "plane_filled.png"), filled)
    cv2.imwrite(str(out_dir / "plane_mask_view.jpg"), mask_view)

    stats: dict[str, Any] = {
        "main_contour": {
            "area_px2": round(area, 2),
            "perimeter_px": round(perimeter, 2),
            "bounding_box_xywh": [int(bx), int(by), int(bw), int(bh)],
            "point_count": int(len(main_contour)),
        },
        "all_valid_contours": {
            "contour_count": len(valid_contours),
            "area_px2": round(all_area, 2),
            "perimeter_px": round(all_perimeter, 2),
            "minimum_area_filter_px2": 100,
        },
        "note": "报告中采用最大主体轮廓的面积和周长；all_valid_contours 仅作参考。",
    }
    write_json(out_dir / "contour_stats.json", stats)

    show_images(
        [
            ("Detected contour", main_overlay),
            ("All contours", all_overlay),
            ("Plane mask", mask_view),
        ],
        show,
    )
    log(
        f"Task 3 完成：主轮廓面积 {area:.1f} px^2，"
        f"周长 {perimeter:.1f} px"
    )
    return stats


def detect_vehicles(
    mask: np.ndarray,
    min_area: int = 60,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], float]:
    """清理运动掩膜，提取外接框，并返回前景像素占比。"""
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel3, iterations=1)
    foreground_ratio = float(np.count_nonzero(cleaned) / cleaned.size)

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # 去掉过于细长的路面边缘，减少白线、接缝造成的误检。
        if w < 8 or h < 8:
            continue
        if max(w, h) / max(1, min(w, h)) > 7:
            continue
        boxes.append((x, y, w, h))
    return cleaned, boxes, foreground_ratio


def box_fill_ratio(
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> float:
    """统计所有目标框内前景像素的占比，用来衡量掩膜是否“空心”。"""
    box_area = 0
    foreground = 0
    for x, y, w, h in boxes:
        roi = mask[y : y + h, x : x + w]
        box_area += roi.size
        foreground += int(np.count_nonzero(roi))
    return float(foreground / box_area) if box_area else 0.0


def draw_detection_panel(
    frame: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    title: str,
    foreground_ratio: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    panel = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
    # 左上角用运动掩膜做一个小预览，方便观察前景区域。
    preview_w = max(120, panel.shape[1] // 4)
    preview_h = max(70, panel.shape[0] // 4)
    preview = cv2.resize(mask, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    panel[8 : 8 + preview_h, 8 : 8 + preview_w] = preview
    cv2.rectangle(
        panel,
        (8, 8),
        (8 + preview_w, 8 + preview_h),
        (255, 255, 255),
        1,
    )
    put_label(panel, title, (12, preview_h + 38), 0.62)
    put_label(
        panel,
        f"FG={foreground_ratio * 100:.2f}%  boxes={len(boxes)}",
        (12, preview_h + 66),
        0.55,
    )
    return panel


def task4_motion_detection(assets: Path, output: Path, show: bool) -> dict[str, Any]:
    """
    Task 4：比较帧差法与 MOG2 背景减除。

    “未使用背景减除”用相邻两帧的灰度绝对差实现：能快速找到运动边缘，
    但目标内部常常是空的，对慢速目标和短暂停顿不敏感。
    MOG2 会在线学习每个像素的背景分布，因此可以得到更完整的运动目标区域。
    """
    video_path = require_file(assets / "example2.mp4")
    out_dir = prepare_output(output, 4)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 在 960x540 上检测，远处的小车不会因为分辨率过低而消失；
    # 并排输出的两个面板也保持 960x540，便于观察目标轮廓。
    proc_width, proc_height = 960, 540

    video_writer = cv2.VideoWriter(
        str(out_dir / "motion_comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (proc_width * 2, proc_height),
    )
    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建输出视频，请检查 OpenCV 的视频编码支持")

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=1000,
        varThreshold=16,
        detectShadows=True,
    )

    previous_gray: np.ndarray | None = None
    frame_index = 0
    warmup_frames = 30
    metrics: list[dict[str, float]] = []
    keyframe_saved = False
    first_frame = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
        if first_frame is None:
            first_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1) 帧差法：没有显式的背景模型，只比较当前帧和上一帧。
        if previous_gray is None:
            diff_mask = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray, previous_gray)
            _, diff_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        diff_mask, diff_boxes, diff_ratio = detect_vehicles(diff_mask)
        diff_fill = box_fill_ratio(diff_mask, diff_boxes)

        # 2) MOG2：detectShadows=True 时阴影像素值为 127，清零可减少阴影误检。
        mog_mask = mog2.apply(frame)
        mog_mask[mog_mask == 127] = 0
        _, mog_mask = cv2.threshold(mog_mask, 200, 255, cv2.THRESH_BINARY)
        mog_mask, mog_boxes, mog_ratio = detect_vehicles(mog_mask)
        mog_fill = box_fill_ratio(mog_mask, mog_boxes)

        if frame_index >= warmup_frames:
            metrics.append(
                {
                    "diff_foreground_ratio": diff_ratio,
                    "mog_foreground_ratio": mog_ratio,
                    "diff_box_count": float(len(diff_boxes)),
                    "mog_box_count": float(len(mog_boxes)),
                    "diff_box_fill_ratio": diff_fill,
                    "mog_box_fill_ratio": mog_fill,
                }
            )

        left = draw_detection_panel(
            frame,
            diff_mask,
            diff_boxes,
            "Frame difference (no background model)",
            diff_ratio,
            (0, 0, 255),
        )
        right = draw_detection_panel(
            frame,
            mog_mask,
            mog_boxes,
            "MOG2 background subtraction",
            mog_ratio,
            (0, 255, 0),
        )
        comparison = np.hstack([left, right])
        put_label(
            comparison,
            f"frame {frame_index:04d} / {frame_count}",
            (proc_width + 12, 26),
            0.55,
        )
        video_writer.write(comparison)

        if not keyframe_saved and frame_index >= min(frame_count - 1, 330):
            cv2.imwrite(str(out_dir / "comparison_keyframe.jpg"), comparison)
            keyframe_saved = True

        previous_gray = gray
        frame_index += 1
        if frame_index % 120 == 0:
            log(f"Task 4 进度：{frame_index}/{frame_count}")

    capture.release()
    video_writer.release()

    if first_frame is None:
        raise RuntimeError("视频中没有可读取的帧")

    def average(key: str) -> float:
        return float(np.mean([item[key] for item in metrics])) if metrics else 0.0

    summary = {
        "video": video_path.name,
        "fps": round(fps, 3),
        "frame_count": frame_index,
        "duration_s": round(frame_count / fps, 3),
        "processed_size": [proc_width, proc_height],
        "warmup_frames_excluded": warmup_frames,
        "frame_difference": {
            "mean_foreground_ratio": round(average("diff_foreground_ratio"), 5),
            "mean_box_count": round(average("diff_box_count"), 3),
            "mean_box_fill_ratio": round(average("diff_box_fill_ratio"), 5),
        },
        "mog2": {
            "mean_foreground_ratio": round(average("mog_foreground_ratio"), 5),
            "mean_box_count": round(average("mog_box_count"), 3),
            "mean_box_fill_ratio": round(average("mog_box_fill_ratio"), 5),
        },
        "conclusion": (
            "帧差法主要保留运动边缘，前景占比低且目标容易碎片化；"
            "MOG2 能学习背景，目标区域更完整，对慢速运动和短时停顿更稳定，"
            "但也会有残影（ghost）和对光照变化敏感的问题。"
        ),
    }
    write_json(out_dir / "motion_metrics.json", summary)

    if show:
        show_images(
            [
                ("Task 4 keyframe", cv2.imread(str(out_dir / "comparison_keyframe.jpg")))
            ],
            True,
        )
    log(
        "Task 4 完成：帧差平均前景 "
        f"{average('diff_foreground_ratio') * 100:.2f}% / "
        f"{average('diff_box_count'):.2f} 个框 / "
        f"填充率 {average('diff_box_fill_ratio') * 100:.2f}%；MOG2 平均前景 "
        f"{average('mog_foreground_ratio') * 100:.2f}% / "
        f"{average('mog_box_count'):.2f} 个框 / "
        f"填充率 {average('mog_box_fill_ratio') * 100:.2f}%"
    )
    return summary


def create_csrt_tracker() -> cv2.Tracker:
    """优先使用 CSRT；如果当前 OpenCV 构建缺少 CSRT，则退回 KCF/MIL。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    raise RuntimeError("当前 OpenCV 没有可用的传统目标追踪器")


def detect_red_ball(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """在 3 号视频中利用红色分割得到小球框，用于自动初始化和校正。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 60), (12, 255, 255))
    mask |= cv2.inRange(hsv, (168, 100, 60), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 300:
        return None
    return cv2.boundingRect(contour)


def detect_white_ball(
    frame: np.ndarray,
    reference_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """
    用 HSV 颜色和形状先验检测 example4 中的白球。

    白球在灯光下偏冷色（H 约 100），而木纹背景偏黄。检测在 960x540 上进行，
    再按比例映射回原始分辨率。reference_bbox 用来在多个候选目标中优先选择
    离上一帧小球最近、尺寸最接近的候选，减少白色纸巾、盘子等干扰。
    """
    original_height, original_width = frame.shape[:2]
    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 0, 115), (125, 80, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    reference = None
    if reference_bbox is not None:
        rx, ry, rw, rh = reference_bbox
        scale_x, scale_y = 960 / original_width, 540 / original_height
        reference = (
            rx * scale_x,
            ry * scale_y,
            max(1.0, rw * scale_x),
            max(1.0, rh * scale_y),
        )

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 700 or area > 26000:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        # 白球在 960x540 画布上的尺寸约为 80~160 像素。左边的手和衣服
        # 可能形成大块浅色区域，贴着画面边缘且尺寸过大，需要排除。
        if w < 45 or h < 45 or w > 210 or h > 210:
            continue
        if x < 5 or y < 5 or x + w > 955 or y + h > 535:
            continue
        aspect = w / h
        fill = area / float(w * h)
        if circularity < 0.42 or not 0.65 <= aspect <= 1.55 or fill < 0.50:
            continue

        # 面积是重要线索：白球通常是画面中最大的“圆而实”的浅色目标，
        # 可以借此排除远处的纸巾、盘子和反光小块。
        score = circularity + 0.35 * min(area / 9000.0, 1.2)
        if reference is not None:
            rx, ry, rw, rh = reference
            reference_center = (rx + rw / 2.0, ry + rh / 2.0)
            candidate_center = (x + w / 2.0, y + h / 2.0)
            distance = math.dist(reference_center, candidate_center)
            size_ratio = math.sqrt(area / max(1.0, rw * rh))
            # 离上一帧位置越近、尺寸越接近，得分越高。
            score -= 0.20 * distance / 500.0
            score -= 0.20 * abs(math.log(max(1e-3, size_ratio)))
            if distance > 500:
                continue
            if not 0.55 <= size_ratio <= 1.9:
                continue
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        return None
    _, (x, y, w, h) = max(candidates, key=lambda item: item[0])
    scale_x, scale_y = original_width / 960.0, original_height / 540.0
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(w * scale_x)),
        int(round(h * scale_y)),
    )


def boxes_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def track_video(
    video_path: Path,
    output_path: Path,
    stats_path: Path,
    initial_bbox: tuple[int, int, int, int] | None,
    detector: str,
    correction_interval: int = 0,
) -> dict[str, Any]:
    """
    用 OpenCV CSRT 跟踪视频中的小球。

    detector:
        - "red"：每 correction_interval 帧用红色分割检查一次，偏移过大时重初始化；
        - "white"：每 correction_interval 帧用白色球形目标检测器校正；
        - "none"：仅使用追踪器。
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"视频没有可读取的帧：{video_path}")

    bbox = initial_bbox
    if bbox is None and detector == "red":
        bbox = detect_red_ball(first_frame)
    if bbox is None and detector == "white":
        bbox = detect_white_ball(first_frame)
    if bbox is None:
        capture.release()
        raise RuntimeError(
            "没有初始目标框。请用 --roi4 x,y,w,h 指定 example4.mp4 的初始框。"
        )

    tracker = create_csrt_tracker()
    tracker.init(first_frame, bbox)
    initial_w = max(1, int(bbox[2]))
    initial_h = max(1, int(bbox[3]))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建追踪结果视频")

    csv_path = output_path.with_suffix(".csv")
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame", "time_s", "x", "y", "w", "h", "center_x", "center_y", "ok"]
    )

    trail: deque[tuple[int, int]] = deque(maxlen=max(10, int(fps * 1.5)))
    tracked_frames = 0
    lost_frames = 0
    reinit_count = 0
    path_length = 0.0
    last_path_center: tuple[float, float] | None = None
    frame_index = 0
    keyframe_index = max(0, frame_count // 2)

    # 第一帧已经用于初始化，这里重新写入输出，保证视频完整。
    while True:
        if frame_index == 0:
            frame = first_frame
            current_box = bbox
            ok_tracking = True
        else:
            ok_read, frame = capture.read()
            if not ok_read:
                break
            ok_tracking, current_box = tracker.update(frame)

        x, y, w, h = map(int, current_box) if ok_tracking else (0, 0, 0, 0)
        if ok_tracking:
            # 框越界或尺寸异常时视为丢失，避免在画面上画到错误区域。
            if w <= 2 or h <= 2 or x + w < 0 or y + h < 0 or x >= width or y >= height:
                ok_tracking = False

        suspicious = bool(
            ok_tracking
            and (
                w < 0.45 * initial_w
                or h < 0.45 * initial_h
                or w > 2.5 * initial_w
                or h > 2.5 * initial_h
            )
        )

        # 尺寸异常时不必等到固定校正周期，立即用检测器重新找球。
        if suspicious and detector in {"red", "white"}:
            detected = (
                detect_red_ball(frame)
                if detector == "red"
                else detect_white_ball(frame)
            )
            if detected is not None:
                tracker = create_csrt_tracker()
                tracker.init(frame, detected)
                current_box = detected
                x, y, w, h = map(int, detected)
                reinit_count += 1
                trail.clear()
                last_path_center = None

        # 对 3、4 号视频定期用检测器校正 CSRT，降低遮挡和快速运动造成的漂移。
        if (
            ok_tracking
            and detector in {"red", "white"}
            and correction_interval > 0
            and frame_index > 0
            and frame_index % correction_interval == 0
        ):
            if detector == "red":
                detected = detect_red_ball(frame)
            else:
                detected = detect_white_ball(frame, current_box)
            if detected is not None:
                current_center = (
                    current_box[0] + current_box[2] / 2.0,
                    current_box[1] + current_box[3] / 2.0,
                )
                detected_center = (
                    detected[0] + detected[2] / 2.0,
                    detected[1] + detected[3] / 2.0,
                )
                center_gap = math.dist(current_center, detected_center)
            else:
                center_gap = 0.0
            if (
                detected is not None
                and (
                    boxes_iou(current_box, detected) < 0.25
                    or center_gap > 0.55 * max(current_box[2], current_box[3])
                )
            ):
                tracker = create_csrt_tracker()
                tracker.init(frame, detected)
                current_box = detected
                x, y, w, h = map(int, detected)
                reinit_count += 1
                trail.clear()
                last_path_center = None

        # 追踪器报告失败时，尝试用检测器恢复目标，而不是直接放弃。
        if not ok_tracking and detector in {"red", "white"}:
            recovered = (
                detect_red_ball(frame)
                if detector == "red"
                else detect_white_ball(frame)
            )
            if recovered is not None:
                tracker = create_csrt_tracker()
                tracker.init(frame, recovered)
                current_box = recovered
                x, y, w, h = map(int, recovered)
                ok_tracking = True
                reinit_count += 1
                trail.clear()
                last_path_center = None

        if ok_tracking:
            center = (x + w // 2, y + h // 2)
            trail.append(center)
            tracked_frames += 1
            if last_path_center is not None:
                path_length += math.hypot(
                    center[0] - last_path_center[0],
                    center[1] - last_path_center[1],
                )
            last_path_center = (float(center[0]), float(center[1]))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            if len(trail) >= 2:
                points = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, (0, 255, 255), 3)
            label = f"CSRT  frame={frame_index}  center=({center[0]},{center[1]})"
        else:
            lost_frames += 1
            trail.clear()
            last_path_center = None
            label = f"CSRT  frame={frame_index}  LOST"

        put_label(frame, label, (18, 42), 0.7)
        writer.write(frame)

        csv_writer.writerow(
            [
                frame_index,
                round(frame_index / fps, 4),
                x,
                y,
                w,
                h,
                x + w / 2 if ok_tracking else "",
                y + h / 2 if ok_tracking else "",
                int(ok_tracking),
            ]
        )

        if frame_index == keyframe_index:
            cv2.imwrite(str(output_path.with_name(output_path.stem + "_keyframe.jpg")), frame)
        frame_index += 1
        if frame_index % 120 == 0:
            log(f"  追踪进度 {video_path.name}: {frame_index}/{frame_count}")

    capture.release()
    writer.release()
    csv_file.close()

    duration = frame_index / fps if fps else 0.0
    summary = {
        "video": video_path.name,
        "tracker": tracker.__class__.__name__,
        "fps": round(fps, 3),
        "frame_count": frame_index,
        "duration_s": round(duration, 3),
        "tracked_frames": tracked_frames,
        "lost_frames": lost_frames,
        "tracking_rate": round(tracked_frames / max(1, frame_index), 4),
        "reinit_count": reinit_count,
        "path_length_px": round(path_length, 2),
        "mean_speed_px_per_s": round(path_length / duration, 2) if duration else 0.0,
        "csv": csv_path.name,
    }
    write_json(stats_path, summary)
    log(
        f"Task 5 完成：{video_path.name} 追踪 {tracked_frames}/{frame_index} 帧，"
        f"丢失 {lost_frames} 帧"
    )
    return summary


def task5_tracking(
    assets: Path,
    output: Path,
    show: bool,
    roi4: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Task 5：对 example3.mp4 和 example4.mp4 中的小球进行 CSRT 跟踪。"""
    out_dir = prepare_output(output, 5)

    example3 = track_video(
        video_path=require_file(assets / "example3.mp4"),
        output_path=out_dir / "example3_tracking.mp4",
        stats_path=out_dir / "example3_tracking_stats.json",
        initial_bbox=None,
        detector="red",
        correction_interval=30,
    )
    example4 = track_video(
        video_path=require_file(assets / "example4.mp4"),
        output_path=out_dir / "example4_tracking.mp4",
        stats_path=out_dir / "example4_tracking_stats.json",
        initial_bbox=roi4,
        detector="white",
        correction_interval=10,
    )
    combined = {"example3.mp4": example3, "example4.mp4": example4}
    write_json(out_dir / "tracking_stats.json", combined)

    if show:
        keyframe3 = out_dir / "example3_tracking_keyframe.jpg"
        keyframe4 = out_dir / "example4_tracking_keyframe.jpg"
        previews = []
        if keyframe3.exists():
            previews.append(("example3", cv2.imread(str(keyframe3))))
        if keyframe4.exists():
            previews.append(("example4", cv2.imread(str(keyframe4))))
        show_images(previews, True)
    return combined


def make_report(output: Path, results: dict[int, Any]) -> None:
    """把本次运行的关键数字汇总成 Markdown，方便直接写实验报告。"""
    lines = [
        "# OpenCV + Pillow 图像处理作业结果",
        "",
        "## 1. 图像读取与通道分离",
        "",
        "OpenCV 使用 BGR 顺序，Pillow 使用 RGB 顺序；两者通过 `cvtColor` 转换。",
        "",
    ]
    if 1 in results:
        stats = results[1]
        lines += [
            f"- 图像尺寸（高, 宽, 通道）：{stats['shape_bgr']}",
            f"- 数据类型：{stats['dtype']}",
            f"- OpenCV 左上角像素 BGR：{stats['opencv_first_pixel_bgr']}",
            f"- Pillow 左上角像素 RGB：{stats['pillow_first_pixel_rgb']}",
            "",
        ]

    lines += ["## 2. 几何变换", ""]
    if 2 in results:
        stats = results[2]
        lines += [
            f"- 原始尺寸：{stats['original_size']}",
            f"- 0.6 倍缩放后尺寸：{stats['scaled_size']}",
            f"- 旋转角度：{stats['rotation_angle_deg']}°",
            f"- 平移量：{stats['translation_px']} 像素",
            "",
        ]

    lines += ["## 3. 轮廓检测", ""]
    if 3 in results:
        main = results[3]["main_contour"]
        all_info = results[3]["all_valid_contours"]
        lines += [
            f"- 主体轮廓面积：**{main['area_px2']} px²**",
            f"- 主体轮廓周长：**{main['perimeter_px']} px**",
            f"- 主体外接框（x, y, w, h）：{main['bounding_box_xywh']}",
            f"- 有效轮廓数量：{all_info['contour_count']}，"
            f"总面积 {all_info['area_px2']} px²，总周长 {all_info['perimeter_px']} px",
            "",
        ]

    lines += ["## 4. 帧差法与 MOG2 对比", ""]
    if 4 in results:
        stats = results[4]
        diff = stats["frame_difference"]
        mog = stats["mog2"]
        lines += [
            f"- 帧差法：平均前景占比 {diff['mean_foreground_ratio'] * 100:.2f}%，"
            f"平均 {diff['mean_box_count']:.2f} 个检测框，"
            f"框内平均填充率 {diff['mean_box_fill_ratio'] * 100:.2f}%",
            f"- MOG2：平均前景占比 {mog['mean_foreground_ratio'] * 100:.2f}%，"
            f"平均 {mog['mean_box_count']:.2f} 个检测框，"
            f"框内平均填充率 {mog['mean_box_fill_ratio'] * 100:.2f}%",
            "",
            "帧差法只对相邻帧之间的变化敏感，通常只能得到运动物体的边缘，"
            "目标内部会出现空洞；MOG2 通过背景模型能得到更完整的车辆区域，"
            "对慢速目标和短时停顿更稳定，但可能出现残影并受光照变化影响。",
            "",
        ]

    lines += ["## 5. 小球追踪", ""]
    if 5 in results:
        for name, stats in results[5].items():
            lines += [
                f"### {name}",
                "",
                f"- 追踪器：{stats['tracker']}",
                f"- 追踪成功：{stats['tracked_frames']}/{stats['frame_count']} 帧"
                f"（{stats['tracking_rate'] * 100:.2f}%）",
                f"- 丢失帧数：{stats['lost_frames']}",
                f"- 重初始化次数：{stats['reinit_count']}",
                f"- 轨迹总长度：{stats['path_length_px']} px",
                f"- 平均速度：{stats['mean_speed_px_per_s']} px/s",
                "",
            ]
        lines += [
            "提高追踪准确度的方向：",
            "",
            "1. 用颜色/圆形检测先定位小球，再把检测框交给 CSRT，避免手工框选误差；",
            "2. CSRT 与检测器交替工作，检测框和追踪框偏差过大时重新初始化；",
            "3. 加入 Kalman 滤波预测小球位置，处理遮挡和运动模糊；",
            "4. 对高帧率视频适当降采样或缩小搜索区域，减少相邻帧位移；",
            "5. 保持光照稳定，必要时做白平衡、对比度增强，并使用多尺度检测应对远近变化。",
            "",
        ]

    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_roi(text: str) -> tuple[int, int, int, int]:
    values = [int(item.strip()) for item in text.split(",")]
    if len(values) != 4 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("ROI 格式应为 x,y,w,h，且均为非负整数")
    return values[0], values[1], values[2], values[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV + Pillow 计算机视觉基础作业")
    parser.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_ASSETS,
        help="素材目录，默认 ./assets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出目录，默认 ./outputs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="要运行的任务编号，例如 --tasks 1 2 3",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="运行结束后弹出 OpenCV 窗口；无桌面环境不要使用",
    )
    parser.add_argument(
        "--roi4",
        type=parse_roi,
        default=(1370, 790, 330, 290),
        help="example4.mp4 第一帧中白球的初始框，格式 x,y,w,h",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    results: dict[int, Any] = {}
    if 1 in args.tasks:
        results[1] = task1_channels(assets, output, args.show)
    if 2 in args.tasks:
        results[2] = task2_transforms(assets, output, args.show)
    if 3 in args.tasks:
        results[3] = task3_contours(assets, output, args.show)
    if 4 in args.tasks:
        results[4] = task4_motion_detection(assets, output, args.show)
    if 5 in args.tasks:
        results[5] = task5_tracking(assets, output, args.show, args.roi4)

    # 如果只运行部分任务，保留上一次 report 中已有的内容意义不大，
    # 因此 report 只汇总本次运行涉及的任务。
    make_report(output, results)
    log(f"全部完成，结果目录：{output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        raise
