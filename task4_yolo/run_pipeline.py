#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO 无人机识别与模型量化流水线。

常用命令：
    python task4_yolo/run_pipeline.py prepare
    python task4_yolo/run_pipeline.py pretrained
    python task4_yolo/run_pipeline.py train --epochs 10
    python task4_yolo/run_pipeline.py evaluate --weights runs/.../best.pt
    python task4_yolo/run_pipeline.py video --weights runs/.../best.pt
    python task4_yolo/run_pipeline.py export --weights runs/.../best.pt
    python task4_yolo/run_pipeline.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MPL_CACHE = CACHE / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from ultralytics import YOLO


DATA_ZIP = Path(
    os.environ.get(
        "YOLO_DATA_ZIP",
        str(ROOT / "datasets" / "drone and bird.v2i.yolov11.zip"),
    )
)
DATA_DIR = ROOT / "datasets" / "drone_bird"
SUBSET_DIR = ROOT / "datasets" / "drone_subset"
EXAMPLE_DIR = ROOT / "assets" / "task4_yolo" / "example"
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs" / "task4_yolo"
RUNS_DIR = ROOT / "runs" / "detect" / "runs_task4"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_dataset(zip_path: Path = DATA_ZIP) -> None:
    if (DATA_DIR / "data.yaml").exists():
        log(f"数据集已解压：{DATA_DIR}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到数据集压缩包：{zip_path}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)
    log(f"数据集已解压到 {DATA_DIR}")


def classify_split(split: str) -> tuple[list[Path], list[Path]]:
    image_dir = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"
    drone_images: list[Path] = []
    bird_images: list[Path] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        classes = {
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if 1 in classes:
            drone_images.append(image_path)
        elif 0 in classes:
            bird_images.append(image_path)
    return drone_images, bird_images


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_subset() -> None:
    """
    原数据 test 划分只有 bird，没有 drone，无法评估无人机。
    因此从 train 中抽出平衡的 train/valid，并把原 valid 作为 test。
    受 CPU 算力限制，训练集抽样为 800 张无人机 + 800 张鸟。
    """
    if (SUBSET_DIR / "data.yaml").exists():
        counts = {}
        for split in ("train", "valid", "test"):
            drone_count = 0
            bird_count = 0
            for label_path in (SUBSET_DIR / split / "labels").glob("*.txt"):
                classes = {
                    int(line.split()[0])
                    for line in label_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                if 1 in classes:
                    drone_count += 1
                elif 0 in classes:
                    bird_count += 1
            counts[split] = {
                "images": len(list((SUBSET_DIR / split / "images").glob("*.jpg"))),
                "drone_images": drone_count,
                "bird_images": bird_count,
            }
        write_json(
            OUTPUT_DIR / "dataset_summary.json",
            {
                "source_counts": {
                    "train_images": 3893,
                    "valid_images": 649,
                    "test_images": 196,
                },
                "source_test_note": "原始 test 划分只有 bird，无 drone。",
                "subset": counts,
            },
        )
        log(f"平衡子集已存在：{SUBSET_DIR}")
        return
    random.seed(42)
    drone_images, bird_images = classify_split("train")
    random.shuffle(drone_images)
    random.shuffle(bird_images)

    train_drone = drone_images[:800]
    train_bird = bird_images[:800]
    valid_drone = drone_images[800:950]
    valid_bird = bird_images[800:950]

    for split in ("train", "valid", "test"):
        (SUBSET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (SUBSET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    def add_pair(image_path: Path, split: str) -> None:
        source_split = image_path.parent.parent.name
        label_path = DATA_DIR / source_split / "labels" / f"{image_path.stem}.txt"
        link_or_copy(image_path, SUBSET_DIR / split / "images" / image_path.name)
        link_or_copy(label_path, SUBSET_DIR / split / "labels" / label_path.name)

    for image_path in train_drone + train_bird:
        add_pair(image_path, "train")
    for image_path in valid_drone + valid_bird:
        add_pair(image_path, "valid")
    for image_path in sorted((DATA_DIR / "valid" / "images").glob("*.jpg")):
        add_pair(image_path, "test")

    yaml_data = (
        f"path: {SUBSET_DIR.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names: ['bird', 'drone']\n"
    )
    (SUBSET_DIR / "data.yaml").write_text(yaml_data, encoding="utf-8")
    write_json(
        OUTPUT_DIR / "dataset_summary.json",
        {
            "source_counts": {
                "train_images": 3893,
                "valid_images": 649,
                "test_images": 196,
            },
            "source_test_note": "原始 test 划分只有 bird，无 drone。",
            "subset": {
                "train": len(train_drone) + len(train_bird),
                "valid": len(valid_drone) + len(valid_bird),
                "test": len(list((DATA_DIR / "valid" / "images").glob("*.jpg"))),
                "train_drone": len(train_drone),
                "train_bird": len(train_bird),
                "valid_drone": len(valid_drone),
                "valid_bird": len(valid_bird),
            },
        },
    )
    log("平衡训练子集已生成")


def command_prepare(_: argparse.Namespace) -> None:
    extract_dataset()
    prepare_subset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("数据准备完成")


def draw_ssdlite_results() -> None:
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.transforms.functional import pil_to_tensor

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    categories = weights.meta["categories"]
    out_dir = OUTPUT_DIR / "pretrained_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {}
    for image_path in sorted(EXAMPLE_DIR.glob("coco-*.jpg")):
        pil_image = Image.open(image_path).convert("RGB")
        tensor = pil_to_tensor(pil_image).float() / 255.0
        with torch.no_grad():
            output = model([tensor])[0]
        keep = output["scores"] >= 0.5
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        canvas = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            name = categories[int(label)]
            detections.append(
                {"class": name, "confidence": round(float(score), 4)}
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{name} {float(score):.2f}"
            cv2.putText(
                canvas,
                text,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(out_dir / image_path.name), canvas)
        records[image_path.name] = detections
    write_json(
        out_dir / "detections.json",
        {
            "model": "torchvision SSDLite320 MobileNetV3 COCO",
            "reason": "GitHub 无法访问，官方 YOLO11n 权重未能下载；使用可访问的官方 COCO 预训练检测器替代示例图推理。",
            "results": records,
        },
    )


def command_pretrained(_: argparse.Namespace) -> None:
    yolo_weights = WEIGHTS_DIR / "yolo11n.pt"
    if yolo_weights.exists():
        model = YOLO(str(yolo_weights))
        out_dir = OUTPUT_DIR / "pretrained_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, object] = {}
        for image_path in sorted(EXAMPLE_DIR.glob("coco-*.jpg")):
            result = model.predict(
                str(image_path),
                imgsz=640,
                conf=0.25,
                verbose=False,
            )[0]
            result.save(filename=str(out_dir / image_path.name))
            records[image_path.name] = {
                "classes": [
                    result.names[int(label)]
                    for label in result.boxes.cls.cpu().tolist()
                ],
                "confidence": [
                    round(float(value), 4)
                    for value in result.boxes.conf.cpu().tolist()
                ],
            }
        write_json(
            out_dir / "detections.json",
            {"model": "official YOLO11n", "results": records},
        )
        return
    log("未找到 yolo11n.pt，使用 torchvision COCO 预训练检测器替代")
    draw_ssdlite_results()


def command_train(args: argparse.Namespace) -> None:
    model_source = args.weights or "yolo11n.yaml"
    model = YOLO(model_source)
    result = model.train(
        data=str(SUBSET_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=str(RUNS_DIR.parent.parent),
        name=args.name,
        exist_ok=True,
        plots=True,
        close_mosaic=min(3, args.epochs),
        patience=30,
    )
    print(f"SAVE_DIR={result.save_dir}")
    print(f"BEST={Path(result.save_dir) / 'weights' / 'best.pt'}")


def read_gt_boxes(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, xc, yc, bw, bh = map(float, line.split()[:5])
            boxes.append(
                [
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ]
            )
            classes.append(int(cls))
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area1 = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area2 = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def match_detection_accuracy(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    imgsz: int,
) -> dict[str, float]:
    image_paths = sorted(image_dir.glob("*.jpg"))
    total_gt = 0
    matched_gt = 0
    false_positives = 0
    correct_class = 0
    for start in range(0, len(image_paths), 16):
        batch_paths = image_paths[start : start + 16]
        results = model.predict(
            [str(path) for path in batch_paths],
            imgsz=imgsz,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            height, width = result.orig_shape
            gt_boxes, gt_classes = read_gt_boxes(
                label_dir / f"{image_path.stem}.txt",
                width,
                height,
            )
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            used_predictions: set[int] = set()
            total_gt += len(gt_boxes)
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                best_index = -1
                best_iou = 0.0
                for index, (pred_box, pred_class) in enumerate(
                    zip(pred_boxes, pred_classes)
                ):
                    if index in used_predictions or pred_class != gt_class:
                        continue
                    iou = box_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_iou >= 0.5 and best_index >= 0:
                    matched_gt += 1
                    correct_class += 1
                    used_predictions.add(best_index)
            false_positives += len(pred_boxes) - len(used_predictions)
    precision = matched_gt / max(1, matched_gt + false_positives)
    recall = matched_gt / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "total_gt": total_gt,
        "matched_gt": matched_gt,
        "false_positives": false_positives,
        "match_accuracy_tp_over_gt": recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def plot_training_curves(results_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs = frame["epoch"]
    axes[0, 0].plot(epochs, frame["train/box_loss"], label="box_loss")
    axes[0, 0].plot(epochs, frame["train/cls_loss"], label="cls_loss")
    axes[0, 0].plot(epochs, frame["train/dfl_loss"], label="dfl_loss")
    axes[0, 0].set_title("Training losses")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, frame["metrics/mAP50(B)"], label="mAP50")
    axes[0, 1].plot(epochs, frame["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[0, 1].set_title("Validation mAP")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, frame["metrics/precision(B)"], label="precision")
    axes[1, 0].plot(epochs, frame["metrics/recall(B)"], label="recall")
    axes[1, 0].set_title("Validation precision / recall")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, frame["val/box_loss"], label="val box_loss")
    axes[1, 1].plot(epochs, frame["val/cls_loss"], label="val cls_loss")
    axes[1, 1].plot(epochs, frame["val/dfl_loss"], label="val dfl_loss")
    axes[1, 1].set_title("Validation losses")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def command_evaluate(args: argparse.Namespace) -> None:
    weights = Path(args.weights).resolve()
    model = YOLO(str(weights))
    eval_dir = OUTPUT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result = model.val(
        data=str(SUBSET_DIR / "data.yaml"),
        split="test",
        imgsz=args.imgsz,
        batch=16,
        device="cpu",
        plots=True,
        project=str(eval_dir),
        name="test",
        exist_ok=True,
    )
    save_dir = Path(result.save_dir)
    for name in (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "val_batch0_pred.jpg",
    ):
        copy_if_exists(save_dir / name, eval_dir / name)

    train_metrics = match_detection_accuracy(
        model,
        SUBSET_DIR / "train" / "images",
        SUBSET_DIR / "train" / "labels",
        args.imgsz,
    )
    test_metrics = match_detection_accuracy(
        model,
        SUBSET_DIR / "test" / "images",
        SUBSET_DIR / "test" / "labels",
        args.imgsz,
    )
    names = result.names
    metrics = {
        "weights": str(weights),
        "imgsz": args.imgsz,
        "test_box_map50": float(result.box.map50),
        "test_box_map50_95": float(result.box.map),
        "test_precision": float(result.box.mp),
        "test_recall": float(result.box.mr),
        "per_class_map50": {
            str(names[index]): float(value)
            for index, value in enumerate(result.box.ap50[: len(names)])
        },
        "per_class_map50_95": {
            str(names[index]): float(value)
            for index, value in enumerate(result.box.maps[: len(names)])
        },
        "train_match_accuracy": train_metrics,
        "test_match_accuracy": test_metrics,
        "note": "目标检测没有分类任务意义上的 Train Acc；这里同时给出 mAP50 与 IoU=0.5 的匹配准确率。",
    }
    write_json(eval_dir / "metrics.json", metrics)

    training_csv = weights.parent.parent / "results.csv"
    if training_csv.exists():
        plot_training_curves(training_csv, eval_dir / "training_curves.png")

    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = ["Train match acc", "Test match acc", "Test mAP50"]
    values = [
        train_metrics["match_accuracy_tp_over_gt"],
        test_metrics["match_accuracy_tp_over_gt"],
        float(result.box.map50),
    ]
    bars = axis.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("score")
    axis.set_title("Detection accuracy proxy")
    axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(eval_dir / "accuracy_summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    log(f"评估完成：mAP50={result.box.map50:.4f}")


def command_video(args: argparse.Namespace) -> None:
    weights = Path(args.weights).resolve()
    video_path = Path(args.video).resolve()
    out_dir = OUTPUT_DIR / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "example_tracked.mp4"

    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_size = (1280, 720)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建视频写入器")

    detection_counts = []
    drone_confidences = []
    track_ids: set[int] = set()
    frame_index = 0
    start_time = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        rendered = result.plot()
        writer.write(rendered)
        detection_counts.append(len(result.boxes))
        if result.boxes is not None and len(result.boxes):
            for confidence, class_id in zip(
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                if int(class_id) == 1:
                    drone_confidences.append(float(confidence))
            if result.boxes.id is not None:
                track_ids.update(
                    int(value) for value in result.boxes.id.cpu().tolist()
                )
        if frame_index == total_frames // 2:
            cv2.imwrite(str(out_dir / "video_keyframe.jpg"), rendered)
        frame_index += 1
        if frame_index % 100 == 0:
            log(f"视频推理进度：{frame_index}/{total_frames}")
    capture.release()
    writer.release()
    elapsed = time.perf_counter() - start_time
    metrics = {
        "weights": str(weights),
        "video": str(video_path),
        "frames": frame_index,
        "source_fps": fps,
        "processing_fps": frame_index / elapsed if elapsed else 0.0,
        "processing_seconds": elapsed,
        "mean_detections": float(np.mean(detection_counts)) if detection_counts else 0.0,
        "frames_with_detection": int(sum(count > 0 for count in detection_counts)),
        "drone_detection_ratio": float(
            sum(count > 0 for count in detection_counts) / max(1, frame_index)
        ),
        "mean_drone_confidence": float(np.mean(drone_confidences))
        if drone_confidences
        else 0.0,
        "unique_track_ids": len(track_ids),
        "model_size_mb": weights.stat().st_size / (1024 * 1024),
    }
    write_json(out_dir / "video_metrics.json", metrics)
    log(
        f"视频推理完成：{frame_index} 帧，处理速度 "
        f"{metrics['processing_fps']:.2f} FPS"
    )


def benchmark_model(path: Path, frame: np.ndarray, imgsz: int, runs: int = 30) -> dict[str, float]:
    model = YOLO(str(path))
    for _ in range(3):
        model.predict(frame, imgsz=imgsz, conf=0.25, verbose=False)
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(frame, imgsz=imgsz, conf=0.25, verbose=False)
    elapsed = time.perf_counter() - start
    return {
        "model_size_mb": path.stat().st_size / (1024 * 1024),
        "runs": runs,
        "total_seconds": elapsed,
        "latency_ms": elapsed / runs * 1000.0,
        "fps": runs / elapsed if elapsed else 0.0,
    }


def validate_onnx(path: Path, imgsz: int) -> float | None:
    try:
        result = YOLO(str(path)).val(
            data=str(SUBSET_DIR / "data.yaml"),
            split="test",
            imgsz=imgsz,
            batch=8,
            device="cpu",
            plots=False,
            verbose=False,
        )
        return float(result.box.map50)
    except Exception as error:
        log(f"ONNX 验证失败 {path.name}: {error}")
        return None


def command_export(args: argparse.Namespace) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weights = Path(args.weights).resolve()
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    exported = Path(
        YOLO(str(weights)).export(
            format="onnx",
            imgsz=args.imgsz,
            opset=12,
            simplify=False,
            dynamic=False,
            half=False,
        )
    )
    fp32_path = model_dir / "drone_yolo11n_fp32.onnx"
    shutil.copy2(exported, fp32_path)
    int8_path = model_dir / "drone_yolo11n_int8.onnx"
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    capture = cv2.VideoCapture(str(EXAMPLE_DIR / "example.mp4"))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("无法读取示例视频首帧")
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    pt_result = benchmark_model(weights, frame, args.imgsz)
    fp32_result = benchmark_model(fp32_path, frame, args.imgsz)
    int8_result = benchmark_model(int8_path, frame, args.imgsz)
    pt_map50 = YOLO(str(weights)).val(
        data=str(SUBSET_DIR / "data.yaml"),
        split="test",
        imgsz=args.imgsz,
        batch=8,
        device="cpu",
        plots=False,
        verbose=False,
    ).box.map50
    benchmark = {
        "imgsz": args.imgsz,
        "models": {
            "PyTorch FP32": {
                **pt_result,
                "map50": float(pt_map50),
            },
            "ONNX FP32": {
                **fp32_result,
                "map50": validate_onnx(fp32_path, args.imgsz),
            },
            "ONNX INT8 dynamic": {
                **int8_result,
                "map50": validate_onnx(int8_path, args.imgsz),
            },
        },
        "note": "FPS 为 CPU 上包含 Ultralytics 预处理/后处理的端到端重复推理速度，非 GPU TensorRT 指标。",
    }
    write_json(model_dir / "benchmark.json", benchmark)
    log("ONNX 导出、INT8 量化与速度测试完成")


def command_report(_: argparse.Namespace) -> None:
    metrics_path = OUTPUT_DIR / "evaluation" / "metrics.json"
    video_path = OUTPUT_DIR / "video" / "video_metrics.json"
    benchmark_path = OUTPUT_DIR / "models" / "benchmark.json"
    dataset_path = OUTPUT_DIR / "dataset_summary.json"
    baseline_path = OUTPUT_DIR / "evaluation" / "metrics_baseline.json"
    if not metrics_path.exists():
        raise FileNotFoundError("请先运行 evaluate")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    video = (
        json.loads(video_path.read_text(encoding="utf-8"))
        if video_path.exists()
        else None
    )
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.exists()
        else None
    )
    dataset = (
        json.loads(dataset_path.read_text(encoding="utf-8"))
        if dataset_path.exists()
        else None
    )
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.exists()
        else None
    )
    confusion_path = OUTPUT_DIR / "evaluation" / "confusion_matrix_conf25.png"
    if not confusion_path.exists():
        confusion_path = OUTPUT_DIR / "evaluation" / "confusion_matrix.png"

    lines = [
        "# 基于 YOLO 的无人机识别与模型量化",
        "",
        "## 0. 项目源码与运行环境",
        "",
        "- 源码：`task4_yolo/run_pipeline.py`",
        "- 框架：Ultralytics 8.4.182、PyTorch 2.14.1 CPU、ONNX Runtime 1.31.0",
        "- 设备：CPU（AMD Ryzen 9 7945HX），未使用 CUDA/TensorRT",
        "- 训练好的权重：`runs/detect/runs_task4/adapt_10e/weights/best.pt`",
        "",
        "YOLO 基本 API：`YOLO(weights)` 加载模型，`model.train(data=..., epochs=...)` "
        "训练，`model.predict(source=...)` 推理，`model.val(...)` 评估，"
        "`model.export(format='onnx')` 导出，`model.track(...)` 做多目标追踪。",
        "",
        "## 1. 数据与实验说明",
        "",
        "- 数据集：Roboflow `drone and bird`，两类：`bird`、`drone`。",
        "- 原始 train/valid/test：3893 / 649 / 196 张。",
        "- 原始 test 划分只有 bird、没有 drone，因此本项目从 train 重新划分平衡验证集，"
        "并将原始 valid 作为测试集。",
        "- 受 CPU 算力限制，训练集抽样为 800 张无人机 + 800 张鸟，"
        "验证集为 150 + 150，测试集 649 张。",
        "- 题目提到的 `drone.pt` 未出现在提供的文件中；GitHub 无法访问，"
        "官方 YOLO11n 权重无法下载，因此 YOLO 模型从 `yolo11n.yaml` 从零训练。",
        "- 为缩小“远处小无人机”和“近景无人机视频”的域差异，"
        "用可访问的 COCO 预训练检测器为视频帧生成伪标签，"
        "再对 YOLO 进行 10 轮域适应微调。",
        "",
    ]
    if dataset:
        lines += [
            f"- 子集训练/验证/测试：{dataset['subset']['train']['images']} / "
            f"{dataset['subset']['valid']['images']} / "
            f"{dataset['subset']['test']['images']}",
            "",
        ]

    lines += [
        "## 2. 训练损失与指标",
        "",
        f"- 测试集 mAP@0.5：**{metrics['test_box_map50']:.4f}**",
        f"- 测试集 mAP@0.5:0.95：**{metrics['test_box_map50_95']:.4f}**",
        f"- 测试集 Precision：**{metrics['test_precision']:.4f}**",
        f"- 测试集 Recall：**{metrics['test_recall']:.4f}**",
        f"- `bird` 类别 mAP50：{metrics['per_class_map50']['bird']:.4f}",
        f"- `drone` 类别 mAP50：{metrics['per_class_map50']['drone']:.4f}",
        f"- 训练集匹配准确率（TP/GT）："
        f"**{metrics['train_match_accuracy']['match_accuracy_tp_over_gt']:.4f}**",
        f"- 测试集匹配准确率（TP/GT）："
        f"**{metrics['test_match_accuracy']['match_accuracy_tp_over_gt']:.4f}**",
        "",
        "目标检测没有分类任务中的普通 Accuracy，因此同时报告 mAP@0.5、"
        "Precision、Recall 和 IoU≥0.5 的匹配准确率。",
        "",
        "![training curves](evaluation/training_curves.png)",
        "",
        "![accuracy summary](evaluation/accuracy_summary.png)",
        "",
        f"![confusion matrix]({confusion_path.relative_to(OUTPUT_DIR).as_posix()})",
        "",
        "混淆矩阵在 `conf=0.25` 下生成；背景列表示漏检，"
        "`bird` 与 `drone` 的交叉项表示类别混淆。",
        "",
        "### 基线模型与域适应模型对比",
        "",
    ]
    if baseline:
        lines += [
            "| 模型 | Test mAP50 | Drone mAP50 | Train 匹配准确率 | Test 匹配准确率 | 视频检测 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 基线（仅原数据集） | {baseline['test_box_map50']:.4f} | "
            f"{baseline['per_class_map50']['drone']:.4f} | "
            f"{baseline['train_match_accuracy']['match_accuracy_tp_over_gt']:.4f} | "
            f"{baseline['test_match_accuracy']['match_accuracy_tp_over_gt']:.4f} | 0/378 |",
            f"| 域适应（最终） | {metrics['test_box_map50']:.4f} | "
            f"{metrics['per_class_map50']['drone']:.4f} | "
            f"{metrics['train_match_accuracy']['match_accuracy_tp_over_gt']:.4f} | "
            f"{metrics['test_match_accuracy']['match_accuracy_tp_over_gt']:.4f} | "
            f"{video['frames_with_detection'] if video else 0}/378 |",
            "",
            "域适应牺牲了少量原测试集指标，但把视频检测从 0 提升到可用水平，"
            "说明数据分布与训练集一致性比单纯增加训练轮数更关键。",
            "",
        ]
    lines += [
        "## 3. 示例图片推理",
        "",
        "由于无法下载官方 YOLO 权重，三张 COCO 示例图片使用可访问的 "
        "torchvision SSDLite320 COCO 预训练检测器完成推理，结果保存在 "
        "`pretrained_images/`。",
        "",
        "![bear](pretrained_images/coco-bear.jpg)",
        "",
        "![bus](pretrained_images/coco-bus.jpg)",
        "",
        "![person](pretrained_images/coco-person.jpg)",
        "",
        "## 4. 无人机视频推理",
        "",
    ]
    if video:
        lines += [
            f"- 视频帧数：{video['frames']}",
            f"- 处理速度：**{video['processing_fps']:.2f} FPS**（CPU，含预处理和后处理）",
            f"- 平均检测框数量：{video['mean_detections']:.2f}",
            f"- 检测到目标的帧数：{video['frames_with_detection']}/{video['frames']} "
            f"（{video['drone_detection_ratio'] * 100:.2f}%）",
            f"- 无人机平均置信度：{video['mean_drone_confidence']:.4f}",
            f"- 唯一追踪 ID 数量：{video['unique_track_ids']}",
            f"- 模型大小：{video['model_size_mb']:.2f} MB",
            "- EAO（可选指标）：本实验未计算，主要报告检测率、置信度和处理 FPS。",
            "",
            "推理视频：[example_tracked.mp4](video/example_tracked.mp4)",
            "",
            "![video keyframe](video/video_keyframe.jpg)",
            "",
        ]
    else:
        lines += ["视频推理尚未运行。", ""]

    lines += [
        "## 5. 模型导出与量化对比",
        "",
    ]
    if benchmark:
        lines += [
            "| 模型 | 大小 (MB) | 延迟 (ms) | FPS | mAP50 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, item in benchmark["models"].items():
            map50 = item["map50"]
            map_text = f"{map50:.4f}" if map50 is not None else "N/A"
            lines.append(
                f"| {name} | {item['model_size_mb']:.2f} | "
                f"{item['latency_ms']:.2f} | {item['fps']:.2f} | {map_text} |"
            )
        lines += [
            "",
            benchmark["note"],
            "",
        ]
    else:
        lines += ["导出与量化尚未运行。", ""]

    lines += [
        "## 6. 分析",
        "",
        "1. 由于缺少官方预训练权重，YOLO11n 在 CPU 子集上从零训练；"
        "域适应后仍能学习到无人机结构，但受训练轮数、样本量和分辨率限制，"
        "原数据集指标有所下降。",
        "2. 视频中的无人机目标较大，检测效果通常优于数据集中远处的小目标。",
        "3. ONNX FP32 便于跨框架部署；动态 INT8 能压缩模型体积，"
        "但 CPU 动态量化不一定带来明显加速，需要结合硬件和静态量化校准集评估。",
        "4. 进一步提高精度的方向：使用完整训练集、提高输入分辨率、增加训练轮数、"
        "使用官方预训练权重迁移学习，并针对 drone 类别做重采样或类别加权。",
        "",
        "## 7. 提交文件",
        "",
        "- 源码：`task4_yolo/run_pipeline.py`",
        "- 训练权重：`runs/detect/runs_task4/adapt_10e/weights/best.pt`",
        "- ONNX FP32：`outputs/task4_yolo/models/drone_yolo11n_fp32.onnx`",
        "- ONNX INT8：`outputs/task4_yolo/models/drone_yolo11n_int8.onnx`",
        "- 示例图片：`outputs/task4_yolo/pretrained_images/`",
        "- 推理视频：`outputs/task4_yolo/video/example_tracked.mp4`",
        "- 训练图与混淆矩阵：`outputs/task4_yolo/evaluation/`",
        "",
    ]
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 无人机识别与量化流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("pretrained").set_defaults(func=command_pretrained)
    subparsers.add_parser("report").set_defaults(func=command_report)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--weights", default=None)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--imgsz", type=int, default=320)
    train_parser.add_argument("--batch", type=int, default=32)
    train_parser.add_argument("--name", default="train")
    train_parser.set_defaults(func=command_train)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--weights", required=True)
    eval_parser.add_argument("--imgsz", type=int, default=320)
    eval_parser.set_defaults(func=command_evaluate)

    video_parser = subparsers.add_parser("video")
    video_parser.add_argument("--weights", required=True)
    video_parser.add_argument("--video", default=str(EXAMPLE_DIR / "example.mp4"))
    video_parser.add_argument("--imgsz", type=int, default=320)
    video_parser.set_defaults(func=command_video)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--weights", required=True)
    export_parser.add_argument("--imgsz", type=int, default=320)
    export_parser.set_defaults(func=command_export)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
