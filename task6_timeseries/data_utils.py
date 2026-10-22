#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Air Passengers 数据读取、切分、归一化与滑动窗口构造。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class ForecastCase:
    """一个预测起点对应的输入、季节锚点和真实值。"""

    origin: int
    origin_date: pd.Timestamp
    baseline_input: torch.Tensor
    feature_input: torch.Tensor
    seasonal_reference: torch.Tensor
    target_scaled: torch.Tensor
    target_original: np.ndarray
    target_dates: pd.DatetimeIndex


@dataclass
class PreparedData:
    """训练所需全部张量，以及用于解释和反归一化的元数据。"""

    dataframe: pd.DataFrame
    values: np.ndarray
    log_values: np.ndarray
    scaled_values: np.ndarray
    features: np.ndarray
    log_mean: float
    log_std: float
    train_end: int
    val_end: int
    window: int
    horizon: int
    baseline_x: torch.Tensor
    baseline_y: torch.Tensor
    direct_x: torch.Tensor
    direct_y: torch.Tensor
    val_long_case: ForecastCase
    val_short_cases: list[ForecastCase]
    test_cases: list[ForecastCase]


def load_air_passengers(path: Path) -> pd.DataFrame:
    """读取并校验题目推荐的 Air Passengers CSV。"""
    frame = pd.read_csv(path)
    if "Month" not in frame.columns or "Passengers" not in frame.columns:
        raise ValueError("CSV 必须包含 Month 和 Passengers 两列。")
    frame = frame[["Month", "Passengers"]].copy()
    frame["Month"] = pd.to_datetime(frame["Month"], format="%Y-%m")
    frame["Passengers"] = pd.to_numeric(frame["Passengers"], errors="coerce")
    frame = frame.sort_values("Month").reset_index(drop=True)
    if frame["Passengers"].isna().any():
        raise ValueError("Passengers 列存在缺失值或非数值。")
    if frame["Month"].duplicated().any():
        raise ValueError("Month 列存在重复月份。")
    month_gaps = frame["Month"].diff().dropna().dt.days
    if not (month_gaps.between(28, 31)).all():
        raise ValueError("时间序列不是连续的月度数据。")
    return frame


def _inverse_scaled(
    scaled_values: np.ndarray,
    log_mean: float,
    log_std: float,
) -> np.ndarray:
    """把归一化后的对数客流量还原为原始人数。"""
    log_values = scaled_values * log_std + log_mean
    return np.expm1(log_values)


def prepare_data(
    path: Path,
    window: int = 24,
    horizon: int = 12,
    train_end: int = 108,
    val_end: int = 120,
) -> PreparedData:
    """构造训练窗口，并严格按时间顺序切分。

    train_end 和 val_end 使用“不含右端点”的索引：
    - 0..107：1949-01 至 1957-12，训练集；
    - 108..119：1958 年，验证集；
    - 120..143：1959-01 至 1960-12，测试集。
    """
    frame = load_air_passengers(path)
    if len(frame) != 144:
        raise ValueError(f"预期 144 个月，实际读取到 {len(frame)} 个月。")
    if train_end <= window + horizon or train_end >= val_end:
        raise ValueError("训练集、验证集、滑动窗口和预测步长设置不合理。")
    if val_end >= len(frame):
        raise ValueError("测试集为空。")

    values = frame["Passengers"].to_numpy(dtype=np.float64)
    log_values = np.log1p(values)
    log_mean = float(log_values[:train_end].mean())
    log_std = float(log_values[:train_end].std(ddof=0))
    if log_std <= 1e-12:
        raise ValueError("训练集对数序列方差过小。")
    scaled_values = (log_values - log_mean) / log_std

    month_angle = 2.0 * np.pi * frame["Month"].dt.month.to_numpy() / 12.0
    month_sin = np.sin(month_angle)
    month_cos = np.cos(month_angle)

    # 同比变化和近三个月动量都属于历史特征，在预测起点及以前均可计算。
    seasonal_change = np.zeros_like(scaled_values)
    seasonal_change[12:] = scaled_values[12:] - scaled_values[:-12]
    momentum_3m = np.zeros_like(scaled_values)
    momentum_3m[3:] = scaled_values[3:] - scaled_values[:-3]
    # 时间趋势只使用“未来时刻已知”的日历索引，不读取未来客流值。
    time_trend = np.arange(len(scaled_values), dtype=np.float64) / (
        train_end - 1
    )
    features = np.stack(
        [
            scaled_values,
            month_sin,
            month_cos,
            seasonal_change,
            momentum_3m,
            time_trend,
        ],
        axis=1,
    ).astype(np.float32)

    baseline_x = []
    baseline_y = []
    for origin in range(window - 1, train_end - 1):
        baseline_x.append(scaled_values[origin - window + 1 : origin + 1])
        baseline_y.append(scaled_values[origin + 1] - scaled_values[origin - 11])

    direct_x = []
    direct_y = []
    for origin in range(window - 1, train_end - horizon):
        direct_x.append(features[origin - window + 1 : origin + 1])
        reference = scaled_values[origin + 1 - horizon : origin + 1]
        target = scaled_values[origin + 1 : origin + 1 + horizon]
        direct_y.append(target - reference)

    def make_case(origin: int, case_horizon: int) -> ForecastCase:
        reference = scaled_values[origin + 1 - case_horizon : origin + 1]
        target = scaled_values[origin + 1 : origin + 1 + case_horizon]
        target_original = _inverse_scaled(target, log_mean, log_std)
        return ForecastCase(
            origin=origin,
            origin_date=pd.Timestamp(frame.loc[origin, "Month"]),
            baseline_input=torch.tensor(
                scaled_values[origin - window + 1 : origin + 1, None],
                dtype=torch.float32,
            ),
            feature_input=torch.tensor(
                features[origin - window + 1 : origin + 1],
                dtype=torch.float32,
            ),
            seasonal_reference=torch.tensor(reference, dtype=torch.float32),
            target_scaled=torch.tensor(target, dtype=torch.float32),
            target_original=target_original,
            target_dates=pd.date_range(
                frame.loc[origin + 1, "Month"],
                periods=case_horizon,
                freq="MS",
            ),
        )

    # 验证集：一个完整年度预测 + 12 个滚动一步预测。
    val_origin = train_end - 1
    val_long_case = make_case(val_origin, horizon)
    val_short_cases = [
        make_case(origin, 1)
        for origin in range(train_end - 1, val_end - 1)
    ]

    # 测试集按年度重新锚定；每个预测起点都是一个完整年度的最后一个月。
    test_origins = list(
        range(val_end - 1, len(frame) - 1, horizon)
    )
    test_cases = [make_case(origin, horizon) for origin in test_origins]
    if not test_cases or test_cases[0].origin != val_end - 1:
        raise ValueError(f"测试预测起点异常：{[case.origin for case in test_cases]}")

    return PreparedData(
        dataframe=frame,
        values=values,
        log_values=log_values,
        scaled_values=scaled_values,
        features=features,
        log_mean=log_mean,
        log_std=log_std,
        train_end=train_end,
        val_end=val_end,
        window=window,
        horizon=horizon,
        baseline_x=torch.tensor(np.stack(baseline_x)[..., None], dtype=torch.float32),
        baseline_y=torch.tensor(np.asarray(baseline_y), dtype=torch.float32),
        direct_x=torch.tensor(np.stack(direct_x), dtype=torch.float32),
        direct_y=torch.tensor(np.stack(direct_y), dtype=torch.float32),
        val_long_case=val_long_case,
        val_short_cases=val_short_cases,
        test_cases=test_cases,
    )


def inverse_scaled(
    scaled_values: np.ndarray | torch.Tensor,
    log_mean: float,
    log_std: float,
) -> np.ndarray:
    """公开的反归一化函数，供训练和评估脚本共用。"""
    array = np.asarray(scaled_values, dtype=np.float64)
    return _inverse_scaled(array, log_mean, log_std)
