#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Air Passengers 时序预测使用的三个 PyTorch 模型。

所有模型都输出“相对于同一月份上一年观测值的增量”，这样既保留了
季节基线，又让网络专注于学习趋势变化。时间序列输入只包含历史窗口，
不会读取未来真实值。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecursiveLSTM(nn.Module):
    """基线模型：单层 LSTM + 单步预测 + 递归生成未来序列。"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """训练时输入一个窗口，输出下一个月的同比增量。"""
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)

    @torch.no_grad()
    def forecast(
        self,
        x: torch.Tensor,
        horizon: int,
        seasonal_reference: torch.Tensor,
    ) -> torch.Tensor:
        """递归预测。

        参数
        ----
        x:
            形状为 [batch, window, 1] 的归一化历史水平值。
        horizon:
            预测步数。
        seasonal_reference:
            形状为 [batch, horizon]，第 h 列是预测起点后第 h 个月
            在上一年的真实观测值，作为季节锚点。
        """
        output, (hidden, cell) = self.lstm(x)
        last_hidden = output[:, -1]
        predictions: list[torch.Tensor] = []
        for step in range(horizon):
            delta = self.head(last_hidden).squeeze(-1)
            next_level = seasonal_reference[:, step] + delta
            predictions.append(next_level)
            next_input = next_level[:, None, None]
            _, (hidden, cell) = self.lstm(next_input, (hidden, cell))
            last_hidden = hidden[-1]
        return torch.stack(predictions, dim=1)


class DirectGRU(nn.Module):
    """消融模型：多变量输入 + GRU + 直接预测未来 12 个月。"""

    def __init__(
        self,
        n_features: int = 6,
        hidden_size: int = 16,
        horizon: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, hidden_size)
        self.activation = nn.GELU()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_projection(x))
        output, _ = self.gru(x)
        return self.head(output[:, -1])


class CausalDepthwiseBlock(nn.Module):
    """带残差连接的因果深度可分离卷积块。

    左侧填充保证位置 t 只使用 t 及更早的输入，避免信息泄漏。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.depthwise(x)
        x = F.gelu(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return F.gelu(x + residual)


class TemporalAttention(nn.Module):
    """对历史窗口中的每个时间片学习权重，并汇总成上下文向量。"""

    def __init__(self, hidden_size: int, attention_size: int = 24) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context


class AttnConvGRU(nn.Module):
    """改进模型：多尺度因果卷积 + BiGRU + 时间注意力 + 直接多步预测。"""

    def __init__(
        self,
        n_features: int = 6,
        conv_channels: int = 8,
        gru_hidden: int = 8,
        horizon: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=1,
        )
        self.conv_blocks = nn.Sequential(
            CausalDepthwiseBlock(conv_channels, kernel_size=3, dilation=1),
            CausalDepthwiseBlock(conv_channels, kernel_size=3, dilation=2),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(gru_hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, window, features] -> [batch, features, window]
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.conv_blocks(x)
        sequence, _ = self.gru(x.transpose(1, 2))
        context = self.attention(sequence)
        return self.head(context)


def build_model(
    name: str,
    n_features: int,
    horizon: int,
    seed: int,
) -> nn.Module:
    """按实验名称构建模型，并完成确定性的参数初始化。"""
    torch.manual_seed(seed)
    if name == "LSTM-Recursive":
        return RecursiveLSTM(input_size=1, hidden_size=48, dropout=0.1)
    if name == "Direct-GRU":
        return DirectGRU(
            n_features=n_features,
            hidden_size=16,
            horizon=horizon,
            dropout=0.3,
        )
    if name == "Attn-CNN-GRU":
        return AttnConvGRU(
            n_features=n_features,
            conv_channels=8,
            gru_hidden=8,
            horizon=horizon,
            dropout=0.3,
        )
    raise ValueError(f"未知模型：{name}")


def count_parameters(model: nn.Module) -> int:
    """统计模型中可训练参数的数量。"""
    return sum(parameter.numel() for parameter in model.parameters())
