"""Shared DFMS-IQ components extracted from the validated research model."""

import torch
from torch import nn
from torch.nn import functional as F

class MultiScaleDepthwiseResidual1d(nn.Module):
    """Parameter-efficient local, medium, and long temporal evidence block."""

    def __init__(
        self,
        channels,
        kernels=(3, 7, 15),
        dilations=(1, 2, 4),
        dropout=0.0,
    ):
        super().__init__()
        channels = int(channels)
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have the same length")
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernels, dilations):
            kernel_size = int(kernel_size)
            dilation = int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * len(self.branches), channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout1d(float(dropout))

    def forward(self, values):
        encoded = torch.cat([branch(values) for branch in self.branches], dim=1)
        encoded = self.dropout(self.fusion(encoded))
        return F.silu(values + encoded, inplace=True)


class GlobalStatisticsPooling1d(nn.Module):
    """Unweighted mean/std path that attention cannot suppress."""

    def __init__(self, channels):
        super().__init__()
        self.output_dim = int(channels) * 2

    def forward(self, values):
        mean = values.mean(dim=-1)
        standard_deviation = torch.sqrt(
            torch.clamp(values.pow(2).mean(dim=-1) - mean.pow(2), min=1e-5)
        )
        return torch.cat([mean, standard_deviation], dim=1)


def _statistics_projection(input_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(int(input_dim), int(output_dim)),
        nn.LayerNorm(int(output_dim)),
        nn.GELU(),
        nn.Dropout(float(dropout)),
    )
