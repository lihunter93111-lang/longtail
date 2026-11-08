"""Residual boundaries using either global vectors or unpooled time positions."""
import math
import torch
from torch import nn


class GlobalResidual(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.hidden = nn.Linear(128, width)
        self.output = nn.Linear(width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(torch.nn.functional.gelu(self.hidden(x)))


class TemporalResidual(nn.Module):
    """Three label queries pool contextual time positions independently.

    Positions come from the frozen temporal backbone. This module adds no
    positional encoding and makes no claim to recover discarded signal samples.
    """
    def __init__(self, width=32):
        super().__init__()
        self.value = nn.Linear(128, width)
        self.query = nn.Parameter(torch.empty(3, width))
        self.output_weight = nn.Parameter(torch.zeros(3, width))
        self.output_bias = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.query, std=.02)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 128:
            raise ValueError('Expected [batch,128,time]')
        value = torch.nn.functional.gelu(self.value(x.transpose(1, 2)))
        score = torch.einsum('btd,cd->bct', value, self.query) / math.sqrt(value.shape[-1])
        pooled = torch.einsum('bct,btd->bcd', score.softmax(-1), value)
        return (pooled * self.output_weight).sum(-1) + self.output_bias


def build_residual(kind):
    if kind == 'global32':
        return GlobalResidual(32)
    if kind == 'global33':
        return GlobalResidual(33)
    if kind == 'temporal':
        return TemporalResidual(32)
    raise ValueError(kind)


def unpooled_features(model, signal):
    """Exact existing eval forward path, exposing its pre-pooling tensor."""
    real, imag = model.complex_encoder(signal[:, :1], signal[:, 1:2])
    temporal = model.temporal_backbone(model.late_fusion(real, imag))
    global_feature = model.feature_projection(model.global_pool(temporal))
    return global_feature, temporal
