"""Multi-resolution statistics and label-specific residual classifiers."""
import torch
from torch import nn


def segment_statistics(values, levels=(1, 2, 4)):
    if values.ndim != 3 or not levels or min(levels) < 1 or max(levels) > values.shape[-1]:
        raise ValueError("Expected [batch, channels, time] and nonempty time bins")
    statistics = []
    for count in levels:
        for part in torch.tensor_split(values, count, dim=-1):
            mean = part.mean(-1)
            std = (part.square().mean(-1) - mean.square()).clamp_min(1e-5).sqrt()
            statistics.extend((mean, std))
    return torch.cat(statistics, dim=1)


class ResidualReadout(nn.Module):
    def __init__(self, dimensions=128, hidden=32, outputs=1):
        super().__init__()
        self.hidden = nn.Linear(dimensions, hidden)
        self.output = nn.Linear(hidden, outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values):
        return self.output(nn.functional.gelu(self.hidden(values)))
