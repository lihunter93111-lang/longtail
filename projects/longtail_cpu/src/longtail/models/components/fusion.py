"""Shared DFMS-IQ components extracted from the validated research model."""

import torch
from torch import nn
from torch.nn import functional as F

from .complex import complex_magnitude


def _lagged(values, lag):
    lag = int(lag)
    if lag <= 0:
        raise ValueError("phase lag must be positive")
    if values.shape[-1] <= lag:
        return values[..., :1].expand_as(values)
    return torch.cat(
        [values[..., :1].expand(*values.shape[:-1], lag), values[..., :-lag]],
        dim=-1,
    )


class DelayedCartesianMagnitudePhaseFusion1d(nn.Module):
    """Late residual fusion without a single destructive concatenation bottleneck."""

    def __init__(
        self,
        complex_channels,
        output_channels,
        phase_hidden=32,
        phase_lags=(1, 4, 16),
        include_phase=True,
        initial_phase_scale=0.1,
    ):
        super().__init__()
        complex_channels = int(complex_channels)
        output_channels = int(output_channels)
        self.phase_lags = tuple(int(value) for value in phase_lags)
        self.include_phase = bool(include_phase)
        self.cartesian_projection = nn.Sequential(
            nn.Conv1d(
                complex_channels * 2, output_channels, 1, bias=False
            ),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(inplace=True),
        )
        self.magnitude_projection = nn.Sequential(
            nn.Conv1d(complex_channels, output_channels, 1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(inplace=True),
        )
        if self.include_phase:
            phase_input_channels = (
                complex_channels * 2 * len(self.phase_lags)
            )
            self.phase_projection = nn.Sequential(
                nn.Conv1d(
                    phase_input_channels, int(phase_hidden), 1, bias=False
                ),
                nn.BatchNorm1d(int(phase_hidden)),
                nn.SiLU(inplace=True),
                nn.Conv1d(
                    int(phase_hidden), output_channels, 1, bias=False
                ),
                nn.BatchNorm1d(output_channels),
            )
            self.phase_scale = nn.Parameter(
                torch.tensor(float(initial_phase_scale))
            )
        else:
            self.phase_projection = None
            self.register_parameter("phase_scale", None)
        self.output_channels = output_channels

    def _phase_features(self, real, imag, magnitude):
        normalized_real = real / magnitude
        normalized_imag = imag / magnitude
        features = []
        for lag in self.phase_lags:
            previous_real = _lagged(normalized_real, lag)
            previous_imag = _lagged(normalized_imag, lag)
            features.extend(
                [
                    normalized_real * previous_real
                    + normalized_imag * previous_imag,
                    normalized_imag * previous_real
                    - normalized_real * previous_imag,
                ]
            )
        return torch.cat(features, dim=1)

    def forward_components(self, real, imag):
        """Return the auditable late-fusion branches before residual addition."""

        magnitude = complex_magnitude(real, imag)
        cartesian = self.cartesian_projection(torch.cat([real, imag], dim=1))
        stable_magnitude = self.magnitude_projection(magnitude)
        phase = None
        if self.phase_projection is not None:
            phase = self.phase_projection(
                self._phase_features(real, imag, magnitude)
            )
        return cartesian, stable_magnitude, phase

    def combine_components(self, cartesian, stable_magnitude, phase=None):
        fused = cartesian + stable_magnitude
        if phase is not None:
            fused = fused + torch.tanh(self.phase_scale) * phase
        return F.silu(fused, inplace=True)

    def forward(self, real, imag):
        return self.combine_components(*self.forward_components(real, imag))
