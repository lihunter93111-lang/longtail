"""DFMS-IQ baseline, with the original 257,511-parameter state-dict layout."""
from torch import nn
from .components.encoder import DelayedComplexMultiScaleEncoder1d
from .components.fusion import DelayedCartesianMagnitudePhaseFusion1d
from .components.temporal import (
    MultiScaleDepthwiseResidual1d, GlobalStatisticsPooling1d, _statistics_projection,
)


class DFMSIQ(nn.Module):
    def __init__(self, num_classes=6, width=16, dropout=0.2,
                 include_phase=True, phase_lags=(1, 4, 16),
                 anti_aliased_downsampling=False):
        super().__init__()
        if width < 4 or num_classes < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid model width, class count or dropout")
        self.feature_dim = width * 8
        self.complex_encoder = DelayedComplexMultiScaleEncoder1d(
            width, anti_aliased_downsampling=anti_aliased_downsampling)
        # Keep the research checkpoint alias: both names refer to one module.
        self.complex_stem = self.complex_encoder
        self.complex_real_bn = nn.Identity()
        self.complex_imag_bn = nn.Identity()
        self.late_fusion = DelayedCartesianMagnitudePhaseFusion1d(
            self.complex_encoder.out_channels, self.feature_dim,
            phase_hidden=width * 2, phase_lags=phase_lags,
            include_phase=include_phase, initial_phase_scale=0.1)
        self.temporal_backbone = nn.Sequential(
            MultiScaleDepthwiseResidual1d(self.feature_dim, dropout=dropout))
        self.global_pool = GlobalStatisticsPooling1d(self.feature_dim)
        self.feature_projection = _statistics_projection(
            self.global_pool.output_dim, self.feature_dim, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))

    def forward_features(self, values):
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[-1] < 8:
            raise ValueError("DFMS-IQ expects [batch, 2, length >= 8]")
        real, imag = self.complex_encoder(values[:, :1], values[:, 1:2])
        temporal = self.temporal_backbone(self.late_fusion(real, imag))
        return self.feature_projection(self.global_pool(temporal))

    def forward(self, values):
        return self.classifier(self.forward_features(values))


build_model = DFMSIQ
