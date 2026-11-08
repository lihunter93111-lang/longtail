"""Route-1 controls: fused evidence and genuinely trainable uniform pooling.

The original DFMSQueryIQ implementation is kept unchanged for old checkpoints.
At width=16, dimension=64 and feedforward=128 all three added heads have the
same 300,211 total parameters, without retaining unused attention parameters.
"""
import math
import torch
from torch import nn
from .dfms import DFMSIQ
from .dfms_query import DFMSQueryIQ


class DFMSFusedQueryIQ(DFMSQueryIQ):
    """Reuse the computed phase/magnitude-fused temporal features as memory."""

    def __init__(self, num_classes=6, width=16, dropout=0.2,
                 include_phase=True, phase_lags=(1, 4, 16),
                 anti_aliased_downsampling=False, query_dimension=64,
                 query_heads=4, query_feedforward=128, query_layers=1,
                 query_dropout=0.1, memory_levels=(3,),
                 query_scale_init=0.1, use_position=True):
        if tuple(memory_levels) != (3,):
            raise ValueError('Fused evidence is defined only for the deepest level')
        super().__init__(num_classes, width, dropout, include_phase, phase_lags,
                         anti_aliased_downsampling, query_dimension, query_heads,
                         query_feedforward, query_layers, query_dropout,
                         memory_levels, query_scale_init, use_position)

    def _forward(self, values, return_attention):
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[-1] < 8:
            raise ValueError('DFMSFusedQueryIQ expects [batch, 2, length >= 8]')
        real, imag = self.complex_encoder(values[:, :1], values[:, 1:2])
        temporal = self.temporal_backbone(self.late_fusion(real, imag))
        global_logits = self.classifier(self.feature_projection(self.global_pool(temporal)))
        tokens = self.memory_norm(self.memory_projections[0](temporal).transpose(1, 2))
        position = self._position(tokens.shape[1], tokens) + self.level_embeddings.weight[0][None, None, :]
        queries = self.label_queries.weight[None].expand(len(values), -1, -1)
        target = torch.zeros_like(queries);attention = None
        for block in self.query_decoder:
            target, attention = block(target, queries, tokens, position, return_attention)
        query_logits = (target * self.query_classifier_weight).sum(-1) + self.query_classifier_bias
        logits = global_logits + self.query_scale.tanh() * query_logits
        return logits, {'global_logits': global_logits, 'query_logits': query_logits,
                        'query_scale': self.query_scale.tanh(), 'attention': attention,
                        'memory_lengths': [tokens.shape[1]]}


class DFMSUniformIQ(DFMSIQ):
    """Uniform deep-token mean plus a parameter-matched, active residual MLP.

Two real MLP blocks replace the attention/readout computation. Every parameter
participates in the forward pass; no Q/K matrices or positional buffers remain.
The memory source matches the original deep complex query arm.
"""

    def __init__(self, num_classes=6, width=16, dropout=0.2,
                 include_phase=True, phase_lags=(1, 4, 16),
                 anti_aliased_downsampling=False, query_dimension=64,
                 query_feedforward=128, query_dropout=0.1, query_scale_init=0.1):
        super().__init__(num_classes, width, dropout, include_phase, phase_lags,
                         anti_aliased_downsampling)
        if (query_dimension < 4 or query_feedforward < 1 or not 0 <= query_dropout < 1
                or not math.isfinite(query_scale_init)):
            raise ValueError('Invalid uniform-pooling configuration')
        dimension = query_dimension
        self.memory_levels = (3,)
        self.memory_projection = nn.Conv1d(width * 8, dimension, 1)
        self.memory_norm = nn.LayerNorm(dimension)
        self.pooled_norm = nn.LayerNorm(dimension)
        def feedforward():
            return nn.Sequential(nn.Linear(dimension, query_feedforward), nn.GELU(),
                nn.Dropout(query_dropout), nn.Linear(query_feedforward, dimension))
        self.pool_transform = feedforward()
        self.pool_transform_norm = nn.LayerNorm(dimension)
        self.label_queries = nn.Embedding(num_classes, dimension)
        self.class_transform = feedforward()
        self.class_transform_norm = nn.LayerNorm(dimension)
        self.output_dropout = nn.Dropout(query_dropout)
        self.query_classifier_weight = nn.Parameter(torch.empty(num_classes, dimension))
        self.query_classifier_bias = nn.Parameter(torch.zeros(num_classes))
        self.query_scale = nn.Parameter(torch.full((num_classes,), float(query_scale_init)))
        nn.init.normal_(self.label_queries.weight, std=0.02)
        nn.init.xavier_uniform_(self.query_classifier_weight)

    def _read_tokens(self, tokens):
        pooled = self.pooled_norm(tokens.mean(1))
        pooled = self.pool_transform_norm(pooled + self.output_dropout(self.pool_transform(pooled)))
        target = pooled[:, None, :] + self.label_queries.weight[None]
        target = self.class_transform_norm(target + self.output_dropout(self.class_transform(target)))
        return (target * self.query_classifier_weight).sum(-1) + self.query_classifier_bias

    def _forward(self, values, return_attention):
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[-1] < 8:
            raise ValueError('DFMSUniformIQ expects [batch, 2, length >= 8]')
        real, imag = self.complex_encoder(values[:, :1], values[:, 1:2])
        temporal = self.temporal_backbone(self.late_fusion(real, imag))
        global_logits = self.classifier(self.feature_projection(self.global_pool(temporal)))
        tokens = self.memory_norm(self.memory_projection(torch.cat((real, imag), 1)).transpose(1, 2))
        query_logits = self._read_tokens(tokens)
        attention = (tokens.new_full((len(values), self.label_queries.num_embeddings, tokens.shape[1]),
                                     1 / tokens.shape[1]) if return_attention else None)
        return global_logits + self.query_scale.tanh() * query_logits, {
            'global_logits': global_logits, 'query_logits': query_logits,
            'query_scale': self.query_scale.tanh(), 'attention': attention,
            'memory_lengths': [tokens.shape[1]]}

    def forward(self, values):
        return self._forward(values, False)[0]

    def forward_diagnostics(self, values):
        return self._forward(values, True)
