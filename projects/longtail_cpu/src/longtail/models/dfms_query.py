"""Category queries over a complex feature pyramid, with the DFMS global path.

The label-query/cross-attention/group-wise classification design follows the
Query2Label idea (https://arxiv.org/abs/2107.10834). This is a small 1-D
adaptation, not a reproduction of its image backbone or training recipe.
"""
import math
import torch
from torch import nn
from .dfms import DFMSIQ


class QueryDecoderBlock(nn.Module):
    """Six queries read time tokens; there is no token or query self-attention."""

    def __init__(self, dimension, heads, feedforward, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(feedforward, dimension))
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, target, queries, memory, position, return_attention=False):
        evidence, attention = self.cross_attention(
            target + queries, memory + position, memory,
            need_weights=return_attention, average_attn_weights=True)
        target = self.attention_norm(target + self.attention_dropout(evidence))
        target = self.output_norm(target + self.output_dropout(self.feedforward(target)))
        return target, attention


class DFMSQueryIQ(DFMSIQ):
    def __init__(self, num_classes=6, width=16, dropout=0.2,
                 include_phase=True, phase_lags=(1, 4, 16),
                 anti_aliased_downsampling=False, query_dimension=64,
                 query_heads=4, query_feedforward=128, query_layers=1,
                 query_dropout=0.1, memory_levels=(1, 2, 3),
                 query_scale_init=0.1, use_position=True):
        # Construct the common backbone first: identical seeds give identical
        # initial DFMS weights in every architecture arm.
        super().__init__(num_classes, width, dropout, include_phase, phase_lags,
                         anti_aliased_downsampling)
        levels = tuple(memory_levels)
        if (not levels or len(set(levels)) != len(levels)
                or any(type(i) is not int or i not in range(4) for i in levels)):
            raise ValueError('memory_levels must be unique indices from 0 to 3')
        if (query_dimension < 4 or query_dimension % 2
                or query_heads < 1 or query_dimension % query_heads
                or query_feedforward < 1 or query_layers < 1
                or not 0 <= query_dropout < 1
                or not math.isfinite(query_scale_init)):
            raise ValueError('Invalid query decoder configuration')
        self.memory_levels = levels
        self.query_dimension = query_dimension
        self.use_position = bool(use_position)
        # Real and imaginary channels are concatenated, preserving both streams.
        complex_channels = (width, width * 2, width * 3, width * 4)
        self.memory_projections = nn.ModuleList([
            nn.Conv1d(2 * complex_channels[i], query_dimension, 1) for i in levels])
        self.memory_norm = nn.LayerNorm(query_dimension)
        self.level_embeddings = nn.Embedding(len(levels), query_dimension)
        self.label_queries = nn.Embedding(num_classes, query_dimension)
        self.query_decoder = nn.ModuleList([
            QueryDecoderBlock(query_dimension, query_heads, query_feedforward, query_dropout)
            for _ in range(query_layers)])
        self.query_classifier_weight = nn.Parameter(torch.empty(num_classes, query_dimension))
        self.query_classifier_bias = nn.Parameter(torch.zeros(num_classes))
        self.query_scale = nn.Parameter(torch.full((num_classes,), float(query_scale_init)))
        nn.init.normal_(self.label_queries.weight, std=0.02)
        nn.init.normal_(self.level_embeddings.weight, std=0.02)
        nn.init.xavier_uniform_(self.query_classifier_weight)

    def _position(self, length, reference):
        if not self.use_position:
            return reference.new_zeros((1, length, self.query_dimension))
        # Centers are expressed on the same normalized PRI for every scale.
        # Unlike a fixed image-grid buffer this also handles odd signal lengths.
        time = ((torch.arange(length, device=reference.device, dtype=torch.float32)
                 + 0.5) / length * (2 * math.pi))[:, None]
        frequency = torch.exp(torch.arange(
            0, self.query_dimension, 2, device=reference.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.query_dimension))
        position = torch.stack(((time * frequency).sin(),
                                (time * frequency).cos()), dim=-1).flatten(1)
        return position.to(dtype=reference.dtype).unsqueeze(0)

    def _query_memory(self, pyramid):
        memories, positions, lengths = [], [], []
        for index, (level, projection) in enumerate(zip(self.memory_levels, self.memory_projections)):
            real, imag = pyramid[level]
            tokens = self.memory_norm(projection(torch.cat((real, imag), dim=1)).transpose(1, 2))
            memories.append(tokens)
            level_position = self.level_embeddings.weight[index][None, None, :]
            positions.append(self._position(tokens.shape[1], tokens) + level_position)
            lengths.append(tokens.shape[1])
        return torch.cat(memories, dim=1), torch.cat(positions, dim=1), lengths

    def _forward(self, values, return_attention):
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[-1] < 8:
            raise ValueError('DFMSQueryIQ expects [batch, 2, length >= 8]')
        pyramid = self.complex_encoder.forward_pyramid(values[:, :1], values[:, 1:2])
        real, imag = pyramid[-1]
        temporal = self.temporal_backbone(self.late_fusion(real, imag))
        global_logits = self.classifier(self.feature_projection(self.global_pool(temporal)))
        memory, position, lengths = self._query_memory(pyramid)
        queries = self.label_queries.weight[None].expand(len(values), -1, -1)
        target = torch.zeros_like(queries)
        attention = None
        for block in self.query_decoder:
            target, attention = block(target, queries, memory, position, return_attention)
        query_logits = (target * self.query_classifier_weight).sum(-1) + self.query_classifier_bias
        logits = global_logits + self.query_scale.tanh() * query_logits
        return logits, {'global_logits': global_logits, 'query_logits': query_logits,
                        'query_scale': self.query_scale.tanh(), 'attention': attention,
                        'memory_lengths': lengths}

    def forward(self, values):
        return self._forward(values, False)[0]

    def forward_diagnostics(self, values):
        """Inspection only; training and checkpoint evaluation use forward()."""
        return self._forward(values, True)


build_model = DFMSQueryIQ
