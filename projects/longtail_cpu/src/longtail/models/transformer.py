"""A non-convolutional I/Q baseline using linear patches and self-attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class IQTransformer(nn.Module):
    def __init__(self, num_classes=6, patch_size=32, dimension=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        if patch_size < 1 or dimension < 2 or dimension % 2 or heads < 1 or dimension % heads or layers < 1:
            raise ValueError('Invalid Transformer dimensions')
        self.patch_size = patch_size
        self.projection = nn.Linear(2 * patch_size, dimension)
        block = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, dropout,
                                            batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dimension), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension, num_classes))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1] != 2 or signal.shape[-1] < 1:
            raise ValueError('Expected [batch, 2, time]')
        signal = F.pad(signal, (0, (-signal.shape[-1]) % self.patch_size))
        patches = signal.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)
        tokens = self.projection(patches)
        position = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, tokens.shape[-1], 2, device=tokens.device) * (-math.log(10000) / tokens.shape[-1]))
        encoding = torch.zeros(tokens.shape[1:], device=tokens.device)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        return self.head(self.encoder(tokens + encoding.to(tokens.dtype)).mean(1))


build_model = IQTransformer
