"""Minimal complex-valued layers used by the validated I/Q interface."""

import torch
from torch import nn
from torch.nn import functional as F


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.real_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.imag_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_channels))
            self.imag_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("real_bias", None)
            self.register_parameter("imag_bias", None)

    def forward(self, real, imag):
        output_real = self.real_conv(real) - self.imag_conv(imag)
        output_imag = self.real_conv(imag) + self.imag_conv(real)
        if self.real_bias is not None:
            output_real = output_real + self.real_bias.view(1, -1, 1)
            output_imag = output_imag + self.imag_bias.view(1, -1, 1)
        return output_real, output_imag


def complex_magnitude(real, imag, epsilon=1e-8):
    return torch.sqrt(real.pow(2) + imag.pow(2) + float(epsilon))


class ComplexBatchNormActivation1d(nn.Module):
    def __init__(self, channels, activation=True):
        super().__init__()
        self.real_bn = nn.BatchNorm1d(int(channels))
        self.imag_bn = nn.BatchNorm1d(int(channels))
        self.activation = bool(activation)

    def forward(self, real, imag):
        real = self.real_bn(real)
        imag = self.imag_bn(imag)
        if self.activation:
            real = F.silu(real, inplace=True)
            imag = F.silu(imag, inplace=True)
        return real, imag

class ComplexResidualBlock1d(nn.Module):
    """Two complex convolutions with a complex residual projection."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = ComplexConv1d(
            int(in_channels), int(out_channels), int(kernel_size), padding=padding
        )
        self.norm1 = ComplexBatchNormActivation1d(out_channels, activation=True)
        self.conv2 = ComplexConv1d(
            int(out_channels), int(out_channels), int(kernel_size), padding=padding
        )
        self.norm2 = ComplexBatchNormActivation1d(out_channels, activation=False)
        self.shortcut = None
        self.shortcut_norm = None
        if int(in_channels) != int(out_channels):
            self.shortcut = ComplexConv1d(
                int(in_channels), int(out_channels), kernel_size=1, padding=0
            )
            self.shortcut_norm = ComplexBatchNormActivation1d(
                out_channels, activation=False
            )

    def forward(self, real, imag):
        residual_real, residual_imag = real, imag
        real, imag = self.conv1(real, imag)
        real, imag = self.norm1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.norm2(real, imag)
        if self.shortcut is not None:
            residual_real, residual_imag = self.shortcut(
                residual_real, residual_imag
            )
            residual_real, residual_imag = self.shortcut_norm(
                residual_real, residual_imag
            )
        return (
            F.silu(real + residual_real, inplace=True),
            F.silu(imag + residual_imag, inplace=True),
        )
