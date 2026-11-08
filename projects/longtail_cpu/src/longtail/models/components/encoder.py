"""Shared DFMS-IQ components extracted from the validated research model."""

import torch
from torch import nn
from torch.nn import functional as F

from .complex import ComplexConv1d, ComplexBatchNormActivation1d, ComplexResidualBlock1d


class FixedBlurPool1d(nn.Module):
    """Fixed binomial low-pass used immediately before stride-2 pooling.

    The module performs only the anti-alias prefilter.  The existing AvgPool1d
    remains responsible for decimation, so all registered stage lengths stay
    unchanged.  One buffer is expanded across channels; there are no trainable
    parameters and real/imaginary streams use exactly the same coefficients.
    """

    def __init__(self, coefficients=(1.0, 4.0, 6.0, 4.0, 1.0)):
        super().__init__()
        kernel = torch.as_tensor(coefficients, dtype=torch.float32)
        if kernel.ndim != 1 or kernel.numel() % 2 != 1 or kernel.numel() < 3:
            raise ValueError("blur coefficients must be a one-dimensional odd kernel")
        if not torch.isfinite(kernel).all() or float(kernel.sum()) <= 0.0:
            raise ValueError("blur coefficients must be finite with positive sum")
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.padding = int(kernel.numel() // 2)

    def forward(self, values):
        if values.ndim != 3:
            raise ValueError("FixedBlurPool1d expects [batch, channels, length]")
        if values.shape[-1] <= self.padding:
            raise ValueError("input is too short for reflected blur padding")
        weight = self.kernel.to(dtype=values.dtype).expand(values.shape[1], 1, -1)
        padded = F.pad(values, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, weight, groups=values.shape[1])


class ComplexMultiScaleResidualStage1d(nn.Module):
    """Downsample and aggregate short, medium, and long complex evidence."""

    def __init__(
        self,
        in_channels,
        out_channels,
        branch_channels,
        kernels=(3, 7, 15),
        dilations=(1, 2, 4),
        anti_aliased_downsampling=False,
    ):
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have the same length")
        self.pre_downsample = (
            FixedBlurPool1d() if anti_aliased_downsampling else nn.Identity()
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernels, dilations):
            kernel_size = int(kernel_size)
            dilation = int(dilation)
            self.branches.append(
                nn.ModuleDict(
                    {
                        "conv": ComplexConv1d(
                            int(in_channels),
                            int(branch_channels),
                            kernel_size=kernel_size,
                            padding=dilation * (kernel_size // 2),
                            dilation=dilation,
                            bias=False,
                        ),
                        "norm": ComplexBatchNormActivation1d(
                            int(branch_channels), activation=True
                        ),
                    }
                )
            )
        merged_channels = int(branch_channels) * len(self.branches)
        self.projection = ComplexConv1d(
            merged_channels, int(out_channels), kernel_size=1, bias=False
        )
        self.projection_norm = ComplexBatchNormActivation1d(
            int(out_channels), activation=False
        )
        self.shortcut = ComplexConv1d(
            int(in_channels), int(out_channels), kernel_size=1, bias=False
        )
        self.shortcut_norm = ComplexBatchNormActivation1d(
            int(out_channels), activation=False
        )

    def forward(self, real, imag):
        real = self.pre_downsample(real)
        imag = self.pre_downsample(imag)
        real = self.downsample(real)
        imag = self.downsample(imag)
        residual_real, residual_imag = self.shortcut(real, imag)
        residual_real, residual_imag = self.shortcut_norm(
            residual_real, residual_imag
        )
        branch_real = []
        branch_imag = []
        for branch in self.branches:
            output_real, output_imag = branch["conv"](real, imag)
            output_real, output_imag = branch["norm"](
                output_real, output_imag
            )
            branch_real.append(output_real)
            branch_imag.append(output_imag)
        encoded_real, encoded_imag = self.projection(
            torch.cat(branch_real, dim=1), torch.cat(branch_imag, dim=1)
        )
        encoded_real, encoded_imag = self.projection_norm(
            encoded_real, encoded_imag
        )
        return (
            F.silu(encoded_real + residual_real, inplace=True),
            F.silu(encoded_imag + residual_imag, inplace=True),
        )


class DelayedComplexMultiScaleEncoder1d(nn.Module):
    """Keep real/imaginary streams separate through all long-range stages."""

    def __init__(self, width, anti_aliased_downsampling=False):
        super().__init__()
        width = int(width)
        self.block1 = ComplexResidualBlock1d(1, width, kernel_size=7)
        self.pre_downsample = (
            FixedBlurPool1d() if anti_aliased_downsampling else nn.Identity()
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.block2 = ComplexResidualBlock1d(
            width, width * 2, kernel_size=5
        )
        self.stage3 = ComplexMultiScaleResidualStage1d(
            width * 2,
            width * 3,
            branch_channels=width,
            anti_aliased_downsampling=anti_aliased_downsampling,
        )
        self.stage4 = ComplexMultiScaleResidualStage1d(
            width * 3,
            width * 4,
            branch_channels=width + width // 2,
            anti_aliased_downsampling=anti_aliased_downsampling,
        )
        self.out_channels = width * 4

    def forward_pyramid(self, real, imag):
        """Return every complex stage, including the pre-downsample stage1."""
        stage1_real, stage1_imag = self.block1(real, imag)
        stage1_real = self.pre_downsample(stage1_real)
        stage1_imag = self.pre_downsample(stage1_imag)
        real = self.downsample(stage1_real)
        imag = self.downsample(stage1_imag)
        stage2_real, stage2_imag = self.block2(real, imag)
        stage3_real, stage3_imag = self.stage3(stage2_real, stage2_imag)
        stage4_real, stage4_imag = self.stage4(stage3_real, stage3_imag)
        return (
            (stage1_real, stage1_imag),
            (stage2_real, stage2_imag),
            (stage3_real, stage3_imag),
            (stage4_real, stage4_imag),
        )

    def forward_stages(self, real, imag):
        return self.forward_pyramid(real, imag)[1:]

    def forward(self, real, imag):
        return self.forward_stages(real, imag)[-1]
