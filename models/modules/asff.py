"""ASFF module adapted from the original ASFF implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_conv(in_ch: int, out_ch: int, ksize: int, stride: int, leaky: bool = True) -> nn.Sequential:
    """Original ASFF conv-bn-activation helper."""
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=False,
        ),
    )
    stage.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module("leaky", nn.LeakyReLU(0.1))
    else:
        stage.add_module("relu6", nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    """
    Original ASFF fusion-weight computation adapted for pre-aligned inputs.

    In the original repository, ASFF performs both cross-scale resizing and
    weighted fusion. In this project, spatial alignment is handled upstream by
    SDA_Fusion, so this class keeps only the original weight-computation and
    expand stages while assuming the three inputs already share the same shape.
    """

    def __init__(self, c: int, rfb: bool = False, vis: bool = False):
        super().__init__()
        compress_c = 8 if rfb else 16
        self.weight_level_0 = add_conv(c, compress_c, 1, 1)
        self.weight_level_1 = add_conv(c, compress_c, 1, 1)
        self.weight_level_2 = add_conv(c, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.expand = add_conv(c, c, 3, 1)
        self.vis = vis

    def forward(self, inputs):
        x_level_0, x_level_1, x_level_2 = inputs[0], inputs[1], inputs[2]

        level_0_weight_v = self.weight_level_0(x_level_0)
        level_1_weight_v = self.weight_level_1(x_level_1)
        level_2_weight_v = self.weight_level_2(x_level_2)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = (
            x_level_0 * levels_weight[:, 0:1, :, :]
            + x_level_1 * levels_weight[:, 1:2, :, :]
            + x_level_2 * levels_weight[:, 2:, :, :]
        )
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        return out
