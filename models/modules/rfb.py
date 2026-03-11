"""RFB module adapted from the original RFBNet BasicRFB implementation."""

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    """Original RFBNet conv-bn-relu helper."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB(nn.Module):
    """
    Original RFBNet BasicRFB block with the same three-branch topology.

    This class keeps the local project interface `RFB(c1, c2, ...)` so the
    existing Ultralytics YAML and parse_model patch continue to work unchanged.
    """

    def __init__(self, c1: int, c2: int, stride: int = 1, scale: float = 0.1, visual: int = 1):
        super().__init__()
        self.scale = scale
        self.out_channels = c2
        inter_planes = c1 // 8

        self.branch0 = nn.Sequential(
            BasicConv(c1, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=visual,
                dilation=visual,
                relu=False,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=visual + 1,
                dilation=visual + 1,
                relu=False,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(c1, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=2 * visual + 1,
                dilation=2 * visual + 1,
                relu=False,
            ),
        )

        self.ConvLinear = BasicConv(6 * inter_planes, c2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(c1, c2, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
