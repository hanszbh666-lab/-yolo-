"""EMA module adapted from the original EMA-attention-module implementation."""

import torch
from torch import nn


class EMA(nn.Module):
    """
    Original EMA attention module with grouped feature reshaping.

    The local project interface keeps `EMA(c1, c2=None, factor=32)` so the
    current Ultralytics YAML and parse_model patch can continue using `EMA`
    without further changes.
    """

    def __init__(self, c1: int, c2: int = None, factor: int = 32):
        super().__init__()
        channels = c2 if c2 is not None else c1
        self.groups = factor
        assert channels // self.groups > 0, (
            f"通道数 channels={channels} 必须大于分组因子 factor={factor}"
        )
        assert channels % self.groups == 0, (
            f"通道数 channels={channels} 必须能被 factor={factor} 整除"
        )

        group_channels = channels // self.groups
        self.out_channels = channels
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
