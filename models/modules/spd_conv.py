"""
SPDConv: Space-to-Depth Convolution

论文: "No More Strided Convolutions or Pooling: A New CNN Building Block
      for Low-Resolution Images and Small Objects"
链接: https://arxiv.org/abs/2208.03641

核心思想:
    传统步长卷积和池化在下采样时会丢失细粒度特征信息，对低分辨率图像和
    小目标尤为不利。SPDConv 用无损的空间到深度（SPD）重排层替代步长操作，
    将空间信息完整地转移到通道维度，再通过无步长卷积学习特征，实现信息
    无损的下采样。
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution（空间到深度卷积）。

    用 SPD 层 + 无步长卷积替代所有步长卷积/池化，避免小目标特征丢失。

    结构:
        输入 (B, C, H, W)
          └─ SPD 层 (pixel_unshuffle, scale=2)
               └─ (B, 4C, H/2, W/2)
                    └─ Conv(4C → c2, k=3, s=1) + BN + SiLU
                         └─ 输出 (B, c2, H/2, W/2)

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k  (int): 后续标准卷积的 kernel size，默认为 3。
    """

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        # SPD 层将空间采样率降低 2 倍，通道扩大 4 倍（2^2）
        # 后接步长为 1 的标准卷积对通道进行压缩：4*c1 → c2
        self.conv = Conv(c1 * 4, c2, k=k, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        张量 shape 变化:
            输入  : (B, C,   H,   W)
            SPD后 : (B, 4C, H/2, W/2)  ← pixel_unshuffle(downscale_factor=2)
                                          等价于按 stride=2 在空间切片后拼接到通道：
                                          取 (H[0::2], W[0::2]), (H[0::2], W[1::2]),
                                              (H[1::2], W[0::2]), (H[1::2], W[1::2])
                                          四块沿通道维度 concat
            输出  : (B, c2, H/2, W/2)  ← Conv(k=3, s=1) + BN + SiLU

        Args:
            x (Tensor): 输入特征图，shape = (B, C, H, W)。

        Returns:
            Tensor: 下采样后的特征图，shape = (B, c2, H/2, W/2)。
        """
        # --- Step 1: SPD 层（无损下采样） ---
        # 按原作者仓库中的实现顺序显式切片并沿通道拼接：
        # [H偶/W偶, H奇/W偶, H偶/W奇, H奇/W奇]
        # (B, C, H, W) -> (B, 4C, H/2, W/2)
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )

        # --- Step 2: 无步长标准卷积 ---
        # (B, 4C, H/2, W/2) -> (B, c2, H/2, W/2)
        return self.conv(x)
