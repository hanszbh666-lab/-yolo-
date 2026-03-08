"""
RFB: Receptive Field Block

论文: "Receptive Field Block Net for Accurate and Fast Object Detection"
链接: https://arxiv.org/abs/1711.07767  (ECCV 2018)

核心思想:
    模仿人类视觉皮层中感受野（Receptive Field）的偏心率与尺寸关系，用多条
    并行的空洞卷积分支（不同 dilation rate）显式地模拟多尺度感受野，然后
    聚合得到包含丰富尺度信息的特征表示，最后加上 shortcut 残差连接。
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class RFB(nn.Module):
    """
    Receptive Field Block（感受野增强模块）。

    四条并行分支，dilation 分别为 1 / 1 / 3 / 5，拼接后经 1×1 卷积降维，
    再与 shortcut 相加，输出与输入空间尺寸相同的增强特征图。

    结构示意:
                    ┌─ Branch 0: 1×1 Conv ──────────────────────────────────────┐
                    │                                                            │
                    ├─ Branch 1: 1×1 Conv → 3×3 Conv(d=1) ─────────────────────┤
        x(B,C,H,W) ─┤                                                           ├─ Concat(4路)
                    ├─ Branch 2: 1×1 Conv → 3×3 Conv → 3×3 Conv(d=3) ──────────┤  (B, c2, H, W)
                    │                                                            │
                    └─ Branch 3: 1×1 Conv → 3×3 Conv → 3×3 Conv(d=5) ──────────┘
                                                                            ↓
                                                                   1×1 Conv 降维
                                                                            ↓
                                                                      + shortcut
                                                                            ↓
                                                                    输出 (B, c2, H, W)

    Args:
        c1  (int): 输入通道数。
        c2  (int): 输出通道数，内部每条分支通道数为 c2 // 4。
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        # 每条分支的中间通道数，四路拼接后恰好为 c2
        inter_c = c2 // 4

        # --- Branch 0: 1×1 Conv (感受野最小，仅局部点信息) ---
        # (B, C, H, W) -> (B, inter_c, H, W)
        self.branch0 = Conv(c1, inter_c, k=1, s=1)

        # --- Branch 1: 1×1 Conv -> 3×3 Conv, dilation=1 ---
        # (B, C, H, W) -> (B, inter_c, H, W) -> (B, inter_c, H, W)
        self.branch1 = nn.Sequential(
            Conv(c1, inter_c, k=1, s=1),           # 通道压缩
            Conv(inter_c, inter_c, k=3, s=1, d=1), # 普通 3×3，有效 RF=3
        )

        # --- Branch 2: 1×1 Conv -> 3×3 Conv -> 3×3 AtrousConv, dilation=3 ---
        # (B, C, H, W) -> (B, inter_c, H, W) -> (B, inter_c, H, W)
        self.branch2 = nn.Sequential(
            Conv(c1, inter_c, k=1, s=1),           # 通道压缩
            Conv(inter_c, inter_c, k=3, s=1, d=1), # 3×3 感受野预扩展
            Conv(inter_c, inter_c, k=3, s=1, d=3), # 空洞卷积 dilation=3，有效 RF=7
        )

        # --- Branch 3: 1×1 Conv -> 3×3 Conv -> 3×3 AtrousConv, dilation=5 ---
        # (B, C, H, W) -> (B, inter_c, H, W) -> (B, inter_c, H, W)
        self.branch3 = nn.Sequential(
            Conv(c1, inter_c, k=1, s=1),           # 通道压缩
            Conv(inter_c, inter_c, k=3, s=1, d=1), # 3×3 感受野预扩展
            Conv(inter_c, inter_c, k=3, s=1, d=5), # 空洞卷积 dilation=5，有效 RF=11
        )

        # --- 聚合: 将四路 (各 inter_c 通道) Concat 后用 1×1 卷积降维 ---
        # 拼接后通道: 4 * inter_c = (c2 // 4) * 4 = c2
        # Conv 内含 BN + SiLU，不使用 act 以便 shortcut 相加后再激活
        self.conv_cat = Conv(4 * inter_c, c2, k=1, s=1, act=False)

        # --- Shortcut: 若输入输出通道不同，用 1×1 卷积对齐（不加激活函数） ---
        self.shortcut = (
            Conv(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()
        )

        self.act = nn.SiLU(inplace=True)  # 残差相加后统一激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        张量 shape 变化（以 c1=256, c2=256, inter_c=64 为例）:
            输入         : (B, 256, H, W)
            branch0 输出 : (B,  64, H, W)
            branch1 输出 : (B,  64, H, W)
            branch2 输出 : (B,  64, H, W)
            branch3 输出 : (B,  64, H, W)
            Concat       : (B, 256, H, W)  ← 沿 dim=1 拼接四路分支
            conv_cat     : (B, 256, H, W)  ← 1×1 卷积 + BN（无激活）
            shortcut     : (B, 256, H, W)  ← Identity 或 1×1 Conv
            相加后激活    : (B, 256, H, W)  ← SiLU

        Args:
            x (Tensor): 输入特征图，shape = (B, c1, H, W)。

        Returns:
            Tensor: 增强后的特征图，shape = (B, c2, H, W)。
        """
        # 四条分支并行提取不同感受野的特征
        b0 = self.branch0(x)  # (B, inter_c, H, W)
        b1 = self.branch1(x)  # (B, inter_c, H, W)
        b2 = self.branch2(x)  # (B, inter_c, H, W)
        b3 = self.branch3(x)  # (B, inter_c, H, W)

        # 沿通道拼接: (B, inter_c*4, H, W) = (B, c2, H, W)
        out = torch.cat([b0, b1, b2, b3], dim=1)

        # 1×1 卷积聚合多尺度感受野信息
        out = self.conv_cat(out)  # (B, c2, H, W)

        # 残差连接 + 激活
        return self.act(out + self.shortcut(x))  # (B, c2, H, W)
