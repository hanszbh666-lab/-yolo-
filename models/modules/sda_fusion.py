"""
SDA_Fusion: Space-Depth Adaptive Fusion

自设封装模块（非独立论文模块），用于 SDA-STD YOLO11 颈部网络中的三尺度特征融合。

功能:
    接收来自 Backbone/Neck 不同层级的三个特征图（分辨率和通道数各不相同），
    通过 SPDConv（大分辨率下采样）和双线性插值上采样将三路特征对齐到统一的
    中层分辨率，再送入 ASFF 进行自适应空间加权融合，输出单一融合特征图供后
    续检测头使用。

输入特征图约定（以典型 YOLOv11 为例）:
    x1 — 浅层特征（高分辨率、低语义）: 如 P3, (B, C1, 2H, 2W)
    x2 — 中层特征（中分辨率、中语义）: 如 P4, (B, C2,  H,  W)  ← 目标对齐分辨率
    x3 — 深层特征（低分辨率、高语义）: 如 P5, (B, C3, H/2,H/2) 或已上采样
"""


import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

from .spd_conv import SPDConv
from .asff import ASFF


class SDA_Fusion(nn.Module):
    """
    Space-Depth Adaptive Fusion（空间-深度自适应融合模块）。

    三路特征对齐 + ASFF 自适应融合的封装。

    对齐策略:
        x1 (高分辨率) → SPDConv(c1_1, c2) 无损下采样 → (B, c2, H, W)
        x2 (中分辨率) → Conv(c1_2, c2, k=1)  通道对齐 → (B, c2, H, W)
        x3 (低分辨率) → Conv(c1_3, c2, k=1)  通道对齐 → (B, c2, H, W)
                       若 x3 空间尺寸 < x2，则追加 F.interpolate 上采样

    融合:
        [x1_aligned, x2_aligned, x3_aligned] → ASFF → (B, c2, H, W)

    Args:
        c1_list (list[int]): 三个输入特征图的通道数，格式 [c1_1, c1_2, c1_3]。
        c2      (int)      : 对齐后（也是输出）的统一通道数。
    """

    def __init__(self, c1_list: list, c2: int):
        super().__init__()
        assert len(c1_list) == 3, "c1_list 必须包含恰好 3 个通道数"
        c1_1, c1_2, c1_3 = c1_list

        # --- x1 对齐: SPDConv 无损下采样，分辨率减半、通道数压缩为 c2 ---
        # 输入 (B, c1_1, 2H, 2W) → 输出 (B, c2, H, W)
        self.align1 = SPDConv(c1_1, c2)

        # --- x2 对齐: 1×1 Conv 通道对齐，分辨率不变 ---
        # 输入 (B, c1_2, H, W) → 输出 (B, c2, H, W)
        self.align2 = Conv(c1_2, c2, k=1, s=1)

        # --- x3 对齐: 1×1 Conv 通道对齐，空间尺寸不足时内部 interpolate 上采样 ---
        # 输入 (B, c1_3, H', W') → 输出 (B, c2, H, W)，其中 H' <= H
        self.align3 = Conv(c1_3, c2, k=1, s=1)

        # --- 融合: ASFF 对三路已对齐特征进行自适应空间加权求和 ---
        self.asff = ASFF(c2)

    def forward(self, inputs):
        """
        前向传播。

        张量 shape 变化（以 c1_list=[128,256,512], c2=256, H=40,W=40 为例）:
            x1 输入          : (B, 128,  80, 80)  ← 高分辨率浅层特征
            x2 输入          : (B, 256,  40, 40)  ← 中层特征（目标分辨率）
            x3 输入          : (B, 512,  20, 20)  ← 低分辨率深层特征

            x1_aligned (SPDConv) :
                pixel_unshuffle  : (B, 512,  40, 40)  ← 4×128，H/2,W/2
                Conv(k=3,s=1)    : (B, 256,  40, 40)  ← 输出 c2 通道

            x2_aligned (1×1 Conv): (B, 256,  40, 40)  ← 通道对齐，尺寸不变

            x3_aligned (1×1 Conv): (B, 256,  20, 20)  ← 通道对齐
              → interpolate(×2)  : (B, 256,  40, 40)  ← 上采样至 x2 的 H/W

            ASFF 输出           : (B, 256,  40, 40)  ← 自适应加权融合

        Args:
            inputs (list[Tensor]): [x1, x2, x3]，三个不同尺度的特征图。

        Returns:
            Tensor: 融合后的特征图，shape = (B, c2, H_x2, W_x2)。
        """
        x1, x2, x3 = inputs[0], inputs[1], inputs[2]

        # --- Step 1: 通道和分辨率对齐 ---

        # x1: 高分辨率 → SPDConv 无损下采样到 x2 的分辨率
        # (B, c1_1, 2H, 2W) → (B, c2, H, W)
        x1_aligned = self.align1(x1)

        # x2: 中层特征 → 1×1 卷积调整通道数
        # (B, c1_2, H, W) → (B, c2, H, W)
        x2_aligned = self.align2(x2)

        # x3: 深层特征 → 先 1×1 卷积调整通道数，再按需上采样
        # (B, c1_3, H', W') → (B, c2, H', W')
        x3_aligned = self.align3(x3)

        # 若 x3 空间尺寸小于 x2，则双线性插值上采样至 x2 的尺寸
        # (B, c2, H', W') → (B, c2, H, W)
        target_h, target_w = x2_aligned.shape[2], x2_aligned.shape[3]
        if x3_aligned.shape[2] != target_h or x3_aligned.shape[3] != target_w:
            x3_aligned = F.interpolate(
                x3_aligned,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # --- Step 2: ASFF 自适应空间融合 ---
        # 三路 shape 此时均为 (B, c2, H, W)
        return self.asff([x1_aligned, x2_aligned, x3_aligned])  # (B, c2, H, W)
