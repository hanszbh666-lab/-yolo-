"""
EMA: Efficient Multi-Scale Attention Module with Cross-Spatial Learning

论文: "Efficient Multi-Scale Attention Module with Cross-Spatial Learning"
链接: https://arxiv.org/abs/2305.13563  (ISCAS 2023)

核心思想:
    将通道拆为两路（1×1 卷积支路 Y1 和 3×3 深度卷积支路 Y2），并在两路之间
    进行跨空间（Cross-Spatial）交互：Y1 沿 H 维池化后为 Y2 提供 W 方向的
    通道注意力，Y2 沿 W 维池化后为 Y1 提供 H 方向的通道注意力，两路互相
    调制后拼接输出，以极低的计算代价实现多尺度空间注意力。
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention（高效多尺度注意力模块）。

    结构示意:
                            ┌─ 1×1 Conv → Y1 (B, C/2, H, W) ──────────────────────┐
        x (B, C, H, W)      │                         ↑ W 方向注意力 (sigmoid)      │
        split(dim=1)  ──────┤               pool_W(Y2)→1×1 conv→sigmoid            ├─ Concat
                            │                                                        │  (B,C,H,W)
                            └─ 3×3 DWConv → Y2 (B, C/2, H, W) ─────────────────────┘
                                                     ↑ H 方向注意力 (sigmoid)
                                            pool_H(Y1)→1×1 conv→sigmoid

        Concat(Y1_attn, Y2_attn) -> 1×1 Conv -> 输出 (B, C, H, W)

    Args:
        c1      (int): 输入/输出通道数，必须为偶数。
        c2      (int): 输出通道数，默认与 c1 相同（EMA 是通道保持型模块）。
        groups  (int): 通道分组数（当前实现按 groups 分组后在组内操作，
                       默认 1 表示不分组直接对整体通道拆分。若需分组，
                       可将 groups 设为大于 1 的值，要求 c % groups == 0）。
    """

    def __init__(self, c1: int, c2: int = None, groups: int = 1):
        """
        Args:
            c1     (int): 输入通道数（由 parse_model 自动注入）。
            c2     (int): 输出通道数，默认与 c1 相同（EMA 是通道保持型模块）。
                          在 Ultralytics base_modules 机制下由 args[0]*width 缩放后传入。
            groups (int): 通道分组数，默认 1。
        """
        super().__init__()
        # EMA 是通道保持型注意力模块，c2 应等于 c1；
        # 当从 YAML 中由 parse_model 解析时，c2 已经过 width 缩放，使用 c2；
        # 当直接实例化（如 EMA(256)）时，c2 为 None，退回到 c1。
        c = c2 if c2 is not None else c1
        assert c % 2 == 0, f"通道数 c={c} 必须为偶数（需对半拆分）"
        assert c % groups == 0, f"通道数 c={c} 必须能被 groups={groups} 整除"

        self.groups = groups
        self.c_half = c // 2  # 每路的通道数

        # --- Branch 1: 1×1 卷积，捕捉局部逐点信息 ---
        # (B, C/2, H, W) -> (B, C/2, H, W)
        self.conv1x1 = Conv(self.c_half, self.c_half, k=1, s=1)

        # --- Branch 2: 3×3 深度可分离卷积，捕捉局部空间信息 ---
        # g=c_half 即 DWConv：每个通道独立卷积，计算量极低
        # (B, C/2, H, W) -> (B, C/2, H, W)
        self.conv3x3_dw = Conv(self.c_half, self.c_half, k=3, s=1, g=self.c_half)

        # --- 跨空间学习: Y1 → H 方向池化 → 为 Y2 生成 W 方向注意力 ---
        # AdaptiveAvgPool2d((None, 1)): 保留 H 维度，W 聚合为 1
        # (B, C/2, H, W) -> (B, C/2, H, 1)
        self.pool_H = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_H = nn.Conv2d(self.c_half, self.c_half, kernel_size=1, bias=False)

        # --- 跨空间学习: Y2 → W 方向池化 → 为 Y1 生成 H 方向注意力 ---
        # AdaptiveAvgPool2d((1, None)): 保留 W 维度，H 聚合为 1
        # (B, C/2, H, W) -> (B, C/2, 1, W)
        self.pool_W = nn.AdaptiveAvgPool2d((1, None))
        self.conv_W = nn.Conv2d(self.c_half, self.c_half, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # --- 输出投影: 拼接两路后通道恢复为 c ---
        # (B, C, H, W) -> (B, C, H, W)
        self.conv_out = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        张量 shape 变化（以 c=256, H=40, W=40 为例）:
            输入 x          : (B, 256, 40, 40)
            x1, x2 (split)  : each (B, 128, 40, 40)  ← 沿 dim=1 对半拆分

            -- Branch 1 --
            y1 = conv1x1(x1): (B, 128, 40, 40)

            -- Branch 2 --
            y2 = conv3x3(x2): (B, 128, 40, 40)

            -- 跨空间注意力 (Cross-Spatial) --
            pool_H(y1)       : (B, 128, 40,  1)  ← 对 W 维做全局平均池化
            conv_H 后 sigmoid : (B, 128, 40,  1)  ← 生成 H 方向的空间注意力权重
            y2_attn = y2 * attn_H: (B, 128, 40, 40)  ← y2 被 H 方向注意力调制

            pool_W(y2)       : (B, 128,  1, 40)  ← 对 H 维做全局平均池化
            conv_W 后 sigmoid : (B, 128,  1, 40)  ← 生成 W 方向的空间注意力权重
            y1_attn = y1 * attn_W: (B, 128, 40, 40)  ← y1 被 W 方向注意力调制

            -- 拼接 & 输出 --
            out = Concat     : (B, 256, 40, 40)
            conv_out(out)    : (B, 256, 40, 40)

        Args:
            x (Tensor): 输入特征图，shape = (B, C, H, W)。

        Returns:
            Tensor: 注意力增强后的特征图，shape = (B, C, H, W)。
        """
        # --- Step 1: 沿通道对半拆分 ---
        # x1: 前半通道；x2: 后半通道
        x1, x2 = x.chunk(2, dim=1)  # each: (B, C/2, H, W)

        # --- Step 2: 双分支特征提取 ---
        y1 = self.conv1x1(x1)     # (B, C/2, H, W)  — 逐点语义特征
        y2 = self.conv3x3_dw(x2)  # (B, C/2, H, W)  — 局部空间特征

        # --- Step 3: 跨空间注意力（Cross-Spatial Learning）---
        # y1 经 H 方向池化（保留 H，聚合 W）→ 学习 H 维空间权重 → 调制 y2
        # 含义: 告知 y2 在哪些行（H 位置）特征更重要
        attn_for_y2 = self.sigmoid(self.conv_H(self.pool_H(y1)))  # (B, C/2, H, 1)
        y2_attn = y2 * attn_for_y2                                 # (B, C/2, H, W) ← 广播乘

        # y2 经 W 方向池化（保留 W，聚合 H）→ 学习 W 维空间权重 → 调制 y1
        # 含义: 告知 y1 在哪些列（W 位置）特征更重要
        attn_for_y1 = self.sigmoid(self.conv_W(self.pool_W(y2)))  # (B, C/2, 1, W)
        y1_attn = y1 * attn_for_y1                                 # (B, C/2, H, W) ← 广播乘

        # --- Step 4: 拼接两路注意力增强特征并投影 ---
        out = torch.cat([y1_attn, y2_attn], dim=1)  # (B, C, H, W)
        return self.conv_out(out)                    # (B, C, H, W)
