"""
ASFF: Adaptively Spatial Feature Fusion

论文: "Learning Spatial Fusion for Single-Shot Object Detection"
链接: https://arxiv.org/abs/1911.09516

核心思想:
    跨尺度特征融合（如 FPN Add/Concat）对所有空间位置施加固定权重，忽略不同
    位置特征的差异性。ASFF 对三个已对齐的特征图在每个空间位置自适应地学习
    融合权重（alpha/beta/gamma），通过 1×1 卷积预测权重图后 Softmax 归一化，
    再加权求和，从而让不同位置选择最有利的尺度特征参与融合。

注意:
    ASFF 模块本身不处理分辨率/通道对齐，三个输入特征图的空间尺寸和通道数
    必须已经完全一致。对齐操作由上层封装模块（如 SDA_Fusion）负责。
"""

import torch
import torch.nn as nn


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion（自适应空间特征融合）。

    接收三个空间尺寸与通道数均已对齐的特征图，在每个空间位置自适应地
    计算三路融合权重（Softmax 保证归一化），最后加权求和输出融合特征图。

    结构示意:
        x1 (B,C,H,W) ─── w_conv1 ──┐
        x2 (B,C,H,W) ─── w_conv2 ──┼─ Concat(B,3,H,W) ─ Softmax(dim=1)
        x3 (B,C,H,W) ─── w_conv3 ──┘                         ↓
                                              alpha (B,1,H,W) · x1
                                            + beta  (B,1,H,W) · x2
                                            + gamma (B,1,H,W) · x3
                                                         ↓
                                              输出 (B, C, H, W)

    Args:
        c (int): 三个输入特征图的（统一）通道数，也是输出通道数。
    """

    def __init__(self, c: int):
        super().__init__()
        # 三路各用 1×1 卷积将通道压缩为 1，产生单通道的空间权重图
        # 使用 bias=True 以便网络在训练初期能自由调节权重均衡
        self.w_conv1 = nn.Conv2d(c, 1, kernel_size=1, bias=True)
        self.w_conv2 = nn.Conv2d(c, 1, kernel_size=1, bias=True)
        self.w_conv3 = nn.Conv2d(c, 1, kernel_size=1, bias=True)

        # 沿通道维度 (dim=1) 做 Softmax，保证三路权重之和为 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        前向传播。

        张量 shape 变化（以 C=256, H=40, W=40 为例）:
            x1, x2, x3  : each (B, 256, 40, 40)  ← 三路已对齐输入
            w1, w2, w3  : each (B,   1, 40, 40)  ← 1×1 Conv 压缩为单通道权重图
            weights      :      (B,   3, 40, 40)  ← Concat(w1, w2, w3, dim=1)
            after softmax:      (B,   3, 40, 40)  ← 沿 dim=1 归一化，和=1
            alpha/beta/gamma:   (B,   1, 40, 40)  ← 切片得到各路权重
            输出         :      (B, 256, 40, 40)  ← α·x1 + β·x2 + γ·x3

        Args:
            inputs (list[Tensor] | tuple[Tensor]): 三个已对齐的特征图 [x1, x2, x3]，
                shape 均为 (B, C, H, W)。

        Returns:
            Tensor: 自适应融合后的特征图，shape = (B, C, H, W)。
        """
        x1, x2, x3 = inputs[0], inputs[1], inputs[2]

        # --- Step 1: 各路生成单通道权重图 ---
        w1 = self.w_conv1(x1)  # (B, 1, H, W)
        w2 = self.w_conv2(x2)  # (B, 1, H, W)
        w3 = self.w_conv3(x3)  # (B, 1, H, W)

        # --- Step 2: 拼接并 Softmax 归一化 ---
        # (B, 3, H, W) — 三路权重图并排，后续 Softmax 保证每点三权重之和=1
        weights = torch.cat([w1, w2, w3], dim=1)
        weights = self.softmax(weights)  # (B, 3, H, W)

        # 切分出三路自适应权重
        alpha = weights[:, 0:1, :, :]  # (B, 1, H, W) — x1 的采用比例
        beta  = weights[:, 1:2, :, :]  # (B, 1, H, W) — x2 的采用比例
        gamma = weights[:, 2:3, :, :]  # (B, 1, H, W) — x3 的采用比例

        # --- Step 3: 加权逐元素求和 ---
        # 广播乘法: (B,1,H,W) * (B,C,H,W) -> (B,C,H,W)
        return alpha * x1 + beta * x2 + gamma * x3  # (B, C, H, W)
