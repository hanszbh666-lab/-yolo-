"""
SDA-STD YOLO11 自定义网络模块包

包含以下五个模块，可直接在 Ultralytics YAML 中通过模块名引用：

    SPDConv     — 空间到深度无损下采样卷积
                  论文: "No More Strided Convolutions or Pooling" (2022)

    RFB         — 感受野增强模块
                  论文: "Receptive Field Block Net for Accurate and Fast
                         Object Detection" (ECCV 2018)

    ASFF        — 自适应空间特征融合
                  论文: "Learning Spatial Fusion for Single-Shot Object
                         Detection" (2019)

    EMA         — 高效多尺度注意力（跨空间学习）
                  论文: "Efficient Multi-Scale Attention Module with
                         Cross-Spatial Learning" (ISCAS 2023)

    SDA_Fusion  — 空间-深度自适应融合封装模块（三输入特征对齐 + ASFF 融合）

使用方式（在 scripts/ 或 train.py 中注册到 Ultralytics）:
    from models.modules import SPDConv, RFB, ASFF, EMA, SDA_Fusion

    # 注册到 Ultralytics 任务解析器
    from ultralytics.nn.tasks import attempt_load_weights
    import ultralytics.nn.tasks as tasks
    tasks.__dict__["SPDConv"]   = SPDConv
    tasks.__dict__["RFB"]       = RFB
    tasks.__dict__["ASFF"]      = ASFF
    tasks.__dict__["EMA"]       = EMA
    tasks.__dict__["SDA_Fusion"] = SDA_Fusion
"""

from .spd_conv import SPDConv
from .rfb import RFB
from .asff import ASFF
from .ema import EMA
from .sda_fusion import SDA_Fusion

__all__ = [
    "SPDConv",
    "RFB",
    "ASFF",
    "EMA",
    "SDA_Fusion",
]
