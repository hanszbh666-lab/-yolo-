# UAVDT数据集使用指南

## 📋 概述

UAVDT (UAV-Benchmark Dataset) 是一个无人机航拍的目标检测和跟踪数据集，专门用于车辆检测任务。

**数据集特点：**
- 🚁 无人机航拍视角
- 🚗 主要目标：车辆 (汽车、卡车、公交车)
- 🎬 50个视频序列，约4万帧
- 📊 超过79万个标注框

## 📊 数据集统计

根据分析结果：
- **序列数量**: 50个视频序列 (M0101-M1401)
- **总帧数**: 40,409帧
- **标注统计**:
  - 汽车 (car): 755,688个标注
  - 卡车 (truck): 25,086个标注  
  - 公交车 (bus): 18,021个标注

## 🚀 快速开始

### 1. 当前状态
✅ 标注文件已分析  
✅ 数据集目录结构已创建  
✅ 配置文件已生成  
❌ 图像文件需要下载

### 2. 下载图像数据

```bash
# 查看下载说明
python scripts/download_uavdt.py

# 手动下载完成后，验证数据
python scripts/download_uavdt.py --verify path/to/downloaded/data
```

### 3. 转换数据集

```bash
# 将UAVDT格式转换为YOLO格式
python scripts/convert_uavdt.py path/to/downloaded/data --create-config

# 或者先分析数据
python scripts/analyze_uavdt.py path/to/GT/folder --create-dirs
```

### 4. 开始训练

```bash
# 使用UAVDT数据集训练YOLOv8s
python scripts/train.py --data configs/uavdt.yaml --name uavdt_yolov8s

# 使用预训练权重加速训练
python scripts/download_weights.py yolov8s
python scripts/train.py --data configs/uavdt.yaml --model weights/yolov8s.pt --name uavdt_yolov8s_pretrained
```

## 📁 目录结构

### 当前结构
```
datasets/uavdt/
├── images/          # 图像文件 (需要下载)
│   ├── train/      # 训练图像
│   ├── val/        # 验证图像
│   └── test/       # 测试图像
├── labels/          # YOLO格式标注 (转换后生成)
│   ├── train/      # 训练标注
│   ├── val/        # 验证标注
│   └── test/       # 测试标注
└── README.md       # 数据集说明

configs/
└── uavdt.yaml      # UAVDT数据集配置文件
```

### 原始数据结构 (下载后)
```
UAVDT-Raw/
├── sequences/
│   ├── M0101/
│   │   └── img1/           # 序列图像帧
│   │       ├── 000001.jpg
│   │       ├── 000002.jpg
│   │       └── ...
│   ├── M0201/
│   └── ...
└── GT/                     # 标注文件 (已有)
    ├── M0101_gt_whole.txt
    ├── M0201_gt_whole.txt
    └── ...
```

## 🔧 配置文件说明

[configs/uavdt.yaml](configs/uavdt.yaml) 配置：

```yaml
# 数据集路径
path: datasets/uavdt
train: images/train
val: images/val
test: images/test

# 类别配置
nc: 3
names:
  0: car      # 汽车
  1: truck    # 卡车
  2: bus      # 公交车
```

## 🎯 训练建议

### 针对UAVDT数据集的训练参数优化

```bash
# 基础训练
python scripts/train.py \
  --data configs/uavdt.yaml \
  --name uavdt_yolov8s \
  --epochs 300 \
  --batch 16 \
  --imgsz 640

# 无人机场景优化训练
python scripts/train.py \
  --data configs/uavdt.yaml \
  --name uavdt_yolov8s_optimized \
  --epochs 300 \
  --batch 16 \
  --imgsz 1280 \     # 更大输入尺寸，适合小目标
  --patience 100     # 更大耐心值
```

### 建议的训练策略

1. **图像尺寸**: 使用1280x1280以更好检测小目标
2. **数据增强**: 适度使用，避免过度变形无人机视角
3. **学习率**: 可以适当降低，车辆形状相对固定
4. **验证频率**: 增加验证频率，监控小目标检测性能

## 📈 验证和评估

```bash
# 验证训练好的模型
python scripts/val.py \
  --weights runs/train/uavdt_yolov8s/weights/best.pt \
  --data configs/uavdt.yaml \
  --name uavdt_val

# 在测试集上评估
python scripts/val.py \
  --weights runs/train/uavdt_yolov8s/weights/best.pt \
  --data configs/uavdt.yaml \
  --split test \
  --name uavdt_test_eval
```

## 🔍 检测应用

```bash
# 使用训练好的模型进行检测
python scripts/detect.py \
  --weights runs/train/uavdt_yolov8s/weights/best.pt \
  --source path/to/test/images \
  --name uavdt_detection

# 检测视频
python scripts/detect.py \
  --weights runs/train/uavdt_yolov8s/weights/best.pt \
  --source path/to/test/video.mp4 \
  --name uavdt_video_detection
```

## 🆚 与VisDrone的对比

| 特征 | UAVDT | VisDrone |
|------|-------|----------|
| **视角** | 无人机航拍 | 无人机航拍 |
| **主要目标** | 车辆检测 | 多类别小目标 |
| **类别数** | 3类 | 10类 |
| **数据量** | 4万帧 | 约1万张图像 |
| **场景** | 交通场景 | 城市综合场景 |
| **难点** | 车辆密集、尺度变化 | 小目标、类别多样 |

## 📚 相关论文

```
@inproceedings{du2018unmanned,
  title={The unmanned aerial vehicle benchmark: Object detection and tracking},
  author={Du, Dawei and Qi, Yuankai and Yu, Hongyang and Yang, Yifan and Duan, Kaiwen and Li, Guorong and Zhang, Weigang and Huang, Qingming and Tian, Qi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={370--386},
  year={2018}
}
```