# 🚁 YOLO 无人机小目标检测项目

基于YOLOv8/YOLOv11的无人机航拍数据集小目标检测训练项目，支持VisDrone和UAVDT数据集。

## 📋 项目简介

本项目提供完整的YOLOv8训练流程，专门针对VisDrone和UAVDT无人机航拍数据集进行优化。这些数据集包含大量小目标、密集场景和多尺度目标，非常适合训练和评估小目标检测算法。

### 主要特点

- ✅ 支持VisDrone和UAVDT两个主流无人机数据集
- ✅ 完整的数据下载和转换流程
- ✅ 针对RTX4060优化的训练配置
- ✅ YOLOv8和YOLOv11基础模型和改进模板
- ✅ 数据集分析和可视化工具
- ✅ 训练、验证、检测一体化脚本
- ✅ 模型改进接口，方便后续优化
- ✅ 完整的实验流程和文档

## 📁 项目结构

```
yolo/
├── datasets/                   # 数据集目录
│   ├── UAVDT/                 # UAVDT数据集
│   │   ├── images/           # 图像文件
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/           # YOLO格式标签
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── conversion_stats.json
│   └── visdrone/             # VisDrone数据集
│       ├── images/           # 图像文件
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/           # YOLO格式标签
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── annotations/      # 原始标注文件
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/                    # 模型配置文件
│   ├── yolov8s.yaml          # YOLOv8s基础配置
│   ├── yolov8s_improved.yaml # 改进模型配置模板
│   └── README.md             # 模型改进说明
│
├── configs/                   # 数据集配置文件
│   ├── uavdt.yaml            # UAVDT数据集配置
│   └── visdrone.yaml         # VisDrone数据集配置
│
├── scripts/                   # 脚本目录
│   ├── download_uavdt.py     # UAVDT数据下载
│   ├── download_visdrone.py  # VisDrone数据下载
│   ├── reorganize_visdrone.py # VisDrone格式重组
│   ├── data_analysis.py      # 数据分析
│   ├── train.py              # 训练脚本
│   ├── val.py                # 验证脚本
│   ├── detect.py             # 推理脚本
│   └── utils.py              # 工具函数
│
├── weights/                   # 预训练模型权重
├── runs/                      # 实验输出目录
│   ├── train/                # 训练结果
│   │   ├── uavdt_yolov8s/
│   │   └── yolov8s_visdrone/
│   ├── val/                  # 验证结果
│   │   ├── uavdt_yolov8s_val/
│   │   └── visdrone_yolov8s_val/
│   ├── detect/               # 检测结果
│   └── analysis/             # 数据分析结果
│
├── docs/                      # 文档目录
│   ├── 实验管理指南.md
│   └── UAVDT使用指南.md
│
├── notebooks/                 # Jupyter笔记本
├── yolo11n.pt                # YOLOv11预训练模型
├── requirements.txt           # Python依赖
└── README.md                 # 项目说明
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

#### VisDrone数据集
```bash
# 下载VisDrone数据集
python scripts/download_visdrone.py

# 或指定要下载的划分
python scripts/download_visdrone.py --splits train val
```

#### UAVDT数据集
```bash
# 下载UAVDT数据集
python scripts/download_uavdt.py

# 或指定要下载的序列
python scripts/download_uavdt.py --sequences M0203 M0401
```

### 3. 转换数据格式

#### VisDrone格式转换
```bash
# 将VisDrone格式转换为YOLO格式
python scripts/reorganize_visdrone.py

# 转换指定划分
python scripts/reorganize_visdrone.py --splits train val test
```

#### UAVDT格式转换
```bash
# UAVDT数据集会自动转换为YOLO格式
# 无需额外转换步骤
```

### 4. 数据分析（可选）

```bash
# 分析数据集统计信息并生成可视化图表
python scripts/data_analysis.py

# 分析指定划分
python scripts/data_analysis.py --splits train val
```

### 5. 开始训练

#### VisDrone数据集训练
```bash
# 使用YOLOv8s预训练模型训练VisDrone
python scripts/train.py --model models/yolov8s.yaml --data configs/visdrone.yaml --epochs 300

# 自定义参数训练
python scripts/train.py --model models/yolov8s.yaml --data configs/visdrone.yaml --batch 16 --epochs 300 --imgsz 640
```

#### UAVDT数据集训练
```bash
# 使用YOLOv8s预训练模型训练UAVDT
python scripts/train.py --model models/yolov8s.yaml --data configs/uavdt.yaml --epochs 300

# 使用改进模型配置
python scripts/train.py --model models/yolov8s_improved.yaml --data configs/uavdt.yaml --epochs 300
```

### 6. 验证模型

```bash
# 验证训练好的模型
python scripts/val.py --model runs/train/yolov8s_visdrone/weights/best.pt

# 在测试集上验证
python scripts/val.py --model runs/train/yolov8s_visdrone/weights/best.pt --split test
```

### 7. 推理检测

```bash
# 检测单张图像
python scripts/detect.py --source path/to/image.jpg

# 检测文件夹中的所有图像
python scripts/detect.py --source datasets/visdrone/images/val

# 检测视频
python scripts/detect.py --source path/to/video.mp4

# 使用摄像头
python scripts/detect.py --source 0
```

## 🧪 完整实验流程

### 4.1 标准实验流程

以下是完整的模型训练和评估实验流程，适用于VisDrone和UAVDT数据集：

```bash
# 1. 训练模型
python scripts/train.py --name visdrone_yolov8s --model models/yolov8s.yaml --data configs/visdrone.yaml --epochs 300

# 2. 在验证集上详细评估
python scripts/val.py --name visdrone_yolov8s_val --model runs/train/yolov8s_visdrone/weights/best.pt --data configs/visdrone.yaml

# 3. 实际应用检测
python scripts/detect.py --name yolov8_visdrone --source datasets/visdrone/images/test --weights runs/train/yolov8s_visdrone/weights/best.pt
```

### 4.2 UAVDT数据集实验流程

```bash
# 1. 下载和准备UAVDT数据集
python scripts/download_uavdt.py

# 2. 训练UAVDT模型
python scripts/train.py --name uavdt_yolov8s --model models/yolov8s.yaml --data configs/uavdt.yaml --epochs 300

# 3. 验证UAVDT模型
python scripts/val.py --name uavdt_yolov8s_val --model runs/train/uavdt_yolov8s/weights/best.pt --data configs/uavdt.yaml

# 4. UAVDT数据集检测
python scripts/detect.py --name uavdt_yolov8s --source datasets/UAVDT/images/test --weights runs/train/uavdt_yolov8s/weights/best.pt
```

### 4.3 实验参数说明

#### 训练参数
- `--name`: 实验名称，用于区分不同的训练实验
- `--model`: 模型配置文件路径
- `--data`: 数据集配置文件路径
- `--epochs`: 训练轮数，建议200-300
- `--batch`: 批次大小，RTX4060建议16
- `--imgsz`: 输入图像尺寸，640或1280

#### 验证参数
- `--name`: 验证实验名称
- `--model`: 训练好的模型权重路径
- `--data`: 数据集配置文件路径
- `--conf`: 置信度阈值，默认0.25
- `--iou`: NMS IoU阈值，默认0.45

#### 检测参数
- `--name`: 检测实验名称
- `--source`: 输入图像/视频/文件夹路径
- `--weights`: 模型权重文件路径
- `--conf`: 检测置信度阈值
- `--save`: 保存检测结果

### 4.4 实验结果查看

训练完成后，可以在以下位置查看实验结果：

```
runs/
├── train/实验名称/           # 训练结果
│   ├── weights/
│   │   ├── best.pt          # 最佳模型权重
│   │   └── last.pt          # 最新模型权重
│   ├── results.csv          # 训练指标曲线数据
│   ├── confusion_matrix.png # 混淆矩阵
│   └── train_batch*.jpg     # 训练样本可视化
│
├── val/实验名称/             # 验证结果
│   ├── confusion_matrix.png # 验证混淆矩阵
│   ├── F1_curve.png         # F1曲线
│   ├── PR_curve.png         # PR曲线
│   └── val_batch*.jpg       # 验证样本可视化
│
└── detect/实验名称/          # 检测结果
    ├── labels/              # 检测标签文件
    └── *.jpg                # 检测结果图像
```

## 📊 数据集信息

### VisDrone数据集

- **来源**: [VisDrone官网](http://aiskyeye.com/)
- **特点**: 无人机航拍、小目标密集、多尺度目标
- **规模**:
  - 训练集: 6,471张图像
  - 验证集: 548张图像
  - 测试集: 1,610张图像
- **类别**: 10类目标检测

### UAVDT数据集

- **来源**: [UAVDT官网](https://sites.google.com/site/uavdtdataset/)
- **特点**: 无人机视频数据集，包含大量车辆跟踪序列
- **规模**: 
  - 50个视频序列
  - 超过80,000帧图像
  - 3种主要目标类型
- **类别**: 3类目标检测（车、卡车、公交车）

### 支持的数据集类别

#### VisDrone类别（10类）

| ID | 类别名称 | 中文 |
|----|---------|------|
| 0 | pedestrian | 行人 |
| 1 | people | 人 |
| 2 | bicycle | 自行车 |
| 3 | car | 汽车 |
| 4 | van | 货车 |
| 5 | truck | 卡车 |
| 6 | tricycle | 三轮车 |
| 7 | awning-tricycle | 遮阳篷三轮车 |
| 8 | bus | 公交车 |
| 9 | motor | 摩托车 |

#### UAVDT类别（3类）

| ID | 类别名称 | 中文 |
|----|---------|------|
| 0 | car | 汽车 |
| 1 | truck | 卡车 |
| 2 | bus | 公交车 |

## ⚙️ 训练配置

### 针对RTX4060 (8GB)的推荐配置

```python
model = 'yolov8s.pt'      # YOLOv8s模型
epochs = 300              # 训练轮数
batch_size = 16           # 批次大小（可调整为24）
imgsz = 640               # 图像尺寸（可提升至1280）
device = '0'              # GPU设备
workers = 8               # 数据加载线程
amp = True                # 混合精度训练
optimizer = 'SGD'         # 优化器
lr0 = 0.01               # 初始学习率
```

### 训练参数说明

- **batch_size**: RTX4060建议16-24，根据显存使用情况调整
- **imgsz**: 640适合快速训练，1280提高小目标检测但速度慢
- **epochs**: 建议200-300轮，早停patience=50
- **amp**: 自动混合精度训练，节省显存并加速

## 🔧 模型改进

模型配置文件位于 `models/` 目录：

- **yolov8s.yaml**: YOLOv8s基础配置
- **yolov8s_improved.yaml**: 改进模型模板

### 常用改进方向

1. **添加注意力机制**
   - CBAM (Convolutional Block Attention Module)
   - SE (Squeeze-and-Excitation)
   - CA (Coordinate Attention)
   - ECA (Efficient Channel Attention)

2. **改进特征融合**
   - BiFPN (Bidirectional Feature Pyramid Network)
   - ASFF (Adaptively Spatial Feature Fusion)
   - AFPN (Asymptotic Feature Pyramid Network)

3. **增加小目标检测层**
   - 添加P2检测层 (stride=4)
   - 多尺度特征融合

4. **损失函数优化**
   - CIoU Loss
   - Focal Loss
   - Varifocal Loss

详细说明请查看 [models/README.md](models/README.md)

## 📈 性能指标

训练完成后，主要关注以下指标：

- **mAP@0.5**: 在IoU=0.5时的平均精度
- **mAP@0.5:0.95**: COCO标准的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

结果文件保存在 `runs/train/实验名称/` 目录。

## 🛠️ 常用命令

### 训练相关

```bash
# 从头开始训练
python scripts/train.py --model yolov8s.pt --epochs 300

# 继续训练（断点续训）
python scripts/train.py --resume runs/train/yolov8s_visdrone/weights/last.pt

# 使用更大的图像尺寸训练（提高小目标检测）
python scripts/train.py --imgsz 1280 --batch 8

# 使用自定义改进模型
python scripts/train.py --model models/yolov8s_improved.yaml
```

### 验证相关

```bash
# 验证最佳模型
python scripts/val.py --model runs/train/yolov8s_visdrone/weights/best.pt

# 保存JSON结果（用于COCO评估）
python scripts/val.py --model best.pt --save-json

# 调整置信度阈值
python scripts/val.py --model best.pt --conf 0.01 --iou 0.6
```

### 推理相关

```bash
# 检测并保存结果
python scripts/detect.py --source test_image.jpg --conf 0.25

# 批量检测
python scripts/detect.py --source datasets/visdrone/images/val --save

# 不保存结果，仅显示
python scripts/detect.py --source test.jpg --show --nosave
```

## 📝 注意事项

1. **显存不足**：减小batch_size或图像尺寸
2. **训练慢**：使用较小的图像尺寸(640)或减少epoch
3. **数据集路径**：确保configs/visdrone.yaml中的路径正确
4. **预训练权重**：首次运行会自动下载YOLOv8权重

## 🎯 后续改进计划

- [ ] 添加注意力机制模块
- [ ] 实现BiFPN特征融合
- [ ] 增加P2小目标检测层
- [ ] 集成WandB训练监控
- [ ] 模型剪枝和量化
- [ ] TensorRT部署优化

## 📚 参考资料

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [VisDrone Dataset](http://aiskyeye.com/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## 📧 联系方式

如有问题或建议，欢迎提Issue或PR。

## 📄 许可证

MIT License

---

**Happy Training! 🚀**
