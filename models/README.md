# YOLO模型配置说明

## 📁 目录说明

本目录包含YOLO模型的配置文件（yaml格式）。这些配置文件定义了模型的网络结构，包括backbone、neck和head。

## 📄 文件说明

### yolov8s.yaml
- **用途**: YOLOv8s基础模型配置
- **来源**: 从ultralytics官方配置复制
- **说明**: 这是标准的YOLOv8s配置，适合直接训练使用

### yolov8s_improved.yaml
- **用途**: 改进版模型配置文件
- **说明**: 在此文件中实现你的模型改进
- **改进方向**:
  - 添加注意力机制（CBAM、SE、CA等）
  - 改进特征融合网络（BiFPN、ASFF等）
  - 增加小目标检测层（P2层）
  - 替换更强的主干网络
  - 改进检测头结构

## 🎯 模型配置结构说明

YOLO模型yaml配置主要包含三部分：

### 1. Parameters（参数设置）
```yaml
nc: 10  # 类别数量
scales:
  s: [0.33, 0.50, 1024]  # depth, width, max_channels
```

### 2. Backbone（主干网络）
```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # [from, repeats, module, args]
  - [-1, 1, Conv, [128, 3, 2]]
  ...
```
- **from**: 输入来源层编号（-1表示上一层）
- **repeats**: 模块重复次数
- **module**: 模块类型（Conv、C2f、SPPF等）
- **args**: 模块参数（通道数、卷积核大小、步长等）

### 3. Head（检测头）
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # 特征融合
  ...
  - [[15, 18, 21], 1, Detect, [nc]]  # 检测输出
```

## 🔧 如何修改模型

### 添加注意力机制
在backbone或head中插入注意力模块：
```yaml
- [-1, 1, CBAM, [512]]  # 添加CBAM注意力
- [-1, 1, SE, [512]]    # 添加SE注意力
```

### 增加小目标检测层
在head末尾添加P2检测层：
```yaml
- [-1, 1, nn.Upsample, [None, 2, 'nearest']]
- [[-1, 2], 1, Concat, [1]]  # 连接P2特征
- [-1, 3, C2f, [128]]
- [[12, 15, 18, 21], 1, Detect, [nc]]  # 4个检测层
```

### 修改通道数和深度
调整scales参数来控制模型大小：
```yaml
scales:
  # [depth, width, max_channels]
  s: [0.33, 0.50, 1024]  # 小模型
  m: [0.67, 0.75, 768]   # 中等模型
  l: [1.00, 1.00, 512]   # 大模型
```

## 📝 使用方法

### 训练时指定模型
```bash
# 使用基础模型
python scripts/train.py --model models/yolov8s.yaml

# 使用改进模型
python scripts/train.py --model models/yolov8s_improved.yaml
```

### 从预训练模型微调
```bash
# 加载预训练权重
python scripts/train.py --weights weights/yolov8s.pt --cfg models/yolov8s_improved.yaml
```

## 🚀 常用改进策略

### 针对VisDrone小目标检测的改进建议：

1. **增加P2检测层**（检测更小目标）
2. **添加注意力机制**（CBAM、CA等，增强特征表达）
3. **使用BiFPN**（双向特征融合，提升多尺度特征）
4. **增大输入分辨率**（640→1280，提高小目标检测）
5. **改进损失函数**（使用CIoU、Focal Loss等）

## 📚 参考资源

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [VisDrone数据集](http://aiskyeye.com/)

## ⚠️ 注意事项

1. 修改模型结构后，需要从头训练或重新加载预训练权重
2. 增加模型复杂度会提高计算量和显存占用
3. 建议在验证集上测试改进效果后再进行完整训练
4. 保持输入输出维度匹配，避免维度不兼容错误
