"""
工具函数集合
提供常用的辅助功能
"""
import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def load_yaml(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data, yaml_path):
    """保存数据到YAML文件"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)


def load_json(json_path):
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, json_path, indent=2):
    """保存数据到JSON文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def visualize_bbox(image, bboxes, labels=None, scores=None, class_names=None, 
                   color=(0, 255, 0), thickness=2, save_path=None):
    """
    可视化边界框
    
    Args:
        image: 输入图像 (numpy array或路径)
        bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
        labels: 类别标签列表
        scores: 置信度分数列表
        class_names: 类别名称列表
        color: 边界框颜色
        thickness: 线宽
        save_path: 保存路径
    """
    # 读取图像
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    # 绘制边界框
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 添加标签
        label_text = ""
        if labels is not None and i < len(labels):
            label_id = labels[i]
            if class_names is not None:
                label_text = class_names[label_id]
            else:
                label_text = str(label_id)
        
        if scores is not None and i < len(scores):
            label_text += f" {scores[i]:.2f}"
        
        if label_text:
            cv2.putText(img, label_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # 保存或显示
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img


def plot_training_curves(csv_path, save_dir=None):
    """
    绘制训练曲线
    
    Args:
        csv_path: 训练日志CSV文件路径
        save_dir: 保存目录
    """
    import pandas as pd
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # Loss曲线
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # mAP曲线
    if 'metrics/mAP50(B)' in df.columns:
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[0, 1].set_title('mAP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr/pg0' in df.columns:
        axes[1, 1].plot(df['epoch'], df['lr/pg0'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存: {save_path}")
    
    plt.show()


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        iou: IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0.0
    return iou


def yolo_to_xyxy(bbox, img_width, img_height):
    """
    将YOLO格式转换为xyxy格式
    
    Args:
        bbox: [x_center, y_center, width, height] (归一化)
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [x1, y1, x2, y2] (像素坐标)
    """
    x_center, y_center, width, height = bbox
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]


def xyxy_to_yolo(bbox, img_width, img_height):
    """
    将xyxy格式转换为YOLO格式
    
    Args:
        bbox: [x1, y1, x2, y2] (像素坐标)
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [x_center, y_center, width, height] (归一化)
    """
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """计算可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_summary(model, input_size=(640, 640)):
    """创建模型摘要"""
    try:
        from torchinfo import summary
        return summary(model, input_size=(1, 3, *input_size))
    except ImportError:
        print("⚠️  需要安装torchinfo: pip install torchinfo")
        return None


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分数据集
    
    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    import random
    import shutil
    
    random.seed(seed)
    
    data_dir = Path(data_dir)
    images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    for split_name, split_images in splits.items():
        split_dir = data_dir.parent / split_name
        split_dir.mkdir(exist_ok=True)
        
        for img_path in split_images:
            shutil.copy(img_path, split_dir / img_path.name)
    
    print(f"✅ 数据集划分完成:")
    print(f"   训练集: {len(splits['train'])}")
    print(f"   验证集: {len(splits['val'])}")
    print(f"   测试集: {len(splits['test'])}")


if __name__ == '__main__':
    print("工具函数模块")
    print("可导入使用各种辅助功能")
