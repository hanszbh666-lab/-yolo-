"""
通用YOLO数据集分析和可视化脚本
支持通过参数输入不同数据集，仅生成两类图表：
1) 检测目标像素分布图
2) 类别分布与数量图
"""
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None


# 默认VisDrone类别名称（仅在未指定类别名时用作兜底）
DEFAULT_VISDRONE_CLASS_NAMES = [
    'pedestrian',      # 行人
    'people',          # 人
    'bicycle',         # 自行车
    'car',             # 汽车
    'van',             # 货车
    'truck',           # 卡车
    'tricycle',        # 三轮车
    'awning-tricycle', # 遮阳篷三轮车
    'bus',             # 公交车
    'motor'            # 摩托车
]


def parse_yolo_label(label_path, img_width, img_height):
    """
    解析YOLO格式标注文件
    
    Args:
        label_path: 标注文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        annotations: [(class_id, bbox), ...]
    """
    annotations = []
    
    if not Path(label_path).exists():
        return annotations
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        annotations.append({
            'class_id': class_id,
            'bbox': [x_center, y_center, width, height],
            'area': width * height
        })
    
    return annotations


def collect_image_files(images_dir):
    """收集常见格式图像文件。"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def analyze_dataset(dataset_root, split='train'):
    """
    分析数据集统计信息
    
    Args:
        dataset_root: 数据集根目录
        split: 数据集划分 ('train', 'val', 'test')
    
    Returns:
        stats: 统计信息字典
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / 'images' / split
    labels_dir = dataset_root / 'labels' / split
    
    if not images_dir.exists():
        print(f"❌ 错误: 找不到图像目录 {images_dir}")
        return None
    
    # 统计信息（仅保留绘制两类图所需字段）
    stats = {
        'num_images': 0,
        'num_objects': 0,
        'class_counts': defaultdict(int),
        'bbox_sizes': [],
    }
    
    # 遍历所有图像
    image_files = collect_image_files(images_dir)
    
    print(f"\n{'='*60}")
    print(f"📊 分析 {split.upper()} 数据集")
    print(f"{'='*60}")
    print(f"📁 图像目录: {images_dir}")
    print(f"📋 标签目录: {labels_dir}")
    print(f"🖼️  图像数量: {len(image_files)}")
    print(f"{'='*60}\n")
    
    for img_path in tqdm(image_files, desc=f"Analyzing {split}"):
        # 读取图像尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        stats['num_images'] += 1
        
        # 解析标注
        label_path = labels_dir / (img_path.stem + '.txt')
        annotations = parse_yolo_label(label_path, img_width, img_height)
        
        stats['num_objects'] += len(annotations)
        
        # 统计类别和尺寸
        for ann in annotations:
            class_id = ann['class_id']
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            
            stats['class_counts'][class_id] += 1
            if width > 0 and height > 0:
                stats['bbox_sizes'].append((width, height))
    
    return stats


def resolve_class_names(args, data_root):
    """解析类别名，优先级：--class-names > --data-yaml > VisDrone默认 > 自动class_{id}。"""
    if args.class_names:
        return list(args.class_names)

    if args.data_yaml:
        yaml_path = Path(args.data_yaml)

        if not yaml_path.exists():
            print(f"⚠️ 未找到yaml文件: {yaml_path}，将尝试其他类别名来源")
        elif yaml is None:
            print("⚠️ 当前环境未安装PyYAML，无法解析--data-yaml，改用其他类别名来源")
        else:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_cfg = yaml.safe_load(f) or {}

            names = data_cfg.get('names')
            if isinstance(names, list):
                return names
            if isinstance(names, dict):
                return [v for _, v in sorted(names.items(), key=lambda x: int(x[0]))]

    if 'visdrone' in str(data_root).lower():
        return DEFAULT_VISDRONE_CLASS_NAMES

    return None


def class_name_from_id(class_id, class_names):
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f'class_{class_id}'


def plot_pixel_distribution(stats, split='train', save_dir=None, dataset_name='Dataset', show=False):
    """绘制检测目标宽高分布图（Bounding Box Size Distribution）。"""
    bbox_sizes = stats.get('bbox_sizes', [])
    if not bbox_sizes:
        print(f"⚠️ {split} 无有效目标框，跳过尺寸分布图")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    widths = [w for w, _ in bbox_sizes]
    heights = [h for _, h in bbox_sizes]

    ax.scatter(widths, heights, s=2, alpha=0.12, color='#1f77b4')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.set_title(f'{dataset_name} {split.upper()} - Bounding Box Size Distribution')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)

    # small/medium 尺度分界线
    ax.axvline(32, color='red', linestyle='--', alpha=0.8, label='Small (32px)')
    ax.axhline(32, color='red', linestyle='--', alpha=0.8)
    ax.axvline(96, color='orange', linestyle='--', alpha=0.8, label='Medium (96px)')
    ax.axhline(96, color='orange', linestyle='--', alpha=0.8)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{split}_bbox_size_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 尺寸分布图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_class_distribution(stats, class_names, split='train', save_dir=None, dataset_name='Dataset', show=False):
    """绘制类别分布与数量图。"""
    class_ids = sorted(stats['class_counts'].keys())
    if not class_ids:
        print(f"⚠️ {split} 无有效标注，跳过类别分布图")
        return

    labels = [class_name_from_id(i, class_names) for i in class_ids]
    counts = [stats['class_counts'][i] for i in class_ids]

    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(range(len(class_ids)), counts, color='#59A14F', alpha=0.9)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset_name} {split.upper()} - Class Distribution')
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{split}_class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 类别分布图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='通用YOLO数据集分析工具（仅输出像素分布图和类别分布图）')
    parser.add_argument('--data-root', type=str,
                        default='datasets/visdrone',
                        help='数据集根目录')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val'],
                        choices=['train', 'val', 'test'],
                        help='要分析的数据集划分')
    parser.add_argument('--data-yaml', type=str, default=None,
                        help='数据集yaml路径（用于解析类别名，如configs/visdrone.yaml）')
    parser.add_argument('--class-names', nargs='+', default=None,
                        help='直接传入类别名列表，优先级高于--data-yaml')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='图表中显示的数据集名称，默认自动推断')
    parser.add_argument('--save-dir', type=str,
                        default='runs/analysis',
                        help='统计图表保存目录')
    parser.add_argument('--show', action='store_true',
                        help='是否弹窗显示图像（默认仅保存不显示）')
    
    args = parser.parse_args()
    
    # 按用户输入路径使用（相对路径基于当前工作目录）
    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    dataset_name = args.dataset_name or data_root.name
    class_names = resolve_class_names(args, data_root)
    
    print("\n" + "="*60)
    print("📊 通用YOLO数据集分析工具")
    print("="*60)
    print(f"📁 数据集目录: {data_root}")
    print(f"🧾 类别名来源: {'命令行/配置文件' if class_names else '自动class_id占位名'}")
    
    # 分析每个数据集划分
    for split in args.splits:
        stats = analyze_dataset(data_root, split)
        if stats:
            plot_pixel_distribution(stats, split, save_dir, dataset_name, show=args.show)
            plot_class_distribution(stats, class_names, split, save_dir, dataset_name, show=args.show)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
