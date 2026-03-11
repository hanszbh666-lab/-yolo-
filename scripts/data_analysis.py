"""
VisDrone数据集分析和可视化脚本
分析数据集统计信息并生成可视化图表
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


# VisDrone类别名称
CLASS_NAMES = [
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
    
    # 统计信息
    stats = {
        'num_images': 0,
        'num_objects': 0,
        'class_counts': defaultdict(int),
        'bbox_sizes': [],
        'objects_per_image': [],
        'image_sizes': [],
        'small_objects': 0,  # 面积 < 32x32
        'medium_objects': 0,  # 32x32 <= 面积 < 96x96
        'large_objects': 0,   # 面积 >= 96x96
    }
    
    # 遍历所有图像
    image_files = list(images_dir.glob('*.jpg'))
    
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
        stats['image_sizes'].append((img_width, img_height))
        stats['num_images'] += 1
        
        # 解析标注
        label_path = labels_dir / (img_path.stem + '.txt')
        annotations = parse_yolo_label(label_path, img_width, img_height)
        
        num_objects = len(annotations)
        stats['num_objects'] += num_objects
        stats['objects_per_image'].append(num_objects)
        
        # 统计类别和尺寸
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            area = ann['area']
            
            stats['class_counts'][class_id] += 1
            stats['bbox_sizes'].append(bbox[2:])  # [width, height]
            
            # 分类目标尺寸
            if area < 24 * 24:
                stats['small_objects'] += 1
            elif area < 64 * 64:
                stats['medium_objects'] += 1
            else:
                stats['large_objects'] += 1
    
    return stats


def plot_statistics(stats, split='train', save_dir=None):
    """
    绘制统计图表
    
    Args:
        stats: 统计信息
        split: 数据集划分
        save_dir: 保存目录
    """
    if stats is None:
        return
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'VisDrone {split.upper()} Dataset Statistics', fontsize=16)
    
    # 1. 类别分布
    ax1 = plt.subplot(2, 3, 1)
    class_ids = sorted(stats['class_counts'].keys())
    class_counts = [stats['class_counts'][i] for i in class_ids]
    class_labels = [CLASS_NAMES[i] for i in class_ids]
    
    bars = ax1.bar(range(len(class_ids)), class_counts, color='skyblue')
    ax1.set_xticks(range(len(class_ids)))
    ax1.set_xticklabels(class_labels, rotation=45, ha='right')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 2. 每张图像的目标数分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(stats['objects_per_image'], bins=50, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Objects per Image')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Objects per Image Distribution')
    ax2.axvline(np.mean(stats['objects_per_image']), color='red', 
                linestyle='--', label=f'Mean: {np.mean(stats["objects_per_image"]):.2f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. 目标尺寸分布
    ax3 = plt.subplot(2, 3, 3)
    if stats['bbox_sizes']:
        bbox_widths = [size[0] for size in stats['bbox_sizes']]
        bbox_heights = [size[1] for size in stats['bbox_sizes']]
        ax3.scatter(bbox_widths, bbox_heights, alpha=0.1, s=1)
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Height (pixels)')
        ax3.set_title('Bounding Box Size Distribution')
        ax3.set_xlim(0, 500)
        ax3.set_ylim(0, 500)
        ax3.grid(alpha=0.3)
        
        # 添加参考线
        ax3.axhline(32, color='red', linestyle='--', alpha=0.5, label='Small (32px)')
        ax3.axhline(96, color='orange', linestyle='--', alpha=0.5, label='Medium (96px)')
        ax3.axvline(32, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(96, color='orange', linestyle='--', alpha=0.5)
        ax3.legend()
    
    # 4. 目标尺寸类别分布
    ax4 = plt.subplot(2, 3, 4)
    size_categories = ['Small\n(<32²)', 'Medium\n(32²-96²)', 'Large\n(≥96²)']
    size_counts = [
        stats['small_objects'],
        stats['medium_objects'],
        stats['large_objects']
    ]
    colors = ['red', 'orange', 'green']
    bars = ax4.bar(size_categories, size_counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Object Size Category Distribution')
    ax4.grid(axis='y', alpha=0.3)
    
    # 添加百分比
    total = sum(size_counts)
    for bar, count in zip(bars, size_counts):
        height = bar.get_height()
        percentage = count / total * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center', va='bottom')
    
    # 5. 图像尺寸分布
    ax5 = plt.subplot(2, 3, 5)
    if stats['image_sizes']:
        img_widths = [size[0] for size in stats['image_sizes']]
        img_heights = [size[1] for size in stats['image_sizes']]
        ax5.scatter(img_widths, img_heights, alpha=0.3, s=10)
        ax5.set_xlabel('Image Width')
        ax5.set_ylabel('Image Height')
        ax5.set_title('Image Size Distribution')
        ax5.grid(alpha=0.3)
    
    # 6. 数据集摘要
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    📊 Dataset Summary
    {'='*40}
    
    Total Images:     {stats['num_images']:,}
    Total Objects:    {stats['num_objects']:,}
    Avg Objects/Img:  {np.mean(stats['objects_per_image']):.2f}
    
    Object Size Distribution:
      • Small (<32²):   {stats['small_objects']:,} ({stats['small_objects']/stats['num_objects']*100:.1f}%)
      • Medium (32²-96²): {stats['medium_objects']:,} ({stats['medium_objects']/stats['num_objects']*100:.1f}%)
      • Large (≥96²):   {stats['large_objects']:,} ({stats['large_objects']/stats['num_objects']*100:.1f}%)
    
    Top 3 Classes:
    """
    
    # 找出前3个最多的类别
    sorted_classes = sorted(stats['class_counts'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
    for class_id, count in sorted_classes:
        summary_text += f"      {CLASS_NAMES[class_id]}: {count:,}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图表
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{split}_statistics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 统计图表已保存: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VisDrone数据集分析工具')
    parser.add_argument('--data-root', type=str, 
                        default='datasets/visdrone',
                        help='数据集根目录')
    parser.add_argument('--splits', nargs='+', 
                        default=['train', 'val'],
                        choices=['train', 'val', 'test'],
                        help='要分析的数据集划分')
    parser.add_argument('--save-dir', type=str,
                        default='runs/analysis',
                        help='统计图表保存目录')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).parent.parent
    data_root = (script_dir / args.data_root).resolve()
    save_dir = (script_dir / args.save_dir).resolve()
    
    print("\n" + "="*60)
    print("📊 VisDrone数据集分析工具")
    print("="*60)
    
    # 分析每个数据集划分
    for split in args.splits:
        stats = analyze_dataset(data_root, split)
        if stats:
            plot_statistics(stats, split, save_dir)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
