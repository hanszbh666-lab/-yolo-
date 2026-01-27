"""
重组VisDrone数据集结构
将下载的VisDrone2019-DET-{split}文件夹重组为统一的images和annotations目录
"""
import shutil
from pathlib import Path


def reorganize_dataset(root_dir, split):
    """
    重组数据集结构
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分 (train/val/test)
    """
    root_dir = Path(root_dir)
    
    # 确定源文件夹名称
    if split == 'test':
        source_folder = root_dir / 'VisDrone2019-DET-test-dev'
    else:
        source_folder = root_dir / f'VisDrone2019-DET-{split}'
    
    if not source_folder.exists():
        print(f"⚠️  源文件夹不存在: {source_folder}")
        return False
    
    print(f"📂 正在重组 {split} 数据集结构...")
    
    try:
        # 创建目标目录
        target_images_dir = root_dir / 'images' / split
        target_annotations_dir = root_dir / 'annotations' / split
        
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动图像文件
        source_images = source_folder / 'images'
        if source_images.exists():
            moved_count = 0
            for img_file in source_images.iterdir():
                if img_file.is_file():
                    target_path = target_images_dir / img_file.name
                    if not target_path.exists():
                        shutil.move(str(img_file), str(target_path))
                        moved_count += 1
            print(f"  ✅ 移动了 {moved_count} 个图像文件到: images/{split}/")
        
        # 移动标注文件
        source_annotations = source_folder / 'annotations'
        if source_annotations.exists():
            moved_count = 0
            for ann_file in source_annotations.iterdir():
                if ann_file.is_file():
                    target_path = target_annotations_dir / ann_file.name
                    if not target_path.exists():
                        shutil.move(str(ann_file), str(target_path))
                        moved_count += 1
            print(f"  ✅ 移动了 {moved_count} 个标注文件到: annotations/{split}/")
        
        # 删除空的源文件夹
        if source_images.exists() and not any(source_images.iterdir()):
            source_images.rmdir()
        if source_annotations.exists() and not any(source_annotations.iterdir()):
            source_annotations.rmdir()
        
        # 删除原始父文件夹
        if source_folder.exists() and not any(source_folder.iterdir()):
            source_folder.rmdir()
            print(f"  ✅ 已删除空文件夹: {source_folder.name}")
        
        return True
    except Exception as e:
        print(f"❌ 重组失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 数据集根目录
    root_dir = Path(__file__).parent.parent / 'datasets' / 'visdrone'
    
    if not root_dir.exists():
        print(f"❌ 数据集目录不存在: {root_dir}")
        return
    
    print("=" * 60)
    print("🔄 VisDrone数据集结构重组工具")
    print("=" * 60)
    print(f"📁 数据集目录: {root_dir}")
    print("=" * 60)
    
    # 检查需要重组的数据集
    splits_to_reorganize = []
    
    for split, folder_name in [
        ('train', 'VisDrone2019-DET-train'),
        ('val', 'VisDrone2019-DET-val'),
        ('test', 'VisDrone2019-DET-test-dev')
    ]:
        if (root_dir / folder_name).exists():
            splits_to_reorganize.append(split)
            print(f"✓ 发现需要重组的 {split} 数据集")
    
    if not splits_to_reorganize:
        print("\n✅ 没有需要重组的数据集")
        print("当前目录结构已经是正确的格式")
        return
    
    print(f"\n将重组以下数据集: {', '.join(splits_to_reorganize)}")
    confirm = input("\n是否继续？(y/n): ").lower()
    
    if confirm != 'y':
        print("❌ 取消操作")
        return
    
    # 重组数据集
    success_count = 0
    for split in splits_to_reorganize:
        print(f"\n{'='*60}")
        if reorganize_dataset(root_dir, split):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 重组完成！成功重组 {success_count}/{len(splits_to_reorganize)} 个数据集")
    print("=" * 60)
    
    # 显示最终结构
    print("\n📊 当前数据集结构:")
    print(f"  {root_dir}/")
    
    images_dir = root_dir / 'images'
    if images_dir.exists():
        print(f"  ├── images/")
        for split_dir in sorted(images_dir.iterdir()):
            if split_dir.is_dir():
                img_count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
                print(f"  │   ├── {split_dir.name}/ ({img_count} 图像)")
    
    annotations_dir = root_dir / 'annotations'
    if annotations_dir.exists():
        print(f"  └── annotations/")
        for split_dir in sorted(annotations_dir.iterdir()):
            if split_dir.is_dir():
                ann_count = len(list(split_dir.glob('*.txt')))
                print(f"      ├── {split_dir.name}/ ({ann_count} 标注)")
    
    print("\n" + "=" * 60)
    print("📋 下一步操作:")
    print("   运行数据格式转换脚本:")
    print("   python scripts/convert_visdrone.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
