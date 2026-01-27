"""
VisDrone数据集下载脚本
支持从官方源和备用源下载VisDrone-DET数据集
"""
import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


# 数据集URL配置
BASE_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
VISDRONE_URLS = {
    'train': BASE_URL + 'VisDrone2019-DET-train.zip',
    'val': BASE_URL + 'VisDrone2019-DET-val.zip',
    'test': BASE_URL + 'VisDrone2019-DET-test-dev.zip'
}

# 备用下载链接（如果GitHub下载失败）
BACKUP_URLS = {
    'train': 'http://aiskyeye.com/download/VisDrone2019-DET-train.zip',
    'val': 'http://aiskyeye.com/download/VisDrone2019-DET-val.zip',
    'test': 'http://aiskyeye.com/download/VisDrone2019-DET-test-dev.zip'
}


def download_file(url, save_path, desc="Downloading"):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        save_path: 保存路径
        desc: 进度条描述
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
    """
    print(f"📦 正在解压: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def reorganize_dataset(root_dir, split):
    """
    重组数据集结构
    将 VisDrone2019-DET-{split}/annotations 和 images 移动到统一的目录结构
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分 (train/val/test)
    """
    import shutil
    
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
            for img_file in source_images.iterdir():
                if img_file.is_file():
                    shutil.move(str(img_file), str(target_images_dir / img_file.name))
            print(f"  ✅ 图像文件已移动到: images/{split}/")
        
        # 移动标注文件
        source_annotations = source_folder / 'annotations'
        if source_annotations.exists():
            for ann_file in source_annotations.iterdir():
                if ann_file.is_file():
                    shutil.move(str(ann_file), str(target_annotations_dir / ann_file.name))
            print(f"  ✅ 标注文件已移动到: annotations/{split}/")
        
        # 删除原始文件夹
        shutil.rmtree(source_folder)
        print(f"  ✅ 已删除原始文件夹: {source_folder.name}")
        
        return True
    except Exception as e:
        print(f"❌ 重组失败: {e}")
        return False


def download_visdrone(root_dir=None, splits=['train', 'val', 'test']):
    """
    下载VisDrone数据集
    
    Args:
        root_dir: 数据集根目录
        splits: 要下载的数据集划分 ['train', 'val', 'test']
    """
    # 设置数据集根目录
    if root_dir is None:
        root_dir = Path(__file__).parent.parent / 'datasets' / 'visdrone'
    else:
        root_dir = Path(root_dir)
    
    root_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = root_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🚁 VisDrone数据集下载工具")
    print("=" * 60)
    print(f"📁 目标目录: {root_dir}")
    print(f"📊 下载划分: {', '.join(splits)}")
    print("=" * 60)
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"📥 正在下载 {split.upper()} 数据集...")
        print(f"{'='*60}")
        
        # 文件路径
        zip_filename = f"VisDrone2019-DET-{split}.zip" if split != 'test' else "VisDrone2019-DET-test-dev.zip"
        zip_path = temp_dir / zip_filename
        
        # 检查是否已存在
        if zip_path.exists():
            print(f"⚠️  文件已存在: {zip_filename}")
            user_input = input("是否重新下载？(y/n): ").lower()
            if user_input != 'y':
                print("⏭️  跳过下载，使用已存在的文件")
            else:
                zip_path.unlink()
        
        # 尝试从主URL下载
        if not zip_path.exists():
            print(f"🌐 从GitHub下载...")
            success = download_file(
                VISDRONE_URLS[split],
                zip_path,
                desc=f"Downloading {split}"
            )
            
            # 如果失败，尝试备用URL
            if not success:
                print(f"⚠️  GitHub下载失败，尝试备用源...")
                success = download_file(
                    BACKUP_URLS[split],
                    zip_path,
                    desc=f"Downloading {split} (backup)"
                )
            
            if not success:
                print(f"❌ {split}数据集下载失败！")
                print("💡 请手动下载并放置到以下目录:")
                print(f"   {temp_dir}")
                print("   下载地址: https://github.com/VisDrone/VisDrone-Dataset")
                continue
        
        # 解压文件
        extract_success = extract_zip(zip_path, root_dir)
        
        if extract_success:
            # 重组数据集结构
            reorganize_dataset(root_dir, split)
            
            # 删除ZIP文件以节省空间
            delete_input = input(f"是否删除ZIP文件以节省空间？(y/n): ").lower()
            if delete_input == 'y':
                zip_path.unlink()
                print(f"🗑️  已删除: {zip_filename}")
    
    # 清理临时目录
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    
    print("\n" + "=" * 60)
    print("✅ 下载完成！")
    print("=" * 60)
    print("\n📋 下一步操作:")
    print("1. 运行数据格式转换脚本:")
    print("   python scripts/convert_visdrone.py")
    print("\n2. 或者查看数据集统计:")
    print("   python scripts/data_analysis.py")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VisDrone数据集下载工具')
    parser.add_argument('--root', type=str, default=None, 
                        help='数据集根目录 (默认: ../datasets/visdrone)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='要下载的数据集划分 (默认: train val test)')
    
    args = parser.parse_args()
    
    try:
        download_visdrone(root_dir=args.root, splits=args.splits)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)
