#!/usr/bin/env python3
"""
UAVDT数据集下载脚本
自动下载UAVDT数据集的图像文件

注意：由于UAVDT数据集较大，下载可能需要较长时间
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import argparse
from tqdm import tqdm


def download_file(url, output_path, chunk_size=8192):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        output_path: 保存路径
        chunk_size: 分块大小
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"下载 {output_path.name}") as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    解压zip文件
    
    Args:
        zip_path: zip文件路径
        extract_to: 解压目标目录
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取压缩文件中的文件列表
            file_list = zip_ref.namelist()
            
            with tqdm(total=len(file_list), desc=f"解压 {zip_path.name}") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_to)
                    pbar.update(1)
        
        print(f"✅ 解压完成: {extract_to}")
        return True
        
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def download_uavdt_dataset(output_dir="datasets/uavdt_raw"):
    """
    下载UAVDT数据集
    
    Args:
        output_dir: 输出目录
    """
    print("🚀 开始下载UAVDT数据集")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # UAVDT数据集下载链接 (这些链接可能需要更新)
    # 注意：实际的下载链接可能需要从官网获取
    download_urls = {
        "sequences_part1": "https://example.com/uavdt_sequences_part1.zip",  # 示例链接
        "sequences_part2": "https://example.com/uavdt_sequences_part2.zip",  # 示例链接
        # 可能还有更多部分...
    }
    
    print("⚠️  注意：UAVDT数据集需要从官网手动下载")
    print("🔗 官网链接: https://sites.google.com/site/daviddo0323/")
    print("")
    print("📋 下载步骤:")
    print("1. 访问官网")
    print("2. 申请数据集访问权限")
    print("3. 下载以下文件:")
    print("   - UAVDT-Benchmark-M.zip (主要序列)")
    print("   - 其他相关文件")
    print("4. 将下载的文件解压到:", output_path.absolute())
    print("")
    print("📁 预期的目录结构:")
    print(f"{output_path.absolute()}/")
    print("├── M0101/")
    print("│   └── img1/")
    print("│       ├── 000001.jpg")
    print("│       ├── 000002.jpg")
    print("│       └── ...")
    print("├── M0201/")
    print("└── ...")
    print("")
    print("⚡ 下载完成后运行:")
    print(f"   python scripts/convert_uavdt.py {output_path.absolute()} --create-config")
    
    return output_path


def verify_download(data_dir):
    """
    验证下载的数据是否完整
    
    Args:
        data_dir: 数据目录
    """
    print(f"🔍 验证数据目录: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print("❌ 数据目录不存在")
        return False
    
    # 查找序列目录
    sequence_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('M')]
    
    if not sequence_dirs:
        print("❌ 未找到序列目录 (M0101, M0201, ...)")
        return False
    
    print(f"✅ 找到 {len(sequence_dirs)} 个序列目录")
    
    # 检查每个序列是否有图像
    valid_sequences = 0
    for seq_dir in sequence_dirs:
        img_dir = seq_dir / "img1"
        if img_dir.exists():
            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if img_files:
                valid_sequences += 1
                print(f"   ✅ {seq_dir.name}: {len(img_files)} 张图像")
            else:
                print(f"   ⚠️  {seq_dir.name}: 图像目录为空")
        else:
            print(f"   ❌ {seq_dir.name}: 缺少img1目录")
    
    print(f"📊 有效序列: {valid_sequences}/{len(sequence_dirs)}")
    
    return valid_sequences > 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAVDT数据集下载工具')
    parser.add_argument('--output-dir', type=str, default='datasets/uavdt_raw',
                        help='下载输出目录')
    parser.add_argument('--verify', type=str, 
                        help='验证已下载的数据目录')
    
    args = parser.parse_args()
    
    # 确保在项目根目录运行
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    if args.verify:
        verify_download(args.verify)
    else:
        download_uavdt_dataset(args.output_dir)


if __name__ == '__main__':
    main()