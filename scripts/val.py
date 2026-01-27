"""
YOLOv8模型验证脚本
在验证集上评估训练好的模型性能
"""
import os
import sys
from pathlib import Path

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from ultralytics import YOLO


def validate_model(
    model_path='runs/train/uavdt_yolov8s/weights/best.pt',
    data_config='configs/uavdt.yaml',
    imgsz=640,
    batch_size=16,
    device='0',
    split='val',
    save_json=False,
    save_hybrid=False,
    conf=0.001,
    iou=0.6,
    max_det=300,
    project='runs/val',
    name='uavdt_val',
    **kwargs
):
    """
    验证YOLO模型
    
    Args:
        model_path: 模型权重路径
        data_config: 数据集配置文件
        imgsz: 图像尺寸
        batch_size: 批次大小
        device: 设备 ('0' 或 'cpu')
        split: 验证数据集划分 ('val' 或 'test')
        save_json: 是否保存COCO格式的JSON结果
        save_hybrid: 是否保存混合标签
        conf: 置信度阈值
        iou: NMS的IoU阈值
        max_det: 每张图像最大检测数
        project: 保存目录
        name: 实验名称
        **kwargs: 其他参数
    """
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("💡 请先训练模型或指定正确的模型路径")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("📊 YOLOv8 模型验证")
    print("="*80)
    print(f"🎯 模型路径: {model_path}")
    print(f"📊 数据集配置: {data_config}")
    print(f"📝 验证集: {split}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"🎚️  置信度阈值: {conf}")
    print(f"🎚️  IoU阈值: {iou}")
    print("="*80 + "\n")
    
    # 加载模型
    print(f"🔄 加载模型...")
    model = YOLO(model_path)
    print(f"✅ 模型加载完成\n")
    
    # 开始验证
    print("🎯 开始验证...\n")
    print("="*80)
    
    try:
        results = model.val(
            data=data_config,
            split=split,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            save_json=save_json,
            save_hybrid=save_hybrid,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=name,
            plots=True,  # 保存验证图表
            verbose=True,
            **kwargs
        )
        
        print("\n" + "="*80)
        print("✅ 验证完成！")
        print("="*80)
        
        # 打印性能指标
        print("\n📊 性能指标:")
        print("-" * 80)
        
        # mAP指标
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"  mAP@0.5     : {metrics.map50:.4f}")
            print(f"  mAP@0.5:0.95: {metrics.map:.4f}")
            print(f"  Precision   : {metrics.mp:.4f}")
            print(f"  Recall      : {metrics.mr:.4f}")
        
        # 各类别mAP
        if hasattr(results, 'box') and hasattr(metrics, 'maps'):
            print("\n📋 各类别 mAP@0.5:")
            print("-" * 80)
            class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                          'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
            for i, (name, map_val) in enumerate(zip(class_names, metrics.maps)):
                print(f"  {name:20s}: {map_val:.4f}")
        
        print("="*80)
        print(f"📁 结果保存至: {project}/{name}/")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 验证过程中发生错误: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8模型验证脚本')
    
    # 基础参数
    parser.add_argument('--model', '--weights', type=str, 
                        default='runs/train/yolov8s_visdrone/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='configs/visdrone.yaml',
                        help='数据集配置文件')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='验证数据集划分')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0/cpu)')
    
    # 检测参数
    parser.add_argument('--conf', type=float, default=0.001,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=300,
                        help='每张图像最大检测数')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='runs/val',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='visdrone_val',
                        help='实验名称')
    parser.add_argument('--save-json', action='store_true',
                        help='保存COCO格式JSON')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='保存混合标签')
    
    args = parser.parse_args()
    
    # 确保在项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # 开始验证
    validate_model(
        model_path=args.model,
        data_config=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        split=args.split,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
