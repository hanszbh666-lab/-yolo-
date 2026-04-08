"""
YOLO11模型验证脚本
在验证集上评估训练好的模型性能
"""
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保自定义 models 包可被权重反序列化时导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from ultralytics import YOLO

from scripts.size_metrics import (
    DEFAULT_MEDIUM_AREA,
    DEFAULT_SMALL_AREA,
    SIZE_BUCKETS,
    SizeAwareDetectionValidator,
)


def register_custom_modules(verbose=True):
    """注册自定义模块，确保自定义模型权重可被正确加载。"""
    try:
        import ultralytics.nn.tasks as tasks
        from models.modules.ema import EMA
        from models.modules.rfb import RFB
        from models.modules.sda_fusion import SDA_Fusion

        tasks.EMA = EMA
        tasks.RFB = RFB
        tasks.SDA_Fusion = SDA_Fusion

        if verbose:
            print("[SDA-STD] 自定义模块注册成功: EMA, RFB, SDA_Fusion")
        return True
    except Exception as exc:
        if verbose:
            print(f"[SDA-STD] 警告：自定义模块注册失败。若加载标准模型可忽略，当前原因: {exc}")
        return False


def validate_model(
    model_path='runs/train/uavdt_yolo11/weights/best.pt',
    data_config='configs/uavdt.yaml',
    imgsz=640,
    batch_size=16,
    device='0',
    split='val',
    save_json=False,
    conf=0.001,
    iou=0.6,
    max_det=300,
    small_area=DEFAULT_SMALL_AREA,
    medium_area=DEFAULT_MEDIUM_AREA,
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
    print("📊 YOLO11 模型验证")
    print("="*80)
    print(f"🎯 模型路径: {model_path}")
    print(f"📊 数据集配置: {data_config}")
    print(f"📝 验证集: {split}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"🎚️  置信度阈值: {conf}")
    print(f"🎚️  IoU阈值: {iou}")
    print(f"📏 尺寸阈值: small < {small_area:.0f}, medium < {medium_area:.0f}, large >= {medium_area:.0f}")
    print("="*80 + "\n")
    
    # 加载模型
    print(f"🔄 加载模型...")
    register_custom_modules(verbose=True)
    SizeAwareDetectionValidator.configure_thresholds(
        small_area=small_area,
        medium_area=medium_area,
    )
    model = YOLO(model_path)
    print(f"✅ 模型加载完成\n")
    
    # 开始验证
    print("🎯 开始验证...\n")
    print("="*80)
    
    try:
        results = model.val(
            validator=SizeAwareDetectionValidator,
            data=data_config,
            split=split,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            save_json=save_json,
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
            print(f"  mAP         : {metrics.map:.4f}")
            print(f"  mAP@0.5     : {metrics.map50:.4f}")
            print(f"  mAP@0.5:0.95: {metrics.map:.4f}")
            print(f"  Precision   : {metrics.mp:.4f}")
            print(f"  Recall      : {metrics.mr:.4f}")

        if hasattr(results, 'custom_size_metrics'):
            size_metrics = results.custom_size_metrics
            print("\n📏 尺寸分桶指标（自定义，面积阈值参考 COCO）:")
            print("-" * 80)
            for bucket_name in SIZE_BUCKETS:
                bucket = size_metrics[bucket_name]
                print(f"  AP_{bucket_name:6s}: {bucket['map']:.4f}")
                print(f"  AP50_{bucket_name:4s}: {bucket['map50']:.4f}")
                print(f"  Recall_{bucket_name}: {bucket['recall']:.4f}")
                print(f"  Precision_{bucket_name}: {bucket['precision']:.4f}")
                print(f"  Instances_{bucket_name}: {bucket['instances']}")
                print(f"  Predictions_{bucket_name}: {bucket['predictions']}")
        
        # 各类别mAP
        if hasattr(results, 'box') and hasattr(metrics, 'maps'):
            print("\n📋 各类别 mAP@0.5:")
            print("-" * 80)
            class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                          'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
            for class_name, map_val in zip(class_names, metrics.maps):
                print(f"  {class_name:20s}: {map_val:.4f}")
        
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
    
    parser = argparse.ArgumentParser(description='YOLO11模型验证脚本')
    
    # 基础参数
    parser.add_argument('--model', '--weights', type=str, 
                        default='runs/train/sda_std_vs/weights/best.pt',
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
    parser.add_argument('--small-area', type=float, default=DEFAULT_SMALL_AREA,
                        help='small 目标面积上限，默认 32^2（COCO）')
    parser.add_argument('--medium-area', type=float, default=DEFAULT_MEDIUM_AREA,
                        help='medium 目标面积上限，默认 96^2（COCO）')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='runs/val',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='visdrone_val',
                        help='实验名称')
    parser.add_argument('--save-json', action='store_true',
                        help='保存COCO格式JSON')
    
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
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        small_area=args.small_area,
        medium_area=args.medium_area,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
