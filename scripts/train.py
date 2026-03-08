"""
YOLOv8s训练脚本 - VisDrone小目标检测
针对RTX4060 8GB显存优化的训练配置
"""
import os
import sys
from pathlib import Path

# ── 将项目根目录加入 sys.path（必须在 import ultralytics 之前）──────────────
# 使得 tasks.py 中的 `from models.modules.xxx import XXX` 能够找到自定义模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 禁用Ultralytics的自动下载功能
os.environ['YOLO_DOWNLOAD'] = 'False'
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

import torch
from ultralytics import YOLO
import yaml


def register_custom_modules(verbose=True):
    """注册 SDA-STD YOLO11 自定义模块。"""
    try:
        import ultralytics.nn.tasks as _tasks
        from models.modules.ema import EMA
        from models.modules.rfb import RFB
        from models.modules.sda_fusion import SDA_Fusion

        _tasks.EMA = EMA
        _tasks.RFB = RFB
        _tasks.SDA_Fusion = SDA_Fusion

        if verbose:
            print("[SDA-STD] 自定义模块注册成功: EMA, RFB, SDA_Fusion")
        return True
    except Exception as e:
        if verbose:
            print(f"[SDA-STD] 警告：自定义模块注册失败，仅在使用标准模型时可忽略。原因: {e}")
        return False


def train_yolov8(
    data_config='configs/visdrone.yaml',
    model='models/yolov8s.yaml',
    epochs=300,
    batch_size=16,
    imgsz=640,
    device='0',
    workers=8,
    project='runs/train',
    name=None,
    resume=False,
    use_pretrained=False,
    **kwargs
):
    """
    训练YOLOv8模型
    
    Args:
        data_config: 数据集配置文件路径
        model: 模型配置或预训练权重路径
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 输入图像尺寸
        device: GPU设备ID ('0' 或 'cpu')
        workers: 数据加载线程数
        project: 项目保存目录
        name: 实验名称
        resume: 是否从上次中断的地方继续训练
        use_pretrained: 是否使用预训练权重（默认False，从头训练）
        **kwargs: 其他训练参数
    """
    # 自动生成实验名称（如果没有提供）
    if name is None:
        dataset_name = Path(data_config).stem
        if model.endswith('.pt'):
            model_name = Path(model).stem
        else:
            model_name = Path(model).stem
        name = f"{dataset_name}_{model_name}"
    
    # 打印训练配置
    print("\n" + "="*80)
    print("🚀 YOLOv8 VisDrone训练配置")
    print("="*80)
    print(f"📊 数据集配置: {data_config}")
    print(f"🎯 模型: {model}")
    print(f"📈 训练轮数: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"👷 工作线程: {workers}")
    print(f"💾 保存路径: {project}/{name}")
    print("="*80 + "\n")
    
    # 检查CUDA是否可用
    if device != 'cpu':
        if not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，切换到CPU训练")
            device = 'cpu'
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ 检测到GPU: {gpu_name}")
            print(f"💾 显存大小: {gpu_memory:.2f} GB\n")
    
    # 加载模型
    print(f"🔄 加载模型: {model}")

    # 仅在主训练流程中注册自定义模块，避免 Windows 多进程重复导入时反复输出
    register_custom_modules(verbose=True)
    
    # 验证模型文件是否存在
    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件不存在: {model}\n请检查路径是否正确")
    
    # 判断是使用预训练权重还是自定义配置
    if model.endswith('.pt'):
        # 使用预训练权重
        yolo_model = YOLO(model)
        print(f"✅ 已加载预训练权重: {model}")
        print(f"📦 模型类型: 预训练模型\n")
        use_pretrained = True  # .pt文件强制使用预训练权重
    elif model.endswith('.yaml'):
        # 使用自定义配置（完全离线模式）
        # 设置离线模式，防止任何自动下载
        import ultralytics
        ultralytics.checks.check_pip_update_available = lambda: False  # 禁用pip更新检查
        yolo_model = YOLO(model, task='detect')
        print(f"✅ 已加载模型配置: {model}")
        if use_pretrained:
            print(f"📦 模型类型: 使用预训练权重初始化")
            print(f"💡 提示: 将自动下载对应的预训练权重进行初始化")
        else:
            print(f"📦 模型类型: 从头训练（随机初始化权重）")
            print(f"⚠️  注意: 不使用预训练权重，训练时间会更长")
            print(f"💡 提示: 如需使用预训练权重加速训练，请添加 --pretrained 参数")
        print()
    else:
        raise ValueError(f"不支持的模型格式: {model}（仅支持 .pt 或 .yaml）")
    
    # 开始训练
    print("🎯 开始训练...\n")
    print("="*80)
    
    try:
        results = yolo_model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=project,
            name=name,
            resume=resume,
            
            # 优化器配置
            optimizer='SGD',      # 优化器: SGD/Adam/AdamW
            lr0=0.01,            # 初始学习率
            lrf=0.01,            # 最终学习率 (lr0 * lrf)
            momentum=0.937,      # SGD动量
            weight_decay=0.0005, # 权重衰减
            
            # 数据增强
            hsv_h=0.015,         # HSV色调增强
            hsv_s=0.7,           # HSV饱和度增强
            hsv_v=0.4,           # HSV明度增强
            degrees=0.0,         # 旋转角度
            translate=0.1,       # 平移
            scale=0.5,           # 缩放
            shear=0.0,           # 剪切
            perspective=0.0,     # 透视变换
            flipud=0.0,          # 上下翻转概率
            fliplr=0.5,          # 左右翻转概率
            mosaic=1.0,          # Mosaic增强概率
            mixup=0.0,           # Mixup增强概率
            copy_paste=0.0,      # Copy-Paste增强概率
            
            # 训练配置
            save=True,           # 保存检查点
            save_period=-1,      # 每N轮保存一次 (-1为仅保存最后和最佳)
            cache=False,         # 数据集缓存到内存 (True/False/'ram'/'disk')
            exist_ok=True,       # 覆盖已存在的项目
            pretrained=use_pretrained,  # 根据用户选择决定是否使用预训练权重
            verbose=True,        # 详细输出
            
            # 验证配置
            val=True,            # 训练时验证
            plots=True,          # 保存训练图表
            
            # 混合精度训练 (RTX4060支持) - 禁用自动检查避免下载
            amp=False,           # 禁用自动混合精度训练以避免下载
            
            # 其他参数
            **kwargs
        )
        
        print("\n" + "="*80)
        print("✅ 训练完成！")
        print("="*80)
        print(f"📊 最佳模型: {project}/{name}/weights/best.pt")
        print(f"📊 最终模型: {project}/{name}/weights/last.pt")
        print(f"📈 训练结果: {project}/{name}/")
        print("="*80 + "\n")
        
        # 打印最佳结果
        if hasattr(results, 'results_dict'):
            print("📊 最佳性能指标:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"  - mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  - mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
            print("="*80 + "\n")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        print(f"💡 可以使用 --resume 参数继续训练:")
        print(f"   python scripts/train.py --resume {project}/{name}/weights/last.pt")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        raise


def main():
    """主函数 - 解析命令行参数并开始训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8训练脚本 - VisDrone小目标检测')
    
    # 基础参数
    parser.add_argument('--data', type=str, default='configs/visdrone.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='models/yolov8s.yaml',
                        help='模型配置文件(.yaml)或预训练权重(.pt)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小 (RTX4060建议: 16-24)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (640/1280)')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (0/cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='runs/train',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称 (如不指定则自动生成: {数据集名称}_{模型名称})')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续训练')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重（适用于.yaml配置文件）')
    
    # 高级参数
    parser.add_argument('--cache', type=str, default=None,
                        help='数据集缓存 (ram/disk/None)')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    
    args = parser.parse_args()
    
    # 自动生成实验名称
    if args.name is None:
        # 提取数据集名称
        dataset_name = Path(args.data).stem
        # 提取模型名称
        if args.model.endswith('.pt'):
            model_name = Path(args.model).stem
        else:
            model_name = Path(args.model).stem
        args.name = f"{dataset_name}_{model_name}"
        print(f"🏷️  自动生成实验名称: {args.name}")
    
    # 确保路径正确
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # 额外参数
    extra_args = {}
    if args.cache:
        extra_args['cache'] = args.cache
    extra_args['patience'] = args.patience
    
    # 开始训练
    train_yolov8(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        use_pretrained=args.pretrained,
        **extra_args
    )


if __name__ == '__main__':
    main()