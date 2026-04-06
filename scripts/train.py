"""
YOLOv11改进实验训练脚本 - VisDrone小目标检测
针对RTX4060 8GB显存优化的训练配置
"""
import os
import socket
import subprocess
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

from scripts.size_metrics import (
    DEFAULT_MEDIUM_AREA,
    DEFAULT_SMALL_AREA,
    SIZE_BUCKETS,
    SizeAwareDetectionValidator,
)


def _parse_device_ids(device):
    """将 device 参数解析为 GPU ID 列表。"""
    if device is None:
        return []
    if isinstance(device, int):
        return [device] if device >= 0 else []
    if isinstance(device, (list, tuple)):
        parsed = []
        for item in device:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return parsed

    device_str = str(device).strip()
    if not device_str or device_str.lower() in {'cpu', 'mps'}:
        return []

    parsed = []
    for item in device_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue
    return parsed


def _is_multi_gpu_request(device):
    """判断当前是否请求多卡训练。"""
    return len(_parse_device_ids(device)) > 1


def _find_free_port():
    """为 torch.distributed.run 找一个可用端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _launch_manual_ddp(args):
    """在多卡场景下直接用 torch.distributed.run 启动当前脚本。"""
    nproc_per_node = len(_parse_device_ids(args.device))
    script_path = Path(__file__).resolve()

    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nproc_per_node',
        str(nproc_per_node),
        '--master_port',
        str(_find_free_port()),
        str(script_path),
        '--data',
        args.data,
        '--model',
        args.model,
        '--epochs',
        str(args.epochs),
        '--batch',
        str(args.batch),
        '--imgsz',
        str(args.imgsz),
        '--device',
        str(args.device),
        '--workers',
        str(args.workers),
        '--project',
        args.project,
        '--patience',
        str(args.patience),
        '--small-area',
        str(args.small_area),
        '--medium-area',
        str(args.medium_area),
    ]

    if args.name is not None:
        cmd.extend(['--name', args.name])
    if args.cache:
        cmd.extend(['--cache', args.cache])
    if args.resume:
        cmd.append('--resume')
    if args.pretrained:
        cmd.append('--pretrained')

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{_PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get('PYTHONPATH') else _PROJECT_ROOT

    print("[DDP] 检测到多卡请求，改用当前脚本直接启动 torch.distributed.run")
    print(f"[DDP] 使用 GPU: {args.device} | 进程数: {nproc_per_node}")
    subprocess.run(cmd, check=True, env=env)


def _patch_parse_model_for_sda(_tasks, EMA, RFB, SDA_Fusion):
    """
    在运行时动态 patch _tasks.parse_model，使其支持 EMA / RFB / SDA_Fusion，
    无需手动修改 ultralytics 安装包文件。

    兼容两种场景：
      - 标准 Ultralytics 8.4.0（云端 pip install，未做任何改动）
      - 本地已手动修改过的 tasks.py（跳过，幂等安全）

    原理：
      1. 通过 inspect.getsource 获取当前 parse_model 的源代码。
      2. 若源码中尚不含本项目的扩展逻辑，则做两处字符串插入：
           a) 将 EMA、RFB 加入 base_modules frozenset → 使 parse_model 为它们
              自动注入 c1 并对 c2 做 width 缩放（与 Conv、C3k2 等标准模块相同）。
           b) 在 elif 链中插入 SDA_Fusion 分支 → 处理三路输入的 c1_list 构建
              与 c2 缩放（直接用 `else: c2 = ch[f]` 时 f 为列表会引发 TypeError）。
      3. 将修改后的源码在 _tasks 模块自身的命名空间中 compile + exec，
         自动替换 _tasks.parse_model（新函数的 __globals__ 即 tasks 模块 dict，
         因此对 Conv、make_divisible 等的引用均可正常解析）。
    """
    import inspect
    import re
    import textwrap

    # 优先从磁盘文件读取 parse_model 函数（最可靠：云端 tasks.py 始终在磁盘上，
    # 且 inspect.getsource 不能获取 exec 动态编译函数的源码）
    tasks_file = getattr(_tasks, '__file__', None)
    src = None
    if tasks_file:
        try:
            raw = Path(tasks_file).read_text(encoding='utf-8')
            # 只提取 parse_model 函数体（从函数定义行到下一个顶层 def/class 之前）
            pm_start = re.search(r'^def parse_model\(', raw, re.MULTILINE)
            if pm_start:
                # 找到 parse_model 之后第一个顶层 def 或 class
                rest = raw[pm_start.start():]
                next_top = re.search(r'\n(?:def |class )', rest[1:])
                end = (next_top.start() + 1) if next_top else len(rest)
                src = rest[:end].rstrip('\n')
        except Exception:
            src = None
    if src is None:
        try:
            src = textwrap.dedent(inspect.getsource(_tasks.parse_model))
        except OSError:
            return  # 无法获取源码（不影响已修改版 tasks.py 的正常运行）

    already_has_base = ('*([EMA, RFB]' in src) or bool(
        re.search(r'\bEMA\b.*\bRFB\b', src.split('repeat_modules')[0])
    )
    already_has_elif = 'elif SDA_Fusion' in src

    if already_has_base and already_has_elif:
        return  # tasks.py 已是修改版，无需重复 patch

    needs_exec = False

    # ── 1. 将 EMA / RFB 加入 base_modules ────────────────────────────────────
    if not already_has_base:
        # 标准 8.4.0 base_modules 以 "A2C2f," 结尾，后接闭合 "}" 和 ")"
        # 用正则匹配，容忍不同数量空白
        new_src, n = re.subn(
            r'(A2C2f,)(\s*\n\s*}\s*\n\s*\))',
            r'\1\n            EMA,\n            RFB,\2',
            src,
            count=1,
        )
        if n:
            src = new_src
            needs_exec = True

    # ── 2. 在 elif 链中插入 SDA_Fusion 分支 ──────────────────────────────────
    if not already_has_elif:
        SDA_ELIF = (
            "        elif SDA_Fusion is not None and m is SDA_Fusion:\n"
            "            c1_list = [ch[x] for x in f]\n"
            "            c2 = make_divisible(min(args[1], max_channels) * width, 8)\n"
            "            args = [c1_list, c2]\n"
        )
        # 插入在 TorchVision/Index 的 elif 之前（标准 8.4.0 中该行唯一存在）
        TARGET = "        elif m in frozenset({TorchVision, Index}):"
        if TARGET in src:
            src = src.replace(TARGET, SDA_ELIF + TARGET, 1)
            needs_exec = True

    if not needs_exec:
        return  # 无可用锚点（版本不匹配），不做 patch，避免引入错误

    # ── 3. 在 tasks 模块命名空间中编译执行，直接替换 parse_model ─────────────
    # vars(_tasks) 即 _tasks.__dict__，exec 后 parse_model 键值自动更新
    exec(compile(src, _tasks.__file__ or '<sda_parse_model_patch>', 'exec'), vars(_tasks))


def register_custom_modules(verbose=True):
    """
    注册 SDA-STD YOLO11 自定义模块。

    执行两步操作，使任意标准 Ultralytics 8.4.0 环境（包括未修改过 tasks.py
    的云端服务器）都能正确加载本项目的 YAML 模型：

    Step 1 — 名称注入：
        将 EMA / RFB / SDA_Fusion 类写入 ultralytics.nn.tasks 的模块命名空间，
        使 parse_model 内的 `globals()[m_str]` 查找能够找到它们。

    Step 2 — parse_model patch（仅在必要时执行）：
        若当前 tasks.py 尚未包含本项目的扩展逻辑（即从 pip 直接安装的标准版），
        则动态修改内存中的 parse_model：
          · 把 EMA / RFB 纳入 base_modules → 自动获得 c1 注入与 width 缩放
          · 添加 SDA_Fusion elif 分支      → 正确处理三路输入的通道追踪
        修改仅在进程内存中生效，不写入任何磁盘文件。
    """
    try:
        import ultralytics.nn.tasks as _tasks
        from models.modules.ema import EMA
        from models.modules.rfb import RFB
        from models.modules.sda_fusion import SDA_Fusion

        # Step 1: 注入模块类到 tasks 全局命名空间
        _tasks.EMA = EMA
        _tasks.RFB = RFB
        _tasks.SDA_Fusion = SDA_Fusion

        # Step 2: 动态 patch parse_model（幂等，已修改版不会重复执行）
        _patch_parse_model_for_sda(_tasks, EMA, RFB, SDA_Fusion)

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
    small_area=DEFAULT_SMALL_AREA,
    medium_area=DEFAULT_MEDIUM_AREA,
    **kwargs
):
    """
    训练YOLO模型（支持YOLOv11改进实验）
    
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
    print("🚀 YOLOv11改进实验训练配置")
    print("="*80)
    print(f"📊 数据集配置: {data_config}")
    print(f"🎯 模型: {model}")
    print(f"📈 训练轮数: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"👷 工作线程: {workers}")
    print(f"💾 保存路径: {project}/{name}")
    print(f"📏 尺寸阈值: small < {small_area:.0f}, medium < {medium_area:.0f}, large >= {medium_area:.0f}")
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

        best_model_path = Path(project) / name / 'weights' / 'best.pt'
        if best_model_path.exists():
            print("📏 开始对 best.pt 进行尺寸分桶验证...\n")
            size_val_model = YOLO(str(best_model_path))
            size_metrics = size_val_model.val(
                validator=SizeAwareDetectionValidator,
                data=data_config,
                split='val',
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                conf=0.001,
                iou=0.6,
                max_det=300,
                small_area=small_area,
                medium_area=medium_area,
                plots=False,
                verbose=False,
            )
            if hasattr(size_metrics, 'custom_size_metrics'):
                print("📏 Best.pt 尺寸分桶指标（自定义，面积阈值参考 COCO）:")
                print("-" * 80)
                for bucket_name in SIZE_BUCKETS:
                    bucket = size_metrics.custom_size_metrics[bucket_name]
                    print(f"  AP_{bucket_name:6s}: {bucket['map']:.4f}")
                    print(f"  AP50_{bucket_name:4s}: {bucket['map50']:.4f}")
                    print(f"  Recall_{bucket_name}: {bucket['recall']:.4f}")
                    print(f"  Precision_{bucket_name}: {bucket['precision']:.4f}")
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
    
    parser = argparse.ArgumentParser(description='YOLOv11改进实验训练脚本 - VisDrone小目标检测')
    
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
    parser.add_argument('--small-area', type=float, default=DEFAULT_SMALL_AREA,
                        help='small 目标面积上限，默认 32^2（COCO）')
    parser.add_argument('--medium-area', type=float, default=DEFAULT_MEDIUM_AREA,
                        help='medium 目标面积上限，默认 96^2（COCO）')
    
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

    # 对包含自定义模块/验证器的脚本，绕过 Ultralytics 自动生成的 DDP 临时脚本。
    # 直接用 torch.distributed.run 重启当前脚本，确保每个子进程都会执行完整初始化。
    if _is_multi_gpu_request(args.device) and 'LOCAL_RANK' not in os.environ:
        _launch_manual_ddp(args)
        return
    
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
        small_area=args.small_area,
        medium_area=args.medium_area,
        **extra_args
    )


if __name__ == '__main__':
    main()