"""
YOLO11目标检测推理脚本
对图像或视频进行目标检测
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
import cv2
import numpy as np

from scripts.size_metrics import (
    DEFAULT_MEDIUM_AREA,
    DEFAULT_SMALL_AREA,
    SIZE_BUCKETS,
    summarize_prediction_size_distribution,
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


def detect_images(
    model_path='runs/train/sda_std_vs/weights/best.pt',
    source='datasets/visdrone/images/test',
    imgsz=640,
    conf=0.25,
    iou=0.7,
    max_det=300,
    device='0',
    save=True,
    save_txt=False,
    save_conf=False,
    show=False,
    project='runs/detect',
    name='visdrone_detect',
    line_width=2,
    hide_labels=True,
    hide_conf=True,
    small_area=DEFAULT_SMALL_AREA,
    medium_area=DEFAULT_MEDIUM_AREA,
    **kwargs
):
    """
    使用YOLO模型进行目标检测
    
    Args:
        model_path: 模型权重路径
        source: 输入源 (图像路径/文件夹/视频文件/0表示摄像头)
        imgsz: 推理图像尺寸
        conf: 置信度阈值
        iou: NMS IoU阈值
        max_det: 每张图像最大检测数
        device: 设备 ('0' 或 'cpu')
        save: 是否保存结果
        save_txt: 是否保存标签文件
        save_conf: 是否在标签中保存置信度
        show: 是否显示结果
        project: 保存目录
        name: 实验名称
        line_width: 边界框线宽
        hide_labels: 是否隐藏类别标签，默认只显示检测框
        hide_conf: 是否隐藏置信度，默认只显示检测框
        **kwargs: 其他参数
    """
    # 检查模型文件
    if not Path(model_path).exists():
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        sys.exit(1)
    
    # 检查输入源
    source_path = Path(source)
    if not source_path.exists() and source != '0':
        print(f"❌ 错误: 找不到输入源 {source}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎯 YOLO11 目标检测")
    print("="*80)
    print(f"🎯 模型: {model_path}")
    print(f"📁 输入源: {source}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"🎚️  置信度阈值: {conf}")
    print(f"🎚️  IoU阈值: {iou}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"💾 保存结果: {'是' if save else '否'}")
    print(f"📏 尺寸阈值: small < {small_area:.0f}, medium < {medium_area:.0f}, large >= {medium_area:.0f}")
    print("="*80 + "\n")
    
    # 加载模型
    print("🔄 加载模型...")
    register_custom_modules(verbose=True)
    model = YOLO(model_path)
    print("✅ 模型加载完成\n")
    
    # 开始检测
    print("🎯 开始检测...\n")
    print("="*80)
    
    try:
        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            show=show,
            project=project,
            name=name,
            line_width=line_width,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            verbose=True,
            stream=False,  # 返回所有结果
            **kwargs
        )
        
        print("\n" + "="*80)
        print("✅ 检测完成！")
        print("="*80)
        
        # 统计检测结果
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
        
        print(f"\n📊 检测统计:")
        print(f"  - 处理图像数: {len(results)}")
        print(f"  - 检测目标数: {total_detections}")
        print(f"  - 平均每张图: {total_detections/len(results):.2f}")

        size_summary = summarize_prediction_size_distribution(results, small_area=small_area, medium_area=medium_area)
        print("\n⚡ 推理速度:")
        print(f"  - 预处理: {size_summary['speed_ms']['preprocess']:.2f} ms/图")
        print(f"  - 推理: {size_summary['speed_ms']['inference']:.2f} ms/图")
        print(f"  - 后处理: {size_summary['speed_ms']['postprocess']:.2f} ms/图")
        print(f"  - 总时延: {size_summary['latency_ms']:.2f} ms/图")
        print(f"  - FPS: {size_summary['fps']:.2f}")

        print("\n📏 预测框尺寸分布:")
        for bucket_name in SIZE_BUCKETS:
            bucket = size_summary['buckets'][bucket_name]
            print(f"  - {bucket_name:6s}: {bucket['count']:4d} ({bucket['ratio'] * 100:5.1f}%), 平均每图 {bucket['per_image']:.2f}, 平均置信度 {bucket['avg_confidence']:.4f}")
        
        if save:
            print(f"\n💾 结果保存至: {project}/{name}/")
        
        print("="*80 + "\n")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  检测被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检测过程中发生错误: {e}")
        raise


def detect_single_image(
    model_path,
    image_path,
    conf=0.25,
    save_path=None,
    hide_labels=True,
    hide_conf=True,
):
    """
    检测单张图像并显示结果（简化版）
    
    Args:
        model_path: 模型路径
        image_path: 图像路径
        conf: 置信度阈值
        save_path: 保存路径（可选）
    """
    register_custom_modules(verbose=True)
    model = YOLO(model_path)
    results = model.predict(
        image_path,
        conf=conf,
        save=save_path is not None,
        hide_labels=hide_labels,
        hide_conf=hide_conf,
    )
    
    # 显示结果
    for result in results:
        img = result.plot()  # 绘制检测框
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"✅ 结果已保存: {save_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11目标检测推理脚本')
    
    # 基础参数
    parser.add_argument('--model', '--weights', type=str,
                        default='runs/train/sda_std_vs/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str,
                        default='datasets/visdrone/images/test',
                        help='输入源 (图像/文件夹/视频/0=摄像头)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=300,
                        help='每张图像最大检测数')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0/cpu)')
    
    # 保存参数
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存检测结果')
    parser.add_argument('--nosave', action='store_true',
                        help='不保存检测结果')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存标签文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='在标签中保存置信度')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='visdrone_detect',
                        help='实验名称')
    
    # 可视化参数
    parser.add_argument('--line-width', type=int, default=2,
                        help='边界框线宽')
    parser.add_argument('--hide-labels', dest='hide_labels', action='store_true', default=True,
                        help='隐藏标签（默认开启，仅显示检测框）')
    parser.add_argument('--show-labels', dest='hide_labels', action='store_false',
                        help='显示类别标签')
    parser.add_argument('--hide-conf', dest='hide_conf', action='store_true', default=True,
                        help='隐藏置信度（默认开启，仅显示检测框）')
    parser.add_argument('--show-conf', dest='hide_conf', action='store_false',
                        help='显示置信度')
    parser.add_argument('--small-area', type=float, default=DEFAULT_SMALL_AREA,
                        help='small 目标面积上限，默认 32^2（COCO）')
    parser.add_argument('--medium-area', type=float, default=DEFAULT_MEDIUM_AREA,
                        help='medium 目标面积上限，默认 96^2（COCO）')
    
    args = parser.parse_args()
    
    # 确保在项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # 处理save参数
    save = args.save and not args.nosave
    
    # 额外参数
    extra_args = {
        'line_width': args.line_width,
        'hide_labels': args.hide_labels,
        'hide_conf': args.hide_conf,
    }
    
    # 开始检测
    detect_images(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=args.project,
        name=args.name,
        small_area=args.small_area,
        medium_area=args.medium_area,
        **extra_args
    )


if __name__ == '__main__':
    main()
