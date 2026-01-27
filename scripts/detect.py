"""
YOLOv8目标检测推理脚本
对图像或视频进行目标检测
"""
import os
import sys
from pathlib import Path

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from ultralytics import YOLO
import cv2
import numpy as np


def detect_images(
    model_path='runs/train/yolov8s_visdrone/weights/best.pt',
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
    print("🎯 YOLOv8 目标检测")
    print("="*80)
    print(f"🎯 模型: {model_path}")
    print(f"📁 输入源: {source}")
    print(f"🖼️  图像尺寸: {imgsz}")
    print(f"🎚️  置信度阈值: {conf}")
    print(f"🎚️  IoU阈值: {iou}")
    print(f"🔧 设备: {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"💾 保存结果: {'是' if save else '否'}")
    print("="*80 + "\n")
    
    # 加载模型
    print("🔄 加载模型...")
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


def detect_single_image(model_path, image_path, conf=0.25, save_path=None):
    """
    检测单张图像并显示结果（简化版）
    
    Args:
        model_path: 模型路径
        image_path: 图像路径
        conf: 置信度阈值
        save_path: 保存路径（可选）
    """
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, save=save_path is not None)
    
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
    
    parser = argparse.ArgumentParser(description='YOLOv8目标检测推理脚本')
    
    # 基础参数
    parser.add_argument('--model', '--weights', type=str,
                        default='runs/train/yolov8s_visdrone/weights/best.pt',
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
    parser.add_argument('--hide-labels', action='store_true',
                        help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true',
                        help='隐藏置信度')
    
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
        **extra_args
    )


if __name__ == '__main__':
    main()
