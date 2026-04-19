"""
特征响应热力图可视化脚本

用途：
1. 在同一批图片上对比 a1(基线)、a2(EMA)、full(MRA-STD) 三个模型
2. 针对指定层提取特征图并生成热力图叠加图
3. 导出单模型热力图与三模型拼图，便于直观比较模块影响
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# 将项目根目录加入 sys.path，保证可导入自定义模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 解决 OpenMP 冲突（与现有脚本保持一致）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scripts.detect import register_custom_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三模型特征响应热力图对比脚本")
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/visdrone/images/test",
        help="输入图像路径或目录",
    )
    parser.add_argument(
        "--weights-a1",
        type=str,
        default="runs/train/ablation_a1/weights/best.pt",
        help="a1 基线模型权重路径",
    )
    parser.add_argument(
        "--weights-a2",
        type=str,
        default="runs/train/ablation_a2/weights/best.pt",
        help="a2 (EMA) 模型权重路径",
    )
    parser.add_argument(
        "--weights-full",
        type=str,
        default="runs/train/sda_std_vs/weights/best.pt",
        help="MRA-STD 完整模型权重路径",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理输入尺寸",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，如 0 或 cpu",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（仅保持与推理流程一致）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU 阈值（仅保持与推理流程一致）",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-2,
        help="目标层索引（支持负索引，默认 -2）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="最多处理图像数量，<=0 表示处理全部图像",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="热力图叠加透明度 (0~1)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/analysis",
        help="输出根目录",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="feature_heatmap_compare",
        help="输出实验名",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="打印模型层索引与类型后退出",
    )
    return parser.parse_args()


def load_image_unicode(image_path: Path) -> np.ndarray:
    """兼容中文路径读取图像。"""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def save_image_unicode(save_path: Path, image: np.ndarray) -> None:
    """兼容中文路径保存图像。"""
    suffix = save_path.suffix.lower() if save_path.suffix else ".jpg"
    ext = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"图像编码失败: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(save_path))


def collect_images(source: str, max_images: int) -> List[Path]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"输入源不存在: {source}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        return [source_path]

    images = [p for p in source_path.rglob("*") if p.suffix.lower() in exts]
    images = sorted(images)
    if max_images > 0:
        images = images[:max_images]
    return images


def list_model_layers(model: YOLO, tag: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{tag}] 可视化层索引列表")
    print("=" * 90)
    for idx, module in enumerate(model.model.model):
        print(f"{idx:>3d}: {module.__class__.__name__}")


def resolve_layer_module(model: YOLO, layer_idx: int) -> torch.nn.Module:
    modules = model.model.model
    total = len(modules)
    real_idx = layer_idx if layer_idx >= 0 else total + layer_idx
    if real_idx < 0 or real_idx >= total:
        raise IndexError(f"层索引越界: {layer_idx}，可选范围: [{-total}, {total - 1}]")
    return modules[real_idx]


def normalize_map(response_map: np.ndarray) -> np.ndarray:
    response_map = response_map.astype(np.float32)
    min_val = float(response_map.min())
    max_val = float(response_map.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(response_map, dtype=np.float32)
    return (response_map - min_val) / (max_val - min_val)


def feature_to_heatmap(feature: torch.Tensor, size_hw: Tuple[int, int]) -> np.ndarray:
    """将特征图聚合为热力图 (H, W, 3)。"""
    if feature.dim() == 4:
        feature = feature[0]
    if feature.dim() != 3:
        raise ValueError(f"仅支持 3D/4D 特征图，当前维度: {tuple(feature.shape)}")

    # 通道维聚合，突出高响应区域
    response = torch.mean(torch.abs(feature), dim=0).detach().cpu().numpy()
    response = normalize_map(response)

    h, w = size_hw
    response_resized = cv2.resize(response, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(np.clip(response_resized * 255.0, 0, 255))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return heat_color


def overlay_heatmap(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


class FeatureHook:
    def __init__(self, model: YOLO, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.feature = None
        module = resolve_layer_module(model, layer_idx)
        self.hook = module.register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        if isinstance(output, (list, tuple)):
            picked = None
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() >= 3:
                    picked = item
                    break
            self.feature = picked
        elif isinstance(output, torch.Tensor):
            self.feature = output
        else:
            self.feature = None

    def clear(self):
        self.feature = None

    def close(self):
        self.hook.remove()


def build_compare_panel(original: np.ndarray, overlays: Dict[str, np.ndarray]) -> np.ndarray:
    left = add_title(original, "Original")
    a1 = add_title(overlays["a1"], "a1 Baseline")
    a2 = add_title(overlays["a2"], "a2 +EMA")
    full = add_title(overlays["full"], "MRA-STD Full")
    return np.concatenate([left, a1, a2, full], axis=1)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    register_custom_modules(verbose=True)

    weight_paths = {
        "a1": Path(args.weights_a1),
        "a2": Path(args.weights_a2),
        "full": Path(args.weights_full),
    }
    for key, path in weight_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"[{key}] 找不到权重文件: {path}")

    image_paths = collect_images(args.source, args.max_images)
    if not image_paths:
        raise RuntimeError(f"未找到可处理图像: {args.source}")

    print("\n" + "=" * 90)
    print("特征响应热力图对比")
    print("=" * 90)
    print(f"图像数量: {len(image_paths)}")
    print(f"可视化层索引: {args.layer}")
    print(f"输入源: {args.source}")

    models = {
        "a1": YOLO(str(weight_paths["a1"])),
        "a2": YOLO(str(weight_paths["a2"])),
        "full": YOLO(str(weight_paths["full"])),
    }

    if args.list_layers:
        list_model_layers(models["full"], "full")
        return

    hooks = {name: FeatureHook(model, args.layer) for name, model in models.items()}

    out_root = Path(args.project) / args.name
    out_compare = out_root / "compare"
    out_single = out_root / "single"
    out_compare.mkdir(parents=True, exist_ok=True)
    out_single.mkdir(parents=True, exist_ok=True)

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx:>3d}/{len(image_paths)}] 处理: {image_path.name}")
            image = load_image_unicode(image_path)
            h, w = image.shape[:2]

            overlays: Dict[str, np.ndarray] = {}
            for name, model in models.items():
                hook = hooks[name]
                hook.clear()

                model.predict(
                    source=str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    save=False,
                    verbose=False,
                    stream=False,
                )

                if hook.feature is None:
                    print(f"  - [{name}] 警告：该层未捕获到有效特征，使用黑图占位")
                    heat = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    heat = feature_to_heatmap(hook.feature, (h, w))

                overlay = overlay_heatmap(image, heat, args.alpha)
                overlays[name] = overlay

                single_path = out_single / name / image_path.name
                save_image_unicode(single_path, overlay)

            panel = build_compare_panel(image, overlays)
            compare_path = out_compare / image_path.name
            save_image_unicode(compare_path, panel)

        print("\n✅ 完成！")
        print(f"输出目录: {out_root}")
        print(f"对比图目录: {out_compare}")
        print(f"单模型目录: {out_single}")

    finally:
        for hook in hooks.values():
            hook.close()


if __name__ == "__main__":
    main()
