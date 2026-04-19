"""Generate a paper-style ASFF-3 visualization for one image.

This script captures aligned inputs and fusion weights from the target SDA_Fusion
module, then renders a figure similar to the ASFF illustration in the paper.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Keep project import behavior consistent with existing scripts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Workaround for OpenMP conflicts on some Windows Python environments.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scripts.detect import register_custom_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASFF-3 paper-style single-image visualizer")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/sda_std_vs/weights/best.pt",
        help="model weights path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="D:/yolo（大三大创）/datasets/visdrone/images/test/9999938_00000_d_0000017.jpg",
        help="single image path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", type=str, default="0", help="device, e.g. 0 or cpu")
    parser.add_argument(
        "--target-sda-index",
        type=int,
        default=1,
        help="which SDA_Fusion to use (0-based among SDA_Fusion modules)",
    )
    parser.add_argument(
        "--alpha-overlay",
        type=float,
        default=0.45,
        help="overlay alpha in [0, 1]",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=320,
        help="single tile size for final panel",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="runs/analysis/asff3_paper_case",
        help="output directory",
    )
    return parser.parse_args()


def load_image_unicode(image_path: Path) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    return image


def save_image_unicode(save_path: Path, image: np.ndarray) -> None:
    suffix = save_path.suffix.lower() if save_path.suffix else ".png"
    ext = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp"} else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"failed to encode image: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(save_path))


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def tensor_to_response_2d(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"expected 3D/4D tensor, got shape={tuple(t.shape)}")
    return torch.mean(torch.abs(t), dim=0).detach().cpu().numpy()


def map_to_color(map_2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    n = normalize_map(map_2d)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    u8 = np.uint8(np.clip(n * 255.0, 0, 255))
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def overlay_map(image_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)


def make_tile(image: np.ndarray, title: str, tile_size: int) -> np.ndarray:
    tile = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    bar_h = 34
    bar = np.full((bar_h, tile_size, 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    return np.concatenate([bar, tile], axis=0)


def build_formula_tile(tile_size: int, stats: Dict[str, float]) -> np.ndarray:
    bar_h = 34
    tile = np.full((tile_size, tile_size, 3), 252, dtype=np.uint8)
    lines = [
        "ASFF-3 weighted fusion",
        "F = a*X1->3 + b*X2->3 + g*X3->3",
        "",
        f"mean(a)={stats['alpha_mean']:.4f}",
        f"mean(b)={stats['beta_mean']:.4f}",
        f"mean(g)={stats['gamma_mean']:.4f}",
        "",
        "Per-pixel: a+b+g=1",
    ]
    y = 42
    for text in lines:
        cv2.putText(tile, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 1, cv2.LINE_AA)
        y += 34

    bar = np.full((bar_h, tile_size, 3), 245, dtype=np.uint8)
    cv2.putText(bar, "Equation", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    return np.concatenate([bar, tile], axis=0)


def shape_list(t: torch.Tensor) -> List[int]:
    return [int(v) for v in t.shape]


class ASFF3Capture:
    def __init__(self, model: YOLO, target_sda_index: int):
        self.model = model
        self.target_sda_index = target_sda_index
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, torch.Tensor] = {}
        self.module_index: int = -1

        modules = list(model.model.model)
        sda_pairs = [
            (idx, m)
            for idx, m in enumerate(modules)
            if m.__class__.__name__ == "SDA_Fusion" and hasattr(m, "asff")
        ]
        if not sda_pairs:
            raise RuntimeError("no SDA_Fusion modules found")
        if target_sda_index < 0 or target_sda_index >= len(sda_pairs):
            raise IndexError(
                f"target-sda-index out of range: {target_sda_index}, available [0, {len(sda_pairs)-1}]"
            )

        self.module_index, self.sda = sda_pairs[target_sda_index]
        self._register_hooks()

    def _register_hooks(self) -> None:
        def to_cpu_tensor(x: torch.Tensor) -> torch.Tensor:
            return x.detach().cpu().float()

        def sda_input_pre_hook(module, inputs):
            if not inputs:
                return
            first = inputs[0]
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                self.cache["x1_raw"] = to_cpu_tensor(first[0])
                self.cache["x2_raw"] = to_cpu_tensor(first[1])
                self.cache["x3_raw"] = to_cpu_tensor(first[2])

        def align1_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache["x1_to3"] = to_cpu_tensor(output)

        def align2_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache["x2_to3"] = to_cpu_tensor(output)

        def align3_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache["x3_after_conv"] = to_cpu_tensor(output)

        def asff_input_pre_hook(module, inputs):
            if not inputs:
                return
            first = inputs[0]
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                self.cache["x1_asff_in"] = to_cpu_tensor(first[0])
                self.cache["x2_asff_in"] = to_cpu_tensor(first[1])
                self.cache["x3_asff_in"] = to_cpu_tensor(first[2])

        def weight_logits_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache["weight_logits"] = to_cpu_tensor(output)

        def expand_input_pre_hook(module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self.cache["fused_reduced"] = to_cpu_tensor(inputs[0])

        def asff_out_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache["asff_out"] = to_cpu_tensor(output)

        self.handles.append(self.sda.register_forward_pre_hook(sda_input_pre_hook))
        self.handles.append(self.sda.align1.register_forward_hook(align1_hook))
        self.handles.append(self.sda.align2.register_forward_hook(align2_hook))
        self.handles.append(self.sda.align3.register_forward_hook(align3_hook))
        self.handles.append(self.sda.asff.register_forward_pre_hook(asff_input_pre_hook))
        self.handles.append(self.sda.asff.weight_levels.register_forward_hook(weight_logits_hook))
        self.handles.append(self.sda.asff.expand.register_forward_pre_hook(expand_input_pre_hook))
        self.handles.append(self.sda.asff.register_forward_hook(asff_out_hook))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def require_cache(cache: Dict[str, torch.Tensor], keys: List[str]) -> None:
    missing = [k for k in keys if k not in cache]
    if missing:
        raise RuntimeError(f"missing captured tensors: {missing}")


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    register_custom_modules(verbose=True)
    model = YOLO(str(weights_path))

    capture = ASFF3Capture(model, target_sda_index=args.target_sda_index)
    original = load_image_unicode(image_path)
    h, w = original.shape[:2]

    try:
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            device=args.device,
            conf=0.25,
            iou=0.7,
            save=False,
            verbose=False,
            stream=False,
        )
    finally:
        capture.close()

    cache = capture.cache
    require_cache(
        cache,
        ["x1_asff_in", "x2_asff_in", "x3_asff_in", "weight_logits", "fused_reduced", "asff_out"],
    )

    x1 = cache["x1_asff_in"]
    x2 = cache["x2_asff_in"]
    x3 = cache["x3_asff_in"]
    logits = cache["weight_logits"]
    weights = F.softmax(logits, dim=1)

    recon_fused = x1 * weights[:, 0:1] + x2 * weights[:, 1:2] + x3 * weights[:, 2:3]
    fused_reduced = cache["fused_reduced"]
    asff_out = cache["asff_out"]

    alpha_map = weights[0, 0].numpy()
    beta_map = weights[0, 1].numpy()
    gamma_map = weights[0, 2].numpy()

    x1_ov = overlay_map(original, map_to_color(tensor_to_response_2d(x1), (h, w)), args.alpha_overlay)
    x2_ov = overlay_map(original, map_to_color(tensor_to_response_2d(x2), (h, w)), args.alpha_overlay)
    x3_ov = overlay_map(original, map_to_color(tensor_to_response_2d(x3), (h, w)), args.alpha_overlay)

    alpha_ov = overlay_map(original, map_to_color(alpha_map, (h, w)), args.alpha_overlay)
    beta_ov = overlay_map(original, map_to_color(beta_map, (h, w)), args.alpha_overlay)
    gamma_ov = overlay_map(original, map_to_color(gamma_map, (h, w)), args.alpha_overlay)

    fused_ov = overlay_map(original, map_to_color(tensor_to_response_2d(fused_reduced), (h, w)), args.alpha_overlay)
    out_ov = overlay_map(original, map_to_color(tensor_to_response_2d(asff_out), (h, w)), args.alpha_overlay)

    diff_l1 = float(torch.mean(torch.abs(recon_fused - fused_reduced)).item())

    stats = {
        "alpha_mean": float(alpha_map.mean()),
        "beta_mean": float(beta_map.mean()),
        "gamma_mean": float(gamma_map.mean()),
        "alpha_std": float(alpha_map.std()),
        "beta_std": float(beta_map.std()),
        "gamma_std": float(gamma_map.std()),
        "reconstruct_l1": diff_l1,
    }

    top = np.concatenate(
        [
            make_tile(original, "Original", args.tile_size),
            make_tile(x1_ov, "X1->3", args.tile_size),
            make_tile(x2_ov, "X2->3", args.tile_size),
            make_tile(x3_ov, "X3->3", args.tile_size),
            make_tile(out_ov, "ASFF Out", args.tile_size),
        ],
        axis=1,
    )

    bottom = np.concatenate(
        [
            make_tile(alpha_ov, "alpha", args.tile_size),
            make_tile(beta_ov, "beta", args.tile_size),
            make_tile(gamma_ov, "gamma", args.tile_size),
            make_tile(fused_ov, "Fused Reduced", args.tile_size),
            build_formula_tile(args.tile_size, stats),
        ],
        axis=1,
    )

    panel = np.concatenate([top, bottom], axis=0)

    out_root = Path(args.outdir)
    comp_dir = out_root / "asff3_components"
    out_root.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    save_image_unicode(out_root / "asff3_paper_style.png", panel)
    save_image_unicode(comp_dir / "original.png", original)
    save_image_unicode(comp_dir / "x1_to3_overlay.png", x1_ov)
    save_image_unicode(comp_dir / "x2_to3_overlay.png", x2_ov)
    save_image_unicode(comp_dir / "x3_to3_overlay.png", x3_ov)
    save_image_unicode(comp_dir / "alpha_overlay.png", alpha_ov)
    save_image_unicode(comp_dir / "beta_overlay.png", beta_ov)
    save_image_unicode(comp_dir / "gamma_overlay.png", gamma_ov)
    save_image_unicode(comp_dir / "fused_reduced_overlay.png", fused_ov)
    save_image_unicode(comp_dir / "asff_out_overlay.png", out_ov)

    meta = {
        "image": str(image_path),
        "weights": str(weights_path),
        "target_sda_index": int(args.target_sda_index),
        "target_sda_module_index": int(capture.module_index),
        "shape_x1_asff_in": shape_list(x1),
        "shape_x2_asff_in": shape_list(x2),
        "shape_x3_asff_in": shape_list(x3),
        "shape_weight_logits": shape_list(logits),
        "shape_fused_reduced": shape_list(fused_reduced),
        "shape_asff_out": shape_list(asff_out),
        "stats": stats,
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("ASFF-3 paper-style visualization done")
    print("=" * 88)
    print(f"image: {image_path}")
    print(f"weights: {weights_path}")
    print(f"target SDA_Fusion index: {args.target_sda_index} (module idx={capture.module_index})")
    print(f"reconstruct L1 (recon_fused vs fused_reduced): {stats['reconstruct_l1']:.8f}")
    print(f"output figure: {out_root / 'asff3_paper_style.png'}")
    print(f"components: {comp_dir}")
    print(f"meta: {out_root / 'meta.json'}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
