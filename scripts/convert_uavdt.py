#!/usr/bin/env python3
"""
Convert UAVDT raw dataset into YOLO detection format.

Design choices for this project:
- Build splits at sequence level to avoid leakage across train/val/test.
- Use GT "*_gt_whole.txt" files for class IDs.
- Default frame sampling step is 10 (medium-sized subset).
- Output follows Ultralytics YOLO folder layout.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


# UAVDT classes in GT whole files are 1-based for these 3 classes.
CLASS_MAP = {
    1: 0,  # car
    2: 1,  # truck
    3: 2,  # bus
}
CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
}


@dataclass
class ConvertStats:
    total_images: int = 0
    total_labels: int = 0
    total_objects: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_bbox: int = 0
    skipped_unknown_class: int = 0
    skipped_out_of_view: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UAVDT_NEW to YOLO format")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/UAVDT_NEW/UAV-benchmark-M/UAV-benchmark-M"),
        help="Root folder with sequence image dirs (M0101, M0201, ...)",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("datasets/UAVDT_NEW/UAV-benchmark-MOTD_v1.0/UAV-benchmark-MOTD_v1.0/GT"),
        help="Root folder containing *_gt_whole.txt files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/UAVDT"),
        help="Output YOLO dataset root",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Sample every Nth frame inside each sequence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sequence split",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to place images into output folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output images/labels folders before conversion",
    )
    parser.add_argument(
        "--skip-out-of-view-values",
        type=int,
        nargs="*",
        default=[],
        help="Explicit out_of_view values to filter out. Default keeps all values.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        for sub in ["images", "labels"]:
            target = output_root / sub
            if target.exists():
                shutil.rmtree(target)

    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def list_sequences(images_root: Path, gt_root: Path) -> List[str]:
    image_seqs = {p.name for p in images_root.iterdir() if p.is_dir() and p.name.startswith("M")}
    gt_seqs = {p.name.replace("_gt_whole.txt", "") for p in gt_root.glob("*_gt_whole.txt")}
    common = sorted(image_seqs & gt_seqs)
    return common


def split_sequences(
    seqs: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[str]]:
    if not seqs:
        return {"train": [], "val": [], "test": []}

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios: require train>0, val>=0 and train+val<1")

    seqs_copy = list(seqs)
    rng = random.Random(seed)
    rng.shuffle(seqs_copy)

    n = len(seqs_copy)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train <= 0:
        n_train = 1
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    train = sorted(seqs_copy[:n_train])
    val = sorted(seqs_copy[n_train : n_train + n_val])
    test = sorted(seqs_copy[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def parse_gt_whole_file(
    gt_file: Path,
    skip_out_of_view_values: set[int],
    stats: ConvertStats,
    occlusion_counter: Dict[int, int],
    out_of_view_counter: Dict[int, int],
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Returns frame-wise boxes:
        frame_id -> list of (cls_id_yolo, left, top, width, height)
    """
    frame_to_boxes: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)

    with gt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 9:
                continue

            try:
                # UAVDT GT columns (1-based):
                # 1 frame_index, 2 target_id, 3 left, 4 top, 5 width, 6 height,
                # 7 out_of_view, 8 occlusion_level, 9 object_category.
                frame_id = int(parts[0])
                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                out_of_view = int(parts[6])
                occlusion_level = int(parts[7])
                raw_cls = int(parts[8])
            except ValueError:
                continue

            out_of_view_counter[out_of_view] += 1
            if out_of_view in skip_out_of_view_values:
                stats.skipped_out_of_view += 1
                continue

            if raw_cls not in CLASS_MAP:
                # Only keep 3 target classes.
                stats.skipped_unknown_class += 1
                continue

            cls_id = CLASS_MAP[raw_cls]
            frame_to_boxes[frame_id].append((cls_id, left, top, width, height))
            occlusion_counter[occlusion_level] += 1

    return frame_to_boxes


def image_size(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def to_yolo_line(
    cls_id: int,
    left: float,
    top: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> str | None:
    # Clip box to image boundary.
    x1 = max(0.0, min(left, float(img_w)))
    y1 = max(0.0, min(top, float(img_h)))
    x2 = max(0.0, min(left + width, float(img_w)))
    y2 = max(0.0, min(top + height, float(img_h)))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1e-6 or bh <= 1e-6:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h

    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def place_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            # Fall back to copy on cross-device or permission issues.
            pass
    shutil.copy2(src, dst)


def process_split(
    split_name: str,
    seqs: List[str],
    images_root: Path,
    gt_root: Path,
    output_root: Path,
    frame_step: int,
    copy_mode: str,
    skip_out_of_view_values: set[int],
    stats: ConvertStats,
    class_counter: Dict[int, int],
    occlusion_counter: Dict[int, int],
    out_of_view_counter: Dict[int, int],
) -> Dict[str, int]:
    split_image_dir = output_root / "images" / split_name
    split_label_dir = output_root / "labels" / split_name

    split_image_count = 0
    split_label_count = 0

    for seq in tqdm(seqs, desc=f"Converting {split_name}", unit="seq"):
        seq_img_dir = images_root / seq
        gt_file = gt_root / f"{seq}_gt_whole.txt"

        if not seq_img_dir.exists() or not gt_file.exists():
            continue

        frame_to_boxes = parse_gt_whole_file(
            gt_file=gt_file,
            skip_out_of_view_values=skip_out_of_view_values,
            stats=stats,
            occlusion_counter=occlusion_counter,
            out_of_view_counter=out_of_view_counter,
        )

        img_files = sorted(seq_img_dir.glob("img*.jpg"))
        if frame_step > 1:
            img_files = img_files[::frame_step]

        for img_path in img_files:
            stem = img_path.stem  # img000123
            frame_str = stem.replace("img", "")
            if not frame_str.isdigit():
                continue
            frame_id = int(frame_str)

            out_name = f"{seq}_frame_{frame_id:06d}"
            out_img = split_image_dir / f"{out_name}.jpg"
            out_label = split_label_dir / f"{out_name}.txt"

            try:
                img_w, img_h = image_size(img_path)
            except RuntimeError:
                stats.skipped_missing_image += 1
                continue

            yolo_lines: List[str] = []
            for entry in frame_to_boxes.get(frame_id, []):
                cls_id, left, top, width, height = entry
                line = to_yolo_line(cls_id, left, top, width, height, img_w, img_h)
                if line is None:
                    stats.skipped_invalid_bbox += 1
                    continue
                yolo_lines.append(line)
                class_counter[cls_id] += 1

            place_image(img_path, out_img, copy_mode)
            split_image_count += 1
            stats.total_images += 1

            with out_label.open("w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            split_label_count += 1
            stats.total_labels += 1
            stats.total_objects += len(yolo_lines)

    return {
        "images": split_image_count,
        "labels": split_label_count,
    }


def write_conversion_stats(
    output_root: Path,
    split_counts: Dict[str, Dict[str, int]],
    seq_split: Dict[str, List[str]],
    class_counter: Dict[int, int],
    occlusion_counter: Dict[int, int],
    out_of_view_counter: Dict[int, int],
    stats: ConvertStats,
    args: argparse.Namespace,
) -> None:
    payload = {
        "conversion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "images_root": str(args.images_root),
            "gt_root": str(args.gt_root),
            "frame_step": args.frame_step,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "copy_mode": args.copy_mode,
            "skip_out_of_view_values": args.skip_out_of_view_values,
        },
        "total_images": stats.total_images,
        "total_labels": stats.total_labels,
        "total_objects": stats.total_objects,
        "splits": split_counts,
        "sequence_split": seq_split,
        "class_distribution": {
            CLASS_NAMES[k]: class_counter.get(k, 0) for k in sorted(CLASS_NAMES)
        },
        "occlusion_distribution": {
            str(k): occlusion_counter.get(k, 0) for k in sorted(occlusion_counter)
        },
        "out_of_view_distribution": {
            str(k): out_of_view_counter.get(k, 0) for k in sorted(out_of_view_counter)
        },
        "skipped": {
            "missing_or_unreadable_images": stats.skipped_missing_image,
            "invalid_bboxes": stats.skipped_invalid_bbox,
            "unknown_classes": stats.skipped_unknown_class,
            "out_of_view_filtered": stats.skipped_out_of_view,
        },
    }

    out_stats = output_root / "conversion_stats.json"
    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")

    if not args.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")
    if not args.gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {args.gt_root}")

    ensure_output_dirs(args.output_root, args.overwrite)

    seqs = list_sequences(args.images_root, args.gt_root)
    if not seqs:
        raise RuntimeError("No common sequences found between image and GT roots")

    seq_split = split_sequences(
        seqs=seqs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    stats = ConvertStats()
    class_counter: Dict[int, int] = defaultdict(int)
    occlusion_counter: Dict[int, int] = defaultdict(int)
    out_of_view_counter: Dict[int, int] = defaultdict(int)
    split_counts: Dict[str, Dict[str, int]] = {}

    skip_out_of_view_values = set(args.skip_out_of_view_values)

    for split in ["train", "val", "test"]:
        split_counts[split] = process_split(
            split_name=split,
            seqs=seq_split[split],
            images_root=args.images_root,
            gt_root=args.gt_root,
            output_root=args.output_root,
            frame_step=args.frame_step,
            copy_mode=args.copy_mode,
            skip_out_of_view_values=skip_out_of_view_values,
            stats=stats,
            class_counter=class_counter,
            occlusion_counter=occlusion_counter,
            out_of_view_counter=out_of_view_counter,
        )

    write_conversion_stats(
        output_root=args.output_root,
        split_counts=split_counts,
        seq_split=seq_split,
        class_counter=class_counter,
        occlusion_counter=occlusion_counter,
        out_of_view_counter=out_of_view_counter,
        stats=stats,
        args=args,
    )

    print("\nConversion finished.")
    print(f"Output root: {args.output_root}")
    print(f"Total images: {stats.total_images}")
    print(f"Total labels: {stats.total_labels}")
    print(f"Total objects: {stats.total_objects}")
    print(f"Split sequences: train={len(seq_split['train'])}, val={len(seq_split['val'])}, test={len(seq_split['test'])}")
    print("Class distribution:")
    for cls_id in sorted(CLASS_NAMES):
        print(f"  {CLASS_NAMES[cls_id]}: {class_counter.get(cls_id, 0)}")


if __name__ == "__main__":
    main()
