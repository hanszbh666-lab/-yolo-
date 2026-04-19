"""Flatten UAVDT sequence images into one folder, keeping sequence prefix in file names."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten UAVDT images into one folder")
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("datasets/UAVDT_NEW/UAV-benchmark-M/UAV-benchmark-M"),
        help="UAVDT sequence image root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/UAVDT_step1/images/all"),
        help="Output flattened image directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.old_dir.exists():
        raise FileNotFoundError(f"Image root not found: {args.old_dir}")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    seq_dirs = sorted([p for p in args.old_dir.iterdir() if p.is_dir() and p.name.startswith("M")])

    for seq_dir in seq_dirs:
        for img_path in sorted(seq_dir.glob("img*.jpg")):
            out_name = f"{seq_dir.name}_{img_path.name[-10:]}"
            out_path = args.output_dir / out_name
            shutil.copyfile(img_path, out_path)
            copied += 1
        print("image folder copy finished:", seq_dir.name)

    print("all images have been copied into:", args.output_dir)
    print("total images copied:", copied)


if __name__ == "__main__":
    main()


