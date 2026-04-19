"""Generate one YOLO txt per image and place all labels in one folder."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten UAVDT labels into one folder")
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("datasets/UAVDT_NEW/UAV-benchmark-MOTD_v1.0/UAV-benchmark-MOTD_v1.0/GT"),
        help="GT folder containing *_gt_whole.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/UAVDT_step1/labels/all"),
        help="Output flattened label directory",
    )
    parser.add_argument("--img-w", type=int, default=1024, help="Image width")
    parser.add_argument("--img-h", type=int, default=540, help="Image height")
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Write class index 0 for all objects (same as reference script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.old_dir.exists():
        raise FileNotFoundError(f"GT root not found: {args.old_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    gt_files = sorted(args.old_dir.glob("*_gt_whole.txt"))

    for gt_path in gt_files:
        video_name = gt_path.name[:5]
        with gt_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue

                line_ls = line.split(",")
                if len(line_ls) < 6:
                    continue

                img_num = line_ls[0]
                img_six_num = f"{int(img_num):06d}"

                org_xc = int(line_ls[2]) + int(line_ls[4]) / 2
                org_yc = int(line_ls[3]) + int(line_ls[5]) / 2
                org_w = int(line_ls[4])
                org_h = int(line_ls[5])

                xc = float(org_xc / args.img_w)
                yc = float(org_yc / args.img_h)
                w = float(org_w / args.img_w)
                h = float(org_h / args.img_h)

                wrong_label = org_w > (args.img_w / 6) and org_h > (args.img_h / 6)
                if wrong_label:
                    continue

                cls = "0" if args.single_class else line_ls[8].strip() if len(line_ls) > 8 else "0"
                if not args.single_class:
                    # Keep same category mapping style as existing project: 1/2/3 -> 0/1/2
                    if cls in {"1", "2", "3"}:
                        cls = str(int(cls) - 1)
                    else:
                        continue

                new_txt_path = args.output_dir / f"{video_name}_{img_six_num}.txt"
                with new_txt_path.open("a", encoding="utf-8") as wr:
                    wr.write(f"{cls} {xc} {yc} {w} {h}\n")
                total_labels += 1

        print(gt_path.name, "has been parsed")

    print("all txt labels have been saved in:", args.output_dir)
    print("total label lines written:", total_labels)


if __name__ == "__main__":
    main()