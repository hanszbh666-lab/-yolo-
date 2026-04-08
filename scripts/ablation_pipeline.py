"""
Ablation experiment orchestrator for VisDrone.

Pipeline:
1) Read experiment IDs from xlsx.
2) Run train (if needed), val, detect benchmark.
3) Collect metrics and write summary files.
4) Write metrics back to xlsx by experiment ID.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing project entry points.
from scripts.train import train_yolo11
from scripts.val import validate_model
from scripts.detect import detect_images
from scripts.size_metrics import summarize_prediction_size_distribution

from ultralytics import YOLO


@dataclass
class ExperimentSpec:
    exp_id: str
    group: str
    model_yaml: Optional[str]
    baseline_weight: Optional[str]
    skip_train: bool = False


DEFAULT_EXPERIMENT_MAP: Dict[str, ExperimentSpec] = {
    # Table 1: module ablation
    "A0": ExperimentSpec("A0", "A", "models/Ablation_experiments/a0_module.yaml", "weights/yolo11s-visdrone.pt", True),
    "A1": ExperimentSpec("A1", "A", "models/Ablation_experiments/a1_module.yaml", None, False),
    "A2": ExperimentSpec("A2", "A", "models/Ablation_experiments/a2_module.yaml", None, False),
    "A3": ExperimentSpec("A3", "A", "models/Ablation_experiments/MRA-STD YOLO.yaml", "runs/train/sda_std_vs/weights/best.pt", True),
    # Table 2: head ablation
    "B0": ExperimentSpec("B0", "B", "models/Ablation_experiments/yolov11.yaml", "weights/yolo11s-visdrone.pt", True),
    "B1": ExperimentSpec("B1", "B", "models/Ablation_experiments/b1_head.yaml", None, False),
    "B2": ExperimentSpec("B2", "B", "models/Ablation_experiments/b2_head.yaml", None, False),
    "B3": ExperimentSpec("B3", "B", "models/Ablation_experiments/b3_head.yaml", None, False),
}


def _abs(path_like: str) -> Path:
    return (PROJECT_ROOT / path_like).resolve()


def _norm_path(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path_like).resolve())


def _load_workbook(xlsx_path: Path):
    try:
        module = importlib.import_module("openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required for reading/writing xlsx. Install with: pip install openpyxl"
        ) from exc
    return module.load_workbook(xlsx_path)


def _sanitize_device(device: str) -> str:
    """Normalize requested device string to available CUDA devices."""
    device_str = str(device).strip()
    if not device_str:
        return '0' if torch.cuda.is_available() else 'cpu'
    if device_str.lower() in {'cpu', 'mps'}:
        return device_str.lower()

    if not torch.cuda.is_available():
        print("[Ablation] CUDA 不可用，自动切换到 CPU")
        return 'cpu'

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return 'cpu'

    parsed = []
    for item in device_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue

    if not parsed:
        return '0'

    valid = [idx for idx in parsed if 0 <= idx < gpu_count]
    if not valid:
        print(f"[Ablation] 请求设备 {device_str} 无可用 GPU，自动改为 0")
        return '0'

    if len(valid) < len(parsed):
        valid_str = ','.join(str(x) for x in valid)
        print(f"[Ablation] 请求设备 {device_str} 超出可用范围，自动改为 {valid_str}")

    # 去重并保持顺序
    deduped = list(dict.fromkeys(valid))
    return ','.join(str(x) for x in deduped)


def load_train_baseline_args(args_yaml: Path) -> Dict:
    with args_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return {
        "data": _norm_path(cfg.get("data", "configs/visdrone.yaml")),
        "epochs": int(cfg.get("epochs", 200)),
        "batch": int(cfg.get("batch", 16)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "device": str(cfg.get("device", "0")),
        "workers": int(cfg.get("workers", 8)),
        "patience": int(cfg.get("patience", 50)),
        "project": _norm_path(str(cfg.get("project", "runs/train"))),
    }


def collect_ids_from_excel(xlsx_path: Path) -> List[str]:
    wb = _load_workbook(xlsx_path)
    ws = wb.active

    ids: List[str] = []
    for row in range(3, ws.max_row + 1):
        a_id = ws[f"L{row}"].value
        b_id = ws[f"AC{row}"].value
        if isinstance(a_id, str) and a_id.strip().upper().startswith("A"):
            ids.append(a_id.strip().upper())
        if isinstance(b_id, str) and b_id.strip().upper().startswith("B"):
            ids.append(b_id.strip().upper())

    # keep insertion order, deduplicate
    uniq: List[str] = []
    seen = set()
    for exp_id in ids:
        if exp_id not in seen:
            seen.add(exp_id)
            uniq.append(exp_id)
    return uniq


def parse_only_ids(only_arg: str) -> List[str]:
    """Parse --only argument, supporting both English and Chinese commas."""
    if not only_arg:
        return []

    normalized = only_arg.replace("，", ",")
    ids = [x.strip().upper() for x in normalized.split(",") if x.strip()]

    uniq: List[str] = []
    seen = set()
    for exp_id in ids:
        if exp_id not in seen:
            seen.add(exp_id)
            uniq.append(exp_id)
    return uniq


def resolve_weight_for_experiment(spec: ExperimentSpec) -> Optional[Path]:
    if spec.baseline_weight:
        p = _abs(spec.baseline_weight)
        if p.exists():
            return p
    return None


def extract_model_size_metrics(model_source: Path, imgsz: int) -> Dict[str, Optional[float]]:
    model = YOLO(str(model_source))
    params = float(sum(p.numel() for p in model.model.parameters()))

    gflops = None
    try:
        info = model.info(verbose=False)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            gflops = float(info[3])
    except Exception:
        gflops = None

    return {
        "params": params,
        "params_m": params / 1e6,
        "gflops": gflops,
        "imgsz": imgsz,
    }


def train_if_needed(spec: ExperimentSpec, base_args: Dict, train_name: str) -> Path:
    existing = resolve_weight_for_experiment(spec)
    if spec.skip_train and existing is not None:
        return existing

    if spec.model_yaml is None:
        raise ValueError(f"Experiment {spec.exp_id} has no model yaml and no baseline weight.")

    model_yaml = _abs(spec.model_yaml)
    if not model_yaml.exists():
        raise FileNotFoundError(f"Model yaml not found: {model_yaml}")

    train_yolo11(
        data_config=base_args["data"],
        model=str(model_yaml),
        epochs=base_args["epochs"],
        batch_size=base_args["batch"],
        imgsz=base_args["imgsz"],
        device=base_args["device"],
        workers=base_args["workers"],
        project=base_args["project"],
        name=train_name,
        resume=False,
        use_pretrained=False,
        patience=base_args["patience"],
    )

    best_weight = _abs(f"{base_args['project']}/{train_name}/weights/best.pt")
    if not best_weight.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_weight}")
    return best_weight


def evaluate_with_val(weight_path: Path, base_args: Dict, val_name: str) -> Dict[str, float]:
    val_results = validate_model(
        model_path=str(weight_path),
        data_config=base_args["data"],
        imgsz=base_args["imgsz"],
        batch_size=base_args["batch"],
        device=base_args["device"],
        split="val",
        save_json=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        project=_norm_path("runs/val"),
        name=val_name,
    )

    metrics = {
        "map50": float(val_results.box.map50),
        "map50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "maps": None,
        "mapm": None,
        "mapl": None,
    }

    if hasattr(val_results, "custom_size_metrics"):
        size_metrics = val_results.custom_size_metrics
        metrics["maps"] = float(size_metrics["small"]["map"])
        metrics["mapm"] = float(size_metrics["medium"]["map"])
        metrics["mapl"] = float(size_metrics["large"]["map"])

    return metrics


def evaluate_fps(weight_path: Path, base_args: Dict, detect_name: str, source: str) -> Dict[str, float]:
    detect_results = detect_images(
        model_path=str(weight_path),
        source=source,
        imgsz=base_args["imgsz"],
        conf=0.25,
        iou=0.7,
        max_det=300,
        device=base_args["device"],
        save=True,
        show=False,
        project=_norm_path("runs/detect"),
        name=detect_name,
    )

    summary = summarize_prediction_size_distribution(detect_results)
    return {
        "fps": float(summary["fps"]),
        "latency_ms": float(summary["latency_ms"]),
    }


def update_excel_row(ws, exp_id: str, result: Dict[str, Optional[float]]):
    if exp_id.startswith("A"):
        id_col = "L"
        write_cols = {
            "map50": "Q",
            "map50_95": "R",
            "maps": "S",
            "mapm": "T",
            "mapl": "U",
            "params_m": "V",
            "fps": "W",
            "precision": "X",
            "recall": "Y",
        }
    elif exp_id.startswith("B"):
        id_col = "AC"
        write_cols = {
            "map50": "AH",
            "map50_95": "AI",
            "maps": "AJ",
            "mapm": "AK",
            "mapl": "AL",
            "params_m": "AM",
            "fps": "AN",
            "precision": "AO",
            "recall": "AP",
        }
    else:
        return

    target_row = None
    for row in range(3, ws.max_row + 1):
        v = ws[f"{id_col}{row}"].value
        if isinstance(v, str) and v.strip().upper() == exp_id:
            target_row = row
            break

    if target_row is None:
        return

    for key, col in write_cols.items():
        val = result.get(key)
        if val is None:
            continue
        ws[f"{col}{target_row}"] = float(val)


def save_summary_csv(summary_path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "exp_id",
        "weight_path",
        "params_m",
        "gflops",
        "map50",
        "map50_95",
        "maps",
        "mapm",
        "mapl",
        "precision",
        "recall",
        "fps",
        "latency_ms",
        "status",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def run_pipeline(args):
    os.chdir(PROJECT_ROOT)

    xlsx_path = _abs(args.excel)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    args_yaml_path = _abs(args.baseline_args)
    if not args_yaml_path.exists():
        raise FileNotFoundError(f"Baseline args yaml not found: {args_yaml_path}")

    output_dir = _abs("runs/ablation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_path = output_dir / f"{xlsx_path.stem}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    shutil.copy2(xlsx_path, backup_path)

    base_args = load_train_baseline_args(args_yaml_path)
    base_args["epochs"] = args.epochs
    if args.device is not None:
        base_args["device"] = args.device

    base_args["device"] = _sanitize_device(base_args["device"])

    detect_source = _norm_path(args.detect_source)

    if args.only:
        experiment_ids = parse_only_ids(args.only)
    else:
        experiment_ids = collect_ids_from_excel(xlsx_path)

    records: List[Dict[str, object]] = []

    wb = _load_workbook(xlsx_path)
    ws = wb.active

    for exp_id in experiment_ids:
        spec = DEFAULT_EXPERIMENT_MAP.get(exp_id)
        if spec is None:
            records.append({"exp_id": exp_id, "status": "skipped", "error": "No mapping"})
            continue

        train_name = f"ablation_{exp_id.lower()}"
        val_name = f"ablation_{exp_id.lower()}_val"
        detect_name = f"ablation_{exp_id.lower()}_det"

        record = {
            "exp_id": exp_id,
            "status": "ok",
            "error": "",
        }

        try:
            weight_path = train_if_needed(spec, base_args, train_name)
            record["weight_path"] = str(weight_path)

            size_metrics = extract_model_size_metrics(weight_path if weight_path.suffix == ".pt" else _abs(spec.model_yaml), base_args["imgsz"])
            record.update(size_metrics)

            val_metrics = evaluate_with_val(weight_path, base_args, val_name)
            record.update(val_metrics)

            fps_metrics = evaluate_fps(weight_path, base_args, detect_name, detect_source)
            record.update(fps_metrics)

            update_excel_row(ws, exp_id, record)
            wb.save(xlsx_path)

        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)

        records.append(record)

    wb.save(xlsx_path)

    summary_csv = output_dir / "ablation_summary.csv"
    save_summary_csv(summary_csv, records)

    summary_json = output_dir / "ablation_summary.json"
    summary_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("Ablation pipeline finished")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Excel updated: {xlsx_path}")
    print(f"Excel backup: {backup_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments and write back to xlsx")
    parser.add_argument("--excel", type=str, default="消融实验表.xlsx", help="xlsx file path")
    parser.add_argument(
        "--baseline-args",
        type=str,
        default="runs/train/sda_std_vs/args.yaml",
        help="baseline training args yaml",
    )
    parser.add_argument("--detect-source", type=str, default="datasets/visdrone/images/test", help="detect source")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs for ablation runs")
    parser.add_argument("--device", type=str, default=None, help="override device, e.g. 0 or 0")
    parser.add_argument("--only", type=str, default="", help="comma separated experiment IDs, e.g. A1,A2,B3")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
