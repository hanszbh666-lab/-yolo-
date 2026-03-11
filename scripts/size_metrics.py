"""Shared size-aware validation and prediction summary utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou


DEFAULT_SMALL_AREA = 24**2
DEFAULT_MEDIUM_AREA = 64**2
SIZE_BUCKETS = ("small", "medium", "large")


def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Return box areas for xyxy boxes."""
    if boxes.numel() == 0:
        return torch.zeros(0, device=boxes.device)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return widths * heights


def area_to_bucket_ids(
    areas: torch.Tensor,
    small_area: float = DEFAULT_SMALL_AREA,
    medium_area: float = DEFAULT_MEDIUM_AREA,
) -> torch.Tensor:
    """Map box areas to small/medium/large bucket ids."""
    bucket_ids = torch.full((areas.shape[0],), 2, dtype=torch.long, device=areas.device)
    bucket_ids[areas < small_area] = 0
    medium_mask = (areas >= small_area) & (areas < medium_area)
    bucket_ids[medium_mask] = 1
    return bucket_ids


def summarize_prediction_size_distribution(
    results,
    small_area: float = DEFAULT_SMALL_AREA,
    medium_area: float = DEFAULT_MEDIUM_AREA,
) -> dict:
    """Summarize prediction counts, confidence and latency by size bucket."""
    summary = {
        name: {"count": 0, "confidence_sum": 0.0, "area_sum": 0.0} for name in SIZE_BUCKETS
    }
    speed_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    image_count = 0

    for result in results:
        image_count += 1
        if hasattr(result, "speed") and result.speed:
            for key in speed_totals:
                speed_totals[key] += float(result.speed.get(key, 0.0))

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        areas = _box_area_xyxy(boxes)
        bucket_ids = area_to_bucket_ids(areas, small_area, medium_area)

        for bucket_idx, bucket_name in enumerate(SIZE_BUCKETS):
            mask = bucket_ids == bucket_idx
            count = int(mask.sum().item())
            if count == 0:
                continue
            summary[bucket_name]["count"] += count
            summary[bucket_name]["confidence_sum"] += float(conf[mask].sum().item())
            summary[bucket_name]["area_sum"] += float(areas[mask].sum().item())

    total_predictions = sum(summary[name]["count"] for name in SIZE_BUCKETS)
    avg_speed = {
        key: (value / image_count if image_count else 0.0) for key, value in speed_totals.items()
    }
    total_latency = sum(avg_speed.values())

    for bucket_name in SIZE_BUCKETS:
        count = summary[bucket_name]["count"]
        summary[bucket_name]["avg_confidence"] = (
            summary[bucket_name]["confidence_sum"] / count if count else 0.0
        )
        summary[bucket_name]["avg_area"] = summary[bucket_name]["area_sum"] / count if count else 0.0
        summary[bucket_name]["ratio"] = count / total_predictions if total_predictions else 0.0
        summary[bucket_name]["per_image"] = count / image_count if image_count else 0.0

    return {
        "image_count": image_count,
        "total_predictions": total_predictions,
        "speed_ms": avg_speed,
        "latency_ms": total_latency,
        "fps": (1000.0 / total_latency) if total_latency > 0 else 0.0,
        "buckets": summary,
        "thresholds": {
            "small_area": small_area,
            "medium_area": medium_area,
        },
    }


class SizeAwareDetectionValidator(DetectionValidator):
    """Detection validator with custom small/medium/large metrics for YOLO-format datasets."""

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.small_area = float(getattr(self.args, "small_area", DEFAULT_SMALL_AREA))
        self.medium_area = float(getattr(self.args, "medium_area", DEFAULT_MEDIUM_AREA))
        self.size_stats = {
            name: {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
            for name in SIZE_BUCKETS
        }

    def _scale_gt_boxes(self, pbatch: dict) -> torch.Tensor:
        """Scale target boxes from validator image space to original image space."""
        gt_boxes = pbatch["bboxes"].clone()
        if gt_boxes.numel():
            gt_boxes = ops.scale_boxes(
                pbatch["imgsz"],
                gt_boxes,
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            )
        return gt_boxes

    def _scale_pred_boxes(self, predn: dict, pbatch: dict) -> dict:
        """Scale prediction boxes from validator image space to original image space."""
        scaled = {
            key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in predn.items()
        }
        if scaled["bboxes"].numel():
            scaled = self.scale_preds(scaled, pbatch)
        return scaled

    def _assign_prediction_buckets(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each prediction to the bucket of its best same-class GT, else by predicted area."""
        pred_bucket_ids = area_to_bucket_ids(_box_area_xyxy(pred_boxes), self.small_area, self.medium_area)
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return pred_bucket_ids

        iou = box_iou(gt_boxes, pred_boxes)
        class_match = gt_cls[:, None] == pred_cls[None, :]
        iou = iou * class_match
        best_iou, best_gt_idx = iou.max(0)
        matched = best_iou > 0
        if matched.any():
            pred_bucket_ids[matched] = gt_bucket_ids[best_gt_idx[matched]]
        return pred_bucket_ids

    def _update_size_stats(self, predn: dict, gt_cls: torch.Tensor, gt_boxes: torch.Tensor) -> None:
        """Collect bucket-wise TP/conf/class statistics for custom size-aware AP and recall."""
        gt_bucket_ids = area_to_bucket_ids(_box_area_xyxy(gt_boxes), self.small_area, self.medium_area)
        pred_bucket_ids = self._assign_prediction_buckets(
            predn["bboxes"],
            predn["cls"],
            gt_boxes,
            gt_cls,
            gt_bucket_ids,
        )

        for bucket_idx, bucket_name in enumerate(SIZE_BUCKETS):
            gt_mask = gt_bucket_ids == bucket_idx
            pred_mask = pred_bucket_ids == bucket_idx

            bucket_batch = {
                "cls": gt_cls[gt_mask],
                "bboxes": gt_boxes[gt_mask],
            }
            bucket_pred = {
                "cls": predn["cls"][pred_mask],
                "conf": predn["conf"][pred_mask],
                "bboxes": predn["bboxes"][pred_mask],
            }

            no_pred = bucket_pred["cls"].shape[0] == 0
            bucket_cls_np = bucket_batch["cls"].cpu().numpy()
            bucket_tp = self._process_batch(bucket_pred, bucket_batch)["tp"]
            self.size_stats[bucket_name]["tp"].append(bucket_tp)
            self.size_stats[bucket_name]["target_cls"].append(bucket_cls_np)
            self.size_stats[bucket_name]["target_img"].append(np.unique(bucket_cls_np))
            self.size_stats[bucket_name]["conf"].append(
                np.zeros(0, dtype=np.float32) if no_pred else bucket_pred["conf"].cpu().numpy()
            )
            self.size_stats[bucket_name]["pred_cls"].append(
                np.zeros(0, dtype=np.float32) if no_pred else bucket_pred["cls"].cpu().numpy()
            )

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]) -> None:
        """Run default metrics update and additionally collect custom size-aware statistics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0, dtype=np.float32) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0, dtype=np.float32) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            predn_scaled = self._scale_pred_boxes(predn, pbatch)
            gt_boxes_scaled = self._scale_gt_boxes(pbatch)
            self._update_size_stats(predn_scaled, pbatch["cls"], gt_boxes_scaled)

            if no_pred:
                continue

            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def _compute_size_metrics(self) -> tuple[dict, dict]:
        """Compute custom bucket-wise metrics from accumulated stats."""
        flat_results = {}
        custom_results = {
            "thresholds": {
                "small_area": self.small_area,
                "medium_area": self.medium_area,
            }
        }

        for bucket_name in SIZE_BUCKETS:
            stats = {key: np.concatenate(value, 0) for key, value in self.size_stats[bucket_name].items()}
            target_count = int(stats["target_cls"].size)
            prediction_count = int(stats["pred_cls"].size)

            bucket_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map": 0.0,
                "instances": target_count,
                "predictions": prediction_count,
            }

            if target_count:
                _, _, precision, recall, _, ap, _, _, _, _, _, _ = ap_per_class(
                    stats["tp"],
                    stats["conf"],
                    stats["pred_cls"],
                    stats["target_cls"],
                    plot=False,
                    save_dir=self.save_dir,
                    names=self.names,
                    prefix=f"{bucket_name}_",
                )
                if len(ap):
                    bucket_metrics["map50"] = float(ap[:, 0].mean())
                    bucket_metrics["map"] = float(ap.mean())
                if len(precision):
                    bucket_metrics["precision"] = float(precision.mean())
                if len(recall):
                    bucket_metrics["recall"] = float(recall.mean())

            custom_results[bucket_name] = bucket_metrics
            flat_results[f"metrics/mAP_{bucket_name}(B)"] = bucket_metrics["map"]
            flat_results[f"metrics/mAP50_{bucket_name}(B)"] = bucket_metrics["map50"]
            flat_results[f"metrics/Precision_{bucket_name}(B)"] = bucket_metrics["precision"]
            flat_results[f"metrics/Recall_{bucket_name}(B)"] = bucket_metrics["recall"]
            flat_results[f"metrics/Instances_{bucket_name}(B)"] = bucket_metrics["instances"]

        return custom_results, flat_results

    def get_stats(self) -> dict:
        """Return default detection stats plus custom size-bucket metrics."""
        base_results = super().get_stats()
        custom_results, flat_results = self._compute_size_metrics()
        merged = {**base_results, **flat_results}
        self.metrics.custom_size_metrics = custom_results
        self.metrics.custom_results_dict = merged
        return merged