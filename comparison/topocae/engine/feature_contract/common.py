from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p1_e6_controlled_retraining_v1"
PACKAGE = ROOT / "output_generation_p1_e5_web_v2_mixed_9300" / "training_package"
REALDEV_CACHE = ROOT / "reports" / "p1_m2_rib_only_migration" / "realdev" / "realdev_raw.pt"
REALDEV_MANIFEST = REALDEV_CACHE.with_name("realdev_raw_manifest.json")
SEEDS = (20260718, 20260719, 20260720)
FAMILIES = ("core_5300", "mixed_9300")
SPLITS = ("train", "val", "holdout")

MACRO_CLASSES = (
    "stock",
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rib",
    "web",
    "rectangular_boss",
    "fillet",
    "chamfer",
    "slot_v",
    "slot_o",
)
TARGET_CLASSES = MACRO_CLASSES[1:]
RAW_MACRO_IDS = {
    "stock": 0,
    "through_hole_round": 1,
    "blind_hole_round": 2,
    "stepped_hole_round": 3,
    "counterbore_hole": 4,
    "countersink_hole": 5,
    "obround_hole": 6,
    "slot_u": 9,
    "slot_rect": 10,
    "pocket_rect": 11,
    "pocket_circular": 12,
    "cylinder_boss": 14,
    "rib": 20,
    "web": 21,
    "rectangular_boss": 24,
    "fillet": 25,
    "chamfer": 26,
    "slot_v": 33,
    "slot_o": 35,
}
RAW_TO_CONTIGUOUS = {RAW_MACRO_IDS[name]: index for index, name in enumerate(MACRO_CLASSES)}

NODE_DIM = 50
EDGE_DIM = 24
PRIMITIVE_COUNT = 11
SEMANTIC_COUNT = 64

HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.08
EPOCHS = 56
BATCH_SIZE = 192
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001
PATIENCE = 14
LABEL_SMOOTHING = 0.01
GRADIENT_CLIP_NORM = 5.0
LOSS_WEIGHTS = {"macro": 1.0, "semantic": 0.55, "same_instance": 0.65}

SHARED_CANDIDATE_THRESHOLDS = (
    ROOT
    / "reports"
    / "p1_e3c_seed_matched_candidate_v1"
    / "pipelines"
    / "old_same_instance_seed_20260718"
    / "candidate_model"
    / "candidate_threshold_selection.json"
)
SHARED_GLOBAL_DECODE = SHARED_CANDIDATE_THRESHOLDS.parents[1] / "global_decode_selection.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def hashed(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def family_filter(records: Sequence[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    if family == "mixed_9300":
        return list(records)
    if family != "core_5300":
        raise ValueError(f"unknown family: {family}")
    return [
        record
        for record in records
        if str(record.get("geometry_family_id", "")).startswith("core_5300:")
    ]


def _streaming_moments(
    records: Sequence[dict[str, Any]], key: str, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total = torch.zeros(dimension, dtype=torch.float64)
    total_sq = torch.zeros(dimension, dtype=torch.float64)
    count = 0
    for record in records:
        values = record[key].to(dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError(f"{record.get('sample_id')}:{key} dimension {tuple(values.shape)}")
        total += values.sum(dim=0)
        total_sq += (values * values).sum(dim=0)
        count += int(values.shape[0])
    if not count:
        raise ValueError(f"cannot fit normalization on empty {key}")
    mean = total / count
    variance = (total_sq / count - mean * mean).clamp_min(0.0)
    std = variance.sqrt()
    std = torch.where(std > 1.0e-12, std, torch.ones_like(std))
    return mean.float(), std.float(), count


def fit_normalization(records: Sequence[dict[str, Any]], family: str) -> dict[str, Any]:
    node_mean, node_std, node_count = _streaming_moments(records, "node_features", NODE_DIM)
    edge_mean, edge_std, edge_count = _streaming_moments(records, "edge_attr", EDGE_DIM)
    result = {
        "schema_version": "topocae_p1_e6_family_normalization_v1",
        "family": family,
        "fitted_split": "train",
        "validation_or_holdout_used": False,
        "node_count": node_count,
        "edge_count": edge_count,
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
    }
    result["statistics_sha256"] = canonical_json_sha256(result)
    return result


def normalization_tensors(stats: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(stats[key], dtype=torch.float32)
        for key in ("node_mean", "node_std", "edge_mean", "edge_std")
    }


def remap_macro_targets(raw: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(raw, -1)
    for source, target in RAW_TO_CONTIGUOUS.items():
        output[raw == source] = target
    if bool((output < 0).any()):
        unknown = sorted(set(int(value) for value in raw[output < 0].tolist()))
        raise ValueError(f"unknown P1-E5 macro IDs: {unknown}")
    return output


def adapt_record(record: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    norm = normalization_tensors(stats)
    raw_node = record["node_features"].float()
    raw_edge = record["edge_attr"].float()
    primitive = raw_node[:, :PRIMITIVE_COUNT].argmax(dim=1).long()
    macro = remap_macro_targets(record["macro_feature_ids"].long())
    semantic = record["face_class_labels"].long()
    sample_id = str(record["sample_id"])
    return {
        "sample_id": sample_id,
        "source_graph_path": str(record["source_graph_path"]),
        "source_step_path": str(record.get("source_step_path", "")),
        "base_geometry_family_id": str(record.get("geometry_family_id", sample_id)),
        "geometry_family_id": str(record.get("geometry_family_id", sample_id)),
        "derivation_kind": "identity",
        "stock_type": str(record.get("stock_type", "unknown")),
        "split": str(record.get("split", "")),
        "node_features": (raw_node - norm["node_mean"]) / norm["node_std"],
        "primitive_surface_ids": primitive,
        "macro_targets": macro,
        "semantic_targets": semantic,
        "instance_ids": record["instance_ids"].long(),
        "edge_index": record["edge_index"].long(),
        "edge_features": (raw_edge - norm["edge_mean"]) / norm["edge_std"],
        "edge_same_instance_targets": record["edge_same_instance_labels"].float(),
        "edge_supervision_mask": record["edge_supervision_mask"].bool(),
        "feature_sets": list(record.get("feature_sets", [])),
        "face_to_canonical_patch": {str(i): str(i) for i in range(int(raw_node.shape[0]))},
    }


def validate_raw_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    sample_id = str(record.get("sample_id", "unknown"))
    node = record["node_features"]
    edge = record["edge_attr"]
    edge_index = record["edge_index"]
    node_count = int(node.shape[0])
    edge_count = int(edge.shape[0])
    if tuple(node.shape[1:]) != (NODE_DIM,):
        errors.append(f"{sample_id}:node_dim")
    if tuple(edge.shape[1:]) != (EDGE_DIM,):
        errors.append(f"{sample_id}:edge_dim")
    if tuple(edge_index.shape) != (2, edge_count):
        errors.append(f"{sample_id}:edge_index_shape")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count):
        errors.append(f"{sample_id}:edge_index_range")
    face_ids = record["face_ids"]
    if face_ids.tolist() != list(range(node_count)):
        errors.append(f"{sample_id}:face_ids_not_contiguous")
    for key in ("face_class_labels", "macro_feature_ids", "instance_ids"):
        if int(record[key].numel()) != node_count:
            errors.append(f"{sample_id}:{key}_length")
    if int(record["edge_same_instance_labels"].numel()) != edge_count:
        errors.append(f"{sample_id}:edge_label_length")
    if int(record["edge_supervision_mask"].numel()) != edge_count:
        errors.append(f"{sample_id}:edge_mask_length")
    if not bool(torch.isfinite(node).all()) or not bool(torch.isfinite(edge).all()):
        errors.append(f"{sample_id}:nonfinite_feature")
    raw_macro = set(int(value) for value in record["macro_feature_ids"].tolist())
    if not raw_macro <= set(RAW_TO_CONTIGUOUS):
        errors.append(f"{sample_id}:macro_range")
    semantic = record["face_class_labels"]
    if semantic.numel() and (int(semantic.min()) < 0 or int(semantic.max()) >= SEMANTIC_COUNT):
        errors.append(f"{sample_id}:semantic_range")
    return errors


def classification_metrics(
    truth: Sequence[int], prediction: Sequence[int], names: Sequence[str] | None = None
) -> dict[str, Any]:
    count = len(names) if names is not None else max([*truth, *prediction], default=-1) + 1
    matrix = np.zeros((count, count), dtype=np.int64)
    for target, predicted in zip(truth, prediction):
        if 0 <= int(target) < count and 0 <= int(predicted) < count:
            matrix[int(target), int(predicted)] += 1
    per_class: dict[str, Any] = {}
    active_f1: list[float] = []
    for index in range(count):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = names[index] if names is not None else str(index)
        per_class[str(key)] = {
            "class_id": index,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            active_f1.append(f1)
    correct = int(np.trace(matrix))
    total = int(matrix.sum())
    return {
        "sample_count": total,
        "macro_f1": sum(active_f1) / max(len(active_f1), 1),
        "micro_f1": correct / max(total, 1),
        "accuracy": correct / max(total, 1),
        "active_class_count": len(active_f1),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(names) if names is not None else list(range(count)),
            "matrix": matrix.tolist(),
        },
    }


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, Any]:
    tp = sum(int(left == 1 and right == 1) for left, right in zip(truth, prediction))
    fp = sum(int(left == 0 and right == 1) for left, right in zip(truth, prediction))
    fn = sum(int(left == 1 and right == 0) for left, right in zip(truth, prediction))
    tn = sum(int(left == 0 and right == 0) for left, right in zip(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def truth_objects(feature_sets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": int(item.get("instance_id", index + 1)),
            "macro_feature": str(item["macro_feature"]),
            "face_indices": sorted(int(value) for value in item.get("node_indices", [])),
        }
        for index, item in enumerate(feature_sets)
        if str(item.get("macro_feature", "stock")) != "stock"
    ]


def iou(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def object_counts(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    classes: Sequence[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            overlap = iou(truth["face_indices"], prediction["face_indices"])
            if truth["macro_feature"] == prediction["macro_feature"] and overlap >= 0.5:
                pairs.append((overlap, truth_index, prediction_index))
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    exact = 0
    class_tp: Counter[str] = Counter()
    class_truth = Counter(str(item["macro_feature"]) for item in truths)
    class_prediction = Counter(str(item["macro_feature"]) for item in predictions)
    for _, truth_index, prediction_index in sorted(pairs, reverse=True):
        if truth_index in used_truth or prediction_index in used_prediction:
            continue
        used_truth.add(truth_index)
        used_prediction.add(prediction_index)
        truth = truths[truth_index]
        prediction = predictions[prediction_index]
        class_tp[str(truth["macro_feature"])] += 1
        exact += int(set(truth["face_indices"]) == set(prediction["face_indices"]))
    overmerge = sum(
        int(sum(iou(pred["face_indices"], truth["face_indices"]) > 0 for truth in truths) > 1)
        for pred in predictions
    )
    split = sum(
        int(sum(iou(truth["face_indices"], pred["face_indices"]) > 0 for pred in predictions) > 1)
        for truth in truths
    )
    return {
        "truth": len(truths),
        "prediction": len(predictions),
        "tp": len(used_truth),
        "fp": len(predictions) - len(used_prediction),
        "fn": len(truths) - len(used_truth),
        "exact": exact,
        "overmerge": overmerge,
        "split": split,
        "per_class": {
            name: {
                "tp": class_tp[name],
                "truth": class_truth[name],
                "prediction": class_prediction[name],
            }
            for name in classes
        },
    }


def aggregate_object_counts(samples: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    per_class = {name: Counter() for name in TARGET_CLASSES}
    for sample in samples:
        values = sample[key]
        totals.update({name: int(values[name]) for name in ("truth", "prediction", "tp", "fp", "fn", "exact", "overmerge", "split")})
        for name in TARGET_CLASSES:
            per_class[name].update(values["per_class"][name])
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    class_metrics: dict[str, Any] = {}
    active_f1: list[float] = []
    for name in TARGET_CLASSES:
        item = per_class[name]
        p = item["tp"] / item["prediction"] if item["prediction"] else 0.0
        r = item["tp"] / item["truth"] if item["truth"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        class_metrics[name] = {
            "support": item["truth"],
            "prediction": item["prediction"],
            "tp": item["tp"],
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        if item["truth"]:
            active_f1.append(f1)
    micro = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    truth_total = totals["truth"]
    return {
        "truth_count": truth_total,
        "prediction_count": totals["prediction"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_micro_f1": micro,
        "instance_class_macro_f1": sum(active_f1) / max(len(active_f1), 1),
        "complete_face_set_accuracy": totals["exact"] / max(truth_total, 1),
        "exact_count": totals["exact"],
        "overmerge_count": totals["overmerge"],
        "overmerge_rate": totals["overmerge"] / max(totals["prediction"], 1),
        "wrong_split_count": totals["split"],
        "wrong_split_rate": totals["split"] / max(truth_total, 1),
        "missed_instance_count": totals["fn"],
        "missed_instance_rate": totals["fn"] / max(truth_total, 1),
        "per_class": class_metrics,
    }


def finite_or_raise(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_or_raise(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_or_raise(item, f"{context}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite value at {context}: {value}")


__all__ = [name for name in globals() if not name.startswith("_")]
