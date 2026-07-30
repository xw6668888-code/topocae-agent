from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.gnn_baseline import EdgeAwareMultiHeadGNN, MultiHeadGraphSAGE
from utils.instance_edge_metrics import compute_edge_same_instance_metrics
from utils.training_export import DEFAULT_NODE_FEATURE_MODE, node_feature_schema_for_mode


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    output_root = current_dir / "output"

    parser = argparse.ArgumentParser(
        description="Train a multi-head GNN baseline for CAE face-level feature recognition."
    )
    parser.add_argument("--training-dir", type=Path, default=output_root / "training" / "gnn")
    parser.add_argument("--label-map-dir", type=Path, default=output_root / "label_maps")
    parser.add_argument("--output-dir", type=Path, default=output_root / "model_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--primitive-embedding-dim", type=int, default=16)
    parser.add_argument("--model-kind", choices=["graphsage", "edge_aware"], default="graphsage")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--macro-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--derived-loss-weight", type=float, default=1.0)
    parser.add_argument("--idealization-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def adapt_initial_state_dict(
    current_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    adapted = {key: value.clone() for key, value in current_state.items()}
    report = {
        "loaded_tensor_count": 0,
        "expanded_tensor_count": 0,
        "skipped_tensor_count": 0,
    }

    for key, current_value in current_state.items():
        initial_value = initial_state.get(key)
        if initial_value is None:
            report["skipped_tensor_count"] += 1
            continue
        if tuple(initial_value.shape) == tuple(current_value.shape):
            adapted[key] = initial_value.clone()
            report["loaded_tensor_count"] += 1
            continue
        if (
            initial_value.ndim == 2
            and current_value.ndim == 2
            and initial_value.shape[0] == current_value.shape[0]
            and initial_value.shape[1] < current_value.shape[1]
        ):
            expanded = current_value.clone()
            expanded.zero_()
            expanded[:, : initial_value.shape[1]] = initial_value
            adapted[key] = expanded
            report["expanded_tensor_count"] += 1
            continue
        report["skipped_tensor_count"] += 1

    return adapted, report


def load_initial_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    initial_state = checkpoint.get("model_state_dict", checkpoint)
    adapted_state, report = adapt_initial_state_dict(model.state_dict(), initial_state)
    model.load_state_dict(adapted_state)
    return report


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def load_label_map(path: Path) -> tuple[str, dict[str, str], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    layer_name = next(key for key in data.keys() if key not in {"ontology_name", "ontology_path"})
    layer = data[layer_name]
    return layer_name, layer["id_to_key"], layer["key_to_id"]


def load_training_metadata(training_dir: Path) -> dict:
    metadata_path = training_dir.parent / "metadata" / "training_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class GraphRecordDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def infer_multilabel_target_ids(records: list[dict], field_name: str) -> list[int]:
    target_ids = sorted({
        label_id
        for record in records
        for row in record.get(field_name, [])
        for label_id in row
        if label_id > 0
    })
    return target_ids


def compute_feature_normalization(records: list[dict], feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_nodes = 0
    feature_sum = torch.zeros(feature_dim, dtype=torch.float64)
    feature_sumsq = torch.zeros(feature_dim, dtype=torch.float64)

    for record in records:
        node_features = torch.tensor(record["node_features"], dtype=torch.float64)
        total_nodes += node_features.size(0)
        feature_sum += node_features.sum(dim=0)
        feature_sumsq += (node_features * node_features).sum(dim=0)

    mean = feature_sum / max(total_nodes, 1)
    var = (feature_sumsq / max(total_nodes, 1)) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-8))
    return mean.to(torch.float32), std.to(torch.float32)


def compute_edge_feature_normalization(records: list[dict], edge_feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_edges = 0
    feature_sum = torch.zeros(edge_feature_dim, dtype=torch.float64)
    feature_sumsq = torch.zeros(edge_feature_dim, dtype=torch.float64)

    for record in records:
        edge_features = record.get("edge_features", [])
        if not edge_features:
            continue
        tensor = torch.tensor(edge_features, dtype=torch.float64)
        total_edges += tensor.size(0)
        feature_sum += tensor.sum(dim=0)
        feature_sumsq += (tensor * tensor).sum(dim=0)

    if total_edges == 0:
        return torch.zeros(edge_feature_dim, dtype=torch.float32), torch.ones(edge_feature_dim, dtype=torch.float32)

    mean = feature_sum / total_edges
    var = (feature_sumsq / total_edges) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-8))
    return mean.to(torch.float32), std.to(torch.float32)


def compute_single_label_class_weights(records: list[dict], field_name: str, num_classes: int) -> torch.Tensor:
    counter = Counter()
    total = 0
    for record in records:
        labels = record[field_name]
        counter.update(int(label) for label in labels)
        total += len(labels)

    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        count = counter.get(class_id, 0)
        if count > 0:
            weights[class_id] = total / (num_classes * count)
    return weights


def compute_multilabel_pos_weights(records: list[dict], field_name: str, target_ids: list[int]) -> torch.Tensor:
    if not target_ids:
        return torch.zeros(0, dtype=torch.float32)

    target_index = {label_id: idx for idx, label_id in enumerate(target_ids)}
    positives = torch.zeros(len(target_ids), dtype=torch.float32)
    total_nodes = 0
    for record in records:
        total_nodes += len(record[field_name])
        for label_ids in record[field_name]:
            for label_id in label_ids:
                idx = target_index.get(int(label_id))
                if idx is not None:
                    positives[idx] += 1.0

    negatives = max(total_nodes, 1) - positives
    pos_weights = torch.ones(len(target_ids), dtype=torch.float32)
    mask = positives > 0
    pos_weights[mask] = negatives[mask] / positives[mask]
    return pos_weights


class GraphBatchCollator:
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        derived_target_ids: list[int],
        idealization_target_ids: list[int],
        edge_feature_mean: torch.Tensor | None = None,
        edge_feature_std: torch.Tensor | None = None,
    ):
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.edge_feature_mean = edge_feature_mean
        self.edge_feature_std = edge_feature_std
        self.derived_target_index = {label_id: idx for idx, label_id in enumerate(derived_target_ids)}
        self.idealization_target_index = {label_id: idx for idx, label_id in enumerate(idealization_target_ids)}

    def _build_multihot(self, rows: list[list[int]], target_index: dict[int, int]) -> torch.Tensor:
        if not target_index:
            return torch.zeros((len(rows), 0), dtype=torch.float32)

        tensor = torch.zeros((len(rows), len(target_index)), dtype=torch.float32)
        for row_idx, label_ids in enumerate(rows):
            for label_id in label_ids:
                target_col = target_index.get(int(label_id))
                if target_col is not None:
                    tensor[row_idx, target_col] = 1.0
        return tensor

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        sample_ids = []
        node_features = []
        primitive_surface_ids = []
        macro_feature_ids = []
        semantic_subtype_ids = []
        instance_ids = []
        derived_targets = []
        idealization_targets = []
        edge_tensors = []
        edge_features = []
        edge_same_instance_labels = []
        edge_supervision_mask = []

        node_offset = 0
        for record in batch:
            sample_ids.append(record["sample_id"])

            features = torch.tensor(record["node_features"], dtype=torch.float32)
            features = (features - self.feature_mean) / self.feature_std
            node_features.append(features)

            primitive_surface_ids.append(torch.tensor(record["primitive_surface_ids"], dtype=torch.long))
            macro_feature_ids.append(torch.tensor(record["macro_feature_ids"], dtype=torch.long))
            semantic_subtype_ids.append(torch.tensor(record["semantic_subtype_ids"], dtype=torch.long))
            instance_ids.append(torch.tensor(record["instance_ids"], dtype=torch.long))
            derived_targets.append(
                self._build_multihot(record["derived_macro_feature_ids_per_node"], self.derived_target_index)
            )
            idealization_targets.append(
                self._build_multihot(record["idealization_tag_ids_per_node"], self.idealization_target_index)
            )

            if record["edge_index"]:
                edge_index = torch.tensor(record["edge_index"], dtype=torch.long).t().contiguous()
                edge_tensors.append(edge_index + node_offset)
                record_edge_features = torch.tensor(record.get("edge_features", []), dtype=torch.float32)
                if record_edge_features.numel() == 0:
                    edge_feature_dim = int(self.edge_feature_mean.numel()) if self.edge_feature_mean is not None else 0
                    record_edge_features = torch.zeros((edge_index.size(1), edge_feature_dim), dtype=torch.float32)
                if self.edge_feature_mean is not None and self.edge_feature_std is not None and record_edge_features.numel() > 0:
                    record_edge_features = (record_edge_features - self.edge_feature_mean) / self.edge_feature_std
                edge_features.append(record_edge_features)
                edge_same_instance_labels.append(torch.tensor(record.get("edge_same_instance_labels", []), dtype=torch.float32))
                edge_supervision_mask.append(torch.tensor(record.get("edge_supervision_mask", []), dtype=torch.float32))

            node_offset += record["num_nodes"]

        batch_edge_index = torch.cat(edge_tensors, dim=1) if edge_tensors else torch.zeros((2, 0), dtype=torch.long)
        edge_feature_dim = int(self.edge_feature_mean.numel()) if self.edge_feature_mean is not None else 0
        return {
            "sample_ids": sample_ids,
            "node_features": torch.cat(node_features, dim=0),
            "primitive_surface_ids": torch.cat(primitive_surface_ids, dim=0),
            "macro_feature_ids": torch.cat(macro_feature_ids, dim=0),
            "semantic_subtype_ids": torch.cat(semantic_subtype_ids, dim=0),
            "derived_targets": torch.cat(derived_targets, dim=0) if derived_targets else torch.zeros((0, 0)),
            "idealization_targets": torch.cat(idealization_targets, dim=0) if idealization_targets else torch.zeros((0, 0)),
            "instance_ids": torch.cat(instance_ids, dim=0),
            "edge_index": batch_edge_index,
            "edge_features": torch.cat(edge_features, dim=0) if edge_features else torch.zeros((0, edge_feature_dim), dtype=torch.float32),
            "edge_same_instance_labels": torch.cat(edge_same_instance_labels, dim=0) if edge_same_instance_labels else torch.zeros(0, dtype=torch.float32),
            "edge_supervision_mask": torch.cat(edge_supervision_mask, dim=0) if edge_supervision_mask else torch.zeros(0, dtype=torch.float32),
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_single_label_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, float]:
    predictions = predictions.cpu()
    targets = targets.cpu()
    accuracy = (predictions == targets).float().mean().item()

    active_classes = sorted(set(predictions.tolist()) | set(targets.tolist()))
    class_f1_scores = []
    for class_id in active_classes:
        tp = int(((predictions == class_id) & (targets == class_id)).sum().item())
        fp = int(((predictions == class_id) & (targets != class_id)).sum().item())
        fn = int(((predictions != class_id) & (targets == class_id)).sum().item())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        class_f1_scores.append(_safe_divide(2.0 * precision * recall, precision + recall))

    return {
        "accuracy": accuracy,
        "macro_f1_active": sum(class_f1_scores) / max(len(class_f1_scores), 1),
        "active_class_count": float(len(active_classes)),
        "num_classes": float(num_classes),
    }


def compute_multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0 or targets.numel() == 0:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1_active": 0.0,
            "exact_match": 1.0,
            "active_label_count": 0.0,
        }

    predictions = (torch.sigmoid(logits).cpu() >= 0.5).to(torch.int64)
    targets = targets.cpu().to(torch.int64)

    tp = int(((predictions == 1) & (targets == 1)).sum().item())
    fp = int(((predictions == 1) & (targets == 0)).sum().item())
    fn = int(((predictions == 0) & (targets == 1)).sum().item())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    micro_f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    class_f1_scores = []
    for class_idx in range(targets.size(1)):
        class_pred = predictions[:, class_idx]
        class_target = targets[:, class_idx]
        if int(class_pred.sum().item()) == 0 and int(class_target.sum().item()) == 0:
            continue
        class_tp = int(((class_pred == 1) & (class_target == 1)).sum().item())
        class_fp = int(((class_pred == 1) & (class_target == 0)).sum().item())
        class_fn = int(((class_pred == 0) & (class_target == 1)).sum().item())
        class_precision = _safe_divide(class_tp, class_tp + class_fp)
        class_recall = _safe_divide(class_tp, class_tp + class_fn)
        class_f1_scores.append(_safe_divide(2.0 * class_precision * class_recall, class_precision + class_recall))

    exact_match = (predictions == targets).all(dim=1).float().mean().item()
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1_active": sum(class_f1_scores) / max(len(class_f1_scores), 1),
        "exact_match": exact_match,
        "active_label_count": float(len(class_f1_scores)),
    }


def compute_binary_edge_metrics(
    logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> dict[str, Any]:
    edge_metrics = compute_edge_same_instance_metrics(
        scores=logits,
        labels=targets,
        mask=masks,
        threshold=0.5,
        scores_are_logits=True,
    )
    return {
        "edge_supervised_count": float(edge_metrics["valid_edge_count"]),
        "edge_accuracy": float(edge_metrics["accuracy"]),
        "edge_micro_precision": float(edge_metrics["precision"]),
        "edge_micro_recall": float(edge_metrics["recall"]),
        "edge_micro_f1": float(edge_metrics["f1"]),
        "edge_threshold": float(edge_metrics["threshold"]),
        "edge_positive_count": float(edge_metrics["positive_edge_count"]),
        "edge_negative_count": float(edge_metrics["negative_edge_count"]),
        "edge_tp": float(edge_metrics["tp"]),
        "edge_fp": float(edge_metrics["fp"]),
        "edge_tn": float(edge_metrics["tn"]),
        "edge_fn": float(edge_metrics["fn"]),
        "edge_roc_auc": edge_metrics["roc_auc"],
        "edge_pr_auc": edge_metrics["pr_auc"],
        "edge_auc_available": bool(edge_metrics["auc_available"]),
        "edge_auc_reason": edge_metrics["auc_reason"],
        "edge_threshold_scan": edge_metrics["threshold_scan"],
        "edge_best_f1_threshold": edge_metrics["best_f1_threshold"],
        "edge_best_f1": float(edge_metrics["best_f1"]),
    }


def extract_edge_same_instance_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "supervised_count": float(metrics.get("edge_supervised_count", 0.0)),
        "positive_count": float(metrics.get("edge_positive_count", 0.0)),
        "negative_count": float(metrics.get("edge_negative_count", 0.0)),
        "threshold": metrics.get("edge_threshold", 0.5),
        "accuracy": float(metrics.get("edge_accuracy", 0.0)),
        "precision": float(metrics.get("edge_micro_precision", 0.0)),
        "recall": float(metrics.get("edge_micro_recall", 0.0)),
        "micro_f1": float(metrics.get("edge_micro_f1", 0.0)),
        "tp": float(metrics.get("edge_tp", 0.0)),
        "fp": float(metrics.get("edge_fp", 0.0)),
        "tn": float(metrics.get("edge_tn", 0.0)),
        "fn": float(metrics.get("edge_fn", 0.0)),
        "roc_auc": metrics.get("edge_roc_auc"),
        "pr_auc": metrics.get("edge_pr_auc"),
        "auc_available": bool(metrics.get("edge_auc_available", False)),
        "auc_reason": metrics.get("edge_auc_reason"),
        "threshold_scan": metrics.get("edge_threshold_scan", []),
        "best_f1_threshold": metrics.get("edge_best_f1_threshold"),
        "best_f1": float(metrics.get("edge_best_f1", 0.0)),
    }


def build_edge_same_instance_metrics_report(
    *,
    best_epoch: int,
    best_val_selection_score: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    split_metrics = {
        "train": extract_edge_same_instance_metrics(train_metrics),
        "val": extract_edge_same_instance_metrics(val_metrics),
        "test": extract_edge_same_instance_metrics(test_metrics),
    }
    val_edge = split_metrics["val"]
    return {
        "schema": "edge_same_instance_metrics_v1",
        "best_epoch": best_epoch,
        "best_val_selection_score": best_val_selection_score,
        "threshold_selection": {
            "source_split": "val",
            "best_f1_threshold": val_edge["best_f1_threshold"],
            "best_f1": val_edge["best_f1"],
        },
        "splits": split_metrics,
    }


def compute_selection_score(
    metrics: dict[str, float],
    *,
    use_derived: bool,
    use_idealization: bool,
    use_edge: bool,
) -> float:
    score_terms = [
        metrics["macro_macro_f1_active"],
        metrics["semantic_macro_f1_active"],
    ]
    if use_derived:
        score_terms.append(metrics["derived_micro_f1"])
    if use_idealization:
        score_terms.append(metrics["idealization_micro_f1"])
    if use_edge:
        score_terms.append(metrics["edge_micro_f1"])
    return sum(score_terms) / max(len(score_terms), 1)


def run_epoch(
    model: MultiHeadGraphSAGE,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    macro_loss_fn: nn.Module,
    semantic_loss_fn: nn.Module,
    derived_loss_fn: nn.Module | None,
    idealization_loss_fn: nn.Module | None,
    edge_loss_fn: nn.Module | None,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_nodes = 0
    macro_predictions = []
    macro_targets = []
    semantic_predictions = []
    semantic_targets = []
    derived_logits_all = []
    derived_targets_all = []
    idealization_logits_all = []
    idealization_targets_all = []
    edge_logits_all = []
    edge_targets_all = []
    edge_masks_all = []

    for batch in loader:
        node_features = batch["node_features"].to(device)
        primitive_surface_ids = batch["primitive_surface_ids"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_features = batch["edge_features"].to(device)
        macro_feature_ids = batch["macro_feature_ids"].to(device)
        semantic_subtype_ids = batch["semantic_subtype_ids"].to(device)
        derived_targets = batch["derived_targets"].to(device)
        idealization_targets = batch["idealization_targets"].to(device)

        outputs = model(node_features, edge_index, primitive_surface_ids, edge_features=edge_features)
        macro_loss = macro_loss_fn(outputs["macro_logits"], macro_feature_ids)
        semantic_loss = semantic_loss_fn(outputs["semantic_logits"], semantic_subtype_ids)

        loss = (
            loss_weights["macro"] * macro_loss
            + loss_weights["semantic"] * semantic_loss
        )
        if derived_loss_fn is not None:
            derived_loss = derived_loss_fn(outputs["derived_logits"], derived_targets)
            loss = loss + loss_weights["derived"] * derived_loss
        if idealization_loss_fn is not None:
            idealization_loss = idealization_loss_fn(outputs["idealization_logits"], idealization_targets)
            loss = loss + loss_weights["idealization"] * idealization_loss
        if edge_loss_fn is not None and "edge_same_instance_logits" in outputs:
            edge_targets = batch["edge_same_instance_labels"].to(device)
            edge_mask = batch["edge_supervision_mask"].to(device)
            if edge_mask.numel() > 0 and float(edge_mask.sum().item()) > 0.0:
                raw_edge_loss = edge_loss_fn(outputs["edge_same_instance_logits"], edge_targets.clamp_min(0.0))
                edge_loss = (raw_edge_loss * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
                loss = loss + loss_weights["edge"] * edge_loss

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_node_count = node_features.size(0)
        total_loss += float(loss.item()) * batch_node_count
        total_nodes += batch_node_count

        macro_predictions.append(outputs["macro_logits"].argmax(dim=-1).detach().cpu())
        macro_targets.append(macro_feature_ids.detach().cpu())
        semantic_predictions.append(outputs["semantic_logits"].argmax(dim=-1).detach().cpu())
        semantic_targets.append(semantic_subtype_ids.detach().cpu())
        if "derived_logits" in outputs:
            derived_logits_all.append(outputs["derived_logits"].detach().cpu())
            derived_targets_all.append(derived_targets.detach().cpu())
        if "idealization_logits" in outputs:
            idealization_logits_all.append(outputs["idealization_logits"].detach().cpu())
            idealization_targets_all.append(idealization_targets.detach().cpu())
        if "edge_same_instance_logits" in outputs:
            edge_logits_all.append(outputs["edge_same_instance_logits"].detach().cpu())
            edge_targets_all.append(batch["edge_same_instance_labels"].detach().cpu())
            edge_masks_all.append(batch["edge_supervision_mask"].detach().cpu())

    metrics = {
        "loss": total_loss / max(total_nodes, 1),
    }
    macro_predictions = torch.cat(macro_predictions, dim=0)
    macro_targets = torch.cat(macro_targets, dim=0)
    semantic_predictions = torch.cat(semantic_predictions, dim=0)
    semantic_targets = torch.cat(semantic_targets, dim=0)
    metrics.update({f"macro_{k}": v for k, v in compute_single_label_metrics(macro_predictions, macro_targets, int(macro_targets.max().item()) + 1).items()})
    metrics.update({f"semantic_{k}": v for k, v in compute_single_label_metrics(semantic_predictions, semantic_targets, int(semantic_targets.max().item()) + 1).items()})

    if derived_logits_all:
        derived_logits = torch.cat(derived_logits_all, dim=0)
        derived_targets = torch.cat(derived_targets_all, dim=0)
        metrics.update({f"derived_{k}": v for k, v in compute_multilabel_metrics(derived_logits, derived_targets).items()})
    else:
        metrics.update({f"derived_{k}": v for k, v in compute_multilabel_metrics(torch.zeros((0, 0)), torch.zeros((0, 0))).items()})

    if idealization_logits_all:
        idealization_logits = torch.cat(idealization_logits_all, dim=0)
        idealization_targets = torch.cat(idealization_targets_all, dim=0)
        metrics.update({f"idealization_{k}": v for k, v in compute_multilabel_metrics(idealization_logits, idealization_targets).items()})
    else:
        metrics.update({f"idealization_{k}": v for k, v in compute_multilabel_metrics(torch.zeros((0, 0)), torch.zeros((0, 0))).items()})

    metrics.update(compute_binary_edge_metrics(edge_logits_all, edge_targets_all, edge_masks_all))

    metrics["selection_score"] = compute_selection_score(
        metrics,
        use_derived=derived_loss_fn is not None,
        use_idealization=idealization_loss_fn is not None,
        use_edge=edge_loss_fn is not None,
    )
    return metrics


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_records = load_jsonl(args.training_dir / "train.jsonl")
    val_records = load_jsonl(args.training_dir / "val.jsonl")
    test_records = load_jsonl(args.training_dir / "test.jsonl")
    all_records = train_records + val_records + test_records
    if not train_records:
        raise RuntimeError("No training samples found.")
    training_metadata = load_training_metadata(args.training_dir)

    _, primitive_id_to_key, primitive_key_to_id = load_label_map(args.label_map_dir / "primitive_surface_label_map.json")
    _, macro_id_to_key, _ = load_label_map(args.label_map_dir / "macro_feature_label_map.json")
    _, semantic_id_to_key, _ = load_label_map(args.label_map_dir / "face_semantic_subtype_label_map.json")
    _, idealization_id_to_key, _ = load_label_map(args.label_map_dir / "idealization_tag_label_map.json")

    derived_target_ids = infer_multilabel_target_ids(all_records, "derived_macro_feature_ids_per_node")
    idealization_target_ids = infer_multilabel_target_ids(all_records, "idealization_tag_ids_per_node")

    feature_dim = len(train_records[0]["node_features"][0])
    node_feature_mode = (
        training_metadata.get("node_feature_mode")
        or train_records[0].get("node_feature_mode")
        or DEFAULT_NODE_FEATURE_MODE
    )
    node_feature_schema = (
        training_metadata.get("node_feature_schema")
        or train_records[0].get("node_feature_schema")
        or node_feature_schema_for_mode(node_feature_mode)
    )
    if len(node_feature_schema) != feature_dim:
        node_feature_schema = [f"node_feature_{idx}" for idx in range(feature_dim)]
    feature_mean, feature_std = compute_feature_normalization(train_records, feature_dim)
    edge_feature_dim = 0
    for record in train_records:
        if record.get("edge_features"):
            edge_feature_dim = len(record["edge_features"][0])
            break
    edge_feature_mean, edge_feature_std = compute_edge_feature_normalization(train_records, edge_feature_dim) if edge_feature_dim > 0 else (None, None)

    macro_class_weights = compute_single_label_class_weights(train_records, "macro_feature_ids", len(macro_id_to_key))
    semantic_class_weights = compute_single_label_class_weights(train_records, "semantic_subtype_ids", len(semantic_id_to_key))
    derived_pos_weights = compute_multilabel_pos_weights(train_records, "derived_macro_feature_ids_per_node", derived_target_ids)
    idealization_pos_weights = compute_multilabel_pos_weights(train_records, "idealization_tag_ids_per_node", idealization_target_ids)

    collator = GraphBatchCollator(
        feature_mean=feature_mean,
        feature_std=feature_std,
        derived_target_ids=derived_target_ids,
        idealization_target_ids=idealization_target_ids,
        edge_feature_mean=edge_feature_mean,
        edge_feature_std=edge_feature_std,
    )

    train_loader = DataLoader(
        GraphRecordDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        GraphRecordDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        GraphRecordDataset(test_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    if args.model_kind == "edge_aware":
        if edge_feature_dim <= 0:
            raise RuntimeError("edge_aware model requires edge_features in training records.")
        model = EdgeAwareMultiHeadGNN(
            input_dim=feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_primitive_surfaces=len(primitive_id_to_key),
            num_macro_classes=len(macro_id_to_key),
            num_semantic_classes=len(semantic_id_to_key),
            derived_target_dim=len(derived_target_ids),
            idealization_target_dim=len(idealization_target_ids),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            primitive_embedding_dim=args.primitive_embedding_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        model = MultiHeadGraphSAGE(
            input_dim=feature_dim,
            num_primitive_surfaces=len(primitive_id_to_key),
            num_macro_classes=len(macro_id_to_key),
            num_semantic_classes=len(semantic_id_to_key),
            derived_target_dim=len(derived_target_ids),
            idealization_target_dim=len(idealization_target_ids),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            primitive_embedding_dim=args.primitive_embedding_dim,
            dropout=args.dropout,
        ).to(device)

    init_checkpoint_report = None
    if args.init_checkpoint is not None:
        init_checkpoint_report = load_initial_checkpoint(model, args.init_checkpoint, device)
        print(
            "init_checkpoint_loaded: "
            f"{args.init_checkpoint} "
            f"loaded={init_checkpoint_report['loaded_tensor_count']} "
            f"expanded={init_checkpoint_report['expanded_tensor_count']} "
            f"skipped={init_checkpoint_report['skipped_tensor_count']}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    macro_loss_fn = nn.CrossEntropyLoss(weight=macro_class_weights.to(device))
    semantic_loss_fn = nn.CrossEntropyLoss(weight=semantic_class_weights.to(device))
    derived_loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=derived_pos_weights.to(device))
        if len(derived_target_ids) > 0
        else None
    )
    idealization_loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=idealization_pos_weights.to(device))
        if len(idealization_target_ids) > 0
        else None
    )
    edge_loss_fn = nn.BCEWithLogitsLoss(reduction="none") if args.model_kind == "edge_aware" else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"gnn_baseline_{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_kind": args.model_kind,
        "node_feature_mode": node_feature_mode,
        "node_feature_schema": node_feature_schema,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "primitive_embedding_dim": args.primitive_embedding_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_weights": {
            "macro": args.macro_loss_weight,
            "semantic": args.semantic_loss_weight,
            "derived": args.derived_loss_weight,
            "idealization": args.idealization_loss_weight,
            "edge": args.edge_loss_weight,
        },
        "label_spaces": {
            "primitive_surface_id_to_key": primitive_id_to_key,
            "macro_feature_id_to_key": macro_id_to_key,
            "semantic_subtype_id_to_key": semantic_id_to_key,
            "idealization_tag_id_to_key": idealization_id_to_key,
            "derived_target_ids": derived_target_ids,
            "idealization_target_ids": idealization_target_ids,
        },
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "edge_feature_mean": edge_feature_mean.tolist() if edge_feature_mean is not None else [],
            "edge_feature_std": edge_feature_std.tolist() if edge_feature_std is not None else [],
        },
        "edge_feature_dim": edge_feature_dim,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint is not None else None,
        "init_checkpoint_report": init_checkpoint_report,
    }
    save_json(run_dir / "config.json", run_config)

    history = []
    best_score = float("-inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            macro_loss_fn=macro_loss_fn,
            semantic_loss_fn=semantic_loss_fn,
            derived_loss_fn=derived_loss_fn,
            idealization_loss_fn=idealization_loss_fn,
            edge_loss_fn=edge_loss_fn,
            loss_weights=run_config["loss_weights"],
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                macro_loss_fn=macro_loss_fn,
                semantic_loss_fn=semantic_loss_fn,
                derived_loss_fn=derived_loss_fn,
                idealization_loss_fn=idealization_loss_fn,
                edge_loss_fn=edge_loss_fn,
                loss_weights=run_config["loss_weights"],
            )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_macro_f1_active']:.4f} "
            f"val_semantic_f1={val_metrics['semantic_macro_f1_active']:.4f} "
            f"val_derived_f1={val_metrics['derived_micro_f1']:.4f} "
            f"val_idealization_f1={val_metrics['idealization_micro_f1']:.4f} "
            f"val_edge_f1={val_metrics['edge_micro_f1']:.4f} "
            f"val_score={val_metrics['selection_score']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "best.pt")

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            macro_loss_fn=macro_loss_fn,
            semantic_loss_fn=semantic_loss_fn,
            derived_loss_fn=derived_loss_fn,
            idealization_loss_fn=idealization_loss_fn,
            edge_loss_fn=edge_loss_fn,
            loss_weights=run_config["loss_weights"],
        )

    best_train_metrics = best_checkpoint.get("train_metrics", {})
    best_val_metrics = best_checkpoint.get("val_metrics", {})
    if not best_train_metrics and 0 < best_epoch <= len(history):
        best_train_metrics = history[best_epoch - 1]["train"]
    if not best_val_metrics and 0 < best_epoch <= len(history):
        best_val_metrics = history[best_epoch - 1]["val"]

    edge_same_instance_report = build_edge_same_instance_metrics_report(
        best_epoch=best_epoch,
        best_val_selection_score=float(best_score),
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
    )
    save_json(run_dir / "edge_same_instance_metrics.json", edge_same_instance_report)
    summary = {
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "final_test_metrics": test_metrics,
        "history": history,
    }
    save_json(run_dir / "metrics.json", summary)
    print(f"training_complete: {run_dir}")
    print(f"best_epoch: {best_epoch}")
    print(f"test_macro_f1: {test_metrics['macro_macro_f1_active']:.4f}")
    print(f"test_semantic_f1: {test_metrics['semantic_macro_f1_active']:.4f}")
    print(f"test_derived_micro_f1: {test_metrics['derived_micro_f1']:.4f}")
    print(f"test_idealization_micro_f1: {test_metrics['idealization_micro_f1']:.4f}")
    print(f"test_edge_micro_f1: {test_metrics['edge_micro_f1']:.4f}")


if __name__ == "__main__":
    main()
