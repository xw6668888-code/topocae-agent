from __future__ import annotations

"""Train one P1-B face/same-instance GNN ablation with Synthetic-Val selection."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.gnn_baseline import EdgeAwareMultiHeadGNN, MultiHeadGraphSAGE
from utils.p1_b_instance_assembly import select_global_same_instance_threshold


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2"
OUTPUT_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"
RUN_KINDS = ("node_only", "edge_attr", "same_instance")


class Records(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = OUTPUT_ROOT / f"{args.run_kind}_seed_{args.seed}"
    summary_path = run_dir / "metrics.json"
    if summary_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "metrics": str(summary_path)}, ensure_ascii=False))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metadata = _read_json(PACKAGE / "metadata.json")
    split_records = {
        split: torch.load(PACKAGE / f"{split}.pt", map_location="cpu", weights_only=False)
        for split in ("train", "val", "development_test")
    }
    macro_names = list(metadata["macro_classes"])
    dims = {
        "node": len(metadata["node_feature_schema"]),
        "edge": len(metadata["edge_feature_schema"]),
        "primitive": int(metadata["primitive_surface_class_count"]),
        "macro": int(metadata["macro_class_count"]),
        "semantic": int(metadata["semantic_class_count"]),
    }
    use_edge_attr = args.run_kind != "node_only"
    train_same_instance = args.run_kind == "same_instance"
    model = _build_model(args, dims).to(device)
    macro_weights = _class_weights(split_records["train"], "macro_targets", dims["macro"]).to(device)
    semantic_weights = _class_weights(split_records["train"], "semantic_targets", dims["semantic"]).to(device)
    edge_pos, edge_neg = _edge_counts(split_records["train"])
    edge_pos_weight = torch.tensor([edge_neg / max(edge_pos, 1)], dtype=torch.float32, device=device)
    macro_loss = nn.CrossEntropyLoss(weight=macro_weights, label_smoothing=0.01)
    semantic_loss = nn.CrossEntropyLoss(weight=semantic_weights, label_smoothing=0.01)
    edge_loss = nn.BCEWithLogitsLoss(pos_weight=edge_pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.04)
    loaders = {
        split: DataLoader(
            Records(records), batch_size=args.batch_size, shuffle=split == "train",
            num_workers=args.num_workers, collate_fn=_collate, pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + {"train": 0, "val": 1, "development_test": 2}[split]),
        )
        for split, records in split_records.items()
    }
    config = {
        "schema_version": "topocae_p1_b_gnn_run_config_1",
        "run_kind": args.run_kind,
        "seed": args.seed,
        "model": {
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "dropout": args.dropout,
            "dimensions": dims, "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "optimization": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "loss_weights": {"macro": 1.0, "semantic": 0.55, "same_instance": 0.65 if train_same_instance else 0.0},
            "edge_positive_count": edge_pos, "edge_negative_count": edge_neg,
            "edge_pos_weight": float(edge_pos_weight.item()),
        },
        "data": {
            "package": str(PACKAGE.resolve()), "package_manifest_sha256": _sha256(PACKAGE / "manifest.json"),
            "split_counts": {split: len(records) for split, records in split_records.items()},
            "normalization_fitted_split": metadata["normalization"]["fitted_split"],
        },
        "selection_contract": {
            "early_stopping": "Synthetic-Val only", "same_instance_threshold": "Synthetic-Val only/global",
            "real_data_used": False, "holdout_used": False,
        },
        "environment": _environment(device),
    }
    _write_json(run_dir / "config.json", config)

    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_result = _run_epoch(
            model, loaders["train"], device, optimizer, macro_loss, semantic_loss, edge_loss,
            use_edge_attr=use_edge_attr, train_same_instance=train_same_instance,
            macro_names=macro_names, semantic_count=dims["semantic"], collect=False,
        )
        scheduler.step()
        with torch.no_grad():
            val_result = _run_epoch(
                model, loaders["val"], device, None, macro_loss, semantic_loss, edge_loss,
                use_edge_attr=use_edge_attr, train_same_instance=train_same_instance,
                macro_names=macro_names, semantic_count=dims["semantic"], collect=False,
            )
        score = _selection_score(val_result["metrics"], train_same_instance)
        row = {
            "epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "selection_score": score,
            "train": train_result["metrics"], "val": val_result["metrics"],
        }
        history.append(row)
        print(
            f"kind={args.run_kind} seed={args.seed} epoch={epoch:03d} "
            f"loss={train_result['metrics']['loss']:.4f}/{val_result['metrics']['loss']:.4f} "
            f"face_f1={val_result['metrics']['face']['macro_f1_all']:.4f} "
            f"semantic_f1={val_result['metrics']['semantic']['macro_f1_active']:.4f} "
            f"edge_f1={val_result['metrics']['same_instance_edge_at_0_5']['f1']:.4f} score={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "schema_version": "topocae_p1_b_gnn_checkpoint_1", "epoch": epoch,
                "model_state_dict": model.state_dict(), "config": config,
                "val_metrics": val_result["metrics"], "selection_score": score,
            }, run_dir / "best.pt")
        else:
            stale += 1
        if stale >= args.patience and epoch >= 20:
            break
    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prediction_payload: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, Any] = {}
    inference_seconds = 0.0
    for split, loader in loaders.items():
        started = time.perf_counter()
        with torch.no_grad():
            result = _run_epoch(
                model, loader, device, None, macro_loss, semantic_loss, edge_loss,
                use_edge_attr=use_edge_attr, train_same_instance=train_same_instance,
                macro_names=macro_names, semantic_count=dims["semantic"], collect=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        inference_seconds += elapsed
        prediction_payload[split] = result["predictions"]
        evaluation[split] = result["metrics"]
        evaluation[split]["inference_seconds"] = elapsed
        evaluation[split]["inference_ms_per_graph"] = elapsed * 1000.0 / max(len(split_records[split]), 1)

    threshold_selection = None
    if train_same_instance:
        threshold_records = _undirected_edge_threshold_records(prediction_payload["val"], split="synthetic_val")
        threshold_selection = select_global_same_instance_threshold(threshold_records)
        threshold = float(threshold_selection["global_threshold"])
        for split in prediction_payload:
            evaluation[split]["same_instance_edge_selected_threshold"] = _edge_metrics(
                _undirected_edge_values(prediction_payload[split]), threshold
            )
    consistency = {
        split: _consistency_metrics(records) for split, records in prediction_payload.items()
    }
    torch.save(prediction_payload, run_dir / "predictions.pt")
    _write_json(run_dir / "history.json", history)
    if threshold_selection is not None:
        _write_json(run_dir / "same_instance_threshold_selection.json", threshold_selection)
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metrics = {
        "schema_version": "topocae_p1_b_gnn_metrics_1",
        "run_kind": args.run_kind, "seed": args.seed, "best_epoch": best_epoch,
        "best_val_selection_score": best_score, "evaluation": evaluation,
        "invariance": consistency,
        "same_instance_threshold_selection": threshold_selection,
        "resource": {
            "parameter_count": config["model"]["parameter_count"], "training_seconds": training_seconds,
            "total_inference_seconds": inference_seconds, "peak_vram_bytes": peak_vram,
        },
        "contracts": {
            "model_selected_on_synthetic_val_only": True,
            "threshold_selected_on_synthetic_val_only": threshold_selection is None or threshold_selection["selection_split"] == "synthetic_val",
            "real_data_used": False, "holdout_used": False,
        },
    }
    _write_json(summary_path, metrics)
    _write_epoch_csv(run_dir / "history.csv", history)
    print(json.dumps({
        "training_complete": str(run_dir), "best_epoch": best_epoch,
        "val_face_macro_f1": evaluation["val"]["face"]["macro_f1_all"],
        "val_semantic_macro_f1": evaluation["val"]["semantic"]["macro_f1_active"],
        "val_edge_f1": evaluation["val"].get("same_instance_edge_selected_threshold", {}).get("f1"),
        "rotation_consistency": consistency["val"]["rotation_consistency"],
        "scale_consistency": consistency["val"]["scale_consistency"],
    }, ensure_ascii=False), flush=True)
    return 0


def _build_model(args: argparse.Namespace, dims: dict[str, int]) -> nn.Module:
    common = dict(
        input_dim=dims["node"], num_primitive_surfaces=dims["primitive"],
        num_macro_classes=dims["macro"], num_semantic_classes=dims["semantic"],
        derived_target_dim=0, idealization_target_dim=0, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, primitive_embedding_dim=16, dropout=args.dropout,
    )
    if args.run_kind == "node_only":
        return MultiHeadGraphSAGE(**common)
    return EdgeAwareMultiHeadGNN(edge_feature_dim=dims["edge"], **common)


def _collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    node_features, primitive, macro, semantic, instances = [], [], [], [], []
    edges, edge_features, edge_targets, edge_masks = [], [], [], []
    node_slices, edge_slices = [], []
    node_offset = edge_offset = 0
    for record in records:
        node_count = int(record["node_features"].shape[0])
        edge_count = int(record["edge_index"].shape[1])
        node_features.append(record["node_features"])
        primitive.append(record["primitive_surface_ids"])
        macro.append(record["macro_targets"])
        semantic.append(record["semantic_targets"])
        instances.append(record["instance_ids"])
        edges.append(record["edge_index"] + node_offset)
        edge_features.append(record["edge_features"])
        edge_targets.append(record["edge_same_instance_targets"])
        edge_masks.append(record["edge_supervision_mask"])
        node_slices.append((node_offset, node_offset + node_count))
        edge_slices.append((edge_offset, edge_offset + edge_count))
        node_offset += node_count
        edge_offset += edge_count
    return {
        "records": records, "node_slices": node_slices, "edge_slices": edge_slices,
        "node_features": torch.cat(node_features), "primitive": torch.cat(primitive),
        "macro": torch.cat(macro), "semantic": torch.cat(semantic), "instances": torch.cat(instances),
        "edge_index": torch.cat(edges, dim=1), "edge_features": torch.cat(edge_features),
        "edge_targets": torch.cat(edge_targets), "edge_masks": torch.cat(edge_masks),
    }


def _run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None,
    macro_loss_fn: nn.Module, semantic_loss_fn: nn.Module, edge_loss_fn: nn.Module,
    *, use_edge_attr: bool, train_same_instance: bool, macro_names: list[str], semantic_count: int,
    collect: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_nodes = 0
    macro_preds, macro_truth = [], []
    semantic_preds, semantic_truth = [], []
    edge_probs, edge_truth, edge_masks = [], [], []
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        node = batch["node_features"].to(device, non_blocking=True)
        primitive = batch["primitive"].to(device, non_blocking=True)
        edge_index = batch["edge_index"].to(device, non_blocking=True)
        edge_features = batch["edge_features"].to(device, non_blocking=True)
        macro_target = batch["macro"].to(device, non_blocking=True)
        semantic_target = batch["semantic"].to(device, non_blocking=True)
        outputs = model(node, edge_index, primitive, edge_features=edge_features if use_edge_attr else None)
        loss = macro_loss_fn(outputs["macro_logits"], macro_target) + 0.55 * semantic_loss_fn(outputs["semantic_logits"], semantic_target)
        if train_same_instance:
            target = batch["edge_targets"].to(device, non_blocking=True)
            mask = batch["edge_masks"].to(device, non_blocking=True)
            raw = edge_loss_fn(outputs["edge_same_instance_logits"], target.clamp_min(0.0))
            loss = loss + 0.65 * (raw[mask].mean() if bool(mask.any()) else raw.sum() * 0.0)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach().item()) * node.shape[0]
        total_nodes += node.shape[0]
        macro_prediction = outputs["macro_logits"].argmax(dim=1).detach().cpu()
        semantic_prediction = outputs["semantic_logits"].argmax(dim=1).detach().cpu()
        macro_preds.append(macro_prediction)
        macro_truth.append(batch["macro"])
        semantic_preds.append(semantic_prediction)
        semantic_truth.append(batch["semantic"])
        if "edge_same_instance_logits" in outputs:
            probability = torch.sigmoid(outputs["edge_same_instance_logits"]).detach().cpu()
            edge_probs.append(probability)
            edge_truth.append(batch["edge_targets"])
            edge_masks.append(batch["edge_masks"])
        else:
            probability = torch.empty(0)
        if collect:
            embeddings = outputs["node_embeddings"].detach().cpu().to(torch.float16)
            for record, node_slice, edge_slice in zip(batch["records"], batch["node_slices"], batch["edge_slices"]):
                ns, ne = node_slice
                es, ee = edge_slice
                predictions.append({
                    "sample_id": record["sample_id"],
                    "source_graph_path": record["source_graph_path"],
                    "base_geometry_family_id": record["base_geometry_family_id"],
                    "derivation_kind": record["derivation_kind"],
                    "macro_prediction": macro_prediction[ns:ne],
                    "macro_target": batch["macro"][ns:ne],
                    "semantic_prediction": semantic_prediction[ns:ne],
                    "semantic_target": batch["semantic"][ns:ne],
                    "instance_ids": batch["instances"][ns:ne],
                    "node_embeddings": embeddings[ns:ne],
                    "edge_index": record["edge_index"],
                    "edge_probability": probability[es:ee] if probability.numel() else torch.empty(0),
                    "edge_target": batch["edge_targets"][es:ee],
                    "edge_mask": batch["edge_masks"][es:ee],
                    "edge_features": record["edge_features"].to(torch.float16),
                    "feature_sets": record["feature_sets"],
                    "face_to_canonical_patch": record["face_to_canonical_patch"],
                })
    macro_preds_t, macro_truth_t = torch.cat(macro_preds), torch.cat(macro_truth)
    semantic_preds_t, semantic_truth_t = torch.cat(semantic_preds), torch.cat(semantic_truth)
    if edge_probs:
        probs, truth, masks = torch.cat(edge_probs), torch.cat(edge_truth), torch.cat(edge_masks).bool()
        edge_at_half = _binary_metrics(probs[masks], truth[masks].long(), 0.5)
    else:
        edge_at_half = _empty_binary_metrics()
    return {
        "metrics": {
            "loss": total_loss / max(total_nodes, 1),
            "face": _classification_metrics(macro_preds_t, macro_truth_t, macro_names),
            "semantic": _classification_metrics(semantic_preds_t, semantic_truth_t, [str(i) for i in range(semantic_count)]),
            "same_instance_edge_at_0_5": edge_at_half,
        },
        "predictions": predictions,
    }


def _class_weights(records: list[dict[str, Any]], key: str, count: int) -> torch.Tensor:
    values = Counter(int(value) for record in records for value in record[key].tolist())
    total = sum(values.values())
    weights = torch.zeros(count, dtype=torch.float32)
    for index, frequency in values.items():
        weights[index] = math.sqrt(total / (len(values) * frequency))
    active = weights > 0
    weights[active] /= weights[active].mean()
    return weights


def _edge_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    positive = negative = 0
    for record in records:
        mask = record["edge_supervision_mask"].bool()
        labels = record["edge_same_instance_targets"][mask]
        positive += int((labels == 1).sum())
        negative += int((labels == 0).sum())
    return positive, negative


def _classification_metrics(prediction: torch.Tensor, truth: torch.Tensor, names: list[str]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    f1s, recalls = [], []
    active_ids = sorted(set(int(value) for value in truth.tolist()))
    for class_id in active_ids:
        target_positive = truth == class_id
        predicted_positive = prediction == class_id
        tp = int((target_positive & predicted_positive).sum())
        fp = int((~target_positive & predicted_positive).sum())
        fn = int((target_positive & ~predicted_positive).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        name = names[class_id] if class_id < len(names) else str(class_id)
        per_class[name] = {"support": int(target_positive.sum()), "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
        recalls.append(recall)
    feature_f1s = [record["f1"] for name, record in per_class.items() if name != "stock"]
    return {
        "accuracy": float((prediction == truth).float().mean()),
        "macro_accuracy_active": sum(recalls) / max(len(recalls), 1),
        "macro_f1_active": sum(f1s) / max(len(f1s), 1),
        "macro_f1_all": sum(f1s) / max(len(active_ids), 1),
        "macro_f1_feature_classes": sum(feature_f1s) / max(len(feature_f1s), 1),
        "active_class_count": len(active_ids), "per_class": per_class,
    }


def _binary_metrics(probability: torch.Tensor, truth: torch.Tensor, threshold: float) -> dict[str, Any]:
    predicted = probability >= threshold
    truth_bool = truth.bool()
    tp = int((predicted & truth_bool).sum())
    fp = int((predicted & ~truth_bool).sum())
    fn = int((~predicted & truth_bool).sum())
    tn = int((~predicted & ~truth_bool).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _empty_binary_metrics() -> dict[str, Any]:
    return {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _undirected_edge_threshold_records(records: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    return [{"split": split, "probability": probability, "label": label} for probability, label in _undirected_edge_values(records)]


def _undirected_edge_values(records: list[dict[str, Any]]) -> list[tuple[float, int]]:
    values: list[tuple[float, int]] = []
    for record in records:
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        instances = record["instance_ids"].tolist()
        for position, pair in enumerate(record["edge_index"].t().tolist()):
            left, right = int(pair[0]), int(pair[1])
            if left == right or record["edge_probability"].numel() == 0:
                continue
            if not bool(record["edge_mask"][position]):
                continue
            grouped[tuple(sorted((left, right)))].append(float(record["edge_probability"][position]))
        for (left, right), probabilities in grouped.items():
            label = int(instances[left] > 0 and instances[left] == instances[right])
            values.append((sum(probabilities) / len(probabilities), label))
    return values


def _edge_metrics(values: Iterable[tuple[float, int]], threshold: float) -> dict[str, Any]:
    values_list = list(values)
    if not values_list:
        return _empty_binary_metrics()
    return _binary_metrics(
        torch.tensor([value[0] for value in values_list]),
        torch.tensor([value[1] for value in values_list]), threshold,
    )


def _consistency_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[str(record["base_geometry_family_id"])].append(record)
    counters = {"rigid_rotation": [0, 0], "uniform_scale": [0, 0], "legal_face_partition": [0, 0]}
    for members in families.values():
        identities = [record for record in members if record["derivation_kind"] == "identity"]
        if not identities:
            continue
        reference = _patch_prediction(identities[0])
        for record in members:
            kind = str(record["derivation_kind"])
            if kind not in counters or record is identities[0]:
                continue
            derived = _patch_prediction(record)
            shared = set(reference) & set(derived)
            counters[kind][0] += sum(reference[patch] == derived[patch] for patch in shared)
            counters[kind][1] += len(shared)
    result = {
        "rotation_consistency": counters["rigid_rotation"][0] / max(counters["rigid_rotation"][1], 1),
        "scale_consistency": counters["uniform_scale"][0] / max(counters["uniform_scale"][1], 1),
        "canonical_patch_partition_consistency": counters["legal_face_partition"][0] / max(counters["legal_face_partition"][1], 1),
        "comparison_counts": {key: {"matching_patches": value[0], "shared_patches": value[1]} for key, value in counters.items()},
    }
    return result


def _patch_prediction(record: dict[str, Any]) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    mapping = record.get("face_to_canonical_patch", {})
    for index, predicted in enumerate(record["macro_prediction"].tolist()):
        patch = str(mapping.get(str(index), mapping.get(index, index)))
        grouped[patch].append(int(predicted))
    return {patch: Counter(values).most_common(1)[0][0] for patch, values in grouped.items()}


def _selection_score(metrics: dict[str, Any], train_same_instance: bool) -> float:
    terms = [metrics["face"]["macro_f1_all"], metrics["semantic"]["macro_f1_active"]]
    if train_same_instance:
        terms.append(metrics["same_instance_edge_at_0_5"]["f1"])
    return sum(terms) / len(terms)


def _environment(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory, "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor]}
    nvidia_smi = None
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "gpu": gpu, "nvidia_smi": nvidia_smi,
        "numpy": np.__version__, "pid": os.getpid(),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_epoch_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lr", "selection_score", "train_loss", "val_loss", "val_face_macro_f1", "val_semantic_macro_f1", "val_edge_f1"])
        writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"], "lr": row["lr"], "selection_score": row["selection_score"],
                "train_loss": row["train"]["loss"], "val_loss": row["val"]["loss"],
                "val_face_macro_f1": row["val"]["face"]["macro_f1_all"],
                "val_semantic_macro_f1": row["val"]["semantic"]["macro_f1_active"],
                "val_edge_f1": row["val"]["same_instance_edge_at_0_5"]["f1"],
            })


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
