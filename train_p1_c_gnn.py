from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_p1_b_gnn_ablation as baseline
from models.gnn_baseline import EdgeAwareMultiHeadGNN
from utils.p1_b_instance_assembly import select_global_same_instance_threshold


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "output_generation_p1_c_clean_context_7200" / "training_packages"
OUTPUT_ROOT = ROOT / "reports" / "p1_c_training_runs_v1"
PLAN = ROOT / "configs" / "p1_c" / "training_plan_v1.json"
P1_B_CONFIG = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718" / "config.json"
VARIANTS = ("clean_5000", "clean_7200", "single_only", "no_instance_balanced_loss", "no_canonical_patch")
PACKAGE_BY_VARIANT = {
    "clean_5000": "clean_5000",
    "clean_7200": "clean_7200",
    "single_only": "single_only",
    "no_instance_balanced_loss": "clean_7200",
    "no_canonical_patch": "clean_7200",
}


class Records(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training_plan = _read_json(PLAN)
    declared = training_plan["formal_variants"][args.variant]
    if args.seed not in declared["seeds"]:
        raise RuntimeError(f"seed is not in frozen training plan: {args.variant}/{args.seed}")
    run_dir = OUTPUT_ROOT / f"{args.variant}_seed_{args.seed}"
    summary_path = run_dir / "metrics.json"
    if summary_path.exists():
        print(json.dumps({"status": "already_complete", "metrics": str(summary_path)}, ensure_ascii=False))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline._set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    package = PACKAGE_ROOT / PACKAGE_BY_VARIANT[args.variant]
    metadata = _read_json(package / "metadata.json")
    split_records = {
        split: torch.load(package / f"{split}.pt", map_location="cpu", weights_only=False)
        for split in ("train", "val", "development_test")
    }
    dims = {
        "node": len(metadata["node_feature_schema"]),
        "edge": len(metadata["edge_feature_schema"]),
        "primitive": int(metadata["primitive_surface_class_count"]),
        "macro": int(metadata["macro_class_count"]),
        "semantic": int(metadata["semantic_class_count"]),
    }
    reference = _read_json(P1_B_CONFIG)
    reference_dims = reference["model"]["dimensions"]
    expected_dims = {
        "node": int(reference_dims["node"]),
        "edge": int(reference_dims["edge"]),
        "primitive": int(reference_dims["primitive"]),
        "macro": int(reference_dims["macro"]),
        "semantic": int(reference_dims["semantic"]),
    }
    if dims != expected_dims:
        raise RuntimeError(f"P1-C architecture dimensions differ from P1-B: {dims} != {expected_dims}")
    model = EdgeAwareMultiHeadGNN(
        input_dim=dims["node"],
        edge_feature_dim=dims["edge"],
        num_primitive_surfaces=dims["primitive"],
        num_macro_classes=dims["macro"],
        num_semantic_classes=dims["semantic"],
        derived_target_dim=0,
        idealization_target_dim=0,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        primitive_embedding_dim=16,
        dropout=args.dropout,
    ).to(device)
    macro_weights = baseline._class_weights(split_records["train"], "macro_targets", dims["macro"]).to(device)
    semantic_weights = baseline._class_weights(split_records["train"], "semantic_targets", dims["semantic"]).to(device)
    edge_pos, edge_neg = baseline._edge_counts(split_records["train"])
    edge_pos_weight = torch.tensor([edge_neg / max(edge_pos, 1)], dtype=torch.float32, device=device)
    macro_loss = nn.CrossEntropyLoss(weight=macro_weights, label_smoothing=0.01, reduction="none")
    semantic_loss = nn.CrossEntropyLoss(weight=semantic_weights, label_smoothing=0.01, reduction="none")
    edge_loss = nn.BCEWithLogitsLoss(pos_weight=edge_pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.04
    )
    loaders = {
        split: DataLoader(
            Records(records),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(
                args.seed + {"train": 0, "val": 1, "development_test": 2}[split]
            ),
        )
        for split, records in split_records.items()
    }
    balanced_loss = args.variant != "no_instance_balanced_loss"
    canonical_patch = args.variant != "no_canonical_patch"
    config = {
        "schema_version": "topocae_p1_c_gnn_run_config_1",
        "variant": args.variant,
        "seed": args.seed,
        "training_plan_sha256": _sha256(PLAN),
        "p1_b_architecture_reference_sha256": _sha256(P1_B_CONFIG),
        "model": {
            "kind": "EdgeAwareMultiHeadGNN",
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "dimensions": dims,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "instance_balanced_loss": balanced_loss,
            "loss_weights": {"macro": 1.0, "semantic": 0.55, "same_instance": 0.65},
            "edge_positive_count": edge_pos,
            "edge_negative_count": edge_neg,
            "edge_pos_weight": float(edge_pos_weight.item()),
        },
        "representation": {"canonical_patch": canonical_patch},
        "data": {
            "package": str(package.resolve()),
            "package_manifest_sha256": _sha256(package / "manifest.json"),
            "split_counts": {split: len(records) for split, records in split_records.items()},
            "normalization_fitted_split": metadata["normalization"]["fitted_split"],
        },
        "selection_contract": {
            "early_stopping": "Synthetic-Val only",
            "same_instance_threshold": "Synthetic-Val only/global",
            "real_data_used": False,
            "holdout_used": False,
        },
        "environment": baseline._environment(device),
    }
    _write_json(run_dir / "config.json", config)
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_result = _run_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            macro_loss,
            semantic_loss,
            edge_loss,
            macro_names=metadata["macro_classes"],
            semantic_count=dims["semantic"],
            balanced_loss=balanced_loss,
            canonical_patch=canonical_patch,
            collect=False,
        )
        scheduler.step()
        with torch.no_grad():
            val_result = _run_epoch(
                model,
                loaders["val"],
                device,
                None,
                macro_loss,
                semantic_loss,
                edge_loss,
                macro_names=metadata["macro_classes"],
                semantic_count=dims["semantic"],
                balanced_loss=balanced_loss,
                canonical_patch=canonical_patch,
                collect=False,
            )
        score = sum(
            (
                val_result["metrics"]["face"]["macro_f1_all"],
                val_result["metrics"]["semantic"]["macro_f1_active"],
                val_result["metrics"]["same_instance_edge_at_0_5"]["f1"],
            )
        ) / 3.0
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_score": score,
            "train": train_result["metrics"],
            "val": val_result["metrics"],
        }
        history.append(row)
        print(
            f"variant={args.variant} seed={args.seed} epoch={epoch:03d} "
            f"loss={train_result['metrics']['loss']:.4f}/{val_result['metrics']['loss']:.4f} "
            f"face={val_result['metrics']['face']['macro_f1_all']:.4f} "
            f"semantic={val_result['metrics']['semantic']['macro_f1_active']:.4f} "
            f"edge={val_result['metrics']['same_instance_edge_at_0_5']['f1']:.4f} score={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "schema_version": "topocae_p1_c_gnn_checkpoint_1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_result["metrics"],
                    "selection_score": score,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience and epoch >= 20:
            break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, Any] = {}
    inference_seconds = 0.0
    for split, loader in loaders.items():
        split_started = time.perf_counter()
        with torch.no_grad():
            result = _run_epoch(
                model,
                loader,
                device,
                None,
                macro_loss,
                semantic_loss,
                edge_loss,
                macro_names=metadata["macro_classes"],
                semantic_count=dims["semantic"],
                balanced_loss=balanced_loss,
                canonical_patch=canonical_patch,
                collect=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - split_started
        inference_seconds += elapsed
        predictions[split] = result["predictions"]
        evaluation[split] = result["metrics"]
        evaluation[split]["inference_seconds"] = elapsed
        evaluation[split]["inference_ms_per_graph"] = elapsed * 1000.0 / max(len(split_records[split]), 1)
        evaluation[split]["face_by_context"] = _face_by_field(
            result["predictions"], "context", metadata["macro_classes"]
        )
        evaluation[split]["face_by_host_family"] = _face_by_field(
            result["predictions"], "host_family", metadata["macro_classes"]
        )
    threshold_records = baseline._undirected_edge_threshold_records(predictions["val"], split="synthetic_val")
    threshold_selection = select_global_same_instance_threshold(threshold_records)
    threshold = float(threshold_selection["global_threshold"])
    for split in predictions:
        evaluation[split]["same_instance_edge_selected_threshold"] = baseline._edge_metrics(
            baseline._undirected_edge_values(predictions[split]), threshold
        )
    torch.save(predictions, run_dir / "predictions.pt")
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "same_instance_threshold_selection.json", threshold_selection)
    _write_epoch_csv(run_dir / "history.csv", history)
    robustness = _read_json(
        ROOT
        / "output_generation_p1_c_clean_context_7200"
        / "reports"
        / "graph_v2_robustness"
        / "robustness.json"
    )
    metrics = {
        "schema_version": "topocae_p1_c_gnn_metrics_1",
        "variant": args.variant,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "evaluation": evaluation,
        "same_instance_threshold_selection": threshold_selection,
        "representation_robustness": {
            key: robustness[key]
            for key in (
                "rotation_consistency",
                "scale_consistency",
                "partition_consistency",
                "cleanup_probe_signature_consistency",
            )
        },
        "resource": {
            "parameter_count": config["model"]["parameter_count"],
            "training_seconds": training_seconds,
            "total_inference_seconds": inference_seconds,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "contracts": {
            "model_selected_on_synthetic_val_only": True,
            "threshold_selected_on_synthetic_val_only": threshold_selection["selection_split"] == "synthetic_val",
            "real_data_used": False,
            "holdout_used": False,
        },
    }
    _write_json(summary_path, metrics)
    print(
        json.dumps(
            {
                "training_complete": str(run_dir),
                "best_epoch": best_epoch,
                "val_face_macro_f1": evaluation["val"]["face"]["macro_f1_all"],
                "val_semantic_macro_f1": evaluation["val"]["semantic"]["macro_f1_active"],
                "val_edge_f1": evaluation["val"]["same_instance_edge_selected_threshold"]["f1"],
                "threshold": threshold,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    node_features, primitive, macro, semantic, instances, node_weights = [], [], [], [], [], []
    edges, edge_features, edge_targets, edge_masks, edge_weights = [], [], [], [], []
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
        node_weights.append(record["node_instance_balanced_weights"])
        edges.append(record["edge_index"] + node_offset)
        edge_features.append(record["edge_features"])
        edge_targets.append(record["edge_same_instance_targets"])
        edge_masks.append(record["edge_supervision_mask"])
        edge_weights.append(record["edge_sample_balanced_weights"])
        node_slices.append((node_offset, node_offset + node_count))
        edge_slices.append((edge_offset, edge_offset + edge_count))
        node_offset += node_count
        edge_offset += edge_count
    return {
        "records": records,
        "node_slices": node_slices,
        "edge_slices": edge_slices,
        "node_features": torch.cat(node_features),
        "primitive": torch.cat(primitive),
        "macro": torch.cat(macro),
        "semantic": torch.cat(semantic),
        "instances": torch.cat(instances),
        "node_weights": torch.cat(node_weights),
        "edge_index": torch.cat(edges, dim=1),
        "edge_features": torch.cat(edge_features),
        "edge_targets": torch.cat(edge_targets),
        "edge_masks": torch.cat(edge_masks),
        "edge_weights": torch.cat(edge_weights),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    macro_loss_fn: nn.Module,
    semantic_loss_fn: nn.Module,
    edge_loss_fn: nn.Module,
    *,
    macro_names: list[str],
    semantic_count: int,
    balanced_loss: bool,
    canonical_patch: bool,
    collect: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_graphs = 0
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
        outputs = model(node, edge_index, primitive, edge_features=edge_features)
        macro_raw = macro_loss_fn(outputs["macro_logits"], macro_target)
        semantic_raw = semantic_loss_fn(outputs["semantic_logits"], semantic_target)
        if balanced_loss:
            node_weight = batch["node_weights"].to(device, non_blocking=True)
            macro_value = (macro_raw * node_weight).sum() / node_weight.sum().clamp_min(1e-12)
            semantic_value = (semantic_raw * node_weight).sum() / node_weight.sum().clamp_min(1e-12)
        else:
            macro_value = macro_raw.mean()
            semantic_value = semantic_raw.mean()
        target = batch["edge_targets"].to(device, non_blocking=True)
        mask = batch["edge_masks"].to(device, non_blocking=True)
        edge_raw = edge_loss_fn(outputs["edge_same_instance_logits"], target.clamp_min(0.0))
        if bool(mask.any()):
            if balanced_loss:
                edge_weight = batch["edge_weights"].to(device, non_blocking=True)
                edge_value = (edge_raw[mask] * edge_weight[mask]).sum() / edge_weight[mask].sum().clamp_min(1e-12)
            else:
                edge_value = edge_raw[mask].mean()
        else:
            edge_value = edge_raw.sum() * 0.0
        loss = macro_value + 0.55 * semantic_value + 0.65 * edge_value
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        graph_count = len(batch["records"])
        total_loss += float(loss.detach().item()) * graph_count
        total_graphs += graph_count
        macro_prediction = outputs["macro_logits"].argmax(dim=1).detach().cpu()
        semantic_prediction = outputs["semantic_logits"].argmax(dim=1).detach().cpu()
        probability = torch.sigmoid(outputs["edge_same_instance_logits"]).detach().cpu()
        macro_preds.append(macro_prediction)
        macro_truth.append(batch["macro"])
        semantic_preds.append(semantic_prediction)
        semantic_truth.append(batch["semantic"])
        edge_probs.append(probability)
        edge_truth.append(batch["edge_targets"])
        edge_masks.append(batch["edge_masks"])
        if collect:
            embeddings = outputs["node_embeddings"].detach().cpu().to(torch.float16)
            for record, node_slice, edge_slice in zip(
                batch["records"], batch["node_slices"], batch["edge_slices"]
            ):
                ns, ne = node_slice
                es, ee = edge_slice
                predictions.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_graph_path": record["source_graph_path"],
                        "base_geometry_family_id": record["base_geometry_family_id"],
                        "primary_target": record["primary_target"],
                        "context": record["context"],
                        "host_family": record["host_family"],
                        "macro_prediction": macro_prediction[ns:ne],
                        "macro_target": batch["macro"][ns:ne],
                        "semantic_prediction": semantic_prediction[ns:ne],
                        "semantic_target": batch["semantic"][ns:ne],
                        "instance_ids": batch["instances"][ns:ne],
                        "node_embeddings": embeddings[ns:ne],
                        "edge_index": record["edge_index"],
                        "edge_probability": probability[es:ee],
                        "edge_target": batch["edge_targets"][es:ee],
                        "edge_mask": batch["edge_masks"][es:ee],
                        "edge_features": record["edge_features"].to(torch.float16),
                        "feature_sets": record["feature_sets"],
                        "face_to_canonical_patch": record["face_to_canonical_patch"] if canonical_patch else {},
                    }
                )
    macro_preds_t, macro_truth_t = torch.cat(macro_preds), torch.cat(macro_truth)
    semantic_preds_t, semantic_truth_t = torch.cat(semantic_preds), torch.cat(semantic_truth)
    probs, truth, masks = torch.cat(edge_probs), torch.cat(edge_truth), torch.cat(edge_masks).bool()
    return {
        "metrics": {
            "loss": total_loss / max(total_graphs, 1),
            "face": baseline._classification_metrics(macro_preds_t, macro_truth_t, macro_names),
            "semantic": baseline._classification_metrics(
                semantic_preds_t, semantic_truth_t, [str(index) for index in range(semantic_count)]
            ),
            "same_instance_edge_at_0_5": baseline._binary_metrics(probs[masks], truth[masks].long(), 0.5),
        },
        "predictions": predictions,
    }


def _face_by_field(
    records: list[dict[str, Any]], field: str, macro_names: list[str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    result: dict[str, Any] = {}
    for value, members in sorted(groups.items()):
        prediction = torch.cat([record["macro_prediction"] for record in members])
        truth = torch.cat([record["macro_target"] for record in members])
        result[value] = baseline._classification_metrics(prediction, truth, macro_names)
    return result


def _write_epoch_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "lr",
                "selection_score",
                "train_loss",
                "val_loss",
                "val_face_macro_f1",
                "val_semantic_macro_f1",
                "val_edge_f1",
            ),
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "lr": row["lr"],
                    "selection_score": row["selection_score"],
                    "train_loss": row["train"]["loss"],
                    "val_loss": row["val"]["loss"],
                    "val_face_macro_f1": row["val"]["face"]["macro_f1_all"],
                    "val_semantic_macro_f1": row["val"]["semantic"]["macro_f1_active"],
                    "val_edge_f1": row["val"]["same_instance_edge_at_0_5"]["f1"],
                }
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
