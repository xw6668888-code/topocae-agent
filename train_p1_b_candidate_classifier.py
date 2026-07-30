from __future__ import annotations

"""Train/evaluate the ragged P1-B candidate topology classifier in batches."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from utils.p1_b_topology_verifier_v3 import select_global_candidate_thresholds


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"


class CandidateRecords(Dataset):
    def __init__(self, records: list[dict[str, Any]], no_canonical_patch: bool) -> None:
        self.records = records
        self.no_canonical_patch = no_canonical_patch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if not self.no_canonical_patch:
            return record
        revised = dict(record)
        engineered = record["engineered_features"].clone()
        engineered[1] = engineered[0]  # canonical patch count falls back to raw face count
        revised["engineered_features"] = engineered
        return revised


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-canonical-patch", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RUN_ROOT / f"same_instance_seed_{args.seed}"
    package = run_dir / "candidate_training_v1"
    model_name = "candidate_model_no_canonical_patch_v1" if args.no_canonical_patch else "candidate_model_v1"
    output = run_dir / model_name
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "metrics": str(metrics_path)}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed + (100000 if args.no_canonical_patch else 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metadata = _read_json(package / "metadata.json")
    records = {
        split: torch.load(package / f"{split}.pt", map_location="cpu", weights_only=False)
        for split in ("train", "val", "development_test")
    }
    dimensions = CandidateModelDimensions(**metadata["dimensions"], hidden_dim=128)
    model = CandidateTopologyClassifier(dimensions, macro_classes=metadata["macro_classes"]).to(device)
    macro_weights = _positive_macro_weights(records["train"], len(metadata["macro_classes"])).to(device)
    validity_positive = sum(int(record["validity_target"].item()) for record in records["train"])
    validity_negative = len(records["train"]) - validity_positive
    validity_pos_weight = torch.tensor([validity_negative / max(validity_positive, 1)], device=device)
    loaders = {
        split: DataLoader(
            CandidateRecords(split_records, args.no_canonical_patch), batch_size=args.batch_size,
            shuffle=split == "train", num_workers=0, collate_fn=_collate,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + {"train": 11, "val": 12, "development_test": 13}[split]),
        )
        for split, split_records in records.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    config = {
        "schema_version": "topocae_p1_b_candidate_run_config_1",
        "seed": args.seed, "no_canonical_patch": args.no_canonical_patch,
        "source_package": str(package.resolve()), "source_manifest_sha256": _sha256(package / "manifest.json"),
        "dimensions": dimensions.__dict__, "macro_classes": metadata["macro_classes"],
        "parameter_count": model.parameter_count(),
        "optimization": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "loss_weights": {"macro": 1.0, "validity": 0.7, "membership": 0.4, "boundary": 0.4, "semantic_role": 0.25},
            "validity_pos_weight": float(validity_pos_weight.item()),
        },
        "selection_contract": {"model": "Synthetic-Val only", "thresholds": "Synthetic-Val only/global", "real_data_used": False, "holdout_used": False},
        "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    }
    _write_json(output / "config.json", config)
    history = []
    best_score = -math.inf
    best_epoch = stale = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_result = _run_epoch(model, loaders["train"], device, optimizer, macro_weights, validity_pos_weight, collect=False)
        scheduler.step()
        with torch.no_grad():
            val_result = _run_epoch(model, loaders["val"], device, None, macro_weights, validity_pos_weight, collect=False)
        score = _selection_score(val_result["metrics"])
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "selection_score": score, "train": train_result["metrics"], "val": val_result["metrics"]}
        history.append(row)
        print(
            f"candidate seed={args.seed} no_patch={args.no_canonical_patch} epoch={epoch:03d} "
            f"loss={train_result['metrics']['loss']:.4f}/{val_result['metrics']['loss']:.4f} "
            f"validity_f1={val_result['metrics']['validity']['f1']:.4f} "
            f"macro_f1={val_result['metrics']['macro']['macro_f1']:.4f} "
            f"membership_f1={val_result['metrics']['membership']['f1']:.4f} "
            f"boundary_f1={val_result['metrics']['boundary']['f1']:.4f} score={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "schema_version": "topocae_p1_b_candidate_checkpoint_1", "epoch": epoch,
                "model_state_dict": model.state_dict(), "config": config,
                "selection_score": score, "val_metrics": val_result["metrics"],
            }, output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience and epoch >= 14:
            break
    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prediction_payload: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, Any] = {}
    inference_seconds = 0.0
    for split, loader in loaders.items():
        started = time.perf_counter()
        with torch.no_grad():
            result = _run_epoch(model, loader, device, None, macro_weights, validity_pos_weight, collect=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        prediction_payload[split] = result["predictions"]
        evaluation[split] = result["metrics"]
        evaluation[split]["inference_seconds"] = elapsed
        evaluation[split]["inference_ms_per_candidate"] = elapsed * 1000.0 / max(len(records[split]), 1)
        inference_seconds += elapsed
    threshold_records = [_threshold_record(record) for record in prediction_payload["val"]]
    threshold_selection = select_global_candidate_thresholds(threshold_records)
    torch.save(prediction_payload, output / "predictions.pt")
    _write_json(output / "candidate_threshold_selection.json", threshold_selection)
    _write_json(output / "history.json", history)
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "schema_version": "topocae_p1_b_candidate_metrics_1", "seed": args.seed,
        "no_canonical_patch": args.no_canonical_patch, "best_epoch": best_epoch,
        "best_val_selection_score": best_score, "evaluation": evaluation,
        "threshold_selection": threshold_selection,
        "proposal_audit": metadata["splits"],
        "resource": {
            "parameter_count": model.parameter_count(), "training_seconds": training_seconds,
            "total_inference_seconds": inference_seconds, "peak_vram_bytes": peak_vram,
        },
        "contracts": {"model_selected_synthetic_val_only": True, "thresholds_selected_synthetic_val_only": True, "real_data_used": False, "holdout_used": False},
    }
    _write_json(metrics_path, summary)
    _write_history_csv(output / "history.csv", history)
    print(json.dumps({
        "training_complete": str(output), "best_epoch": best_epoch,
        "val_macro_f1": evaluation["val"]["macro"]["macro_f1"],
        "val_validity_f1": evaluation["val"]["validity"]["f1"],
        "val_membership_f1": evaluation["val"]["membership"]["f1"],
        "val_boundary_f1": evaluation["val"]["boundary"]["f1"],
        "val_semantic_role_f1": evaluation["val"]["semantic_role"]["macro_f1"],
    }, ensure_ascii=False), flush=True)
    return 0


def _collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(records)
    max_faces = max(record["face_embeddings"].shape[0] for record in records)
    max_internal = max(max(record["internal_edge_embeddings"].shape[0] for record in records), 1)
    max_boundary = max(max(record["boundary_edge_embeddings"].shape[0] for record in records), 1)
    face_dim = records[0]["face_embeddings"].shape[1]
    edge_dim = records[0]["internal_edge_embeddings"].shape[1]
    faces = torch.zeros((batch_size, max_faces, face_dim), dtype=torch.float32)
    face_mask = torch.zeros((batch_size, max_faces), dtype=torch.bool)
    internal = torch.zeros((batch_size, max_internal, edge_dim), dtype=torch.float32)
    internal_mask = torch.zeros((batch_size, max_internal), dtype=torch.bool)
    boundary_edges = torch.zeros((batch_size, max_boundary, edge_dim), dtype=torch.float32)
    boundary_edge_mask = torch.zeros((batch_size, max_boundary), dtype=torch.bool)
    membership = torch.zeros((batch_size, max_faces), dtype=torch.float32)
    boundary = torch.zeros((batch_size, max_faces), dtype=torch.float32)
    semantic = torch.zeros((batch_size, max_faces), dtype=torch.long)
    for row, record in enumerate(records):
        face_count = record["face_embeddings"].shape[0]
        internal_count = record["internal_edge_embeddings"].shape[0]
        boundary_count = record["boundary_edge_embeddings"].shape[0]
        faces[row, :face_count] = record["face_embeddings"].float()
        face_mask[row, :face_count] = True
        if internal_count:
            internal[row, :internal_count] = record["internal_edge_embeddings"].float()
            internal_mask[row, :internal_count] = True
        if boundary_count:
            boundary_edges[row, :boundary_count] = record["boundary_edge_embeddings"].float()
            boundary_edge_mask[row, :boundary_count] = True
        membership[row, :face_count] = record["membership_target"]
        boundary[row, :face_count] = record["boundary_target"]
        semantic[row, :face_count] = record["semantic_role_target"]
    return {
        "records": records, "faces": faces, "face_mask": face_mask,
        "internal": internal, "internal_mask": internal_mask,
        "boundary_edges": boundary_edges, "boundary_edge_mask": boundary_edge_mask,
        "engineered": torch.stack([record["engineered_features"] for record in records]),
        "validity": torch.stack([record["validity_target"] for record in records]),
        "macro": torch.stack([record["macro_target"] for record in records]),
        "membership": membership, "boundary": boundary, "semantic": semantic,
    }


def _batch_forward(model: CandidateTopologyClassifier, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    faces, face_mask = batch["faces"], batch["face_mask"]
    scores = model.face_attention(faces).squeeze(-1).masked_fill(~face_mask, -1e9)
    attention = torch.softmax(scores, dim=1)
    face_count = face_mask.sum(dim=1, keepdim=True).clamp_min(1)
    face_mean = (faces * face_mask.unsqueeze(-1)).sum(dim=1) / face_count
    face_max = faces.masked_fill(~face_mask.unsqueeze(-1), -1e9).max(dim=1).values
    face_attention = (attention.unsqueeze(-1) * faces).sum(dim=1)
    internal_mean, internal_max = _masked_mean_max(batch["internal"], batch["internal_mask"])
    boundary_mean, boundary_max = _masked_mean_max(batch["boundary_edges"], batch["boundary_edge_mask"])
    context = model.context_encoder(torch.cat([
        face_mean, face_max, face_attention, internal_mean, internal_max,
        boundary_mean, boundary_max, batch["engineered"],
    ], dim=1))
    face_hidden = F.gelu(model.face_projection(faces) + model.face_context(context).unsqueeze(1))
    return {
        "macro_logits": model.macro_head(context),
        "validity_logits": model.validity_head(context).squeeze(-1),
        "membership_logits": model.membership_head(face_hidden).squeeze(-1),
        "boundary_logits": model.boundary_head(face_hidden).squeeze(-1),
        "semantic_logits": model.semantic_role_head(face_hidden),
        "attention": attention,
    }


def _masked_mean_max(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (values * mask.unsqueeze(-1)).sum(dim=1) / count
    maximum = values.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def _run_epoch(
    model: CandidateTopologyClassifier, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None, macro_weights: torch.Tensor,
    validity_pos_weight: torch.Tensor, *, collect: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in (
        "validity_probability", "validity_target", "macro_prediction", "macro_probability", "macro_target",
        "membership_probability", "membership_target", "boundary_probability", "boundary_target",
        "semantic_prediction", "semantic_target", "face_mask",
    )}
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        tensor_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        outputs = _batch_forward(model, tensor_batch)
        valid = tensor_batch["validity"] > 0.5
        face_valid_mask = tensor_batch["face_mask"] & valid.unsqueeze(1)
        validity_loss = F.binary_cross_entropy_with_logits(outputs["validity_logits"], tensor_batch["validity"], pos_weight=validity_pos_weight)
        macro_loss = F.cross_entropy(outputs["macro_logits"][valid], tensor_batch["macro"][valid], weight=macro_weights) if bool(valid.any()) else outputs["macro_logits"].sum() * 0.0
        membership_loss = F.binary_cross_entropy_with_logits(outputs["membership_logits"][face_valid_mask], tensor_batch["membership"][face_valid_mask]) if bool(face_valid_mask.any()) else outputs["membership_logits"].sum() * 0.0
        boundary_loss = F.binary_cross_entropy_with_logits(outputs["boundary_logits"][face_valid_mask], tensor_batch["boundary"][face_valid_mask]) if bool(face_valid_mask.any()) else outputs["boundary_logits"].sum() * 0.0
        semantic_loss = F.cross_entropy(outputs["semantic_logits"][face_valid_mask], tensor_batch["semantic"][face_valid_mask]) if bool(face_valid_mask.any()) else outputs["semantic_logits"].sum() * 0.0
        loss = macro_loss + 0.7 * validity_loss + 0.4 * membership_loss + 0.4 * boundary_loss + 0.25 * semantic_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach()) * len(batch["records"])
        total_count += len(batch["records"])
        macro_probabilities = torch.softmax(outputs["macro_logits"], dim=1).detach().cpu()
        validity_probability = torch.sigmoid(outputs["validity_logits"]).detach().cpu()
        membership_probability = torch.sigmoid(outputs["membership_logits"]).detach().cpu()
        boundary_probability = torch.sigmoid(outputs["boundary_logits"]).detach().cpu()
        semantic_probability = torch.softmax(outputs["semantic_logits"], dim=2).detach().cpu()
        collected["validity_probability"].append(validity_probability)
        collected["validity_target"].append(batch["validity"])
        collected["macro_prediction"].append(macro_probabilities.argmax(dim=1))
        collected["macro_probability"].append(macro_probabilities)
        collected["macro_target"].append(batch["macro"])
        collected["membership_probability"].append(membership_probability[batch["face_mask"]])
        collected["membership_target"].append(batch["membership"][batch["face_mask"]])
        collected["boundary_probability"].append(boundary_probability[batch["face_mask"]])
        collected["boundary_target"].append(batch["boundary"][batch["face_mask"]])
        collected["semantic_prediction"].append(semantic_probability.argmax(dim=2)[batch["face_mask"]])
        collected["semantic_target"].append(batch["semantic"][batch["face_mask"]])
        collected["face_mask"].append(batch["face_mask"].flatten())
        if collect:
            for row, record in enumerate(batch["records"]):
                count = len(record["face_indices"])
                predictions.append({
                    "sample_id": record["sample_id"], "source_graph_path": record["source_graph_path"],
                    "base_geometry_family_id": record["base_geometry_family_id"], "derivation_kind": record["derivation_kind"],
                    "candidate_id": record["candidate_id"], "origin": record["origin"],
                    "proposal": record["proposal"], "face_indices": record["face_indices"],
                    "macro_logits": outputs["macro_logits"][row].detach().cpu().tolist(),
                    "macro_probabilities": macro_probabilities[row].tolist(),
                    "validity_score": float(validity_probability[row]),
                    "membership_probabilities": membership_probability[row, :count].tolist(),
                    "boundary_probabilities": boundary_probability[row, :count].tolist(),
                    "semantic_role_probabilities": semantic_probability[row, :count].tolist(),
                    "validity_target": int(record["validity_target"].item()),
                    "macro_target": int(record["macro_target"].item()),
                    "membership_labels": record["membership_target"].tolist(),
                    "boundary_labels": record["boundary_target"].tolist(),
                    "semantic_role_labels": record["semantic_role_target"].tolist(),
                    "matched_truth_instance_id": record["matched_truth_instance_id"],
                    "matched_truth_macro": record["matched_truth_macro"],
                    "matched_truth_face_indices": record["matched_truth_face_indices"],
                    "matched_truth_iou": record["matched_truth_iou"],
                })
    validity_probability = torch.cat(collected["validity_probability"])
    validity_target = torch.cat(collected["validity_target"]).long()
    macro_prediction = torch.cat(collected["macro_prediction"])
    macro_target = torch.cat(collected["macro_target"])
    positive = validity_target.bool()
    membership_probability = torch.cat(collected["membership_probability"])
    membership_target = torch.cat(collected["membership_target"]).long()
    boundary_probability = torch.cat(collected["boundary_probability"])
    boundary_target = torch.cat(collected["boundary_target"]).long()
    semantic_prediction = torch.cat(collected["semantic_prediction"])
    semantic_target = torch.cat(collected["semantic_target"])
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "validity": _binary_metrics(validity_probability, validity_target, 0.5),
        "macro": _multiclass_metrics(macro_prediction[positive], macro_target[positive], len(model.macro_classes)),
        "membership": _binary_metrics(membership_probability, membership_target, 0.5),
        "boundary": _binary_metrics(boundary_probability, boundary_target, 0.5),
        "semantic_role": _multiclass_metrics(semantic_prediction, semantic_target, model.dimensions.semantic_role_count),
    }
    return {"metrics": metrics, "predictions": predictions}


def _positive_macro_weights(records: list[dict[str, Any]], class_count: int) -> torch.Tensor:
    counts = Counter(int(record["macro_target"]) for record in records if int(record["validity_target"].item()) == 1)
    total = sum(counts.values())
    weights = torch.zeros(class_count)
    for index, count in counts.items():
        weights[index] = math.sqrt(total / (len(counts) * count))
    weights[weights > 0] /= weights[weights > 0].mean()
    return weights


def _binary_metrics(probabilities: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, Any]:
    predicted, truth = probabilities >= threshold, target.bool()
    tp = int((predicted & truth).sum()); fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum()); tn = int((~predicted & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _multiclass_metrics(prediction: torch.Tensor, target: torch.Tensor, class_count: int) -> dict[str, Any]:
    per_class, f1s = {}, []
    active = sorted(set(int(value) for value in target.tolist()))
    for class_id in active:
        truth_positive, prediction_positive = target == class_id, prediction == class_id
        tp = int((truth_positive & prediction_positive).sum()); fp = int((~truth_positive & prediction_positive).sum())
        fn = int((truth_positive & ~prediction_positive).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(class_id)] = {"support": int(truth_positive.sum()), "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {
        "accuracy": float((prediction == target).float().mean()) if target.numel() else 0.0,
        "macro_f1": sum(f1s) / max(len(f1s), 1), "active_class_count": len(active),
        "declared_class_count": class_count, "per_class": per_class,
    }


def _threshold_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": "synthetic_val", "macro_confidence": max(record["macro_probabilities"]),
        "validity_score": record["validity_score"], "accept_label": record["validity_target"],
        "membership_probabilities": record["membership_probabilities"], "membership_labels": record["membership_labels"],
        "boundary_probabilities": record["boundary_probabilities"], "boundary_labels": record["boundary_labels"],
    }


def _selection_score(metrics: dict[str, Any]) -> float:
    return sum([
        metrics["validity"]["f1"], metrics["macro"]["macro_f1"], metrics["membership"]["f1"],
        metrics["boundary"]["f1"], metrics["semantic_role"]["macro_f1"],
    ]) / 5.0


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["epoch", "lr", "selection_score", "train_loss", "val_loss", "validity_f1", "macro_f1", "membership_f1", "boundary_f1", "semantic_role_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"], "lr": row["lr"], "selection_score": row["selection_score"],
                "train_loss": row["train"]["loss"], "val_loss": row["val"]["loss"],
                "validity_f1": row["val"]["validity"]["f1"], "macro_f1": row["val"]["macro"]["macro_f1"],
                "membership_f1": row["val"]["membership"]["f1"], "boundary_f1": row["val"]["boundary"]["f1"],
                "semantic_role_f1": row["val"]["semantic_role"]["macro_f1"],
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
