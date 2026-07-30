from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

import train_p1_b_candidate_classifier as protocol
from utils.p1_c_run_paths import LegacySeedRunRoot


protocol.RUN_ROOT = LegacySeedRunRoot()
_original_collate = protocol._collate


class SampleInstanceBalancedCandidateRecords(protocol.Dataset):
    def __init__(self, records: list[dict[str, Any]], no_canonical_patch: bool) -> None:
        self.no_canonical_patch = no_canonical_patch
        by_sample: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            by_sample[str(record["sample_id"])].append(index)
        weights = [0.0] * len(records)
        for indices in by_sample.values():
            positive_groups: dict[str, list[int]] = defaultdict(list)
            negatives: list[int] = []
            for index in indices:
                record = records[index]
                if int(record["validity_target"].item()) == 1:
                    key = str(record.get("matched_truth_instance_id", "unknown"))
                    positive_groups[key].append(index)
                else:
                    negatives.append(index)
            if positive_groups and negatives:
                positive_mass, negative_mass = 0.5, 0.5
            elif positive_groups:
                positive_mass, negative_mass = 1.0, 0.0
            else:
                positive_mass, negative_mass = 0.0, 1.0
            if positive_groups:
                per_instance = positive_mass / len(positive_groups)
                for group in positive_groups.values():
                    for index in group:
                        weights[index] = per_instance / len(group)
            if negatives:
                for index in negatives:
                    weights[index] = negative_mass / len(negatives)
            if abs(sum(weights[index] for index in indices) - 1.0) > 1e-6:
                raise AssertionError("candidate weights do not sum to one per sample")
        self.records = []
        for record, weight in zip(records, weights):
            revised = dict(record)
            revised["candidate_loss_weight"] = torch.tensor(weight, dtype=torch.float32)
            self.records.append(revised)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if not self.no_canonical_patch:
            return record
        revised = dict(record)
        engineered = record["engineered_features"].clone()
        engineered[1] = engineered[0]
        revised["engineered_features"] = engineered
        return revised


def _balanced_collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = _original_collate(records)
    batch["candidate_weights"] = torch.stack(
        [record["candidate_loss_weight"] for record in records]
    )
    return batch


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _balanced_run_epoch(
    model, loader, device, optimizer, macro_weights, validity_pos_weight, *, collect: bool
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_weight = 0.0
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in (
        "validity_probability", "validity_target", "macro_prediction", "macro_probability", "macro_target",
        "membership_probability", "membership_target", "boundary_probability", "boundary_target",
        "semantic_prediction", "semantic_target", "face_mask",
    )}
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        tensor_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        outputs = protocol._batch_forward(model, tensor_batch)
        candidate_weights = tensor_batch["candidate_weights"]
        valid = tensor_batch["validity"] > 0.5
        face_mask = tensor_batch["face_mask"]
        face_count = face_mask.sum(dim=1).clamp_min(1)

        validity_raw = F.binary_cross_entropy_with_logits(
            outputs["validity_logits"], tensor_batch["validity"],
            pos_weight=validity_pos_weight, reduction="none",
        )
        validity_loss = _weighted_mean(validity_raw, candidate_weights)

        if bool(valid.any()):
            macro_raw = F.cross_entropy(
                outputs["macro_logits"][valid], tensor_batch["macro"][valid],
                weight=macro_weights, reduction="none",
            )
            macro_loss = _weighted_mean(macro_raw, candidate_weights[valid])

            membership_raw = F.binary_cross_entropy_with_logits(
                outputs["membership_logits"], tensor_batch["membership"], reduction="none"
            )
            boundary_raw = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"], tensor_batch["boundary"], reduction="none"
            )
            semantic_raw = F.cross_entropy(
                outputs["semantic_logits"].transpose(1, 2), tensor_batch["semantic"], reduction="none"
            )
            membership_per_candidate = (membership_raw * face_mask).sum(dim=1) / face_count
            boundary_per_candidate = (boundary_raw * face_mask).sum(dim=1) / face_count
            semantic_per_candidate = (semantic_raw * face_mask).sum(dim=1) / face_count
            membership_loss = _weighted_mean(membership_per_candidate[valid], candidate_weights[valid])
            boundary_loss = _weighted_mean(boundary_per_candidate[valid], candidate_weights[valid])
            semantic_loss = _weighted_mean(semantic_per_candidate[valid], candidate_weights[valid])
        else:
            zero = outputs["macro_logits"].sum() * 0.0
            macro_loss = membership_loss = boundary_loss = semantic_loss = zero

        loss = (
            macro_loss + 0.7 * validity_loss + 0.4 * membership_loss
            + 0.4 * boundary_loss + 0.25 * semantic_loss
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        batch_weight = float(candidate_weights.sum().detach())
        total_loss += float(loss.detach()) * batch_weight
        total_weight += batch_weight

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
        "loss": total_loss / max(total_weight, 1e-12),
        "validity": protocol._binary_metrics(validity_probability, validity_target, 0.5),
        "macro": protocol._multiclass_metrics(macro_prediction[positive], macro_target[positive], len(model.macro_classes)),
        "membership": protocol._binary_metrics(membership_probability, membership_target, 0.5),
        "boundary": protocol._binary_metrics(boundary_probability, boundary_target, 0.5),
        "semantic_role": protocol._multiclass_metrics(semantic_prediction, semantic_target, model.dimensions.semantic_role_count),
    }
    return {"metrics": metrics, "predictions": predictions}


protocol.CandidateRecords = SampleInstanceBalancedCandidateRecords
protocol._collate = _balanced_collate
protocol._run_epoch = _balanced_run_epoch


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
