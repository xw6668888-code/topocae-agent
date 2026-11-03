"""A4: train a complete-object scorer on frozen D0-train proposals only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from g09_common import ARTIFACT, REPORT, append_jsonl, checked, descriptor, now, read, write_json
from object_features import NUMERIC_FEATURE_NAMES
from train_candidate_data import source_identity


class ObjectScorer(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def balanced_part_loss(logits, labels, part_indices, part_count):
    """Positive and real negative proposals contribute equally within each part."""
    losses = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    positive = (labels == 1).to(losses.dtype)
    negative = (labels == 0).to(losses.dtype)
    positive_sum = losses.new_zeros(part_count).scatter_add(0, part_indices, losses * positive)
    negative_sum = losses.new_zeros(part_count).scatter_add(0, part_indices, losses * negative)
    positive_count = losses.new_zeros(part_count).scatter_add(0, part_indices, positive)
    negative_count = losses.new_zeros(part_count).scatter_add(0, part_indices, negative)
    present = (positive_count + negative_count) > 0
    if not bool(present.any()):
        raise ValueError("Empty object training batch")
    positive_mean = positive_sum / positive_count.clamp_min(1)
    negative_mean = negative_sum / negative_count.clamp_min(1)
    both = (positive_count > 0) & (negative_count > 0)
    per_part = torch.where(both, 0.5 * (positive_mean + negative_mean),
                           positive_mean + negative_mean)
    return per_part[present].mean()


def load_training_data(folder: Path):
    seal = read(folder / "COMPLETE.json")
    if seal.get("schema") != "g09_train_candidate_data_v1" or seal.get("identity", {}).get("split") != "train":
        raise ValueError("Only completed D0 train candidate data is eligible for gradient")
    if seal.get("count") != 7440 or seal["identity"].get("seed") != 20260719 or seal.get("pilot"):
        raise ValueError("D0 train denominator/seed changed")
    expected = source_identity(20260719, 7440)
    if seal["identity"] != expected:
        raise ValueError("Candidate training identity changed")
    for field in ("training_data", "architecture", "checkpoint", "normalization",
                  "prediction_entry", "sample_manifest", "stage_capture",
                  "object_features", "data_builder"):
        checked(seal["identity"][field])
    outputs = {Path(row["path"]).name: checked(row) for row in seal["outputs"]}
    candidate_path, label_path = outputs["candidates.jsonl"], outputs["labels.jsonl"]
    if candidate_path.parent != folder.resolve() or label_path.parent != folder.resolve():
        raise ValueError("Candidate and labels must be bound to this completed train directory")
    vectors, targets, part_indices = [], [], []
    counts = Counter()
    expected_part = -1
    with candidate_path.open(encoding="utf-8") as candidates, label_path.open(encoding="utf-8") as labels:
        for expected_part, (candidate_line, label_line) in enumerate(zip(candidates, labels, strict=True)):
            candidate = json.loads(candidate_line)
            sidecar = json.loads(label_line)
            if candidate["part_index"] != expected_part or sidecar["part_index"] != expected_part:
                raise ValueError("Train candidate/label ordering changed")
            rows = candidate["candidates"]
            if len(rows) != len(sidecar["labels"]):
                raise ValueError("Candidate label count changed")
            for index, (row, target) in enumerate(zip(rows, sidecar["labels"])):
                if target["candidate_index"] != index:
                    raise ValueError("Candidate label index changed")
                vector = row["vector"]
                if len(vector) != len(NUMERIC_FEATURE_NAMES) or not all(math.isfinite(value) for value in vector):
                    raise ValueError("Malformed feature vector")
                vectors.append(vector)
                targets.append(int(target["exact_positive"]))
                part_indices.append(expected_part)
                counts["positive" if target["exact_positive"] else "negative"] += 1
    if expected_part + 1 != 7440 or counts["positive"] == 0 or counts["negative"] == 0:
        raise ValueError("Incomplete or degenerate train candidate set")
    return seal, torch.tensor(vectors, dtype=torch.float32), torch.tensor(targets, dtype=torch.long), torch.tensor(part_indices, dtype=torch.long), counts


def train(folder: Path, *, epochs=30, seed=20260719, batch_parts=64, device="cpu"):
    folder = folder.resolve()
    output = ARTIFACT / "object_scorer/A4_seed_20260719"
    if (output / "COMPLETE.json").exists():
        raise FileExistsError(output / "COMPLETE.json")
    protocol_path = REPORT / "A4_SCORER_PROTOCOL.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        if (protocol["epochs"], protocol["seed"], protocol["train_cache"]["path"]) != (
            epochs, seed, str((folder / "COMPLETE.json").as_posix())
        ):
            raise ValueError("A4 training protocol is already registered with another configuration")
        checked(protocol["train_cache"])
    else:
        protocol = {
            "schema": "g09_a4_object_scorer_protocol_v1", "at_utc": now(),
            "hypothesis": "A train-only learned complete-object score can separate real split/overmerge/duplicate candidates that the initial deterministic score cannot.",
            "unique_primary_change": "64/32 two-layer MLP on fixed A candidate pool and existing frozen R probabilities",
            "matched_control": "A1/A2 deterministic scorer and identical candidate pool on D0 validation/engineering",
            "train_cache": descriptor(folder / "COMPLETE.json"), "feature_names": list(NUMERIC_FEATURE_NAMES),
            "loss": "equal-positive-negative BCE within each part, then mean across parts; empty group reweights valid group",
            "epochs": epochs, "seed": seed, "batch_parts": batch_parts,
            "optimizer": "AdamW lr=0.001 weight_decay=0.0001",
            "validation_gradients": False, "engineering_gradients": False,
            "gpu_seconds_budget": 0 if device == "cpu" else 3600,
            "scorer_configurations_used": 1,
        }
        write_json(protocol_path, protocol)
        append_jsonl(REPORT / "EXPERIMENT_QUEUE.jsonl", {
            "schema": "g09_experiment_event_v1", "event": "registered", "at_utc": now(),
            "experiment_id": "A3-learned-object-score", "route": "A", "status": "planned",
            "hypothesis": protocol["hypothesis"], "unique_primary_change": protocol["unique_primary_change"],
            "parent_configuration": "A2-constrained-selection",
            "expected_error_change": "Reduce 35 engineering FN and 84 FP relative to failed A2; recover D0 validation selection losses without creating missing candidates.",
            "identity": protocol["train_cache"], "matching_control": protocol["matched_control"],
            "budget": {"gpu_seconds_max": protocol["gpu_seconds_budget"], "scorer_configs": 1},
            "actual_result": None, "instance_error_change": None,
            "next_action": "Evaluate unchanged candidate pool on D0 validation and engineering; route absent candidates to B/E/D.",
        })
    began = time.perf_counter()
    seal, x, y, groups, counts = load_training_data(folder)
    if seed != 20260719 or epochs < 1 or epochs > 100 or batch_parts < 1:
        raise ValueError("Unregistered scorer training schedule")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x = ((x - mean) / std).clamp(-8, 8)
    model = ObjectScorer(len(NUMERIC_FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    indices = torch.arange(len(y))
    positions = torch.searchsorted(groups, torch.arange(7441))
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(7440)
        totals = []
        for block in order.split(batch_parts):
            batch_indices = torch.cat([indices[positions[p]:positions[p + 1]] for p in block])
            local_parts = torch.repeat_interleave(torch.arange(len(block)), positions[block + 1] - positions[block])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch_indices].to(device))
            loss = balanced_part_loss(logits, y[batch_indices].to(device), local_parts.to(device), len(block))
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()) * len(block))
        average = sum(totals) / 7440
        losses.append(average)
        print(f"A4 epoch={epoch + 1}/{epochs} train_balanced_bce={average:.6f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "scorer.pt"
    temporary_checkpoint = checkpoint.with_name(f"{checkpoint.name}.{os.getpid()}.tmp")
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "feature_names": list(NUMERIC_FEATURE_NAMES),
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "training_data_identity": seal,
        "protocol": protocol,
    }, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint)
    result = {
        "schema": "g09_a4_scorer_training_v1", "at_utc": now(),
        "checkpoint": descriptor(checkpoint), "protocol": descriptor(protocol_path),
        "training_cache": descriptor(folder / "COMPLETE.json"),
        "parts": 7440, "candidates": len(y), "positive": counts["positive"],
        "negative": counts["negative"], "train_loss_per_epoch": losses,
        "elapsed_seconds": time.perf_counter() - began,
        "device": device, "validation_seen": False, "engineering_seen": False,
    }
    write_json(output / "COMPLETE.json", result)
    append_jsonl(REPORT / "EXPERIMENT_QUEUE.jsonl", {
        "schema": "g09_experiment_event_v1", "event": "trained", "at_utc": now(),
        "experiment_id": "A3-learned-object-score", "status": "pending_matched_validation",
        "actual_result": {"train_candidates": len(y), "train_positive": counts["positive"],
                          "train_negative": counts["negative"], "train_loss_final": losses[-1]},
        "instance_error_change": None,
        "next_action": "Evaluate the frozen scorer against A1/A2 with identical D0 validation and engineering candidates.",
        "evidence": descriptor(output / "COMPLETE.json"),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    train(args.train_cache, device=args.device)


if __name__ == "__main__":
    main()
