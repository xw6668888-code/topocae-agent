"""E2 one-factor experiment: train-only R-final true-object retention loss.

The ordinary balanced BCE is unchanged for every real proposal, including
R-final false positives and Q-only complements. There is no R bypass at infer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from e_joint_scorer import (FEATURE_NAMES, FOLDER, OUTPUT as E1_OUTPUT, RULE,
                            load_scorer as load_e1, load_train)
from g09_common import ARTIFACT, REPORT, checked, descriptor, now, read, write_json
from train_object_scorer import ObjectScorer, balanced_part_loss


OUTPUT = ARTIFACT / "e_joint_scorer_retention/seed_20260719"
PROTOCOL = REPORT / "E2_RETENTION_PROTOCOL.json"
MARGIN = 8.0
WEIGHT = 0.01


def r_final_mask(folder=FOLDER):
    seal = read(Path(folder) / "COMPLETE.json")
    path = checked(next(row for row in seal["outputs"]
                        if Path(row["path"]).name == "candidates.jsonl"))
    if path.parent != Path(folder).resolve():
        raise ValueError("E2 origin mask is not bound to frozen joint train cache")
    mask, parts = [], 0
    with path.open(encoding="utf-8") as stream:
        for parts, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row["part_index"] != parts - 1:
                raise ValueError("R-final source mask part ordering changed")
            mask.extend(any(source["branch"] == "R" and source["stage"] == "final"
                            for source in candidate["provenance"])
                        for candidate in row["candidates"])
    if parts != 7440 or len(mask) != seal["stats"]["candidates"]:
        raise ValueError("Incomplete R-final source mask")
    return torch.tensor(mask, dtype=torch.bool)


def retention_loss(logits, labels, origin, part_indices, part_count):
    """Average positive R-final deficit per present part; never mask negatives."""
    true_r = origin & labels.bool()
    hinge = F.relu(MARGIN - logits) * true_r.float()
    sums = hinge.new_zeros(part_count).scatter_add(0, part_indices, hinge)
    counts = hinge.new_zeros(part_count).scatter_add(0, part_indices, true_r.float())
    present = counts > 0
    return (sums[present] / counts[present]).mean() if bool(present.any()) else logits.sum() * 0


def train():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Retained E2 scorer attempt exists")
    if PROTOCOL.exists():
        raise FileExistsError("E2 one-factor protocol already registered")
    e1 = read(E1_OUTPUT / "COMPLETE.json")
    if e1["schema"] != "g09_e_joint_scorer_v1" or e1["parts"] != 7440:
        raise ValueError("E1 matched control is not sealed")
    seal, x, y, groups, counts = load_train()
    origin = r_final_mask()
    if len(origin) != len(y):
        raise ValueError("E2 mask and exact labels differ")
    positive_count = int((origin & y.bool()).sum())
    negative_count = int((origin & ~y.bool()).sum())
    e1_model, e1_mean, e1_std = load_e1()
    torch.set_num_threads(4)
    with torch.no_grad():
        e1_scores = e1_model(((x - e1_mean) / e1_std).clamp(-8, 8))
        e1_true = e1_scores[origin & y.bool()]
    train_distribution = {
        "R_final_exact_positive": positive_count, "R_final_negative": negative_count,
        "R_final_positive_q25": float(torch.quantile(e1_true, 0.25)),
        "R_final_positive_median": float(torch.quantile(e1_true, 0.5)),
        "R_final_positive_below_margin": int((e1_true < MARGIN).sum()),
        "R_final_positive_nonpositive": int((e1_true <= 0).sum()),
    }
    if positive_count != 28771 or negative_count != 1786:
        raise ValueError("E2 train-only R-final source denominators changed")
    protocol = {"schema": "g09_e2_retention_protocol_v1", "created_utc": now(),
                "hypothesis": "A modest positive-retention margin on exact R-final D0 train objects may reduce engineering domain-shift rejection without privileging R-negative or overriding Q-only candidates.",
                "one_change_from_E1": "add 0.01 times per-part ReLU(8-logit) on exact-positive R-final train candidates to unchanged balanced BCE",
                "controls": "E1 same 66 features/64-32 MLP/30 epochs/AdamW seed/batch, R and Q frozen final, direct union, A4 R-only",
                "train_only_margin_basis": train_distribution,
                "margin": MARGIN, "loss_weight": WEIGHT,
                "E1_control": descriptor(E1_OUTPUT / "COMPLETE.json"),
                "E1_training_cache": descriptor(FOLDER / "COMPLETE.json"),
                "features": list(FEATURE_NAMES),
                "fixed_inference_rule": RULE["inference"],
                "R_final_negative_retains_balanced_BCE": True,
                "Q_only_positive_retains_balanced_BCE": True,
                "validation_gradients": False, "engineering_gradients": False,
                "scorer_slot": 3}
    write_json(PROTOCOL, protocol)
    started = time.perf_counter()
    torch.manual_seed(RULE["seed"])
    random.seed(RULE["seed"])
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x = ((x - mean) / std).clamp(-8, 8)
    model = ObjectScorer(len(FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positions = torch.searchsorted(groups, torch.arange(7441))
    indices = torch.arange(len(y))
    losses, components = [], []
    for epoch in range(RULE["epochs"]):
        order = torch.randperm(7440)
        total, regularizer = 0.0, 0.0
        for block in order.split(RULE["batch_parts"]):
            chosen = torch.cat([indices[positions[p]:positions[p + 1]] for p in block])
            local = torch.repeat_interleave(torch.arange(len(block)), positions[block + 1] - positions[block])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[chosen])
            base = balanced_part_loss(logits, y[chosen], local, len(block))
            retained = retention_loss(logits, y[chosen], origin[chosen], local, len(block))
            loss = base + WEIGHT * retained
            loss.backward()
            optimizer.step()
            total += float(base.detach()) * len(block)
            regularizer += float(retained.detach()) * len(block)
        losses.append(total / 7440)
        components.append(regularizer / 7440)
        print(f"E2 epoch={epoch+1}/30 train_balanced_bce={losses[-1]:.6f} "
              f"R_positive_retention={components[-1]:.6f}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT / "scorer.pt"
    temporary = checkpoint.with_name(f"scorer.pt.{os.getpid()}.tmp")
    torch.save({"model_state_dict": model.cpu().state_dict(),
                "feature_names": list(FEATURE_NAMES), "feature_mean": mean.tolist(),
                "feature_std": std.tolist(), "train_seal": seal, "protocol": protocol}, temporary)
    os.replace(temporary, checkpoint)
    status = {"schema": "g09_e2_retention_scorer_v1", "checkpoint": descriptor(checkpoint),
              "protocol": descriptor(PROTOCOL), "train_seal": descriptor(FOLDER / "COMPLETE.json"),
              "E1_control": descriptor(E1_OUTPUT / "COMPLETE.json"),
              "parts": 7440, "candidates": len(y), "positive": counts["positive"],
              "negative": counts["negative"], "train_balanced_BCE": losses,
              "train_retention_loss": components, "elapsed_seconds": time.perf_counter() - started,
              "validation_seen": False, "engineering_seen": False}
    write_json(OUTPUT / "COMPLETE.json", status)
    return status


def load_scorer(folder=OUTPUT):
    folder = Path(folder)
    complete = read(folder / "COMPLETE.json")
    if complete["schema"] != "g09_e2_retention_scorer_v1" or complete["parts"] != 7440:
        raise ValueError("E2 scorer lacks complete train seal")
    checkpoint = torch.load(checked(complete["checkpoint"]), map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != list(FEATURE_NAMES) or checkpoint["protocol"] != read(
        checked(complete["protocol"])
    ) or checkpoint["train_seal"] != read(checked(complete["train_seal"])):
        raise ValueError("E2 checkpoint/protocol/training identity mismatch")
    model = ObjectScorer(len(FEATURE_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch.tensor(checkpoint["feature_mean"]), torch.tensor(checkpoint["feature_std"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", required=True)
    parser.parse_args()
    result = train()
    print(json.dumps({"train": {key: result[key] for key in (
        "parts", "candidates", "positive", "negative", "elapsed_seconds")},
        "last_balanced_BCE": result["train_balanced_BCE"][-1],
        "last_retention_loss": result["train_retention_loss"][-1]}), flush=True)
