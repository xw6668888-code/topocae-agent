"""One preregistered branch-aware complete-object scorer on paired D0 train.

The R and Q evidence vectors are jointly standardized using train candidates;
the two source flags are learned features, not branch-favoring inference rules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch

from e_joint_train_data import FEATURE_NAMES, source_identity_joint
from g09_common import ARTIFACT, REPORT, checked, descriptor, now, read, write_json
from train_object_scorer import ObjectScorer, balanced_part_loss


FOLDER = ARTIFACT / "e_joint_train/full"
OUTPUT = ARTIFACT / "e_joint_scorer/seed_20260719"
PROTOCOL = REPORT / "E_JOINT_SCORER_PROTOCOL.json"
RULE = {"architecture": "MLP_66_to_64_to_32_to_1", "epochs": 30,
        "batch_parts": 64, "seed": 20260719,
        "optimizer": "AdamW_lr_0.001_weight_decay_0.0001",
        "normalization": "D0_train_candidate_mean_std_clamp_z_8",
        "loss": "equal_positive_negative_BCE_per_part_then_mean_parts",
        "inference": "one_joint_logit_per_canonical_candidate; independent-set packing; logit > 0; exact_limit 32; beam_width 128",
        "validation_gradients": False, "engineering_gradients": False,
        "branch_prior": "none; R and Q source flags and predicted evidence learned from D0 train only"}


def load_train(folder=FOLDER):
    folder = Path(folder).resolve()
    seal = read(folder / "COMPLETE.json")
    if seal.get("schema") != "g09_e_joint_train_data_v1" or seal.get("identity") != source_identity_joint(7440):
        raise ValueError("Joint scorer accepts only the frozen full D0 train R+Q extraction")
    if seal["stats"]["parts"] != 7440 or seal["identity"]["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Joint feature order or train denominator changed")
    outputs = {Path(item["path"]).name: checked(item) for item in seal["outputs"]}
    if set(outputs) != {"candidates.jsonl", "labels.jsonl"} or any(
        path.parent != folder for path in outputs.values()
    ):
        raise ValueError("Joint candidate/label sidecars are not bound to the train seal")
    n = seal["stats"]["candidates"]
    x = torch.empty((n, len(FEATURE_NAMES)), dtype=torch.float32)
    y = torch.empty(n, dtype=torch.long)
    groups = torch.empty(n, dtype=torch.long)
    totals = Counter()
    cursor = 0
    with outputs["candidates.jsonl"].open(encoding="utf-8") as features, \
         outputs["labels.jsonl"].open(encoding="utf-8") as labels:
        for part, (fline, lline) in enumerate(zip(features, labels, strict=True)):
            frow, lrow = json.loads(fline), json.loads(lline)
            if frow["part_index"] != part or lrow["part_index"] != part:
                raise ValueError("Joint candidates and labels are not paired by part")
            candidates, targets = frow["candidates"], lrow["labels"]
            if len(candidates) != len(targets):
                raise ValueError("Joint candidates/labels differ in count")
            truth = {(obj["macro_feature"], tuple(obj["face_indices"]))
                     for obj in lrow["truth_objects"]}
            vectors = []
            for index, (candidate, target) in enumerate(zip(candidates, targets)):
                vector = candidate["vector"]
                if len(vector) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in vector):
                    raise ValueError("Joint feature vector malformed")
                branches = {origin["branch"] for origin in candidate["provenance"]}
                if not branches or not branches <= {"R", "Q"} or vector[-2:] != [
                    float("R" in branches), float("Q" in branches)
                ]:
                    raise ValueError("Joint source flag disagrees with provenance")
                correct = ((candidate["macro_feature"], tuple(candidate["face_indices"])) in truth)
                if target["candidate_index"] != index or target["exact_positive"] is not correct:
                    raise ValueError("Joint target does not match complete class+face truth")
                vectors.append(vector)
                totals["positive" if correct else "negative"] += 1
                y[cursor + index] = int(correct)
            end = cursor + len(vectors)
            if end > n:
                raise ValueError("Joint candidate denominator overflow")
            if vectors:
                x[cursor:end] = torch.tensor(vectors, dtype=torch.float32)
                groups[cursor:end] = part
            cursor = end
            totals["parts"] += 1
    if totals["parts"] != 7440 or cursor != n or totals["positive"] != seal["stats"]["positive"]:
        raise ValueError("Incomplete joint D0 train extraction")
    return seal, x, y, groups, dict(totals)


def train():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("E joint scorer attempt already exists; preserve it")
    seal, x, y, groups, counts = load_train()
    protocol = {"schema": "g09_e_joint_scorer_protocol_v1", "created_utc": now(),
                "hypothesis": "Train-only joint R/Q class and relation evidence with proposal source flags can accept complementary Q-only complete objects while suppressing Q false positives.",
                "matched_control": "Frozen R, frozen Q, direct union, and A4 R-only on the same D0 seeds and engineering inputs",
                "unique_change": "joint R+Q candidate origin and both predicted evidence vectors",
                "train_seal": descriptor(FOLDER / "COMPLETE.json"),
                "feature_names": list(FEATURE_NAMES), "fixed_rule": RULE}
    if PROTOCOL.exists():
        raise FileExistsError("E joint protocol already registered; preserve it")
    write_json(PROTOCOL, protocol)
    started = time.perf_counter()
    torch.manual_seed(RULE["seed"])
    random.seed(RULE["seed"])
    torch.set_num_threads(4)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x = ((x - mean) / std).clamp(-8, 8)
    model = ObjectScorer(len(FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positions = torch.searchsorted(groups, torch.arange(7441))
    indices = torch.arange(len(y))
    losses = []
    for epoch in range(RULE["epochs"]):
        order = torch.randperm(7440)
        weighted = 0.0
        for block in order.split(RULE["batch_parts"]):
            chosen = torch.cat([indices[positions[p]:positions[p + 1]] for p in block])
            local = torch.repeat_interleave(torch.arange(len(block)), positions[block + 1] - positions[block])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[chosen])
            loss = balanced_part_loss(logits, y[chosen], local, len(block))
            loss.backward()
            optimizer.step()
            weighted += float(loss.detach()) * len(block)
        losses.append(weighted / 7440)
        print(f"E joint epoch={epoch + 1}/30 train_balanced_bce={losses[-1]:.6f}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT / "scorer.pt"
    temporary = checkpoint.with_name(f"scorer.pt.{os.getpid()}.tmp")
    torch.save({"model_state_dict": model.cpu().state_dict(),
                "feature_names": list(FEATURE_NAMES), "feature_mean": mean.tolist(),
                "feature_std": std.tolist(), "train_seal": seal, "protocol": protocol}, temporary)
    os.replace(temporary, checkpoint)
    result = {"schema": "g09_e_joint_scorer_v1", "checkpoint": descriptor(checkpoint),
              "protocol": descriptor(PROTOCOL), "train_seal": descriptor(FOLDER / "COMPLETE.json"),
              "parts": counts["parts"], "candidates": len(y), "positive": counts["positive"],
              "negative": counts["negative"], "losses": losses,
              "elapsed_seconds": time.perf_counter() - started,
              "validation_seen": False, "engineering_seen": False}
    write_json(OUTPUT / "COMPLETE.json", result)
    return result


def load_scorer(folder=OUTPUT):
    folder = Path(folder)
    complete = read(folder / "COMPLETE.json")
    if complete["schema"] != "g09_e_joint_scorer_v1" or complete["parts"] != 7440:
        raise ValueError("Joint scorer has no complete D0 train seal")
    checkpoint = torch.load(checked(complete["checkpoint"]), map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != list(FEATURE_NAMES) or checkpoint["protocol"] != read(
        checked(complete["protocol"])
    ) or checkpoint["train_seal"] != read(checked(complete["train_seal"])):
        raise ValueError("Joint checkpoint/feature/protocol/training identity mismatch")
    model = ObjectScorer(len(FEATURE_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch.tensor(checkpoint["feature_mean"]), torch.tensor(checkpoint["feature_std"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(train(), ensure_ascii=False), flush=True)
