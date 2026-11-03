"""Matched E2 versus E1 on identical D0 seeds and engineering inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from e_joint_eval import (CONFIG, FOLDER as E1_FOLDER, engineering, validation)
from e_joint_retention import OUTPUT, load_scorer
from g09_common import ARTIFACT, SEEDS, checked, descriptor, read, write_json
from route_a4_eval import load_scorer as load_a4
from route_a_eval import _CANDIDATE, _GLOBAL


FOLDER = ARTIFACT / "e_joint_retention_eval"
CONTROL_PATHS = {
    "engineering_seed_20260719": E1_FOLDER / "engineering_seed_20260719.json",
    "D0_seed_20260718": E1_FOLDER / "D0_seed_20260718.json",
    "D0_seed_20260719": E1_FOLDER / "D0_seed_20260719_attempt_3.json",
    "D0_seed_20260720": E1_FOLDER / "D0_seed_20260720.json",
}
MODES = ("R_final", "Q_final", "direct_union", "A4_R_packing", "E_joint_packing")


def compare(control, new, name):
    """Paired same-object controls and per-part gains/losses, not only means."""
    if control["domain"] not in (name, "D0_seed_20260719_attempt_3") \
            or control["fixed_rule"] != CONFIG or control["result"]["parts"] != new["parts"]:
        raise ValueError("E1/E2 control identity or part denominator mismatched")
    for field in ("R_final", "Q_final", "direct_union", "A4_R_packing"):
        for metric in ("tp", "fp", "fn", "macro_f1", "all_exact_parts"):
            if abs(control["result"][field][metric] - new[field][metric]) > 1e-10:
                raise ValueError("Frozen E1/E2 R/Q/union/A4 controls differ")
    return {"E2_minus_E1": {field: new["E_joint_packing"][field] - control["result"]["E_joint_packing"][field]
                             for field in ("tp", "fp", "fn", "macro_f1", "all_exact_parts")},
            "E1_Q_only_positive_selected": control["result"]["Q_only_positive_selected"],
            "E2_Q_only_positive_selected": new["Q_only_positive_selected"],
            "E1_Q_only_selected_negative": control["result"]["Q_only_selected_negative"],
            "E2_Q_only_selected_negative": new["Q_only_selected_negative"]}


def paired_rows(control_path, new_path, engineering_set):
    control = read(control_path)
    old_path = checked(control["per_sample"])
    paired = {"parts_improved_TP": 0, "parts_regressed_TP": 0,
              "parts_improved_FP": 0, "parts_regressed_FP": 0,
              "E1_exact_E2_not": [], "E2_exact_E1_not": []}
    with old_path.open(encoding="utf-8") as original, new_path.open(encoding="utf-8") as trial:
        for left, right in zip(original, trial, strict=True):
            a, b = json.loads(left), json.loads(right)
            identity = "part" if engineering_set else "sample_id"
            if a[identity] != b[identity] or a["truth_count"] != b["truth_count"]:
                raise ValueError("E1/E2 per-sample identity mismatch")
            for mode in MODES[:-1]:
                if a["metrics"][mode] != b["metrics"][mode]:
                    raise ValueError("E1/E2 per-instance control mismatch")
            old, current = a["metrics"]["E_joint_packing"], b["metrics"]["E_joint_packing"]
            paired["parts_improved_TP"] += current["tp"] > old["tp"]
            paired["parts_regressed_TP"] += current["tp"] < old["tp"]
            paired["parts_improved_FP"] += current["fp"] < old["fp"]
            paired["parts_regressed_FP"] += current["fp"] > old["fp"]
            old_exact, new_exact = old["fp"] == old["fn"] == 0, current["fp"] == current["fn"] == 0
            if old_exact and not new_exact:
                paired["E1_exact_E2_not"].append(a[identity])
            if new_exact and not old_exact:
                paired["E2_exact_E1_not"].append(a[identity])
    return paired


def run(*, seed=None, include_engineering=False):
    if (seed is None) == (not include_engineering) or (seed is not None and seed not in SEEDS):
        raise ValueError("Specify one frozen D0 validation seed or engineering development")
    name = "engineering_seed_20260719" if include_engineering else f"D0_seed_{seed}"
    control_path = CONTROL_PATHS[name]
    control = read(control_path)
    if control["joint_scorer"] != descriptor(E1_FOLDER.parent / "e_joint_scorer/seed_20260719/COMPLETE.json"):
        raise ValueError("E1 control scorer identity differs")
    if (FOLDER / f"{name}.json").exists() or (FOLDER / f"{name}.jsonl").exists():
        raise FileExistsError("Retained E2 evaluation exists; no overwrite")
    scorer, mean, std = load_scorer()
    a4, a4_mean, a4_std = load_a4(ARTIFACT / "object_scorer/A4_seed_20260719")
    models = (scorer, mean, std, a4, a4_mean, a4_std)
    configs = read(_CANDIDATE), read(_GLOBAL)
    torch.set_num_threads(4)
    FOLDER.mkdir(parents=True, exist_ok=True)
    partial = FOLDER / f"{name}.jsonl.partial"
    if partial.exists():
        raise FileExistsError("Retained E2 partial output exists")
    started = time.perf_counter()
    try:
        with partial.open("w", encoding="utf-8") as stream:
            result = engineering(models, configs, stream) if include_engineering else \
                     validation(seed, models, configs, stream)
        new_path = FOLDER / f"{name}.jsonl"
        partial.replace(new_path)
        changes = compare(control, result, name)
        changes["paired_parts"] = paired_rows(control_path, new_path, include_engineering)
        status = {"schema": "g09_e2_matched_evaluation_v1", "domain": name,
                  "fixed_inference_rule": CONFIG, "E2_scorer": descriptor(OUTPUT / "COMPLETE.json"),
                  "E1_control": descriptor(control_path),
                  "per_sample": descriptor(new_path), "result": result,
                  "paired_changes": changes, "elapsed_seconds": time.perf_counter() - started,
                  "truth_used_in_inference": False, "final_test_exposed": False}
        write_json(FOLDER / f"{name}.json", status)
        return status
    except Exception as exc:
        write_json(FOLDER / f"{name}_FAILED.json", {"error": repr(exc),
            "partial_retained": partial.exists(), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    status = run(seed=args.seed, include_engineering=args.engineering)
    print(json.dumps({"domain": status["domain"],
        "E2": {name: status["result"]["E_joint_packing"][name]
               for name in ("tp", "fp", "fn", "macro_f1", "all_exact_parts")},
        "changes": status["paired_changes"], "elapsed_seconds": status["elapsed_seconds"]},
        ensure_ascii=False), flush=True)
