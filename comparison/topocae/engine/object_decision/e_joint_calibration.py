"""E3: one train-only hierarchical Bayesian class/source prior correction.

No neural retraining, engineering-specific switch, class-specific hand value,
or validation-informed threshold. The E1 scorer and object selector are frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from e_joint_eval import (CONFIG, FOLDER as E1_FOLDER, engineering,
                          infer as infer_e1, validation)
from e_joint_retention_eval import CONTROL_PATHS, compare, paired_rows
from e_joint_source_priors import OUTPUT as PRIOR_PATH, source_group
from e_joint_scorer import OUTPUT as SCORER_PATH, load_scorer
from g09_common import (ARTIFACT, REPORT, SEEDS, checked, descriptor, now,
                        read, write_json)
from object_select import select_objects
from route_a4_eval import load_scorer as load_a4
from route_a_eval import _CANDIDATE, _GLOBAL


FOLDER = ARTIFACT / "e_joint_calibration_eval"
PROTOCOL = REPORT / "E3_BAYESIAN_CALIBRATION_PROTOCOL.json"
KAPPA = 20.0


def logit(value):
    if not 0 < value < 1 or not math.isfinite(value):
        raise ValueError("Prior odds must be strictly finite")
    return math.log(value / (1 - value))


def class_source_prior(data, macro, group):
    """Twenty source-specific pseudo-observations shrink rare class cells."""
    source = data["by_source"][group]["jeffreys_mean"]
    cell = data["by_class"][macro][group]
    return (cell["positive"] + KAPPA * source) / (cell["candidates"] + KAPPA)


def calibrated_score(raw_logit, candidate, prior, temperature):
    """Only predicted class/provenance and frozen E1 logit affect inference."""
    if not math.isfinite(raw_logit):
        raise ValueError("Nonfinite frozen E1 object logit")
    group = source_group(candidate)
    value = class_source_prior(prior, candidate["macro_feature"], group)
    return raw_logit / temperature + logit(value) - logit(prior["overall"]["jeffreys_mean"])


def register():
    prior = read(PRIOR_PATH)
    if prior["schema"] != "g09_e_joint_train_source_priors_v1" \
            or prior["no_validation_or_engineering_labels"] is not True:
        raise ValueError("E3 priors not restricted to frozen D0 train labels")
    e1 = read(SCORER_PATH / "COMPLETE.json")
    if prior["train_seal"] != e1["train_seal"] or e1["schema"] != "g09_e_joint_scorer_v1":
        raise ValueError("E3 prior denominator differs from E1 scorer training set")
    e2_protocol = read(REPORT / "E2_RETENTION_PROTOCOL.json")
    if e2_protocol["E1_control"] != descriptor(SCORER_PATH / "COMPLETE.json"):
        raise ValueError("Train-only E1 positive score quantile disagrees with E3 scorer")
    q25 = e2_protocol["train_only_margin_basis"]["R_final_positive_q25"]
    r_rate = prior["by_source"]["R_final"]["jeffreys_mean"]
    ratio = q25 / logit(r_rate)
    temperature = math.floor(ratio + .5)
    if temperature != 3 or temperature < 1:
        raise ValueError("Registered train-only E3 temperature changed")
    fixed = {"formula": "E1_logit / T + logit(p_class_source) - logit(p_all)",
             "source_partition": prior["group_rule"],
             "p_class_source": "(class_source_exact_positive + 20 * Jeffreys_source_mean) / (class_source_candidates + 20)",
             "p_all": "Jeffreys Beta(0.5,0.5) overall training proposal mean",
             "T": temperature, "T_from_train": "round_half_up(E1_R_final_exact_positive_q25 / logit(Jeffreys_R_final_train_precision))",
             "T_train_ratio": ratio, "q25_train": q25,
             "R_final_train_precision": r_rate, "kappa": KAPPA,
             "threshold": 0.0, "selector_exact_limit": 32, "selector_beam_width": 128,
             "no_identity_or_truth_in_inference": True,
             "no_eng_or_validation_tuned_constants": True}
    if PROTOCOL.exists():
        protocol = read(PROTOCOL)
        if protocol["fixed_rule"] != fixed or protocol["prior"] != descriptor(PRIOR_PATH) \
                or protocol["E1_scorer"] != descriptor(SCORER_PATH / "COMPLETE.json"):
            raise ValueError("Registered E3 prior rule/identity differs")
    else:
        protocol = {"schema": "g09_e3_bayesian_calibration_protocol_v1",
                    "created_utc": now(),
                    "hypothesis": "Train-only class/source proposal precision corrects miscalibrated E1 complete-object logits under rare transition and branch-origin shift while retaining Q-only complement candidates.",
                    "unique_change": "one fixed hierarchical Bayesian class/source log-odds correction and train-derived temperature, no neural retrain",
                    "matched_controls": "E1/E2 paired R/Q/direct union/A4 on D0 three seeds and 13 engineering development parts",
                    "prior": descriptor(PRIOR_PATH),
                    "E1_scorer": descriptor(SCORER_PATH / "COMPLETE.json"),
                    "train_q25_source": descriptor(REPORT / "E2_RETENTION_PROTOCOL.json"),
                    "fixed_rule": fixed,
                    "final_test_exposed": False, "candidate_pool_or_features_changed": False,
                    "scorer_configuration_slot": 4}
        write_json(PROTOCOL, protocol)
    return protocol, prior


def infer_e3(graph, r_pred, q_pred, r_same, q_same, configs,
             model, mean, std, a4, a4_mean, a4_std, *, prior, temperature,
             r_recovery=None, q_recovery=None):
    """Runs frozen E1 inference, then exactly one registered score transform."""
    started = time.perf_counter()
    result = infer_e1(graph, r_pred, q_pred, r_same, q_same, configs,
                      model, mean, std, a4, a4_mean, a4_std,
                      r_recovery=r_recovery, q_recovery=q_recovery)
    adjusted = []
    for row in result["candidates"]:
        adjusted.append({**row, "raw_E1_logit": row["score"],
                         "score": calibrated_score(row["score"], row, prior, temperature)})
    selected = select_objects(adjusted, exact_limit=32, beam_width=128)
    result["candidates"] = adjusted
    result["objects"]["E_joint_packing"] = selected["selected_objects"]
    result["selector"] = selected["diagnostics"]
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def run(*, seed=None, include_engineering=False):
    if (seed is None) == (not include_engineering) or (seed is not None and seed not in SEEDS):
        raise ValueError("Specify one frozen seed or engineering development")
    name = "engineering_seed_20260719" if include_engineering else f"D0_seed_{seed}"
    protocol, prior = register()
    if (FOLDER / f"{name}.json").exists() or (FOLDER / f"{name}.jsonl").exists():
        raise FileExistsError("Retained E3 evaluation exists; no overwrite")
    e1 = read(CONTROL_PATHS[name])
    if e1["joint_scorer"] != protocol["E1_scorer"] or e1["fixed_rule"] != CONFIG:
        raise ValueError("E3/E1 paired control is not the registered model and rule")
    scorer, mean, std = load_scorer()
    a4, a4_mean, a4_std = load_a4(ARTIFACT / "object_scorer/A4_seed_20260719")
    models = (scorer, mean, std, a4, a4_mean, a4_std)
    configs = read(_CANDIDATE), read(_GLOBAL)
    temperature = protocol["fixed_rule"]["T"]

    def inference(*args, **kwargs):
        return infer_e3(*args, prior=prior, temperature=temperature, **kwargs)

    torch.set_num_threads(4)
    FOLDER.mkdir(parents=True, exist_ok=True)
    temporary = FOLDER / f"{name}.jsonl.partial"
    if temporary.exists():
        raise FileExistsError("Retained E3 partial output exists")
    started = time.perf_counter()
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            result = engineering(models, configs, stream, infer_fn=inference) if include_engineering else \
                     validation(seed, models, configs, stream, infer_fn=inference)
        final = FOLDER / f"{name}.jsonl"
        temporary.replace(final)
        changes = compare(e1, result, e1["domain"])
        changes["paired_parts"] = paired_rows(CONTROL_PATHS[name], final, include_engineering)
        status = {"schema": "g09_e3_bayesian_calibration_evaluation_v1",
                  "domain": name, "protocol": descriptor(PROTOCOL),
                  "prior": descriptor(PRIOR_PATH),
                  "E1_scorer": protocol["E1_scorer"], "E1_control": descriptor(CONTROL_PATHS[name]),
                  "per_sample": descriptor(final), "result": result, "paired_changes": changes,
                  "elapsed_seconds": time.perf_counter() - started,
                  "final_test_exposed": False, "truth_in_inference": False}
        write_json(FOLDER / f"{name}.json", status)
        return status
    except Exception as exc:
        write_json(FOLDER / f"{name}_FAILED.json", {"error": repr(exc),
            "partial_retained": temporary.exists(),
            "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    result = run(seed=args.seed, include_engineering=args.engineering)
    print(json.dumps({"domain": result["domain"], "E3": {key: result["result"]["E_joint_packing"][key]
        for key in ("tp", "fp", "fn", "macro_f1", "all_exact_parts")},
        "R_final_selected": {key: result["result"][key] for key in (
            "R_final_selected_TP", "R_final_selected_FP", "R_final_positive_rejected")},
        "Q_only_selected": {key: result["result"][key] for key in (
            "Q_only_positive_selected", "Q_only_selected_negative")},
        "paired_changes": result["paired_changes"], "elapsed_seconds": result["elapsed_seconds"]},
        ensure_ascii=False), flush=True)
