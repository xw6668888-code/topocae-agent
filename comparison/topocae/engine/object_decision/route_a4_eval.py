"""Matched D0/engineering evaluation of the train-only A4 object scorer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch

from g09_common import (ARTIFACT, G03_ARTIFACT, G07_REPORT, REPORT, SEEDS,
                        append_jsonl, checked, descriptor, now, read, write_json)
from m0_replay import _engineering_truth, _truth_objects, aggregate, d0_identity, exact_counts, object_key
from object_features import NUMERIC_FEATURE_NAMES
from object_select import select_objects
from route_a_eval import (_CANDIDATE, _GLOBAL, _ENGINEERING_SAME, _D0_RUNS,
                          infer as initial_infer)
from train_object_scorer import ObjectScorer


MODES = ("old_full", "score_packing", "learned_independent", "learned_packing")


def load_scorer(path: Path):
    complete = read(path / "COMPLETE.json")
    if complete["schema"] != "g09_a4_scorer_training_v1" or complete["parts"] != 7440:
        raise ValueError("Only complete D0-train A4 scorer is eligible")
    checkpoint = torch.load(checked(complete["checkpoint"]), map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != list(NUMERIC_FEATURE_NAMES):
        raise ValueError("A4 feature order changed")
    if checkpoint["training_data_identity"] != read(checked(complete["training_cache"])):
        raise ValueError("A4 train-only seal changed")
    if checkpoint["protocol"] != read(checked(complete["protocol"])):
        raise ValueError("A4 registered protocol changed")
    model = ObjectScorer(len(NUMERIC_FEATURE_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch.tensor(checkpoint["feature_mean"]), torch.tensor(checkpoint["feature_std"])


def infer(graph, prediction, same, candidate_config, global_config, scorer, mean, std, *, recovery=None):
    """No truth, sample ID, path, or engineering-set flag enters selection."""
    initial = initial_infer(graph, prediction, same, candidate_config, global_config, recovery=recovery)
    candidates = initial["candidates"]
    if candidates:
        vectors = torch.tensor([row["features"]["vector"] for row in candidates], dtype=torch.float32)
        with torch.no_grad():
            logits = scorer(((vectors - mean) / std).clamp(-8, 8)).tolist()
    else:
        logits = []
    learned = [{**row, "score": score} for row, score in zip(candidates, logits)]
    packed = select_objects(learned, exact_limit=32, beam_width=128)
    initial["objects"]["learned_independent"] = [
        {"macro_feature": row["macro_feature"], "face_indices": row["face_indices"]}
        for row in learned if row["score"] > 1e-9
    ]
    initial["objects"]["learned_packing"] = packed["selected_objects"]
    initial["learned_selection"] = packed["diagnostics"]
    return initial


def score_one(truth, result):
    pool = {object_key(row) for row in result["candidates"]}
    score = {mode: exact_counts(truth, result["objects"][mode]) for mode in MODES}
    missed = {}
    for mode in MODES:
        selected = Counter(object_key(row) for row in result["objects"][mode])
        missed[mode] = [
            {"macro_feature": object_key(row)[0], "face_indices": list(object_key(row)[1]),
             "pool_present": object_key(row) in pool}
            for row in truth if not selected[object_key(row)]
        ]
    return score, {"truth_count": len(truth), "candidate_count": len(pool),
                   "candidate_coverage": sum(object_key(row) in pool for row in truth),
                   "counts": {mode: {key: score[mode][key] for key in ("tp", "fp", "fn")}
                              for mode in MODES},
                   "missing": missed, "learned_selection": result["learned_selection"]}


def summarize(rows, details):
    result = {mode: aggregate(rows[mode]) for mode in MODES}
    result["candidate_coverage"] = sum(row["candidate_coverage"] for row in details)
    result["candidate_missing_fn"] = sum(row["truth_count"] - row["candidate_coverage"] for row in details)
    result["pool_size"] = sum(row["candidate_count"] for row in details)
    result["learned_selection_lost_fn"] = sum(
        item["pool_present"] for row in details for item in row["missing"]["learned_packing"])
    result["beam_uncertified_parts"] = sum(
        not row["learned_selection"]["all_optimality_certified"] for row in details)
    return result


def engineering(candidate_config, global_config, scorer, mean, std, output):
    truth, _ = _engineering_truth()
    baseline_path = G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json"
    same = read(_ENGINEERING_SAME)
    scores, details = {mode: [] for mode in MODES}, []
    for part in read(baseline_path)["per_part"]:
        folder = checked(part["prediction"]).parent
        graph = read(folder / "guard_input_graph.json")
        prediction = read(folder / "probabilities.json")["guard"]
        result = infer(graph, prediction, same, candidate_config, global_config, scorer, mean, std)
        per_mode, detail = score_one(_truth_objects(truth, part["part"]), result)
        for mode in MODES:
            scores[mode].append(per_mode[mode])
        details.append(detail)
        output.write(json.dumps({"domain": "engineering_development", "part": part["part"],
                                 **detail}, ensure_ascii=False) + "\n")
    return {**summarize(scores, details), "source": descriptor(baseline_path)}


def validation(seed, candidate_config, global_config, scorer, mean, std, output):
    identity = d0_identity(seed)
    folder = G03_ARTIFACT / f"baseline_replay/seed_{seed}"
    cache = read(folder / "prediction_manifest.json")
    predictions = torch.load(checked(cache["predictions"]), map_location="cpu", weights_only=False)
    trace_path = checked(read(folder / "summary.json")["traces"])
    membership = set(read(checked(identity["sample_manifest"]))["identities"]["val"])
    same = read(_D0_RUNS / f"final_8370_seed_{seed}/same_instance_threshold_selection.json")
    scores, details = {mode: [] for mode in MODES}, []
    with trace_path.open(encoding="utf-8") as stream:
        for index, (prediction, line) in enumerate(zip(predictions, stream, strict=True)):
            stage = json.loads(line)
            record = prediction["record"]
            sample_id = record["sample_id"]
            if sample_id not in membership or sample_id != stage["sample_id"]:
                raise ValueError("D0 validation prediction/trace membership mismatch")
            graph = read(record["source_graph_path"])
            result = infer(graph, {key: prediction[key] for key in ("faces", "edge_probabilities")},
                           same, candidate_config, global_config, scorer, mean, std,
                           recovery={"objects": stage["objects"], "trace": stage["trace"]})
            per_mode, detail = score_one(stage["truth_objects"], result)
            for mode in MODES:
                scores[mode].append(per_mode[mode])
            details.append(detail)
            output.write(json.dumps({"domain": "D0_validation", "seed": seed,
                                     "sample_id": sample_id, **detail}, ensure_ascii=False) + "\n")
            if (index + 1) % 200 == 0:
                print(f"A4 seed={seed} {index + 1}/930", flush=True)
    if len(details) != 930:
        raise ValueError("D0 validation denominator changed")
    return {**summarize(scores, details), "identity": identity, "trace": descriptor(trace_path)}


def run(scorer_folder: Path):
    path = REPORT / "A4_MATCHED_EVALUATION.json"
    if path.exists():
        raise FileExistsError(path)
    complete = descriptor(scorer_folder / "COMPLETE.json")
    model, mean, std = load_scorer(scorer_folder)
    config, global_config = read(_CANDIDATE), read(_GLOBAL)
    append_jsonl(REPORT / "EXPERIMENT_QUEUE.jsonl", {
        "schema": "g09_experiment_event_v1", "event": "evaluation_registered", "at_utc": now(),
        "experiment_id": "A3-learned-object-score", "parent_configuration": "A2-constrained-selection",
        "unique_primary_change": "Train-only learned object score replaces fixed score on the same eight-stage R pool",
        "matching_control": "Old R full and A2 packing on identical D0 seeds and engineering inputs",
        "identity": complete, "expected_error_change": "Reduce pool-present FN and FP without changing candidate ceiling",
        "next_action": "Route absent truth candidates to B/E; do not sweep engineering thresholds"})
    per_sample = ARTIFACT / "m1/A4_MATCHED_PER_SAMPLE.jsonl"
    per_sample.parent.mkdir(parents=True, exist_ok=True)
    temporary = per_sample.with_suffix(".jsonl.tmp")
    began = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as output:
        eng = engineering(config, global_config, model, mean, std, output)
        val = {str(seed): validation(seed, config, global_config, model, mean, std, output)
               for seed in SEEDS}
    temporary.replace(per_sample)
    result = {"schema": "g09_a4_matched_evaluation_v1", "at_utc": now(),
              "scorer": complete, "candidate_source": descriptor(_CANDIDATE),
              "global_decode": descriptor(_GLOBAL), "engineering": eng, "D0_validation": val,
              "modes": list(MODES), "per_sample": descriptor(per_sample),
              "truth_used_in_inference": False, "final_test_exposed": False,
              "elapsed_seconds": time.perf_counter() - began}
    write_json(path, result)
    append_jsonl(REPORT / "EXPERIMENT_QUEUE.jsonl", {
        "schema": "g09_experiment_event_v1", "event": "completed", "at_utc": now(),
        "experiment_id": "A3-learned-object-score", "evidence": descriptor(path),
        "actual_result": {"engineering": {mode: [eng[mode][key] for key in ("tp", "fp", "fn", "all_exact_parts")]
                                          for mode in MODES},
                          "D0_mean_macro_f1": {mode: statistics.mean(val[str(seed)][mode]["macro_f1"] for seed in SEEDS)
                                               for mode in MODES}},
        "instance_error_change": {"engineering": {key: eng["learned_packing"][key] - eng["old_full"][key]
                                                  for key in ("tp", "fp", "fn")},
                                  "D0_seed_20260719": {key: val["20260719"]["learned_packing"][key] -
                                                        val["20260719"]["old_full"][key]
                                                        for key in ("tp", "fp", "fn")}},
        "next_action": "Choose A scorer by matched validation and engineering regression, route candidate absence to B/E."})
    print(json.dumps({"engineering": {mode: [eng[mode][key] for key in ("tp", "fp", "fn", "all_exact_parts")]
                                          for mode in MODES},
                      "D0_mean_macro_f1": {mode: statistics.mean(val[str(seed)][mode]["macro_f1"] for seed in SEEDS)
                                           for mode in MODES},
                      "missing_candidates": {seed: row["candidate_missing_fn"] for seed, row in val.items()}}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-folder", required=True, type=Path)
    run(parser.parse_args().scorer_folder)
