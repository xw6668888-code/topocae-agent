"""Matched A-route ablations on frozen sparse-D0 probabilities."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch

from candidate_pool import build_candidate_pool, capture_legacy_stages
from g09_common import (
    ARTIFACT, FROZEN, G03_ARTIFACT, G07_ARTIFACT, G07_REPORT, REPORT,
    SEEDS, append_jsonl, checked, descriptor, now, read, write_json,
)
from m0_replay import (
    _engineering_truth, _truth_objects, aggregate, d0_identity,
    exact_counts, g07_decoder, object_key,
)
from object_features import deterministic_score, extract_object_features
from object_select import select_objects


MODES = ("old_full", "old_full_plus_pre_boundary", "score_independent", "score_packing")
_CANDIDATE = (
    FROZEN / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718"
    / "candidate_model/candidate_threshold_selection.json"
)
_GLOBAL = _CANDIDATE.parent.parent / "global_decode_selection.json"
_ENGINEERING_SAME = FROZEN / "model_packages/p1_e12_reviewed_legacy_v1/same_instance_threshold.json"
_D0_RUNS = FROZEN / "reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs"


def _unique(*groups):
    values = {}
    for group in groups:
        for obj in group:
            key = object_key(obj)
            values[key] = {"macro_feature": key[0], "face_indices": list(key[1])}
    return list(values.values())


def infer(graph: dict, prediction: dict, same: dict, candidate_config: dict,
          global_config: dict, *, recovery: dict | None = None) -> dict:
    """The inference boundary receives no truth, sample ID, file path, or labels."""
    graph = g07_decoder.clean_graph(graph)
    prediction = g07_decoder.clean_prediction(prediction)
    stages = capture_legacy_stages(
        graph, prediction, same, candidate_config, global_config, recovery=recovery
    )
    pool = build_candidate_pool({"R": stages}, face_count=len(graph["nodes"]))
    candidates = []
    for candidate in pool:
        features = extract_object_features(candidate, prediction, graph)
        candidates.append({**candidate, "features": features,
                           "score": deterministic_score(features)})
    selected = select_objects(candidates, exact_limit=32, beam_width=128)
    return {
        "candidates": candidates,
        "stages": stages,
        "objects": {
            "old_full": stages["final"],
            "old_full_plus_pre_boundary": _unique(stages["final"], stages["pre_boundary"]),
            "score_independent": [
                {"macro_feature": row["macro_feature"], "face_indices": row["face_indices"]}
                for row in candidates if row["score"] > 1e-9
            ],
            "score_packing": selected["selected_objects"],
        },
        "selection_diagnostics": selected["diagnostics"],
    }


def score_sample(truth: list[dict], output: dict) -> tuple[dict, dict]:
    pool_keys = {object_key(row) for row in output["candidates"]}
    scores = {mode: exact_counts(truth, output["objects"][mode]) for mode in MODES}
    missed = {}
    for mode in MODES:
        selected = Counter(object_key(row) for row in output["objects"][mode])
        missed[mode] = [
            {"macro_feature": object_key(row)[0], "face_indices": list(object_key(row)[1]),
             "pool_present": object_key(row) in pool_keys}
            for row in truth if not selected[object_key(row)]
        ]
    details = {
        "truth_count": len(truth), "candidate_count": len(pool_keys),
        "candidate_coverage": sum(object_key(row) in pool_keys for row in truth),
        "scores": {mode: {key: scores[mode][key] for key in ("tp", "fp", "fn")} for mode in MODES},
        "missing": missed,
        "selection": output["selection_diagnostics"],
    }
    return scores, details


def _summarize(scores: dict[str, list[dict]], parts: list[dict]) -> dict:
    result = {mode: aggregate(scores[mode]) for mode in MODES}
    result["candidate_coverage"] = sum(row["candidate_coverage"] for row in parts)
    result["candidate_missing_fn"] = sum(row["truth_count"] - row["candidate_coverage"] for row in parts)
    result["pool_size"] = sum(row["candidate_count"] for row in parts)
    result["beam_uncertified_parts"] = sum(
        not row["selection"]["all_optimality_certified"] for row in parts
    )
    result["beam_truncation_components"] = sum(
        row["selection"]["beam_truncation_count"] for row in parts
    )
    result["ambiguous_candidates"] = sum(
        component["ambiguous_objects"]
        for row in parts for component in row["selection"]["components"]
    )
    result["packing_selection_lost_fn"] = sum(
        sum(item["pool_present"] for item in row["missing"]["score_packing"])
        for row in parts
    )
    return result


def _engineering(candidate_config, global_config, output):
    truth, _ = _engineering_truth()
    baseline_path = G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json"
    baseline = read(baseline_path)
    same = read(_ENGINEERING_SAME)
    scores = {mode: [] for mode in MODES}
    parts = []
    began = time.perf_counter()
    for part_row in baseline["per_part"]:
        archive = checked(part_row["prediction"]).parent
        source = read(archive / "guard_input_graph.json")
        probability = read(archive / "probabilities.json")["guard"]
        result = infer(source, probability, same, candidate_config, global_config)
        per_mode, detail = score_sample(_truth_objects(truth, part_row["part"]), result)
        for mode in MODES:
            scores[mode].append(per_mode[mode])
        parts.append(detail)
        output.write(json.dumps({"domain": "engineering_development", "part": part_row["part"],
                                 **detail}, ensure_ascii=False) + "\n")
    summary = _summarize(scores, parts)
    summary["elapsed_seconds"] = time.perf_counter() - began
    summary["source"] = descriptor(baseline_path)
    return summary


def _d0_seed(seed, candidate_config, global_config, output):
    identity = d0_identity(seed)
    folder = G03_ARTIFACT / f"baseline_replay/seed_{seed}"
    manifest = read(folder / "prediction_manifest.json")
    predictions = torch.load(checked(manifest["predictions"]), map_location="cpu", weights_only=False)
    trace_path = checked(read(folder / "summary.json")["traces"])
    membership = set(read(checked(identity["sample_manifest"]))["identities"]["val"])
    same = read(_D0_RUNS / f"final_8370_seed_{seed}/same_instance_threshold_selection.json")
    scores = {mode: [] for mode in MODES}
    parts = []
    began = time.perf_counter()
    with trace_path.open(encoding="utf-8") as stream:
        for index, (prediction, line) in enumerate(zip(predictions, stream, strict=True)):
            stage_record = json.loads(line)
            record = prediction["record"]
            sample_id = record["sample_id"]
            if sample_id not in membership or sample_id != stage_record["sample_id"]:
                raise ValueError("D0 graph/probability/stage membership mismatch")
            graph = read(Path(record["source_graph_path"]))
            result = infer(
                graph,
                {key: prediction[key] for key in ("faces", "edge_probabilities")},
                same, candidate_config, global_config,
                recovery={"objects": stage_record["objects"], "trace": stage_record["trace"]},
            )
            per_mode, detail = score_sample(stage_record["truth_objects"], result)
            for mode in MODES:
                scores[mode].append(per_mode[mode])
            parts.append(detail)
            output.write(json.dumps({"domain": "D0_validation", "seed": seed,
                                     "sample_id": sample_id, **detail}, ensure_ascii=False) + "\n")
            if (index + 1) % 200 == 0:
                print(f"A seed={seed} {index + 1}/930", flush=True)
    if len(parts) != 930:
        raise ValueError("D0 development denominator changed")
    summary = _summarize(scores, parts)
    summary["elapsed_seconds"] = time.perf_counter() - began
    summary["identity"] = identity
    summary["trace"] = descriptor(trace_path)
    return summary


def run(engineering_only=False):
    path = REPORT / ("SELECTION_ABLATIONS_ENGINEERING_PILOT.json" if engineering_only
                     else "SELECTION_ABLATIONS_FULL.json")
    if path.exists():
        raise FileExistsError(path)
    candidate_config, global_config = read(_CANDIDATE), read(_GLOBAL)
    output_path = ARTIFACT / ("m1/A_ROUTE_ENGINEERING_PILOT.jsonl" if engineering_only
                              else "m1/A_ROUTE_FULL_PER_SAMPLE.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".jsonl.tmp")
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as output:
        engineering = _engineering(candidate_config, global_config, output)
        print("Engineering A result:", json.dumps({
            mode: {key: engineering[mode][key] for key in ("tp", "fp", "fn", "all_exact_parts")}
            for mode in MODES}, ensure_ascii=False), flush=True)
        validation = {} if engineering_only else {
            str(seed): _d0_seed(seed, candidate_config, global_config, output)
            for seed in SEEDS
        }
    temporary.replace(output_path)
    result = {
        "schema": "g09_route_a_first_registered_ablation_v1", "at_utc": now(),
        "modes": list(MODES), "data": "Frozen sparse D0 three seed validation and 13 engineering development parts",
        "single_fixed_initial_scoring_configuration": True, "truth_used_in_inference": False,
        "candidate_sources": "R eight stages; no Q and no new model training",
        "score_formula_source": descriptor(Path(__file__).with_name("object_features.py")),
        "selector_source": descriptor(Path(__file__).with_name("object_select.py")),
        "candidate_source": descriptor(Path(__file__).with_name("candidate_pool.py")),
        "candidate_threshold": descriptor(_CANDIDATE), "global_decode": descriptor(_GLOBAL),
        "engineering_same_instance": descriptor(_ENGINEERING_SAME),
        "engineering": engineering, "D0_validation": validation,
        "per_sample": descriptor(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_seconds": 0.0,
        "final_test_exposed": False,
    }
    write_json(path, result)
    if not engineering_only:
        for experiment_id, mode, control in (
            ("A1-deterministic-score", "score_independent", "old_full_plus_pre_boundary"),
            ("A2-constrained-selection", "score_packing", "score_independent"),
        ):
            append_jsonl(REPORT / "EXPERIMENT_QUEUE.jsonl", {
                "schema": "g09_experiment_event_v1", "at_utc": now(),
                "event": "completed", "experiment_id": experiment_id,
                "hypothesis": "Complete-object scoring/selection may reduce errors without changing candidate ceiling.",
                "control": control, "actual_result": {
                    "engineering": {key: engineering[mode][key] for key in ("tp", "fp", "fn", "all_exact_parts")},
                    "validation_macro_f1": {seed: val[mode]["macro_f1"] for seed, val in validation.items()},
                    "validation_tp_fp_fn": {seed: [val[mode][key] for key in ("tp", "fp", "fn")]
                                            for seed, val in validation.items()},
                },
                "instance_error_change": {
                    "engineering": {key: engineering[mode][key] - engineering[control][key]
                                    for key in ("tp", "fp", "fn")},
                    "D0_seed_20260719": {key: validation["20260719"][mode][key] - validation["20260719"][control][key]
                                          for key in ("tp", "fp", "fn")},
                },
                "next_action": "Check candidate absence versus selection and engineering regression; do not tune this result against the final test.",
                "evidence": descriptor(path),
            })
    print(json.dumps({
        "engineering": {mode: [engineering[mode][key] for key in ("tp", "fp", "fn", "all_exact_parts")]
                        for mode in MODES},
        "D0_mean_macro_f1": {mode: statistics.mean(validation[str(seed)][mode]["macro_f1"] for seed in SEEDS)
                             for mode in MODES} if validation else None,
        "candidate_coverage": {seed: val["candidate_coverage"] for seed, val in validation.items()},
    }, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-only", action="store_true")
    run(parser.parse_args().engineering_only)
