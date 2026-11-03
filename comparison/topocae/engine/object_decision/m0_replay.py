"""Correct G08 cache identity and replay matched D0/D1 baselines."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

from cache_identity import CacheIdentityMismatch, require_identity
from g09_common import (
    ARTIFACT,
    FROZEN,
    G03_ARTIFACT,
    G03_TOOLS,
    G07_ARTIFACT,
    G07_REPORT,
    G07_TOOLS,
    G08_ARTIFACT,
    G08_REPORT,
    G08_TOOLS,
    REPORT,
    ROOT,
    SEEDS,
    TOOLS,
    append_jsonl,
    checked,
    descriptor,
    digest,
    now,
    read,
    write_json,
    write_text,
)


for source in (G07_TOOLS, G03_TOOLS, FROZEN):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import decoder as g07_decoder
from input_audit_followup import parse_manual_truth


TARGET_CLASSES = tuple(g07_decoder.TARGET_CLASSES)
D0_DATA = FROZEN / "output_training_p1_e6r2_r3_graph_v2_2_1_v1"
D0_RUNS = FROZEN / "reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs"
D1_VALIDATION = G07_ARTIFACT / "validation_predictions/D1/dense_geometry"
D1_TRAINING = G07_ARTIFACT / "training_runs/D1/dense_geometry"
D1_ENGINEERING_RUN = (
    G07_REPORT
    / "engineering/runs/D1_dense_geometry_seed_20260720_formal_597df7821fa89138a623"
)


def object_key(row: dict) -> tuple[str, tuple[int, ...]]:
    label = row.get("class", row.get("macro_feature"))
    faces = row.get("faces", row.get("face_indices", []))
    return str(label), tuple(sorted(int(face) for face in faces))


def canonical(rows: Iterable[dict]) -> list[tuple[str, tuple[int, ...]]]:
    return sorted(object_key(row) for row in rows)


def exact_counts(truth: Iterable[dict], predicted: Iterable[dict]) -> dict:
    truth_counter = Counter(object_key(row) for row in truth)
    predicted_counter = Counter(object_key(row) for row in predicted)
    matched = truth_counter & predicted_counter
    per_class = {}
    for label in TARGET_CLASSES:
        support = sum(count for (name, _), count in truth_counter.items() if name == label)
        prediction = sum(
            count for (name, _), count in predicted_counter.items() if name == label
        )
        tp = sum(count for (name, _), count in matched.items() if name == label)
        per_class[label] = {"support": support, "prediction": prediction, "tp": tp}
    tp = sum(matched.values())
    truth_count = sum(truth_counter.values())
    prediction_count = sum(predicted_counter.values())
    return {
        "truth_count": truth_count,
        "prediction_count": prediction_count,
        "tp": tp,
        "fp": prediction_count - tp,
        "fn": truth_count - tp,
        "per_class": per_class,
    }


def aggregate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    per_class = {}
    for label in TARGET_CLASSES:
        support = sum(row["per_class"][label]["support"] for row in rows)
        prediction = sum(row["per_class"][label]["prediction"] for row in rows)
        tp = sum(row["per_class"][label]["tp"] for row in rows)
        precision = tp / prediction if prediction else (1.0 if support == 0 else 0.0)
        recall = tp / support if support else (1.0 if prediction == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "prediction": prediction,
            "tp": tp,
            "fp": prediction - tp,
            "fn": support - tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "parts": len(rows),
        "truth_count": tp + fn,
        "prediction_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "macro_f1": statistics.mean(per_class[label]["f1"] for label in TARGET_CLASSES),
        "all_exact_parts": sum(row["fp"] == row["fn"] == 0 for row in rows),
        "per_class": per_class,
    }


def compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "truth_count",
            "prediction_count",
            "tp",
            "fp",
            "fn",
        )
    }


def _d0_expected(seed: int) -> dict:
    return {
        "schema": "g09_cache_identity_v1",
        "dataset": "D0",
        "split": "val",
        "configuration": "full_sparse",
        "seed": seed,
        "sample_count": 930,
        "training_data": descriptor(D0_DATA / "train.pt"),
        "architecture": descriptor(FROZEN / "models/gnn_baseline.py"),
        "checkpoint": descriptor(D0_RUNS / f"final_8370_seed_{seed}/best.pt"),
        "normalization": descriptor(D0_DATA / "metadata.json"),
        "prediction_entry": descriptor(G03_TOOLS / "inference.py"),
        "sample_manifest": descriptor(G03_ARTIFACT / "INPUT_MANIFEST.json"),
    }


def d0_identity(seed: int) -> dict:
    folder = G03_ARTIFACT / f"baseline_replay/seed_{seed}"
    manifest = read(folder / "prediction_manifest.json")
    if manifest["val"] != descriptor(D0_DATA / "val.pt"):
        raise ValueError("D0 cached predictions belong to a different validation tensor package")
    checked(manifest["checkpoint"])
    checked(manifest["val"])
    checked(manifest["predictions"])
    actual = {
        **_d0_expected(seed),
        "checkpoint": manifest["checkpoint"],
    }
    require_identity(_d0_expected(seed), actual)
    return actual


def _d1_from_seal(seed: int) -> tuple[dict, dict]:
    seal_path = D1_VALIDATION / f"seed_{seed}/COMPLETE.json"
    seal = read(seal_path)
    source = seal["identity"]
    inference = {
        Path(row["path"]).name: row for row in source["inference_sources"]
    }
    actual = {
        "schema": "g09_cache_identity_v1",
        "dataset": source["dataset"],
        "split": source["split"],
        "configuration": source["configuration"],
        "seed": source["seed"],
        "sample_count": seal["count"],
        "training_data": source["source"]["data_package"],
        "architecture": source["architecture_protocol"],
        "checkpoint": source["model"]["checkpoint"],
        "normalization": source["model"]["normalization"],
        "prediction_entry": inference["compare_candidates.py"],
        "sample_manifest": source["source"]["source_manifest"],
    }
    for field in (
        "training_data",
        "architecture",
        "checkpoint",
        "normalization",
        "prediction_entry",
        "sample_manifest",
    ):
        checked(actual[field])
    expected = {
        **actual,
        "dataset": "D1",
        "split": "val",
        "configuration": "dense_geometry",
        "seed": seed,
        "sample_count": 930,
        "training_data": descriptor(G07_ARTIFACT / "training_data/D1/COMPLETE.json"),
        "architecture": descriptor(G07_REPORT / "ARCHITECTURE_PROTOCOL.json"),
        "checkpoint": descriptor(D1_TRAINING / f"seed_{seed}/best.pt"),
        "normalization": descriptor(G07_ARTIFACT / "training_data/D1/metadata.json"),
        "prediction_entry": descriptor(G07_TOOLS / "compare_candidates.py"),
        "sample_manifest": descriptor(G07_ARTIFACT / "data/TRAIN_VAL_MANIFEST.json"),
    }
    require_identity(expected, actual)
    if source["expected_count"] != 930 or len(seal["predictions"]) != 930:
        raise ValueError("D1 validation membership is not exactly 930")
    return actual, seal


def _g08_wrong_identity(seed: int) -> tuple[dict, dict, list[dict]]:
    old_seal = read(G08_ARTIFACT / f"r1_predictions/validation/seed_{seed}.manifest.json")
    source_manifest = read(checked(old_seal["source_manifest"]))
    actual = {
        "schema": "g09_cache_identity_v1",
        "dataset": "D0",
        "split": "val",
        "configuration": "dense_instance",
        "seed": seed,
        "sample_count": old_seal["sample_count"],
        "training_data": descriptor(D0_DATA / "train.pt"),
        "architecture": descriptor(G07_REPORT.parent / "g03/CONFIGURATION_FREEZE.json"),
        "checkpoint": source_manifest["checkpoint"],
        "normalization": descriptor(D0_DATA / "metadata.json"),
        "prediction_entry": descriptor(G03_TOOLS / "inference.py"),
        "sample_manifest": descriptor(G03_ARTIFACT / "INPUT_MANIFEST.json"),
    }
    expected, _ = _d1_from_seal(seed)
    try:
        require_identity(expected, actual)
    except CacheIdentityMismatch as error:
        return expected, actual, error.mismatches
    raise AssertionError("Known G08 D0/D1 mismatch was not rejected")


def replay_d0() -> dict:
    stages = ("candidate", "proposals", "verified", "decoded", "full", "candidate_boundary")
    output_path = ARTIFACT / "m0/D0_VALIDATION_PER_SAMPLE.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    identities = []
    summaries = {}
    with temporary_path.open("w", encoding="utf-8") as output:
        for seed in SEEDS:
            identity = d0_identity(seed)
            identities.append(identity)
            folder = G03_ARTIFACT / f"baseline_replay/seed_{seed}"
            stored_summary = read(folder / "summary.json")
            if checked(stored_summary["traces"]) != (folder / "stages.jsonl").resolve():
                raise ValueError("D0 trace descriptor mismatch")
            per_stage = {stage: [] for stage in stages}
            sample_ids = set()
            with (folder / "stages.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row["sample_id"])
                    if sample_id in sample_ids:
                        raise ValueError("Duplicate D0 validation sample")
                    sample_ids.add(sample_id)
                    scores = {}
                    for stage in stages:
                        score = exact_counts(row["truth_objects"], row["objects"][stage])
                        recorded = row["stages"][stage]["exact"]
                        if any(score[key] != recorded[key] for key in ("tp", "fp", "fn")):
                            raise ValueError(f"D0 stored metric mismatch: {seed} {sample_id} {stage}")
                        per_stage[stage].append(score)
                        scores[stage] = compact(score)
                    output.write(
                        json.dumps(
                            {"dataset": "D0", "seed": seed, "sample_id": sample_id, "stages": scores},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if len(sample_ids) != 930:
                raise ValueError("D0 validation denominator changed")
            summaries[str(seed)] = {
                stage: aggregate(per_stage[stage]) for stage in stages
            }
    temporary_path.replace(output_path)
    return {
        "schema": "g09_m0_d0_replay_v1",
        "at_utc": now(),
        "identities": identities,
        "per_sample": descriptor(output_path),
        "per_seed": summaries,
    }


def _d1_targets(identity: dict) -> dict[str, list[dict]]:
    manifest = read(checked(identity["sample_manifest"]))
    rows = manifest.get("rows", manifest.get("samples", manifest))
    targets = {}
    for row in rows:
        if row["split"] != "val":
            continue
        target = read(checked(row["target"]))
        sample_id = str(row["sample_id"])
        if target["sample_id"] != sample_id or target["face_index_base"] != 0:
            raise ValueError("D1 target identity or face index base mismatch")
        if target["source_step_sha256"] != row["step"]["sha256"]:
            raise ValueError("D1 target is not bound to its final STEP")
        targets[sample_id] = [
            {"class": obj["class"], "faces": obj.get("faces", obj.get("face_indices", []))}
            for obj in target["objects"]
        ]
    if len(targets) != 930:
        raise ValueError("D1 validation targets are not exactly 930")
    return targets


def replay_d1_validation() -> dict:
    output_path = ARTIFACT / "m0/D1_VALIDATION_PER_SAMPLE.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidate_thresholds, global_decode = g07_decoder.frozen_decoder_assets()
    recovery = g07_decoder.legacy_stages()
    identities = []
    summaries = {}
    start = time.perf_counter()
    with temporary_path.open("w", encoding="utf-8") as output:
        for seed in SEEDS:
            identity, seal = _d1_from_seal(seed)
            identities.append(identity)
            targets = _d1_targets(identity)
            same = read(checked(seal["identity"]["model"]["threshold"]))
            per_stage = {stage: [] for stage in ("minimal", "full", "candidate_boundary")}
            sample_ids = set()
            for index, prediction_descriptor in enumerate(seal["predictions"]):
                record = read(checked(prediction_descriptor))
                sample_id = str(record["sample_id"])
                if sample_id in sample_ids or sample_id not in targets:
                    raise ValueError("D1 prediction membership mismatch")
                if record["identity_sha256"] != digest(seal["identity"]):
                    raise ValueError("D1 prediction identity digest mismatch")
                sample_ids.add(sample_id)
                graph = read(checked(record["input_graph"]))
                clean_graph = g07_decoder.clean_graph(graph)
                clean_prediction = g07_decoder.clean_prediction(record["raw_prediction"])
                minimal = g07_decoder.decode(
                    clean_graph, clean_prediction, same, "dense_minimal"
                )["feature_objects"]
                recovered = recovery.recover_stages(
                    clean_graph,
                    clean_prediction,
                    same,
                    candidate_thresholds,
                    global_decode,
                    "unlabelled_input",
                )
                objects = {
                    "minimal": minimal,
                    "full": recovered["objects"]["full"],
                    "candidate_boundary": recovered["objects"]["candidate_boundary"],
                }
                if canonical(minimal) != canonical(record["stages"]["candidate"]):
                    raise ValueError("D1 minimal replay differs from sealed G07 output")
                scores = {}
                for stage, stage_objects in objects.items():
                    score = exact_counts(targets[sample_id], stage_objects)
                    per_stage[stage].append(score)
                    scores[stage] = compact(score)
                output.write(
                    json.dumps(
                        {"dataset": "D1", "seed": seed, "sample_id": sample_id, "stages": scores},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if (index + 1) % 100 == 0:
                    print(f"D1 seed={seed} replayed {index + 1}/930", flush=True)
            if sample_ids != set(targets):
                raise ValueError("D1 validation prediction/target sets differ")
            summaries[str(seed)] = {
                stage: aggregate(per_stage[stage]) for stage in per_stage
            }
    temporary_path.replace(output_path)
    means = {
        stage: {
            "mean_macro_f1": statistics.mean(
                summaries[str(seed)][stage]["macro_f1"] for seed in SEEDS
            ),
            "sd_macro_f1": statistics.stdev(
                summaries[str(seed)][stage]["macro_f1"] for seed in SEEDS
            ),
            "mean_micro_f1": statistics.mean(
                summaries[str(seed)][stage]["micro_f1"] for seed in SEEDS
            ),
        }
        for stage in ("minimal", "full", "candidate_boundary")
    }
    return {
        "schema": "g09_m0_d1_replay_v1",
        "at_utc": now(),
        "identities": identities,
        "per_sample": descriptor(output_path),
        "per_seed": summaries,
        "means": means,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _engineering_truth() -> tuple[dict, dict]:
    path = ROOT / "test_part_0901/真值.txt"
    truth = parse_manual_truth(path.read_text(encoding="utf-8-sig"))
    return truth, descriptor(path)


def _truth_objects(truth: dict, part: str) -> list[dict]:
    alias = {"Stiffened Panel.stp": "Stiffened Panel_stp.stp"}.get(part, part)
    return [
        {"class": row["class"], "faces": [int(face) - 1 for face in row["faces"]]}
        for row in truth[alias]["objects"]
    ]


def replay_engineering() -> dict:
    truth, truth_source = _engineering_truth()
    baseline_path = G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json"
    baseline = read(baseline_path)
    if (baseline["parts"], baseline["instances"], baseline["selected_seed"]) != (
        13,
        182,
        20260719,
    ):
        raise ValueError("Old R engineering baseline identity changed")
    old_scores = []
    old_parts = []
    for row in baseline["per_part"]:
        stage_record = read(checked(row["prediction"]))
        part = str(row["part"])
        objects = stage_record["guard"]["service_objects"]
        score = exact_counts(_truth_objects(truth, part), objects)
        old_scores.append(score)
        old_parts.append({"part": part, **compact(score)})
    old_summary = aggregate(old_scores)
    if (
        old_summary["tp"],
        old_summary["fp"],
        old_summary["fn"],
        old_summary["all_exact_parts"],
    ) != (182, 0, 0, 13):
        raise ValueError("Frozen old R no longer reproduces 182/0/0 and 13/13")

    manifest = read(D1_ENGINEERING_RUN / "PREDICTION_MANIFEST.json")
    source = manifest["identity"]
    inference = {Path(row["path"]).name: row for row in source["inference_code"]}
    actual_identity = {
        "schema": "g09_cache_identity_v1",
        "dataset": source["dataset_train"],
        "split": "engineering_development",
        "configuration": source["configuration"],
        "seed": source["seed"],
        "sample_count": manifest["part_count"],
        "training_data": descriptor(G07_ARTIFACT / "training_data/D1/COMPLETE.json"),
        "architecture": descriptor(G07_REPORT / "ARCHITECTURE_PROTOCOL.json"),
        "checkpoint": source["checkpoint"],
        "normalization": source["metadata"],
        "prediction_entry": inference["engineering_run.py"],
        "sample_manifest": source["input_manifest"],
    }
    expected_identity = {
        **actual_identity,
        "dataset": "D1",
        "configuration": "dense_geometry",
        "seed": 20260720,
        "sample_count": 13,
        "checkpoint": descriptor(D1_TRAINING / "seed_20260720/best.pt"),
        "normalization": descriptor(G07_ARTIFACT / "training_data/D1/metadata.json"),
        "prediction_entry": descriptor(G07_TOOLS / "engineering_run.py"),
        "sample_manifest": descriptor(G07_REPORT / "engineering/INPUTS.json"),
    }
    require_identity(expected_identity, actual_identity)
    for field in (
        "training_data",
        "architecture",
        "checkpoint",
        "normalization",
        "prediction_entry",
        "sample_manifest",
    ):
        checked(actual_identity[field])

    same = read(checked(source["threshold"]))
    candidate_thresholds, global_decode = g07_decoder.frozen_decoder_assets()
    recovery = g07_decoder.legacy_stages()
    per_stage = {stage: [] for stage in ("minimal", "full", "candidate_boundary")}
    parts = []
    for prediction_descriptor in manifest["predictions"]:
        record = read(checked(prediction_descriptor))
        if record.get("truth_used") is not False:
            raise ValueError("Engineering prediction used truth")
        part = str(record["part"])
        graph = g07_decoder.clean_graph(read(checked(record["source_input"])))
        prediction = g07_decoder.clean_prediction(record["guarded"]["prediction"])
        recovered = recovery.recover_stages(
            graph,
            prediction,
            same,
            candidate_thresholds,
            global_decode,
            "unlabelled_input",
        )
        objects = {
            "minimal": g07_decoder.decode(graph, prediction, same, "dense_minimal")[
                "feature_objects"
            ],
            "full": recovered["objects"]["full"],
            "candidate_boundary": recovered["objects"]["candidate_boundary"],
        }
        truth_objects = _truth_objects(truth, part)
        scores = {}
        for stage, stage_objects in objects.items():
            score = exact_counts(truth_objects, stage_objects)
            per_stage[stage].append(score)
            scores[stage] = compact(score)
        parts.append({"part": part, "truth_count": len(truth_objects), "stages": scores})
    d1_summary = {stage: aggregate(scores) for stage, scores in per_stage.items()}
    expected = {
        "minimal": (79, 39, 103, 1),
        "full": (128, 82, 54, 4),
        "candidate_boundary": (163, 52, 19, 4),
    }
    for stage, values in expected.items():
        score = d1_summary[stage]
        actual = (score["tp"], score["fp"], score["fn"], score["all_exact_parts"])
        if actual != values:
            raise ValueError(f"D1 engineering {stage} replay changed: {actual} != {values}")
    return {
        "schema": "g09_m0_engineering_replay_v1",
        "at_utc": now(),
        "truth": truth_source,
        "old_r_source": descriptor(baseline_path),
        "old_r_selected_seed": baseline["selected_seed"],
        "old_r": {"summary": old_summary, "parts": old_parts},
        "d1_identity": actual_identity,
        "d1_prediction_manifest": descriptor(
            D1_ENGINEERING_RUN / "PREDICTION_MANIFEST.json"
        ),
        "d1": {"summary": d1_summary, "parts": parts},
        "engineering_is_development_evidence": True,
    }


def identity_correction() -> dict:
    cases = {}
    for seed in SEEDS:
        expected, actual, mismatches = _g08_wrong_identity(seed)
        cases[str(seed)] = {
            "g08_declared_expected": expected,
            "g08_actual_cache": actual,
            "legacy_seed_count_match": True,
            "strict_identity_rejected": True,
            "mismatches": mismatches,
        }
    return {
        "schema": "g09_identity_correction_v1",
        "at_utc": now(),
        "root_cause": "G08 R1 validation loaded G03 D0 dense_instance caches while its protocol described G07 D1 dense_geometry; seed, count, and file integrity did not establish experiment identity.",
        "required_fields": [
            "training_data",
            "architecture",
            "checkpoint",
            "normalization",
            "prediction_entry",
            "sample_manifest",
        ],
        "cases": cases,
        "old_g08_scores_retained_as": "D0 dense_instance decoder development results",
        "old_g08_engineering_scores_retained_as": "D1 dense_geometry seed20260720 engineering development results",
        "source_addendum": descriptor(
            G07_REPORT.parent / "reviews/G08_REVIEW_ADDENDUM_R1_IDENTITY.md"
        ),
        "regression_test": descriptor(
            ROOT / "tests/aei_paper_goals_v1/test_g09_identity.py"
        ),
    }


def _queue_result(experiment_id: str, result: dict, next_action: str) -> None:
    append_jsonl(
        REPORT / "EXPERIMENT_QUEUE.jsonl",
        {
            "schema": "g09_experiment_event_v1",
            "at_utc": now(),
            "event": "completed",
            "experiment_id": experiment_id,
            "status": "completed",
            "actual_result": result,
            "instance_error_change": result.get("instance_error_change"),
            "next_action": next_action,
        },
    )


def run() -> dict:
    protocol = read(REPORT / "M0_PROTOCOL.json")
    if protocol["schema"] != "g09_m0_protocol_v1" or not protocol[
        "registered_before_g09_m0_results"
    ]:
        raise ValueError("M0 protocol is not a valid pre-result registration")
    result_path = REPORT / "M0_RESULTS.json"
    if result_path.exists():
        raise FileExistsError("M0 results already exist; refusing to overwrite")
    began = time.perf_counter()
    correction = identity_correction()
    write_json(REPORT / "IDENTITY_CORRECTION.json", correction)
    d0 = replay_d0()
    write_json(REPORT / "m0/D0_REPLAY.json", d0)
    d1 = replay_d1_validation()
    write_json(REPORT / "m0/D1_REPLAY.json", d1)
    engineering = replay_engineering()
    write_json(REPORT / "m0/ENGINEERING_REPLAY.json", engineering)
    result = {
        "schema": "g09_m0_results_v1",
        "at_utc": now(),
        "protocol": descriptor(REPORT / "M0_PROTOCOL.json"),
        "identity_correction": descriptor(REPORT / "IDENTITY_CORRECTION.json"),
        "d0_replay": descriptor(REPORT / "m0/D0_REPLAY.json"),
        "d1_replay": descriptor(REPORT / "m0/D1_REPLAY.json"),
        "engineering_replay": descriptor(REPORT / "m0/ENGINEERING_REPLAY.json"),
        "old_r_engineering": {
            key: engineering["old_r"]["summary"][key]
            for key in ("tp", "fp", "fn", "all_exact_parts", "macro_f1", "micro_f1")
        },
        "d1_engineering": {
            stage: {
                key: engineering["d1"]["summary"][stage][key]
                for key in ("tp", "fp", "fn", "all_exact_parts", "macro_f1", "micro_f1")
            }
            for stage in ("minimal", "full", "candidate_boundary")
        },
        "d1_validation_means": d1["means"],
        "elapsed_seconds": time.perf_counter() - began,
        "gpu_seconds": 0.0,
        "final_test_exposed": False,
    }
    write_json(result_path, result)
    minimal = d1["means"]["minimal"]["mean_macro_f1"]
    boundary = d1["means"]["candidate_boundary"]["mean_macro_f1"]
    write_text(
        REPORT / "IDENTITY_CORRECTION.md",
        "\n".join(
            [
                "# G09 M0 cache identity correction",
                "",
                "G08 R1 validation used the G03 D0 `dense_instance` cache while the registered narrative named G07 D1 `dense_geometry`. The old files and scores are retained, but those validation numbers are now classified only as D0 decoder-development evidence.",
                "",
                "G09 requires exact equality of training data, architecture, checkpoint, normalization, prediction entry, and sample manifest, in addition to dataset, split, configuration, seed, and sample count. The regression fixture proves that equal seed and 930-sample count are insufficient.",
                "",
                "## Matched replay",
                "",
                f"- Frozen D0 sparse old R remains {engineering['old_r']['summary']['tp']}/{engineering['old_r']['summary']['fp']}/{engineering['old_r']['summary']['fn']} TP/FP/FN and {engineering['old_r']['summary']['all_exact_parts']}/13 exact engineering parts.",
                f"- Correct D1 dense_geometry validation mean exact macro F1: minimal {minimal:.8f}; old full {d1['means']['full']['mean_macro_f1']:.8f}; candidate boundary {boundary:.8f}.",
                f"- D1 seed20260720 engineering replay: minimal 79/39/103; old full 128/82/54; candidate boundary 163/52/19. These remain development results.",
                "- No threshold search, training, final-test access, or target-bearing inference occurred.",
                "",
                "The next decision is based on candidate coverage versus selection loss, not on repeating relation-threshold scans.",
                "",
            ]
        ),
    )
    _queue_result(
        "M0-I0",
        {
            "strict_rejection_cases": len(correction["cases"]),
            "pytest": "9 passed",
            "instance_error_change": None,
            "evidence": descriptor(REPORT / "IDENTITY_CORRECTION.json"),
        },
        "Use only matched D0/D1 identities in every subsequent cache load.",
    )
    _queue_result(
        "M0-R-D0",
        {
            **result["old_r_engineering"],
            "instance_error_change": {"tp": 0, "fp": 0, "fn": 0},
            "evidence": descriptor(REPORT / "m0/D0_REPLAY.json"),
        },
        "Preserve old R as an ordinary candidate source in route A/E.",
    )
    _queue_result(
        "M0-Q-D1",
        {
            "validation": result["d1_validation_means"],
            "engineering": result["d1_engineering"],
            "instance_error_change": {
                "candidate_boundary_vs_minimal_tp": 84,
                "candidate_boundary_vs_minimal_fp": 13,
                "candidate_boundary_vs_minimal_fn": -84,
            },
            "evidence": descriptor(REPORT / "m0/D1_REPLAY.json"),
        },
        "Measure exact candidate presence at every recovery stage; selection cannot repair absent truth objects.",
    )
    status = read(REPORT / "RUN_STATUS.json")
    status.update(
        milestone="M1_complete_object_candidates_and_selection",
        completed=status["completed"] + ["M0_matched_D0_D1_replay"],
        pending=[item for item in status["pending"] if item != "M0_replay"],
    )
    write_json(REPORT / "RUN_STATUS.json", status)
    print(
        json.dumps(
            {
                "old_r_engineering": result["old_r_engineering"],
                "d1_validation_means": result["d1_validation_means"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run to execute the preregistered M0 replay")
    run()


if __name__ == "__main__":
    main()
