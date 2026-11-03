"""Frozen-R, D0-train-only candidate examples for G09 object scoring.

Prediction and object features are completed before the training-only truth
sidecar is opened for exact complete-object labels. This module never trains a
model and never reads validation or engineering targets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cache_identity import require_identity
from candidate_pool import build_candidate_pool, capture_legacy_stages, _legacy_dependencies
from g09_common import (ARTIFACT, FROZEN, G03_ARTIFACT, G03_TOOLS, REPORT,
                        SEEDS, checked, descriptor, digest, now, read, sha256, write_json)
from object_features import NUMERIC_FEATURE_NAMES, extract_object_features


DATA = FROZEN / "output_training_p1_e6r2_r3_graph_v2_2_1_v1"
GRAPHS = FROZEN / "output_generation_p1_e6r2_r1_3_r1_complete_r2_mixed_9300_v1/graphs"
G03_MANIFEST = G03_ARTIFACT / "INPUT_MANIFEST.json"
G03_FREEZE = REPORT.parent / "g03/CONFIGURATION_FREEZE.json"


def _verified_frozen_asset(path: Path, frozen_files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [row for row in frozen_files if Path(row["path"]).resolve() == path.resolve()]
    if len(expected) != 1:
        raise ValueError(f"G03 freeze does not bind exactly one source: {path}")
    checked(expected[0])
    return expected[0]


def source_identity(seed: int, sample_count: int) -> dict[str, Any]:
    """Reject changed D0 train, checkpoint, architecture, normalization or entry."""
    if seed not in SEEDS or not 1 <= sample_count <= 7440:
        raise ValueError("Only original D0 train seeds and sample counts are allowed")
    freeze = read(G03_FREEZE)
    frozen_files = freeze["frozen_files"]
    manifest_file = _verified_frozen_asset(G03_MANIFEST, frozen_files)
    manifest = read(G03_MANIFEST)
    if len(manifest["identities"]["train"]) != 7440 or len(manifest["identities"]["val"]) != 930:
        raise ValueError("Original D0 membership counts changed")
    if set(manifest["identities"]["train"]) & set(manifest["identities"]["val"]):
        raise ValueError("D0 train/validation membership overlaps")
    selected_ids = select_train_ids(manifest["identities"]["train"], sample_count)
    paths = {Path(row["path"]).resolve(): row for row in manifest["inputs"]}
    for asset in (DATA / "train.pt", DATA / "metadata.json"):
        if asset.resolve() not in paths:
            raise ValueError(f"Missing D0 manifest asset: {asset}")
        checked(paths[asset.resolve()])
        _verified_frozen_asset(asset, frozen_files)
    run = FROZEN / "reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs" / f"final_8370_seed_{seed}"
    frozen_run = [row for row in freeze["baseline_runs"] if row["seed"] == seed]
    if len(frozen_run) != 1 or not frozen_run[0]["reusable"]:
        raise ValueError("Frozen R seed unavailable")
    checkpoint = next((row for row in frozen_run[0]["files"]
                       if Path(row["path"]).resolve() == (run / "best.pt").resolve()), None)
    if checkpoint is None:
        raise ValueError("R checkpoint absent from freeze")
    checked(checkpoint)
    architecture = _verified_frozen_asset(FROZEN / "models/gnn_baseline.py", frozen_files)
    entry = _verified_frozen_asset(G03_TOOLS / "inference.py", frozen_files)
    base = FROZEN / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718"
    decoder = [descriptor(path) for path in (
        run / "same_instance_threshold_selection.json",
        base / "candidate_model/candidate_threshold_selection.json",
        base / "global_decode_selection.json",
    )]
    return {
        "schema": "g09_train_candidate_identity_v1", "dataset": "D0", "split": "train",
        "configuration": "R_sparse_frozen_full", "seed": seed, "sample_count": sample_count,
        "training_data": paths[(DATA / "train.pt").resolve()],
        "architecture": architecture, "checkpoint": checkpoint,
        "normalization": paths[(DATA / "metadata.json").resolve()],
        "prediction_entry": entry, "sample_manifest": manifest_file,
        "decoder": decoder, "stage_capture": descriptor(Path(__file__).with_name("candidate_pool.py")),
        "object_features": descriptor(Path(__file__).with_name("object_features.py")),
        "data_builder": descriptor(__file__),
        "feature_order": list(NUMERIC_FEATURE_NAMES),
        "selected_sample_ids_digest": digest(selected_ids),
        "selection_rule": "complete_train" if sample_count == 7440 else "uniform_midpoints_sorted_train",
    }


def select_train_ids(train_ids: Sequence[str], count: int) -> list[str]:
    if len(train_ids) != 7440 or count < 1 or count > 7440:
        raise ValueError("Selection must come from complete original D0 train")
    ordered = sorted(train_ids)
    if count == 7440:
        return ordered
    return [ordered[(2 * index + 1) * len(ordered) // (2 * count)] for index in range(count)]


def validate_train_membership(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    """The only permissible records are the exact frozen D0 train partition."""
    train = manifest["identities"]["train"]
    val = manifest["identities"]["val"]
    if len(train) != 7440 or len(val) != 930 or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Frozen split cardinality or uniqueness changed")
    if set(train) & set(val):
        raise ValueError("Train/validation membership overlap")
    ids = [str(row["sample_id"]) for row in records]
    if len(ids) != 7440 or set(ids) != set(train) or len(set(ids)) != len(ids):
        raise ValueError("Loaded records differ from exact D0 train manifest")
    if any(row["split"] != "train" for row in records):
        raise ValueError("Non-training record in D0 train package")


def _legacy_inference():
    _legacy_dependencies()  # Adds frozen import roots and checks common.py identity.
    path = G03_TOOLS / "inference.py"
    spec = importlib.util.spec_from_file_location("g09_frozen_g03_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen G03 inference unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    path = GRAPHS / f"{sample_id}.json"
    if Path(record["source_graph_path"]).name != path.name or path.suffix != ".json":
        raise ValueError("Train record graph identity mismatch")
    if sha256(path).lower() != str(record["source_graph_sha256"]).lower():
        raise ValueError("D0 train graph source hash mismatch")
    graph = read(path)
    if len(graph["nodes"]) != len(record["node_features"]):
        raise ValueError("Train graph node count differs from normalized record")
    return graph


def extract_part_features(graph, prediction, same, candidate_thresholds, global_decode):
    """Production-equivalent feature path; no target or sample identity argument."""
    clean_graph = _legacy_dependencies().stages.without_truth(graph)
    stages = capture_legacy_stages(clean_graph, prediction, same, candidate_thresholds, global_decode)
    pool = build_candidate_pool({"R": stages}, face_count=len(clean_graph["nodes"]))
    rows = []
    for candidate in pool:
        evidence = extract_object_features(candidate, prediction, clean_graph)
        rows.append({"macro_feature": candidate["macro_feature"],
                     "face_indices": candidate["face_indices"],
                     "sources": candidate["sources"], "provenance": candidate["provenance"],
                     "vector": list(evidence["vector"]),
                     "internal_relation_status": evidence["internal_relation_status"]})
    return {"face_count": len(clean_graph["nodes"]),
            "stage_counts": {stage: len(objects) for stage, objects in stages.items()},
            "candidates": rows}


def _truth(feature_sets: Sequence[Mapping[str, Any]], face_count: int) -> list[dict[str, Any]]:
    objects = []
    for item in feature_sets:
        macro = str(item.get("macro_feature", "stock"))
        if macro == "stock":
            continue
        faces = sorted(int(face) for face in item["node_indices"])
        if not faces or len(set(faces)) != len(faces) or faces[0] < 0 or faces[-1] >= face_count:
            raise ValueError("Malformed D0 train complete truth object")
        objects.append({"macro_feature": macro, "face_indices": faces})
    if len(set((row["macro_feature"], tuple(row["face_indices"])) for row in objects)) != len(objects):
        raise ValueError("Duplicate D0 truth objects cannot be labelled after pool deduplication")
    return objects


def label_part_candidates(feature_record: Mapping[str, Any], feature_sets: Sequence[Mapping[str, Any]]):
    """Exact positive and multiple hard-negative mechanisms in a separate sidecar."""
    truth = _truth(feature_sets, feature_record["face_count"])
    truth_keys = {(row["macro_feature"], tuple(row["face_indices"])) for row in truth}
    positives = [set(row["face_indices"]) for row in feature_record["candidates"]
                 if (row["macro_feature"], tuple(row["face_indices"])) in truth_keys]
    labelled = []
    for index, row in enumerate(feature_record["candidates"]):
        faces = set(row["face_indices"])
        macro = row["macro_feature"]
        exact = (macro, tuple(sorted(faces))) in truth_keys
        same = [set(t["face_indices"]) for t in truth if t["macro_feature"] == macro]
        overlapping = [group for group in same if group & faces]
        fully_covered = sum(group <= faces for group in same)
        types = []
        if not exact:
            if fully_covered >= 2:
                types.append("overmerge")
            if any(faces < group for group in same):
                types.append("missing_face")
            if any(group < faces for group in same):
                types.append("extra_face")
            if overlapping and not ("missing_face" in types or "extra_face" in types):
                types.append("partial_overlap")
            if any(faces & positive for positive in positives):
                types.append("conflict_with_exact_positive")
            if not overlapping and any(faces & set(t["face_indices"]) for t in truth):
                types.append("class_mismatch")
            if not types:
                types.append("unmatched")
        labelled.append({"candidate_index": index, "exact_positive": exact,
                         "error_types": types})
    return {"truth_objects": truth, "truth_count": len(truth), "labels": labelled,
            "exact_positive_count": sum(row["exact_positive"] for row in labelled),
            "negative_type_counts": dict(sorted(Counter(
                typ for row in labelled for typ in row["error_types"]).items()))}


def run(seed: int, limit: int, *, full: bool = False) -> dict[str, Any]:
    """Write real, unsampled D0 train proposals; pilot is intentionally bounded."""
    if not full and limit > 64:
        raise ValueError("CPU pilot limit is 64; full extraction needs --full")
    count = 7440 if full else limit
    identity = source_identity(seed, count)
    parent = ARTIFACT / "train_candidate_data" / f"seed_{seed}"
    name = "full" if full else f"pilot_{limit}"
    retry_of = None
    for attempt in range(1, 100):
        folder = parent / (name if attempt == 1 else f"{name}_attempt_{attempt}")
        status = folder / "COMPLETE.json"
        if status.exists():
            old = read(status)
            require_identity(identity, old["identity"])
            if old["identity"] != identity:
                raise ValueError("Candidate extraction version differs from existing run")
            for item in old["outputs"]:
                checked(item)
            return old
        if not folder.exists() or not any(folder.iterdir()):
            break
        retry_of = folder.as_posix()
    else:
        raise RuntimeError("Too many retained candidate extraction attempts")
    folder.mkdir(parents=True, exist_ok=True)
    partial_features = folder / "candidates.jsonl.partial"
    partial_labels = folder / "labels.jsonl.partial"
    stats = Counter()
    timings = Counter()
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(4)
        records = torch.load(DATA / "train.pt", map_location="cpu", weights_only=False)
        manifest = read(G03_MANIFEST)
        validate_train_membership(records, manifest)
        selected_ids = select_train_ids([row["sample_id"] for row in records], count)
        selected_by_id = {row["sample_id"]: row for row in records}
        selected = [selected_by_id[sample_id] for sample_id in selected_ids]
        if digest(selected_ids) != identity["selected_sample_ids_digest"]:
            raise ValueError("Selected training membership disagrees with frozen manifest")
        inference = _legacy_inference()
        model, same, checkpoint = inference.load_frozen("full", seed)
        if descriptor(checkpoint) != identity["checkpoint"]:
            raise ValueError("Frozen inference checkpoint identity mismatch")
        candidate_thresholds, global_decode = (read(path) for path in inference.decoder_assets())
        with partial_features.open("w", encoding="utf-8") as feature_file, partial_labels.open("w", encoding="utf-8") as label_file:
            for part_index, raw in enumerate(selected):
                t0 = time.perf_counter()
                graph = _graph_for_record(raw)
                timings["graph_read_seconds"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                tensors = (raw["node_features"].float(), raw["edge_index"].long(),
                           raw["primitive_surface_ids"].long(), raw["edge_attr"].float())
                prediction = inference.predict_tensors(model, tensors, "full", "cpu")
                timings["forward_seconds"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                features = extract_part_features(graph, prediction, same, candidate_thresholds, global_decode)
                timings["feature_seconds"] += time.perf_counter() - t0
                # This is the first access to target-side feature_sets. The model,
                # decoder and numeric features have already completed inference.
                labels = label_part_candidates(features, raw["feature_sets"])
                stats.update(parts=1, candidates=len(features["candidates"]),
                             positives=labels["exact_positive_count"], truths=labels["truth_count"])
                stats.update(labels["negative_type_counts"])
                feature_file.write(json.dumps({"part_index": part_index, **features},
                                              ensure_ascii=False, allow_nan=False) + "\n")
                label_file.write(json.dumps({"part_index": part_index, **labels},
                                            ensure_ascii=False, allow_nan=False) + "\n")
                if (part_index + 1) % 50 == 0:
                    print(json.dumps({"train_candidates": part_index + 1, "of": count,
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        os.replace(partial_features, folder / "candidates.jsonl")
        os.replace(partial_labels, folder / "labels.jsonl")
        output = {"schema": "g09_train_candidate_data_v1", "identity": identity,
                  "created_utc": now(), "count": count, "pilot": not full,
                  "retry_of": retry_of,
                  "stats": dict(sorted(stats.items())), "timings": dict(timings),
                  "elapsed_seconds": time.perf_counter() - started,
                  "estimated_full_cpu_seconds": (time.perf_counter() - started) * 7440 / count,
                  "outputs": [descriptor(folder / name) for name in ("candidates.jsonl", "labels.jsonl")],
                  "contract": {"feature_path_truth_free": True, "labels_from_D0_train_only": True,
                               "every_real_candidate_preserved": True, "gradient_steps": 0}}
        write_json(status, output)
        return output
    except Exception as exc:
        write_json(folder / "FAILED.json", {"identity": identity, "error": repr(exc),
                                           "elapsed_seconds": time.perf_counter() - started,
                                           "partial_outputs_retained": True, "retry_of": retry_of})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=20260719)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", type=int)
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = run(args.seed, 7440 if args.full else args.pilot, full=args.full)
    print(json.dumps({"count": result["count"], "stats": result["stats"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "estimated_full_cpu_seconds": result["estimated_full_cpu_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
