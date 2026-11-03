"""D0-train-only R+Q proposal evidence for a branch-aware object scorer."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from candidate_pool import STAGES, build_candidate_pool, capture_legacy_stages
from g09_common import (ARTIFACT, G03_ARTIFACT, G03_TOOLS, REPORT,
                        checked, descriptor, digest, read, sha256, write_json)
from m0_replay import g07_decoder
from object_features import NUMERIC_FEATURE_NAMES, extract_object_features
from train_candidate_data import (_graph_for_record, _legacy_inference,
                                  label_part_candidates, select_train_ids,
                                  source_identity, validate_train_membership)


SEED = 20260719
R_FULL = ARTIFACT / f"train_candidate_data/seed_{SEED}/full"
Q_RUN = G03_ARTIFACT / f"training_runs/dense_instance/seed_{SEED}"
G03_MANIFEST = G03_ARTIFACT / "INPUT_MANIFEST.json"
G03_FREEZE = REPORT.parent / "g03/CONFIGURATION_FREEZE.json"
FEATURE_NAMES = tuple(f"R_{name}" for name in NUMERIC_FEATURE_NAMES) + tuple(
    f"Q_{name}" for name in NUMERIC_FEATURE_NAMES) + ("proposed_by_R", "proposed_by_Q")


def source_identity_joint(sample_count):
    original = read(R_FULL / "COMPLETE.json")
    if original.get("schema") != "g09_train_candidate_data_v1" or original.get("pilot") \
            or original.get("count") != 7440 or original["identity"] != source_identity(SEED, 7440):
        raise ValueError("R training candidate seal is not the full frozen D0 train set")
    for row in original["outputs"]:
        checked(row)
    q_config = read(Q_RUN / "config.json")
    q_metrics = read(Q_RUN / "metrics.json")
    if read(Q_RUN / "STATUS.json")["status"] != "completed" \
            or q_config["config"] != "dense_instance" or q_config["seed"] != SEED \
            or q_config["data_manifest"] != descriptor(G03_MANIFEST) \
            or q_config["configuration_freeze_sha256"] != sha256(G03_FREEZE):
        raise ValueError("Q training identity is not frozen D0 dense_instance")
    checkpoint = checked(q_metrics["checkpoint"])
    if checkpoint != (Q_RUN / "best.pt").resolve():
        raise ValueError("Q checkpoint and G03 training run disagree")
    train_ids = read(G03_MANIFEST)["identities"]["train"]
    return {"schema": "g09_e_joint_train_identity_v1", "dataset": "D0", "split": "train",
            "seed": SEED, "sample_count": sample_count,
            "R_candidate_seal": descriptor(R_FULL / "COMPLETE.json"),
            "Q_checkpoint": q_metrics["checkpoint"],
            "Q_config": descriptor(Q_RUN / "config.json"),
            "Q_inference": descriptor(G03_TOOLS / "inference.py"),
            "training_data": original["identity"]["training_data"],
            "architecture": {"R": original["identity"]["architecture"],
                             "Q": descriptor(G03_TOOLS / "ablations.py")},
            "normalization": original["identity"]["normalization"],
            "sample_manifest": original["identity"]["sample_manifest"],
            "selected_ids_digest": digest(select_train_ids(train_ids, sample_count)),
            "joint_builder": descriptor(__file__),
            "candidate_pool": descriptor(Path(__file__).with_name("candidate_pool.py")),
            "object_features": descriptor(Path(__file__).with_name("object_features.py")),
            "feature_names": list(FEATURE_NAMES),
            "selection_rule": "complete_train" if sample_count == 7440 else "uniform_midpoints_sorted_train"}


def recover_r_stages(r_feature_record):
    """Rehydrate only general stage/branch provenance, never a sample identity."""
    stages = {stage: [] for stage in STAGES}
    for candidate in r_feature_record["candidates"]:
        for origin in candidate["provenance"]:
            if origin["branch"] != "R" or origin["stage"] not in stages:
                raise ValueError("R train cache stage/branch provenance mismatch")
            stages[origin["stage"]].append({
                "macro_feature": candidate["macro_feature"],
                "face_indices": candidate["face_indices"],
                "proposal_sources": origin["mechanisms"]})
    if any(len(stages[stage]) > r_feature_record["stage_counts"][stage] for stage in STAGES):
        raise ValueError("R deduplicated provenance exceeds original stage count")
    return stages


def extract_joint_features(graph, r_prediction, q_prediction, r_cached,
                           q_same, candidate_config, global_config):
    """Both predicted evidence fields exist on every proposal; no truth input."""
    graph = g07_decoder.clean_graph(graph)
    r_prediction = g07_decoder.clean_prediction(r_prediction)
    q_prediction = g07_decoder.clean_prediction(q_prediction)
    q_stages = capture_legacy_stages(graph, q_prediction, q_same,
                                     candidate_config, global_config)
    r_stages = recover_r_stages(r_cached)
    candidates = build_candidate_pool({"R": r_stages, "Q": q_stages}, face_count=len(graph["nodes"]))
    output = []
    for candidate in candidates:
        proposed = {origin["branch"] for origin in candidate["provenance"]}
        r_features = extract_object_features(candidate, r_prediction, graph)
        q_features = extract_object_features(candidate, q_prediction, graph)
        vector = (*r_features["vector"], *q_features["vector"],
                  float("R" in proposed), float("Q" in proposed))
        output.append({"macro_feature": candidate["macro_feature"],
                       "face_indices": candidate["face_indices"],
                       "sources": candidate["sources"], "provenance": candidate["provenance"],
                       "vector": list(vector)})
    return {"face_count": len(graph["nodes"]), "candidates": output,
            "stage_counts": {"R": r_cached["stage_counts"],
                             "Q": {stage: len(q_stages[stage]) for stage in STAGES}}}


def _r_rows(selected_indices):
    feature_path = checked(next(row for row in read(R_FULL / "COMPLETE.json")["outputs"]
                                if Path(row["path"]).name == "candidates.jsonl"))
    label_path = checked(next(row for row in read(R_FULL / "COMPLETE.json")["outputs"]
                              if Path(row["path"]).name == "labels.jsonl"))
    wanted = set(selected_indices)
    with feature_path.open(encoding="utf-8") as features, label_path.open(encoding="utf-8") as labels:
        for index, (fline, lline) in enumerate(zip(features, labels, strict=True)):
            if index not in wanted:
                continue
            feature, label = json.loads(fline), json.loads(lline)
            if feature["part_index"] != index or label["part_index"] != index:
                raise ValueError("R candidate/label part grouping changed")
            yield index, feature, label


def run(count):
    if count not in (64, 7440):
        raise ValueError("Only preregistered 64-part pilot or full 7440 D0 train allowed")
    identity = source_identity_joint(count)
    parent = ARTIFACT / "e_joint_train"
    name = "pilot_64" if count == 64 else "full"
    for attempt in range(1, 100):
        folder = parent / (name if attempt == 1 else f"{name}_attempt_{attempt}")
        complete = folder / "COMPLETE.json"
        if complete.exists():
            previous = read(complete)
            if previous["identity"] != identity:
                raise ValueError("Completed joint extraction identity differs; no overwrite allowed")
            for output in previous["outputs"]:
                checked(output)
            return previous
        if not folder.exists() or not any(folder.iterdir()):
            break
    else:
        raise RuntimeError("Too many retained joint extraction attempts")
    folder.mkdir(parents=True, exist_ok=True)
    partial_features = folder / "candidates.jsonl.partial"
    partial_labels = folder / "labels.jsonl.partial"
    stats, times = Counter(), Counter()
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(4)
        inference = _legacy_inference()
        raw_records = torch.load(Path(identity["training_data"]["path"]), map_location="cpu", weights_only=False)
        manifest = read(G03_MANIFEST)
        validate_train_membership(raw_records, manifest)
        ids = select_train_ids(manifest["identities"]["train"], count)
        by_id = {record["sample_id"]: record for record in raw_records}
        ordered_all = sorted(manifest["identities"]["train"])
        index_of = {sample_id: index for index, sample_id in enumerate(ordered_all)}
        selected_indices = [index_of[sample_id] for sample_id in ids]
        if digest(ids) != identity["selected_ids_digest"]:
            raise ValueError("D0 train selected identities differ")
        r_model, _, r_checkpoint = inference.load_frozen("full", SEED)
        q_model, q_same, q_checkpoint = inference.load_frozen("dense_instance", SEED)
        if descriptor(r_checkpoint) != read(R_FULL / "COMPLETE.json")["identity"]["checkpoint"] \
                or descriptor(q_checkpoint) != identity["Q_checkpoint"]:
            raise ValueError("R/Q loaded weights disagree with sealed identity")
        candidate_config, global_config = (read(path) for path in inference.decoder_assets())
        with partial_features.open("w", encoding="utf-8") as feature_file, \
             partial_labels.open("w", encoding="utf-8") as label_file:
            for part_index, (raw_index, r_cached, r_sidecar) in enumerate(_r_rows(selected_indices)):
                record = by_id[ordered_all[raw_index]]
                t0 = time.perf_counter()
                graph = _graph_for_record(record)
                tensors = (record["node_features"].float(), record["edge_index"].long(),
                           record["primitive_surface_ids"].long(), record["edge_attr"].float())
                r_prediction = inference.predict_tensors(r_model, tensors, "full", "cpu")
                q_prediction = inference.predict_tensors(q_model, tensors, "dense_instance", "cpu")
                times["read_and_forward_seconds"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                features = extract_joint_features(graph, r_prediction, q_prediction,
                                                  r_cached, q_same, candidate_config, global_config)
                times["joint_feature_seconds"] += time.perf_counter() - t0
                sidecar_feature_sets = [{"macro_feature": obj["macro_feature"],
                                         "node_indices": obj["face_indices"]}
                                        for obj in r_sidecar["truth_objects"]]
                labels = label_part_candidates(features, sidecar_feature_sets)
                if labels["truth_objects"] != r_sidecar["truth_objects"]:
                    raise ValueError("Joint sidecar differs from sealed D0 train truth")
                branch_groups = Counter(tuple(sorted({origin["branch"] for origin in candidate["provenance"]}))
                                        for candidate in features["candidates"])
                stats.update(parts=1, candidates=len(features["candidates"]),
                             positive=labels["exact_positive_count"], truth=labels["truth_count"])
                stats.update({"R_only": branch_groups[("R",)], "Q_only": branch_groups[("Q",)],
                              "both": branch_groups[("Q", "R")]})
                stats.update(labels["negative_type_counts"])
                feature_file.write(json.dumps({"part_index": part_index, **features},
                                              ensure_ascii=False, allow_nan=False) + "\n")
                label_file.write(json.dumps({"part_index": part_index, **labels},
                                            ensure_ascii=False, allow_nan=False) + "\n")
                if (part_index + 1) % 50 == 0:
                    print(json.dumps({"joint_train": part_index + 1, "of": count,
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        if stats["parts"] != count:
            raise ValueError("Joint train part denominator changed")
        os.replace(partial_features, folder / "candidates.jsonl")
        os.replace(partial_labels, folder / "labels.jsonl")
        result = {"schema": "g09_e_joint_train_data_v1", "identity": identity,
                  "stats": dict(sorted(stats.items())), "timings": dict(times),
                  "elapsed_seconds": time.perf_counter() - started,
                  "estimate_full_cpu_seconds": (time.perf_counter() - started) * 7440 / count,
                  "outputs": [descriptor(folder / name) for name in ("candidates.jsonl", "labels.jsonl")],
                  "contract": {"D0_train_only": True, "full_candidate_distribution": True,
                               "truth_sidecar_only_after_inference": True, "gradient_steps": 0}}
        write_json(folder / "COMPLETE.json", result)
        return result
    except Exception as exc:
        write_json(folder / "FAILED.json", {"error": repr(exc), "identity": identity,
                                           "partial_retained": True,
                                           "elapsed_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = run(64 if args.pilot else 7440)
    print(json.dumps({"stats": result["stats"], "elapsed_seconds": result["elapsed_seconds"],
                      "estimate_full_cpu_seconds": result["estimate_full_cpu_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
