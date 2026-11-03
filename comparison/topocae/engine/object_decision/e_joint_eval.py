"""Fixed R+Q branch-aware scorer against frozen R, Q, union, and A4 controls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from candidate_pool import build_candidate_pool, capture_legacy_stages
from e_fusion_analysis import (DATA, G03_ARTIFACT, G07_ARTIFACT, G07_REPORT,
                               OLD_RUNS, aggregate, exact, key, verify_pair)
from e_joint_scorer import FEATURE_NAMES, OUTPUT, load_scorer
from g09_common import ARTIFACT, FROZEN, SEEDS, checked, descriptor, read, write_json
from m0_replay import _engineering_truth, _truth_objects, g07_decoder
from object_features import extract_object_features
from object_select import select_objects
from route_a4_eval import load_scorer as load_a4, infer as infer_a4
from route_a_eval import _CANDIDATE, _ENGINEERING_SAME, _GLOBAL


FOLDER = ARTIFACT / "e_joint_eval"
SOURCE_MANIFEST = FROZEN / ("output_generation_p1_e6r2_r1_3_r1_complete_r2_mixed_9300_v1/"
                            "manifests/sample_manifest.jsonl")
CONFIG = {"candidate_pool": "canonical_eight_stage_R_plus_Q",
          "score": "one_train_only_joint_R_Q_logit_per_complete_object",
          "minimum_positive_logit": 0.0, "exact_limit": 32, "beam_width": 128,
          "R_Q_origin_priors": "none", "validation_or_engineering_specific_rule": False,
          "truth_in_inference": False, "model_selection_after_this_evaluation": False}


def feature_rows(pool, graph, r_prediction, q_prediction):
    """Exactly the training vector contract, without cache/truth/sample identity."""
    result = []
    for candidate in pool:
        branches = {origin["branch"] for origin in candidate["provenance"]}
        if not branches or not branches <= {"R", "Q"}:
            raise ValueError("Unexpected branch in joint candidate")
        r = extract_object_features(candidate, r_prediction, graph)["vector"]
        q = extract_object_features(candidate, q_prediction, graph)["vector"]
        vector = [*r, *q, float("R" in branches), float("Q" in branches)]
        if len(vector) != len(FEATURE_NAMES):
            raise ValueError("Joint scorer and extractor feature order mismatch")
        result.append({**candidate, "vector": vector})
    return result


def infer(graph, r_prediction, q_prediction, r_same, q_same, configs,
          joint, mean, std, a4, a4_mean, a4_std, *, r_recovery=None, q_recovery=None):
    """Same fixed inference path for D0 val and engineering; no target argument."""
    t0 = time.perf_counter()
    graph = g07_decoder.clean_graph(graph)
    r_prediction = g07_decoder.clean_prediction(r_prediction)
    q_prediction = g07_decoder.clean_prediction(q_prediction)
    r_stages = capture_legacy_stages(graph, r_prediction, r_same, *configs, recovery=r_recovery)
    q_stages = capture_legacy_stages(graph, q_prediction, q_same, *configs, recovery=q_recovery)
    pool = build_candidate_pool({"R": r_stages, "Q": q_stages}, face_count=len(graph["nodes"]))
    candidates = feature_rows(pool, graph, r_prediction, q_prediction)
    if candidates:
        with torch.no_grad():
            vectors = torch.tensor([row["vector"] for row in candidates], dtype=torch.float32)
            logits = joint(((vectors - mean) / std).clamp(-8, 8)).tolist()
    else:
        logits = []
    scored = [{**candidate, "score": float(logit), "features": {"vector": candidate["vector"]}}
              for candidate, logit in zip(candidates, logits)]
    selected = select_objects(scored, exact_limit=32, beam_width=128)
    a4_result = infer_a4(graph, r_prediction, r_same, *configs, a4, a4_mean, a4_std,
                         recovery=r_recovery)
    union = [{"macro_feature": cls, "face_indices": list(faces)} for cls, faces in sorted(
        set(map(key, r_stages["final"])) | set(map(key, q_stages["final"]))) ]
    return {"candidates": scored,
            "objects": {"R_final": r_stages["final"], "Q_final": q_stages["final"],
                        "direct_union": union, "A4_R_packing": a4_result["objects"]["learned_packing"],
                        "E_joint_packing": selected["selected_objects"]},
            "selector": selected["diagnostics"], "elapsed_seconds": time.perf_counter() - t0}


def score_one(truth, result):
    truth_keys = set(map(key, truth))
    pool = {key(candidate) for candidate in result["candidates"]}
    selected = set(map(key, result["objects"]["E_joint_packing"]))
    modes = {name: exact(truth, predictions) for name, predictions in result["objects"].items()}
    q_only = [candidate for candidate in result["candidates"] if {
        source["branch"] for source in candidate["provenance"]} == {"Q"}]
    r_final = set(map(key, result["objects"]["R_final"]))
    selected_r = selected & r_final
    return {"metrics": modes, "truth_count": len(truth), "pool_size": len(pool),
            "pool_coverage": len(truth_keys & pool),
            "R_final_selected_TP": len(selected_r & truth_keys),
            "R_final_selected_FP": len(selected_r - truth_keys),
            "R_final_positive_rejected": len((r_final & truth_keys) - selected),
            "missing": [{"macro_feature": cls, "face_indices": list(faces),
                         "pool_present": (cls, faces) in pool}
                        for cls, faces in sorted(truth_keys - selected)],
            "false_positive": [{"macro_feature": cls, "face_indices": list(faces)}
                               for cls, faces in sorted(selected - truth_keys)],
            "Q_only_positive": [{"macro_feature": row["macro_feature"],
                                 "face_indices": row["face_indices"], "logit": row["score"],
                                 "selected": key(row) in selected}
                                for row in q_only if key(row) in truth_keys],
            "Q_only_selected_negative": [{"macro_feature": row["macro_feature"],
                                          "face_indices": row["face_indices"], "logit": row["score"]}
                                         for row in q_only if key(row) not in truth_keys and key(row) in selected],
            "selector": result["selector"], "elapsed_seconds": result["elapsed_seconds"]}


def summarize(rows):
    modes = ("R_final", "Q_final", "direct_union", "A4_R_packing", "E_joint_packing")
    result = {mode: aggregate([row["metrics"][mode] for row in rows]) for mode in modes}
    result.update({"parts": len(rows), "truth_count": sum(row["truth_count"] for row in rows),
                   "pool_size": sum(row["pool_size"] for row in rows),
                   "pool_coverage": sum(row["pool_coverage"] for row in rows),
                   "selection_lost_FN": sum(item["pool_present"] for row in rows for item in row["missing"]),
                   "Q_only_positive": sum(len(row["Q_only_positive"]) for row in rows),
                   "Q_only_positive_selected": sum(sum(item["selected"] for item in row["Q_only_positive"])
                                                   for row in rows),
                   "Q_only_selected_negative": sum(len(row["Q_only_selected_negative"]) for row in rows),
                   "beam_uncertified_parts": sum(not row["selector"]["all_optimality_certified"] for row in rows),
                   "mean_inference_seconds": statistics.mean(row["elapsed_seconds"] for row in rows)})
    for name in ("R_final_selected_TP", "R_final_selected_FP", "R_final_positive_rejected"):
        result[name] = sum(row.get(name, 0) for row in rows)
    return result


def validation(seed, models, configs, output, *, infer_fn=None):
    infer_fn = infer if infer_fn is None else infer_fn
    source = verify_pair(seed)
    with SOURCE_MANIFEST.open(encoding="utf-8") as stream:
        source_graphs = {row["sample_id"]: row for row in map(json.loads, stream)
                         if row.get("split") == "val"}
    if set(source_graphs) != source["manifest_validation_ids"]:
        raise ValueError("Archived graph source and frozen D0 validation membership differ")
    rdir = G03_ARTIFACT / "baseline_replay" / f"seed_{seed}"
    qdir = G03_ARTIFACT / "validation_runs/dense_instance" / f"seed_{seed}"
    rman, qman = [read(path / "prediction_manifest.json") for path in (rdir, qdir)]
    r_predictions = torch.load(checked(rman["predictions"]), map_location="cpu", weights_only=False)
    q_predictions = torch.load(checked(qman["predictions"]), map_location="cpu", weights_only=False)
    r_trace, q_trace = [checked(source["traces"][branch]) for branch in ("R", "Q")]
    r_same = read(OLD_RUNS / f"final_8370_seed_{seed}/same_instance_threshold_selection.json")
    q_same = read(G03_ARTIFACT / "training_runs/dense_instance" / f"seed_{seed}/same_instance_threshold_selection.json")
    rows, ids = [], set()
    with r_trace.open(encoding="utf-8") as rs, q_trace.open(encoding="utf-8") as qs:
        for index, (rp, qp, rline, qline) in enumerate(zip(
            r_predictions, q_predictions, rs, qs, strict=True
        )):
            rr, qr = json.loads(rline), json.loads(qline)
            sid = rr["sample_id"]
            if sid != qr["sample_id"] or sid != rp["record"]["sample_id"] \
                    or sid != qp["record"]["sample_id"] or sid not in source["manifest_validation_ids"] \
                    or sid in ids or Counter(map(key, rr["truth_objects"])) != Counter(map(key, qr["truth_objects"])):
                raise ValueError("Paired R/Q D0 validation record or truth mismatch")
            ids.add(sid)
            if Path(rp["record"]["source_graph_path"]).resolve() != Path(
                qp["record"]["source_graph_path"]
            ).resolve():
                raise ValueError("R/Q D0 source graph paths differ")
            graph_path = Path(rp["record"]["source_graph_path"])
            source_row = source_graphs[sid]
            if graph_path.resolve() != Path(source_row["graph_path"]).resolve() \
                    or descriptor(graph_path)["sha256"].lower() != source_row["graph_sha256"].lower():
                raise ValueError("D0 validation graph bytes differ from archived source manifest")
            graph = read(graph_path)
            for fingerprint in ("graph_structure_sha256", "semantic_graph_sha256"):
                if rp["record"].get(fingerprint) != qp["record"].get(fingerprint):
                    raise ValueError("R/Q frozen record graph fingerprints differ")
            r_pred = {name: rp[name] for name in ("faces", "edge_probabilities")}
            q_pred = {name: qp[name] for name in ("faces", "edge_probabilities", "relation_edges", "relation_probabilities")}
            result = infer_fn(graph, r_pred, q_pred, r_same, q_same, configs, *models,
                           r_recovery={"objects": rr["objects"], "trace": rr["trace"]},
                           q_recovery={"objects": qr["objects"], "trace": qr["trace"]})
            scored = score_one(rr["truth_objects"], result)
            rows.append(scored)
            output.write(json.dumps({"domain": "D0_validation", "seed": seed, "sample_id": sid,
                                     **scored}, ensure_ascii=False, allow_nan=False) + "\n")
            if (index + 1) % 200 == 0:
                print(f"E joint seed={seed} {index + 1}/930", flush=True)
    if len(rows) != 930 or ids != source["manifest_validation_ids"]:
        raise ValueError("D0 validation membership/denominator changed")
    return {"identity": source["identity"],
            "source_generation_manifest": descriptor(SOURCE_MANIFEST), **summarize(rows)}


def engineering(models, configs, output, *, infer_fn=None):
    infer_fn = infer if infer_fn is None else infer_fn
    source = verify_pair(20260719)
    baseline = read(G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json")
    if baseline["parts"] != 13 or baseline["instances"] != 182:
        raise ValueError("Engineering development set denominator changed")
    truth, truth_source = _engineering_truth()
    r_same = read(_ENGINEERING_SAME)
    q_same = read(G03_ARTIFACT / "training_runs/dense_instance/seed_20260719/same_instance_threshold_selection.json")
    rows = []
    for part in baseline["per_part"]:
        folder = checked(part["prediction"]).parent
        qpath = G07_ARTIFACT / "engineering_predictions/D0/seed_20260719" / (folder.name + ".json")
        q = read(qpath)
        if q["sample_id"] != folder.name or q["part"] != part["part"] \
                or q["checkpoint"] != source["identity"]["checkpoint"]["Q"] \
                or q["source_input"] != descriptor(folder / "step_input_graph.json") \
                or q["metadata"] != source["identity"]["normalization"] \
                or q["threshold"] != q_same or q["source_step"] != part["source_step"] \
                or q["truth_used"] is not False:
            raise ValueError("Engineering Q cache identity disagrees with guarded R")
        graph = read(folder / "guard_input_graph.json")
        r_pred = read(folder / "probabilities.json")["guard"]
        q_pred = q["guarded"]["prediction"]
        result = infer_fn(graph, r_pred, q_pred, r_same, q_same, configs, *models)
        if Counter(map(key, result["objects"]["R_final"])) != Counter(map(
            key, read(folder / "stages.json")["guard"]["service_objects"]
        )) or Counter(map(key, result["objects"]["Q_final"])) != Counter(map(
            key, q["guarded"]["old_stages"]["full"]
        )):
            raise ValueError("R/Q historical engineering final regression")
        manual = [{"macro_feature": item["class"], "face_indices": item["faces"]}
                  for item in _truth_objects(truth, part["part"])]
        scored = score_one(manual, result)
        scored["part"] = part["part"]
        rows.append(scored)
        output.write(json.dumps({"domain": "engineering_development", **scored},
                                ensure_ascii=False, allow_nan=False) + "\n")
    return {"identity": source["identity"], "truth_source": truth_source, **summarize(rows)}


def run(*, seed=None, include_engineering=False):
    if (seed is None) == (not include_engineering) or (seed is not None and seed not in SEEDS):
        raise ValueError("Specify one frozen D0 seed or engineering development")
    base_name = "engineering_seed_20260719" if include_engineering else f"D0_seed_{seed}"
    for attempt in range(1, 100):
        name = base_name if attempt == 1 else f"{base_name}_attempt_{attempt}"
        path = FOLDER / f"{name}.json"
        if path.exists():
            raise FileExistsError("Completed joint E evaluation exists; preserve it")
        if not any((FOLDER / f"{name}{suffix}").exists() for suffix in (
            ".jsonl", ".jsonl.partial", "_FAILED.json"
        )):
            break
    else:
        raise RuntimeError("Too many retained E joint evaluation attempts")
    scorer, mean, std = load_scorer(OUTPUT)
    a4, a4_mean, a4_std = load_a4(ARTIFACT / "object_scorer/A4_seed_20260719")
    models = (scorer, mean, std, a4, a4_mean, a4_std)
    configs = read(_CANDIDATE), read(_GLOBAL)
    torch.set_num_threads(4)
    FOLDER.mkdir(parents=True, exist_ok=True)
    sample_path = FOLDER / f"{name}.jsonl"
    partial = FOLDER / f"{name}.jsonl.partial"
    if partial.exists():
        raise FileExistsError("Previous partial E joint evaluation must be retained")
    began = time.perf_counter()
    try:
        with partial.open("w", encoding="utf-8") as output:
            result = engineering(models, configs, output) if include_engineering else \
                     validation(seed, models, configs, output)
        partial.replace(sample_path)
        status = {"schema": "g09_e_joint_evaluation_v1", "domain": name,
                  "fixed_rule": CONFIG, "joint_scorer": descriptor(OUTPUT / "COMPLETE.json"),
                  "A4_control": descriptor(ARTIFACT / "object_scorer/A4_seed_20260719/COMPLETE.json"),
                  "candidate_config": descriptor(_CANDIDATE), "global_config": descriptor(_GLOBAL),
                  "per_sample": descriptor(sample_path), "result": result,
                  "elapsed_seconds": time.perf_counter() - began,
                  "final_test_exposed": False}
        write_json(path, status)
        return status
    except Exception as exc:
        write_json(FOLDER / f"{name}_FAILED.json", {"error": repr(exc),
            "partial_retained": partial.exists(), "elapsed_seconds": time.perf_counter() - began})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    result = run(seed=args.seed, include_engineering=args.engineering)
    print(json.dumps({"domain": result["domain"], "elapsed_seconds": result["elapsed_seconds"],
                      "metrics": {name: {key: value[key] for key in ("tp", "fp", "fn", "macro_f1", "all_exact_parts")}
                                  for name, value in result["result"].items() if name in (
                                      "R_final", "Q_final", "direct_union", "A4_R_packing", "E_joint_packing")}},
                     ensure_ascii=False), flush=True)
