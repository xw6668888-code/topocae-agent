"""Read-only matched D0 R/Q cache audit and bounded fusion diagnostics.

Historical G03 traces do not retain pre-boundary raw geometric proposals.
The stage-pool oracle here is therefore a lower bound on G09's full pool.
Oracle and target data never enter an inference or selector decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from cache_identity import require_identity
from candidate_pool import STAGES, build_candidate_pool, capture_legacy_stages
from g09_common import (ARTIFACT, FROZEN, G03_ARTIFACT, G03_TOOLS,
                        G07_ARTIFACT, G07_REPORT, ROOT, SEEDS, checked,
                        descriptor, read, sha256, write_json)


G03_REPORT = ROOT / "reports/aei_paper_goals_v1/g03"
FREEZE = G03_REPORT / "CONFIGURATION_FREEZE.json"
MANIFEST = G03_ARTIFACT / "INPUT_MANIFEST.json"
DATA = FROZEN / "output_training_p1_e6r2_r3_graph_v2_2_1_v1"
OLD_RUNS = FROZEN / "reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs"
TARGET_CLASSES = (
    "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
    "countersink_hole", "obround_hole", "slot_u", "slot_rect", "pocket_rect",
    "pocket_circular", "cylinder_boss", "rib", "web", "rectangular_boss",
    "fillet", "chamfer", "slot_v", "slot_o",
)


def key(item):
    return str(item["macro_feature"]), tuple(sorted(int(face) for face in item["face_indices"]))


def exact(truth, objects):
    t = Counter(map(key, truth))
    p = Counter(map(key, objects))
    m = t & p
    return {"tp": sum(m.values()), "fp": sum(p.values()) - sum(m.values()),
            "fn": sum(t.values()) - sum(m.values()),
            "class": {name: {"tp": sum(v for (cls, _), v in m.items() if cls == name),
                             "truth": sum(v for (cls, _), v in t.items() if cls == name),
                             "prediction": sum(v for (cls, _), v in p.items() if cls == name)}
                      for name in TARGET_CLASSES}}


def aggregate(rows):
    counts = {name: sum(row[name] for row in rows) for name in ("tp", "fp", "fn")}
    class_f1 = {}
    for name in TARGET_CLASSES:
        tp = sum(row["class"][name]["tp"] for row in rows)
        truth = sum(row["class"][name]["truth"] for row in rows)
        pred = sum(row["class"][name]["prediction"] for row in rows)
        class_f1[name] = 2 * tp / (truth + pred) if truth + pred else 1.0
    counts["macro_f1"] = statistics.mean(class_f1.values())
    counts["micro_f1"] = 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])
    counts["all_exact_parts"] = sum(row["fp"] == row["fn"] == 0 for row in rows)
    counts["per_class_f1"] = class_f1
    return counts


def _stage_proxy(objects):
    return {"initial": objects["candidate"], "geometric": [],
            "pre_validation": objects["proposals"], "post_validation": objects["verified"],
            "global_selection": objects["decoded"], "pre_boundary": objects["decoded"],
            "post_boundary": objects["full"], "final": objects["full"]}


def paired_part(r, q):
    if r["sample_id"] != q["sample_id"] or Counter(map(key, r["truth_objects"])) != Counter(map(key, q["truth_objects"])):
        raise ValueError("R/Q validation targets or sample alignment differ")
    truth, ro, qo = r["truth_objects"], r["objects"], q["objects"]
    final_r, final_q = ro["full"], qo["full"]
    union = [{"macro_feature": macro, "face_indices": list(faces)}
             for macro, faces in sorted(set(map(key, final_r)) | set(map(key, final_q)))]
    pool = build_candidate_pool({"R": _stage_proxy(ro), "Q": _stage_proxy(qo)})
    truth_keys = set(map(key, truth))
    r_hit = truth_keys & set(map(key, final_r))
    q_hit = truth_keys & set(map(key, final_q))
    pool_hit = truth_keys & set(map(key, pool))
    return {"sample_id": r["sample_id"],
            "metrics": {"R_final": exact(truth, final_r), "Q_final": exact(truth, final_q),
                        "direct_deduplicated_union": exact(truth, union)},
            "truth_count": len(truth), "stage_pool_candidate_count": len(pool),
            "stage_pool_hits": len(pool_hit), "R_only_final_hits": len(r_hit - q_hit),
            "Q_only_final_hits": len(q_hit - r_hit),
            "final_union_hits": len(r_hit | q_hit),
            "stage_pool_new_hits_beyond_R_final": len(pool_hit - r_hit),
            "stage_pool_new_hits_beyond_both_finals": len(pool_hit - (r_hit | q_hit)),
            "duplicate_final_objects": len(final_r) + len(final_q) - len(union)}


def _frozen_descriptor(path, frozen_files):
    matches = [row for row in frozen_files if Path(row["path"]).resolve() == path.resolve()]
    if len(matches) != 1:
        raise ValueError(f"Missing unique G03 frozen binding: {path}")
    return checked(matches[0]), matches[0]


def verify_pair(seed):
    if seed not in SEEDS:
        raise ValueError("Only three frozen D0 seeds permitted")
    freeze, manifest = read(FREEZE), read(MANIFEST)
    _frozen_descriptor(MANIFEST, freeze["frozen_files"])
    train = next(x for x in manifest["inputs"] if Path(x["path"]).name == "train.pt")
    val = next(x for x in manifest["inputs"] if Path(x["path"]).name == "val.pt")
    norm = next(x for x in manifest["inputs"] if Path(x["path"]).name == "metadata.json")
    package = next(x for x in manifest["inputs"] if Path(x["path"]).name == "manifest.json")
    for value in (train, val, norm, package):
        checked(value)
        _frozen_descriptor(Path(value["path"]), freeze["frozen_files"])
    if len(manifest["identities"]["train"]) != 7440 or len(manifest["identities"]["val"]) != 930:
        raise ValueError("D0 train/val membership count mismatch")
    if set(manifest["identities"]["train"]) & set(manifest["identities"]["val"]):
        raise ValueError("D0 train/val membership overlap")
    rdir = G03_ARTIFACT / "baseline_replay" / f"seed_{seed}"
    qdir = G03_ARTIFACT / "validation_runs/dense_instance" / f"seed_{seed}"
    qrun = G03_ARTIFACT / "training_runs/dense_instance" / f"seed_{seed}"
    r_config = read(OLD_RUNS / f"final_8370_seed_{seed}" / "config.json")
    q_config = read(qrun / "config.json")
    if r_config["data"]["split_counts"] != {"train": 7440, "val": 930}:
        raise ValueError("R frozen training split changed")
    if r_config["data"]["package_manifest"]["sha256"].lower() != package["sha256"].lower():
        raise ValueError("R package manifest differs from frozen D0")
    if q_config["data_manifest"] != descriptor(MANIFEST) or q_config["configuration_freeze_sha256"] != sha256(FREEZE):
        raise ValueError("Q training used a different D0/freeze identity")
    if read(qrun / "STATUS.json")["status"] != "completed":
        raise ValueError("Q frozen training incomplete")
    r_pred, q_pred = read(rdir / "prediction_manifest.json"), read(qdir / "prediction_manifest.json")
    r_summary, q_summary = read(rdir / "summary.json"), read(qdir / "summary.json")
    r_run = next(x for x in freeze["baseline_runs"] if x["seed"] == seed)
    r_checkpoint = next(x for x in r_run["files"] if Path(x["path"]).name == "best.pt")
    q_checkpoint = read(qrun / "metrics.json")["checkpoint"]
    if r_pred["checkpoint"] != r_checkpoint or q_pred["checkpoint"] != q_checkpoint:
        raise ValueError("R/Q checkpoint does not match training provenance")
    for value in (r_checkpoint, q_checkpoint, r_pred["predictions"], q_pred["predictions"],
                  r_summary["traces"], q_summary["traces"]):
        checked(value)
    if r_pred["val"] != val or q_pred["val"] != val:
        raise ValueError("R/Q validation input differs")
    if q_pred["configuration_freeze_sha256"] != sha256(FREEZE):
        raise ValueError("Q prediction entry freeze changed")
    if r_summary["sample_count"] != q_summary["sample_count"] != 930:
        raise ValueError("R/Q validation trace count changed")
    expected = {"schema": "g09_cache_identity_v1", "dataset": "D0", "split": "val",
                "configuration": "R+Q", "seed": seed, "sample_count": 930,
                "training_data": train, "architecture": {
                    "R": descriptor(FROZEN / "models/gnn_baseline.py"),
                    "Q": descriptor(G03_TOOLS / "ablations.py")},
                "checkpoint": {"R": r_checkpoint, "Q": q_checkpoint},
                "normalization": norm,
                "prediction_entry": {"R": descriptor(G03_TOOLS / "stages.py"),
                                     "Q": descriptor(G03_TOOLS / "evaluate_runs.py")},
                "sample_manifest": descriptor(MANIFEST)}
    return {"identity": expected, "traces": {"R": r_summary["traces"], "Q": q_summary["traces"]},
            "manifest_validation_ids": set(manifest["identities"]["val"])}


def analyze(seed):
    sources = verify_pair(seed)
    rfile = Path(sources["traces"]["R"]["path"])
    qfile = Path(sources["traces"]["Q"]["path"])
    rows = []
    with rfile.open(encoding="utf-8") as rs, qfile.open(encoding="utf-8") as qs:
        for rline, qline in zip(rs, qs, strict=True):
            rows.append(paired_part(json.loads(rline), json.loads(qline)))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 930 or len(set(ids)) != 930 or set(ids) != sources["manifest_validation_ids"]:
        raise ValueError("Paired traces do not match exact frozen D0 val membership")
    metrics = {name: aggregate([row["metrics"][name] for row in rows])
               for name in ("R_final", "Q_final", "direct_deduplicated_union")}
    totals = {key: sum(row[key] for row in rows) for key in (
        "truth_count", "stage_pool_candidate_count", "stage_pool_hits",
        "R_only_final_hits", "Q_only_final_hits", "final_union_hits",
        "stage_pool_new_hits_beyond_R_final", "stage_pool_new_hits_beyond_both_finals",
        "duplicate_final_objects")}
    totals["stage_pool_oracle_recall"] = totals["stage_pool_hits"] / totals["truth_count"]
    result = {"schema": "g09_e_d0_fusion_diagnostic_v1", "seed": seed,
              "identity": sources["identity"], "metrics": metrics, "candidate_totals": totals,
              "limitations": ["D0 val is development evidence, not final test.",
                  "Archived stage traces lack raw pre-base-boundary proposals: pooled oracle recall is conservative.",
                  "Direct union is deduplicated but not conflict selected; oracle is never a deployable score."],
              "per_part": rows}
    output = ARTIFACT / "e_fusion_analysis" / f"D0_seed_{seed}.json"
    if output.exists():
        previous = read(output)
        require_identity(result["identity"], previous["identity"])
        if previous != result:
            raise ValueError("Existing E diagnostic differs; retain it and inspect")
        return previous
    write_json(output, result)
    return result


def _captured_matches(captured, archived):
    mapping = {"initial": "candidate", "pre_validation": "proposals",
               "post_validation": "verified", "global_selection": "decoded",
               "pre_boundary": "decoded", "post_boundary": "full"}
    for stage, old in mapping.items():
        if Counter(map(key, captured[stage])) != Counter(map(key, archived[old])):
            raise ValueError(f"Frozen G03 engineering stage mismatch: {stage}")


def _engineering_score(truth, r_stages, q_stages):
    r_final, q_final = r_stages["final"], q_stages["final"]
    union = [{"macro_feature": name, "face_indices": list(faces)}
             for name, faces in sorted(set(map(key, r_final)) | set(map(key, q_final)))]
    pool = build_candidate_pool({"R": r_stages, "Q": q_stages})
    pool_keys = set(map(key, pool))
    truth_keys = set(map(key, truth))
    return {
        "metrics": {"R_final": exact(truth, r_final), "Q_final": exact(truth, q_final),
                    "direct_deduplicated_union": exact(truth, union)},
        "truth_count": len(truth), "candidate_count": len(pool),
        "candidate_coverage": len(pool_keys & truth_keys),
        "candidate_missing": [list((name, list(faces))) for name, faces in sorted(truth_keys - pool_keys)],
        "R_final_missing": [list((name, list(faces))) for name, faces in sorted(truth_keys - set(map(key, r_final)))],
        "Q_final_missing": [list((name, list(faces))) for name, faces in sorted(truth_keys - set(map(key, q_final)))],
        "R_final_fp": [list((name, list(faces))) for name, faces in sorted(set(map(key, r_final)) - truth_keys)],
        "Q_final_fp": [list((name, list(faces))) for name, faces in sorted(set(map(key, q_final)) - truth_keys)],
        "direct_union_fp": [list((name, list(faces))) for name, faces in sorted(set(map(key, union)) - truth_keys)],
    }


def analyze_engineering():
    """Pair 13 guarded R/Q records before opening independent truth for scoring."""
    seed = 20260719
    sources = verify_pair(seed)
    baseline_path = G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json"
    baseline = read(baseline_path)
    if baseline["parts"] != 13 or baseline["instances"] != 182 or baseline["selected_seed"] != seed:
        raise ValueError("Frozen engineering R baseline changed")
    qrun = G03_ARTIFACT / "training_runs/dense_instance" / f"seed_{seed}"
    qcheckpoint = sources["identity"]["checkpoint"]["Q"]
    metadata = sources["identity"]["normalization"]
    qsame = read(qrun / "same_instance_threshold_selection.json")
    rsame = read(FROZEN / "model_packages/p1_e12_reviewed_legacy_v1/same_instance_threshold.json")
    decoder_base = FROZEN / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718"
    candidate_config = read(decoder_base / "candidate_model/candidate_threshold_selection.json")
    global_config = read(decoder_base / "global_decode_selection.json")
    paired = []
    started = time.perf_counter()
    for row in baseline["per_part"]:
        folder = checked(row["prediction"]).parent
        if folder.name != ("engineering_" + Path(row["part"]).stem).replace(" ", "_"):
            raise ValueError("R engineering part/source folder mismatch")
        r_graph_asset = descriptor(folder / "guard_input_graph.json")
        r_step_graph_asset = descriptor(folder / "step_input_graph.json")
        graph = read(r_graph_asset["path"])
        r_prob = read(folder / "probabilities.json")["guard"]
        r_archive = read(folder / "stages.json")["guard"]
        q_path = G07_ARTIFACT / "engineering_predictions/D0" / f"seed_{seed}" / (folder.name + ".json")
        q = read(q_path)
        if q["dataset_train"] != "D0" or q["method"] != "new" or q["seed"] != seed \
                or q["sample_id"] != folder.name or q["part"] != row["part"] or q["truth_used"] is not False:
            raise ValueError("Q engineering cache semantic identity mismatch")
        if q["source_input"] != r_step_graph_asset or q["source_step"] != row["source_step"]:
            raise ValueError("R/Q source STEP or final-face graph differs")
        if q["checkpoint"] != qcheckpoint or q["metadata"] != metadata or q["threshold"] != qsame:
            raise ValueError("Q engineering model/statistics/threshold mismatch")
        for value in (q["source_input"], q["source_step"], q["checkpoint"], q["metadata"]):
            checked(value)
        r_stages = capture_legacy_stages(graph, r_prob, rsame, candidate_config, global_config,
                                         final_objects=r_archive["service_objects"])
        q_stages = capture_legacy_stages(graph, q["guarded"]["prediction"], qsame,
                                         candidate_config, global_config)
        _captured_matches(r_stages, r_archive["objects"])
        _captured_matches(q_stages, q["guarded"]["old_stages"])
        if Counter(map(key, r_stages["final"])) != Counter(map(key, r_archive["service_objects"])):
            raise ValueError("R service final parity changed")
        paired.append({"part": row["part"], "source_graph": r_graph_asset,
                       "Q_cache": descriptor(q_path), "Q_CPU_seconds": q["CPU_seconds"],
                       "R_timing": read(folder / "result.json")["timings"]["paths"]["guard"],
                       "R_stages": r_stages, "Q_stages": q_stages})
    # The input/prediction/fusion cache pairing above is finished before any
    # independent annotation is opened. Truth is never passed to capture.
    from m0_replay import _engineering_truth, _truth_objects
    manual, truth_source = _engineering_truth()
    scored = []
    for row in paired:
        truth = [{"macro_feature": item["class"], "face_indices": item["faces"]}
                 for item in _truth_objects(manual, row["part"])]
        value = _engineering_score(truth, row["R_stages"], row["Q_stages"])
        scored.append({key: value[key] for key in value} | {
            "part": row["part"], "source_graph": row["source_graph"], "Q_cache": row["Q_cache"],
            "Q_CPU_seconds": row["Q_CPU_seconds"], "R_timing": row["R_timing"]})
    aggregate_metrics = {name: aggregate([row["metrics"][name] for row in scored])
                         for name in ("R_final", "Q_final", "direct_deduplicated_union")}
    result = {"schema": "g09_e_d0_engineering_fusion_diagnostic_v1", "seed": seed,
              "D0_identity": sources["identity"], "R_baseline": descriptor(baseline_path),
              "manual_truth": truth_source, "Q_model": qcheckpoint,
              "metrics": aggregate_metrics,
              "candidate_coverage": sum(row["candidate_coverage"] for row in scored),
              "truth_count": sum(row["truth_count"] for row in scored),
              "candidate_count": sum(row["candidate_count"] for row in scored),
              "elapsed_seconds": time.perf_counter() - started,
              "per_part": scored, "truth_used_in_capture": False,
              "all_13_matched_source_and_archived_stage_parity": True}
    output = ARTIFACT / "e_fusion_analysis/D0_engineering_seed_20260719.json"
    if output.exists():
        previous = read(output)
        require_identity(result["D0_identity"], previous["D0_identity"])
        if previous != result:
            raise ValueError("Existing engineering E diagnostic differs")
        return previous
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    if args.engineering:
        result = analyze_engineering()
        print(json.dumps({"engineering": 13, "metrics": {
            name: {key: value[key] for key in ("tp", "fp", "fn", "all_exact_parts")}
            for name, value in result["metrics"].items()},
            "candidate_coverage": result["candidate_coverage"],
            "truth_count": result["truth_count"],
            "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return
    result = analyze(args.seed)
    print(json.dumps({"seed": args.seed, "metrics": {
        name: {key: value[key] for key in ("tp", "fp", "fn", "macro_f1")}
        for name, value in result["metrics"].items()},
        "candidate_totals": result["candidate_totals"]}), flush=True)


if __name__ == "__main__":
    main()
