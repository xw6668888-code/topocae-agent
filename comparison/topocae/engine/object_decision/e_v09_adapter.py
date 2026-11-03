"""Truth-free V09 paired R/Q adapter for frozen E1, E2 and E3 rules.

Never opens V09 targets. The full 930-run is gated on independent V09 geometry
COMPLETE and exact R/Q prediction seals. A pilot cannot masquerade as full.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch

from candidate_pool import STAGES, build_candidate_pool
from e_fusion_analysis import key, verify_pair
from e_joint_calibration import (PRIOR_PATH, PROTOCOL, calibrated_score)
from e_joint_eval import feature_rows
from e_joint_retention import OUTPUT as E2_PATH, load_scorer as load_e2
from e_joint_scorer import OUTPUT as E1_PATH, load_scorer as load_e1
from g09_common import ARTIFACT, SEEDS, checked, descriptor, now, read, write_json
from m0_replay import g07_decoder
from object_select import select_objects
from v09_predict_topo import V09, source_rows, _frozen_imports


ROUTES = ("E1", "E2", "E3")


def checked_stages(row, branch):
    """Prevent G07 post-processing from silently changing G03-full Q semantics."""
    stages = row["stages"]
    if set(stages) != set(STAGES):
        raise ValueError("V09 prediction lacks one of the eight frozen candidate stages")
    if Counter(map(key, stages["final"])) != Counter(map(key, stages["post_boundary"])):
        raise ValueError("V09 final stage differs from G03 full post-boundary training semantics")
    public = Counter((obj["class"], tuple(sorted(obj["faces"]))) for obj in row["objects"])
    if public != Counter(map(key, stages["final"])):
        raise ValueError("V09 public objects differ from cached final stage")
    expected = build_candidate_pool({branch: stages}, face_count=row["face_count"])
    if expected != row["candidate_pool"]:
        raise ValueError("V09 cached per-branch pool differs from eight-stage reconstruction")
    return stages


def infer_cached(graph, r_prediction, q_prediction, r_stages, q_stages,
                 model, mean, std, *, route, prior=None, temperature=None):
    """Production decision: no sample ID, path, target, or candidate ID input."""
    if route not in ROUTES:
        raise ValueError("Only the three frozen E routes are allowed")
    clean = g07_decoder.clean_graph(graph)
    r = g07_decoder.clean_prediction(r_prediction)
    q = g07_decoder.clean_prediction(q_prediction)
    count = len(clean["nodes"])
    if len(r["faces"]) != count or len(q["faces"]) != count or [
        face["idx"] for face in r["faces"]
    ] != list(range(count)) or [face["idx"] for face in q["faces"]] != list(range(count)):
        raise ValueError("R/Q predicted faces do not match graph STEP indices")
    pool = build_candidate_pool({"R": r_stages, "Q": q_stages}, face_count=count)
    features = feature_rows(pool, clean, r, q)
    if features:
        with torch.no_grad():
            x = torch.tensor([row["vector"] for row in features], dtype=torch.float32)
            scores = model(((x - mean) / std).clamp(-8, 8)).tolist()
    else:
        scores = []
    scored = []
    for candidate, raw in zip(features, scores):
        score = calibrated_score(float(raw), candidate, prior, temperature) if route == "E3" else float(raw)
        scored.append({**candidate, "score": score, "features": {"vector": candidate["vector"]}})
    selection = select_objects(scored, exact_limit=32, beam_width=128)
    return {"objects": [{"class": obj["macro_feature"], "faces": obj["face_indices"]}
                        for obj in selection["selected_objects"]],
            "face_macro_classes": [face["macro_feature"] for face in r["faces"]],
            "candidate_count": len(pool), "selection": selection["diagnostics"]}


def load_route(route):
    if route == "E1":
        return (*load_e1(), None, None, {"scorer": descriptor(E1_PATH / "COMPLETE.json")})
    if route == "E2":
        return (*load_e2(), None, None, {"scorer": descriptor(E2_PATH / "COMPLETE.json")})
    if route != "E3":
        raise ValueError("Unknown frozen route")
    protocol = read(PROTOCOL)
    prior = read(checked(protocol["prior"]))
    if protocol["schema"] != "g09_e3_bayesian_calibration_protocol_v1" \
            or protocol["E1_scorer"] != descriptor(E1_PATH / "COMPLETE.json") \
            or protocol["fixed_rule"]["T"] != 3 or protocol["prior"] != descriptor(PRIOR_PATH):
        raise ValueError("E3 train-derived calibration protocol/identity changed")
    return (*load_e1(), prior, protocol["fixed_rule"]["T"],
            {"scorer": protocol["E1_scorer"], "protocol": descriptor(PROTOCOL),
             "prior": descriptor(PRIOR_PATH)})


def pair_source(seed):
    """Read only sealed unlabeled V09 inputs and frozen prediction identities."""
    if seed not in SEEDS:
        raise ValueError("Only three frozen D0 seeds are valid")
    manifest, inputs = source_rows(pilot=False)  # Refuses missing V09 COMPLETE.
    if len(inputs) != 930:
        raise ValueError("V09 full validation denominator differs")
    frozen = verify_pair(seed)
    sealed = {}
    for branch in ("R", "Q"):
        path = V09 / "predictions/full_930" / f"{branch}_seed_{seed}" / "COMPLETE.json"
        item = read(path)
        identity = item.get("model_identity", {})
        if item.get("schema") != "g09_v09_predictions_v1" \
                or item.get("method") != f"{branch}_seed_{seed}" \
                or item.get("manifest") != manifest or item.get("sample_count") != 930 \
                or item.get("pilot_only") is not False or item.get("truth_used") is not False \
                or identity.get("dataset") != "D0" or identity.get("seed") != seed \
                or identity.get("branch") != branch or identity.get("configuration") != (
                    "full" if branch == "R" else "dense_instance"
                ) or identity.get("checkpoint") != frozen["identity"]["checkpoint"][branch] \
                or identity.get("normalization") != frozen["identity"]["normalization"]:
            raise ValueError("V09 R/Q prediction cache differs from paired frozen D0 identities")
        for asset in (identity["threshold"], *identity["legacy_decoder_assets"],
                      *identity["inference_sources"]):
            checked(asset)
        sealed[branch] = {"seal": descriptor(path), "predictions": checked(item["predictions"])}
    return manifest, inputs, sealed


def run(route, seed):
    if route not in ROUTES or seed not in SEEDS:
        raise ValueError("Select one frozen E route and one frozen D0 seed")
    manifest, inputs, sealed = pair_source(seed)
    model, mean, std, prior, temperature, scorer_identity = load_route(route)
    geometry_only, _, _ = _frozen_imports()
    torch.set_num_threads(4)
    label = f"{route}_seed_{seed}"
    output = V09 / "predictions/full_930" / label
    seal_path = output / "COMPLETE.json"
    if seal_path.exists():
        old = read(seal_path)
        if old["manifest"] != manifest or old["scoring_identity"] != scorer_identity \
                or old["paired_predictions"] != {branch: row["seal"] for branch, row in sealed.items()}:
            raise ValueError("Existing E V09 cache has another source/score identity")
        checked(old["predictions"])
        return old
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete E V09 prediction retained; no overwrite")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "predictions.jsonl.partial"
    count = 0
    started = time.perf_counter()
    try:
        with sealed["R"]["predictions"].open(encoding="utf-8") as r_file, \
             sealed["Q"]["predictions"].open(encoding="utf-8") as q_file, \
             temporary.open("w", encoding="utf-8") as stream:
            for source, rline, qline in zip(inputs, r_file, q_file, strict=True):
                rrow, qrow = json.loads(rline), json.loads(qline)
                if rrow["sample_id"] != qrow["sample_id"] \
                        or rrow["sample_id"] != source["sample_id"] \
                        or rrow["source_step_sha256"] != qrow["source_step_sha256"] \
                        or rrow["source_step_sha256"] != source["step"]["sha256"] \
                        or rrow["face_count"] != qrow["face_count"] \
                        or rrow["face_count"] != source["face_count"] \
                        or rrow.get("truth_used") is not False or qrow.get("truth_used") is not False:
                    raise ValueError("V09 R/Q same STEP mapping or truth-free contract mismatch")
                r_stages, q_stages = checked_stages(rrow, "R"), checked_stages(qrow, "Q")
                graph = geometry_only(read(checked(source["input_graph"])))
                if len(graph["nodes"]) != source["face_count"]:
                    raise ValueError("V09 geometry face count differs from paired cache")
                result = infer_cached(graph, rrow["raw_prediction"], qrow["raw_prediction"],
                                      r_stages, q_stages, model, mean, std,
                                      route=route, prior=prior, temperature=temperature)
                if result["face_macro_classes"] != rrow["face_macro_classes"] \
                        or [face["macro_feature"] for face in qrow["raw_prediction"]["faces"]] != qrow["face_macro_classes"]:
                    raise ValueError("V09 face-class predictions differ from sealed R/Q rows")
                stream.write(json.dumps({"sample_id": source["sample_id"],
                    "source_step_sha256": source["step"]["sha256"],
                    "face_count": source["face_count"],
                    "face_macro_classes": result["face_macro_classes"],
                    "objects": result["objects"], "candidate_count": result["candidate_count"],
                    "selection": result["selection"], "truth_used": False},
                    ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"E V09 {label}: {count}/930", flush=True)
        if count != 930:
            raise ValueError("E V09 output denominator differs from geometry seal")
        final = output / "predictions.jsonl"
        os.replace(temporary, final)
        status = {"schema": "g09_v09_predictions_v1", "at_utc": now(),
                  "method": label, "manifest": manifest, "predictions": descriptor(final),
                  "sample_count": count, "paired_predictions": {branch: row["seal"] for branch, row in sealed.items()},
                  "scoring_identity": scorer_identity,
                  "adapter": descriptor(__file__),
                  "candidate_pool": descriptor(Path(__file__).with_name("candidate_pool.py")),
                  "object_features": descriptor(Path(__file__).with_name("object_features.py")),
                  "selected_face_labels": "R frozen predicted face classes",
                  "elapsed_seconds": time.perf_counter() - started,
                  "pilot_only": False, "truth_used": False, "final_test_exposed": False}
        write_json(seal_path, status)
        return status
    except Exception as exc:
        write_json(output / "FAILED.json", {"method": label, "completed_samples": count,
                    "error": repr(exc), "partial_retained": temporary.exists(),
                    "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    result = run(args.route, args.seed)
    print(json.dumps({"method": result["method"], "sample_count": result["sample_count"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "predictions": result["predictions"], "truth_used": result["truth_used"]}), flush=True)
