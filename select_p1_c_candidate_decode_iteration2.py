from __future__ import annotations

"""Final Stage 6 iteration: joint global confidence/decode selection on Val."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

import evaluate_p1_b_synthetic_pipeline as evaluation
from p1_c.iteration1_provenance import preflight_iteration_model
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_c_iteration1_run_paths import iteration_run_dir
from utils.p1_c_iteration2_run_paths import iteration2_run_dir
from utils.p1_c_strict_run_paths import DATASET_ROOT, ROOT, strict_run_dir, training_metadata_path
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction


PLAN = ROOT / "configs" / "p1_c" / "stage6_iteration2_candidate_verifier_calibration_v1.json"
PLAN_SHA256 = "ca74c8958c18d71ed0294d3c0e68e50723fd1b46468445ebd02ef778f2e4c87f"
CONFIDENCE_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DECODE_GRID = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
POLARITY_GRAPHS = DATASET_ROOT / "graphs_polarity_v1"


def _hash(path: Path) -> str:
    digest = sha256(path.read_bytes())
    return digest.hexdigest().lower()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if _hash(PLAN) != PLAN_SHA256:
        raise RuntimeError("iteration-2 plan drift")
    provenance = preflight_iteration_model(args.variant, args.seed)
    source_run = strict_run_dir(args.variant, args.seed)
    model_run = iteration_run_dir(args.variant, args.seed)
    output = iteration2_run_dir(args.variant, args.seed)
    destination = output / "global_candidate_decode_selection.json"
    if destination.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "selection": str(destination)}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    compact_macros = _read(training_metadata_path(args.variant))["macro_classes"]
    macro_classes = compact_macros[1:]
    gnn = torch.load(source_run / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    candidates = torch.load(model_run / "candidate_model_v1" / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        by_sample.setdefault(str(record["sample_id"]), []).append(record)
    base_thresholds = _read(model_run / "candidate_model_v1" / "candidate_threshold_selection.json")
    graphs = {}
    truths = []
    face_macros = {}
    for gnn_record in gnn:
        sample_id = str(gnn_record["sample_id"])
        graph = _read(POLARITY_GRAPHS / Path(gnn_record["source_graph_path"]).name)
        graphs[sample_id] = graph
        face_macros[sample_id] = [compact_macros[int(value)] for value in gnn_record["macro_prediction"]]
        for record in evaluation._truth_objects(graph, sample_id):
            record["boundary_face_indices"] = sorted(evaluation._boundary_faces(graph, set(record["face_indices"])))
            truths.append(record)
    original_boundary = evaluation._boundary_faces_from_objects
    evaluation._boundary_faces_from_objects = lambda record, sample_id: set(record.get("boundary_face_indices", []))
    scans = []
    try:
        for confidence in CONFIDENCE_GRID:
            thresholds = dict(base_thresholds)
            thresholds["min_candidate_confidence"] = confidence
            verified_by_sample = {}
            for sample_id, graph in graphs.items():
                verified = []
                for candidate in by_sample.get(sample_id, []):
                    verified.append(verify_candidate_prediction(
                        graph,
                        candidate["proposal"],
                        {
                            "macro_classes": macro_classes,
                            "macro_logits": candidate["macro_logits"],
                            "macro_probabilities": candidate["macro_probabilities"],
                            "validity_score": candidate["validity_score"],
                            "membership_probabilities": candidate["membership_probabilities"],
                            "boundary_probabilities": candidate["boundary_probabilities"],
                            "face_macro_predictions": face_macros[sample_id]
                        },
                        thresholds=thresholds
                    ))
                verified_by_sample[sample_id] = verified
            for tolerance in DECODE_GRID:
                objects = []
                subset_dominated = 0
                for sample_id, graph in graphs.items():
                    decoded = decode_feature_objects(
                        graph,
                        verified_by_sample[sample_id],
                        subset_confidence_tolerance=tolerance,
                        selection_sha256="SYNTHETIC_VAL_SELECTION_IN_PROGRESS"
                    )
                    subset_dominated += int(decoded["subset_dominated_count"])
                    objects.extend({**record, "sample_id": sample_id} for record in decoded["feature_objects"])
                metrics = evaluation._object_metrics(truths, objects, macro_classes)
                active = [record["f1"] for record in metrics["per_class"].values() if record["truth"]]
                worst = min(active, default=0.0)
                score = sum((metrics["object_macro_f1"], metrics["object_micro_f1"], metrics["exact_instance_ratio"], metrics["boundary"]["f1"])) / 4.0
                gates = bool(metrics["object_macro_f1"] >= 0.90 and metrics["exact_instance_ratio"] >= 0.88 and worst >= 0.80)
                scans.append({
                    "min_candidate_confidence": confidence,
                    "subset_confidence_tolerance": tolerance,
                    "absolute_object_gates_pass": gates,
                    "worst_active_class_f1": worst,
                    "selection_score": score,
                    "object_macro_f1": metrics["object_macro_f1"],
                    "object_micro_f1": metrics["object_micro_f1"],
                    "exact_instance_ratio": metrics["exact_instance_ratio"],
                    "boundary_f1": metrics["boundary"]["f1"],
                    "prediction_count": metrics["prediction_count"],
                    "subset_dominated_count": subset_dominated
                })
    finally:
        evaluation._boundary_faces_from_objects = original_boundary
    best = max(scans, key=lambda row: (
        row["absolute_object_gates_pass"], row["worst_active_class_f1"], row["selection_score"],
        row["exact_instance_ratio"], -row["min_candidate_confidence"], -row["subset_confidence_tolerance"]
    ))
    result = {
        "schema_version": "topocae_p1_c_stage6_iteration2_global_candidate_decode_selection_1",
        "variant": args.variant,
        "seed": args.seed,
        "selection_split": "synthetic_val",
        "iteration_plan_sha256": PLAN_SHA256,
        "selected": best,
        "scan": scans,
        "provenance": {
            "iteration1_candidate_model_predictions_sha256": provenance["candidate_model_predictions_sha256"],
            "iteration1_candidate_package_manifest_sha256": provenance["candidate_package_manifest_sha256"]
        },
        "contracts": {"one_global_confidence": True, "one_global_decode_tolerance": True, "per_class_or_part_thresholds": False, "real_data_used": False, "holdout_used": False}
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["selection_sha256"] = sha256(canonical.encode("utf-8")).hexdigest().lower()
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    selected_thresholds = {
        **base_thresholds,
        "schema_version": "topocae_p1_c_stage6_iteration2_candidate_thresholds_1",
        "min_candidate_confidence": best["min_candidate_confidence"],
        "selection_split": "synthetic_val",
        "selection_sha256": result["selection_sha256"],
        "per_class_or_part_thresholds": False
    }
    (output / "candidate_thresholds_iteration2.json").write_text(json.dumps(selected_thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "selected": best, "selection_sha256": result["selection_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

