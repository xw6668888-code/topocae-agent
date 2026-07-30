from __future__ import annotations

"""Select one global decode tolerance on Synthetic-Val for a strict P1-C run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

import evaluate_p1_b_synthetic_pipeline as evaluation
from p1_c.strict_candidate_provenance_v2 import preflight_candidate_model
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction
from utils.p1_c_strict_run_paths import candidate_model_name, strict_run_dir, training_metadata_path


TOLERANCES = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    provenance = preflight_candidate_model(args.variant, args.seed)
    run_dir = strict_run_dir(args.variant, args.seed)
    destination = run_dir / "global_decode_v2_selection.json"
    if destination.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "selection": str(destination)}, ensure_ascii=False))
        return 0
    compact_macros = _read_json(training_metadata_path(args.variant))["macro_classes"]
    macro_classes = compact_macros[1:]
    gnn = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    model_dir = run_dir / candidate_model_name(args.variant)
    candidates = torch.load(model_dir / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        by_sample.setdefault(str(record["sample_id"]), []).append(record)
    thresholds = _read_json(model_dir / "candidate_threshold_selection.json")
    verified_by_sample: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    truths = []
    for gnn_record in gnn:
        sample_id = str(gnn_record["sample_id"])
        graph = _read_json(Path(gnn_record["source_graph_path"]))
        raw_macros = [compact_macros[int(value)] for value in gnn_record["macro_prediction"]]
        sample_verified = []
        for candidate in by_sample.get(sample_id, []):
            sample_verified.append(verify_candidate_prediction(
                graph,
                candidate["proposal"],
                {
                    "macro_classes": macro_classes,
                    "macro_logits": candidate["macro_logits"],
                    "macro_probabilities": candidate["macro_probabilities"],
                    "validity_score": candidate["validity_score"],
                    "membership_probabilities": candidate["membership_probabilities"],
                    "boundary_probabilities": candidate["boundary_probabilities"],
                    "face_macro_predictions": raw_macros,
                },
                thresholds=thresholds,
            ))
        verified_by_sample[sample_id] = (graph, sample_verified)
        for record in evaluation._truth_objects(graph, sample_id):
            record["boundary_face_indices"] = sorted(evaluation._boundary_faces(graph, set(record["face_indices"])))
            truths.append(record)
    original_boundary = evaluation._boundary_faces_from_objects
    evaluation._boundary_faces_from_objects = lambda record, sample_id: set(record.get("boundary_face_indices", []))
    scans = []
    try:
        for tolerance in TOLERANCES:
            objects = []
            subset_dominated_count = 0
            for sample_id, (graph, verified) in verified_by_sample.items():
                decoded = decode_feature_objects(
                    graph,
                    verified,
                    subset_confidence_tolerance=tolerance,
                    selection_sha256="SYNTHETIC_VAL_SELECTION_IN_PROGRESS",
                )
                subset_dominated_count += int(decoded["subset_dominated_count"])
                objects.extend({**record, "sample_id": sample_id} for record in decoded["feature_objects"])
            metrics = evaluation._object_metrics(truths, objects, macro_classes)
            score = sum((
                metrics["object_macro_f1"],
                metrics["object_micro_f1"],
                metrics["exact_instance_ratio"],
                metrics["boundary"]["f1"],
            )) / 4.0
            scans.append({
                "subset_confidence_tolerance": tolerance,
                "selection_score": score,
                "object_macro_f1": metrics["object_macro_f1"],
                "object_micro_f1": metrics["object_micro_f1"],
                "exact_instance_ratio": metrics["exact_instance_ratio"],
                "mean_best_instance_iou": metrics["mean_best_instance_iou"],
                "boundary_f1": metrics["boundary"]["f1"],
                "prediction_count": metrics["prediction_count"],
                "subset_dominated_count": subset_dominated_count,
            })
    finally:
        evaluation._boundary_faces_from_objects = original_boundary
    best = max(scans, key=lambda record: (
        record["selection_score"],
        record["exact_instance_ratio"],
        record["object_macro_f1"],
        -record["subset_confidence_tolerance"],
    ))
    result = {
        "schema_version": "topocae_p1_c_strict_global_decode_v2_selection_1",
        "variant": args.variant,
        "seed": args.seed,
        "selection_split": "synthetic_val",
        "objective": "mean_object_macro_micro_exact_boundary",
        "global_subset_confidence_tolerance": best["subset_confidence_tolerance"],
        "best_metrics": {key: value for key, value in best.items() if key != "subset_confidence_tolerance"},
        "scan": scans,
        "provenance": {
            "candidate_plan_sha256": provenance["plan_sha256"],
            "candidate_model_predictions_sha256": provenance["candidate_model_predictions_sha256"],
        },
        "contracts": {
            "single_global_tolerance": True,
            "per_class_or_part_tuning": False,
            "real_data_used": False,
            "holdout_used": False,
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["selection_sha256"] = sha256(canonical.encode("utf-8")).hexdigest().lower()
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "selection_split": "synthetic_val",
        "selected_tolerance": result["global_subset_confidence_tolerance"],
        "best_metrics": result["best_metrics"],
        "selection_sha256": result["selection_sha256"],
        "output": str(destination),
    }, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

