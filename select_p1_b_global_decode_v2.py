from __future__ import annotations

"""Select one global strict-subset near-tie tolerance on Synthetic-Val only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

import evaluate_p1_b_synthetic_pipeline as evaluation
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"
TRAINING_METADATA = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2" / "metadata.json"
TOLERANCES = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_dir = RUN_ROOT / f"same_instance_seed_{args.seed}"
    compact_macros = _read_json(TRAINING_METADATA)["macro_classes"]
    macro_classes = compact_macros[1:]
    gnn = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    candidates = torch.load(run_dir / "candidate_model_v1" / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        by_sample.setdefault(str(record["sample_id"]), []).append(record)
    thresholds = _read_json(run_dir / "candidate_model_v1" / "candidate_threshold_selection.json")
    verified_by_sample: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    truths = []
    for gnn_record in gnn:
        sample_id = str(gnn_record["sample_id"])
        graph = _read_json(Path(gnn_record["source_graph_path"]))
        raw_macros = [compact_macros[int(value)] for value in gnn_record["macro_prediction"]]
        sample_verified = []
        for candidate in by_sample.get(sample_id, []):
            sample_verified.append(verify_candidate_prediction(
                graph, candidate["proposal"], {
                    "macro_classes": macro_classes, "macro_logits": candidate["macro_logits"],
                    "macro_probabilities": candidate["macro_probabilities"], "validity_score": candidate["validity_score"],
                    "membership_probabilities": candidate["membership_probabilities"],
                    "boundary_probabilities": candidate["boundary_probabilities"],
                    "face_macro_predictions": raw_macros,
                }, thresholds=thresholds,
            ))
        verified_by_sample[sample_id] = (graph, sample_verified)
        for record in evaluation._truth_objects(graph, sample_id):
            record["boundary_face_indices"] = sorted(evaluation._boundary_faces(graph, set(record["face_indices"])))
            truths.append(record)
    original_boundary = evaluation._boundary_faces_from_objects
    evaluation._boundary_faces_from_objects = lambda record, sample_id: set(record.get("boundary_face_indices", []))
    scans = []
    placeholder_hash = "SYNTHETIC_VAL_SELECTION_IN_PROGRESS"
    try:
        for tolerance in TOLERANCES:
            objects = []
            subset_dominated_count = 0
            for sample_id, (graph, verified) in verified_by_sample.items():
                decoded = decode_feature_objects(
                    graph, verified, subset_confidence_tolerance=tolerance,
                    selection_sha256=placeholder_hash,
                )
                subset_dominated_count += int(decoded["subset_dominated_count"])
                objects.extend({**record, "sample_id": sample_id} for record in decoded["feature_objects"])
            metrics = evaluation._object_metrics(truths, objects, macro_classes)
            score = sum([
                metrics["object_macro_f1"], metrics["object_micro_f1"],
                metrics["exact_instance_ratio"], metrics["boundary"]["f1"],
            ]) / 4.0
            scans.append({
                "subset_confidence_tolerance": tolerance, "selection_score": score,
                "object_macro_f1": metrics["object_macro_f1"], "object_micro_f1": metrics["object_micro_f1"],
                "exact_instance_ratio": metrics["exact_instance_ratio"], "mean_best_instance_iou": metrics["mean_best_instance_iou"],
                "boundary_f1": metrics["boundary"]["f1"], "prediction_count": metrics["prediction_count"],
                "subset_dominated_count": subset_dominated_count,
            })
    finally:
        evaluation._boundary_faces_from_objects = original_boundary
    best = max(scans, key=lambda record: (
        record["selection_score"], record["exact_instance_ratio"],
        record["object_macro_f1"], -record["subset_confidence_tolerance"],
    ))
    result = {
        "schema_version": "topocae_p1_b_global_decode_v2_selection_1",
        "selection_split": "synthetic_val", "objective": "mean_object_macro_micro_exact_boundary",
        "global_subset_confidence_tolerance": best["subset_confidence_tolerance"],
        "best_metrics": {key: value for key, value in best.items() if key != "subset_confidence_tolerance"},
        "scan": scans,
        "contracts": {"single_global_tolerance": True, "per_class_or_part_tuning": False, "real_data_used": False, "holdout_used": False},
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["selection_sha256"] = sha256(canonical.encode("utf-8")).hexdigest().upper()
    destination = run_dir / "global_decode_v2_selection.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "selection_split": "synthetic_val", "selected_tolerance": result["global_subset_confidence_tolerance"],
        "best_metrics": result["best_metrics"], "selection_sha256": result["selection_sha256"],
        "output": str(destination),
    }, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
