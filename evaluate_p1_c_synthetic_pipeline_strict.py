from __future__ import annotations

"""Strict P1-C synthetic evaluation with context and host stratification."""

import argparse
import json
import sys
from typing import Any

import evaluate_p1_b_synthetic_pipeline as implementation
from p1_c.strict_candidate_provenance_v2 import preflight_candidate_model
from utils.p1_b_global_decode_v2 import decode_feature_objects as decode_v2
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction as verify_v4
from utils.p1_c_strict_run_paths import StrictVariantSeedRunRoot, VARIANT_SPECS, strict_run_dir, training_metadata_path


_original_truth_objects = implementation._truth_objects
_original_gnn_prediction_data = implementation._gnn_prediction_data
_original_object_metrics = implementation._object_metrics
_face_macros_by_sample: dict[str, list[str]] = {}
_sample_metadata: dict[str, dict[str, str]] = {}
_selection: dict[str, Any] = {}


def _extract_wrapper_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def _truth_objects(graph: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    records = _original_truth_objects(graph, sample_id)
    sampling = graph.get("generation", {}).get("sampling_metadata", {})
    _sample_metadata[sample_id] = {
        "context": str(sampling.get("context", "unknown")),
        "host_family": str(sampling.get("host_family", "unknown")),
        "primary_target": str(sampling.get("primary_target", "unknown")),
    }
    for record in records:
        record["boundary_face_indices"] = sorted(implementation._boundary_faces(graph, set(record["face_indices"])))
    return records


def _boundary_faces_from_objects(record: dict[str, Any], sample_id: str) -> set[int]:
    return set(int(index) for index in record.get("boundary_face_indices", []))


def _gnn_prediction_data(record: dict[str, Any], macros: list[str]) -> dict[str, Any]:
    values = [macros[int(value)] for value in record["macro_prediction"]]
    _face_macros_by_sample[str(record["sample_id"])] = values
    return _original_gnn_prediction_data(record, macros)


def _verify(graph_data: dict[str, Any], proposal: dict[str, Any], model_prediction: dict[str, Any], *, thresholds: dict[str, Any]) -> dict[str, Any]:
    augmented = dict(model_prediction)
    augmented["face_macro_predictions"] = _face_macros_by_sample.get(str(graph_data.get("sample_id", "")), [])
    return verify_v4(graph_data, proposal, augmented, thresholds=thresholds)


def _decode(graph_data: dict[str, Any], verified_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return decode_v2(
        graph_data,
        verified_candidates,
        subset_confidence_tolerance=float(_selection["global_subset_confidence_tolerance"]),
        selection_sha256=str(_selection["selection_sha256"]),
    )


def _subset_metrics(truths: list[dict[str, Any]], predictions: list[dict[str, Any]], classes: list[str], sample_ids: set[str]) -> dict[str, Any]:
    return _original_object_metrics(
        [record for record in truths if str(record["sample_id"]) in sample_ids],
        [record for record in predictions if str(record["sample_id"]) in sample_ids],
        classes,
    )


def _object_metrics(truths: list[dict[str, Any]], predictions: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    result = _original_object_metrics(truths, predictions, classes)
    present = {str(record["sample_id"]) for record in truths}
    context: dict[str, Any] = {}
    for name in ("target_only", "pair", "triple"):
        ids = {sample_id for sample_id in present if _sample_metadata.get(sample_id, {}).get("context") == name}
        if ids:
            context[name] = _subset_metrics(truths, predictions, classes, ids)
    single_ids = {sample_id for sample_id in present if _sample_metadata.get(sample_id, {}).get("context") == "target_only"}
    multi_ids = present - single_ids
    context_group = {}
    if single_ids:
        context_group["single_context"] = _subset_metrics(truths, predictions, classes, single_ids)
    if multi_ids:
        context_group["multi_context"] = _subset_metrics(truths, predictions, classes, multi_ids)
    host_family = {}
    host_names = sorted({_sample_metadata.get(sample_id, {}).get("host_family", "unknown") for sample_id in present})
    for name in host_names:
        ids = {sample_id for sample_id in present if _sample_metadata.get(sample_id, {}).get("host_family", "unknown") == name}
        host_family[name] = _subset_metrics(truths, predictions, classes, ids)
    result["stratified"] = {"context": context, "context_group": context_group, "host_family": host_family}
    return result


def main() -> int:
    global _selection
    variant, seed = _extract_wrapper_args()
    preflight_candidate_model(variant, seed)
    run_dir = strict_run_dir(variant, seed)
    _selection = json.loads((run_dir / "global_decode_v2_selection.json").read_text(encoding="utf-8"))
    if _selection.get("selection_split") != "synthetic_val":
        raise RuntimeError("global decode selection is not Synthetic-Val-only")
    implementation.RUN_ROOT = StrictVariantSeedRunRoot(variant)
    implementation.TRAINING_METADATA = training_metadata_path(variant)
    implementation._truth_objects = _truth_objects
    implementation._boundary_faces_from_objects = _boundary_faces_from_objects
    implementation._gnn_prediction_data = _gnn_prediction_data
    implementation._object_metrics = _object_metrics
    implementation.verify_candidate_prediction = _verify
    implementation.decode_feature_objects = _decode
    if not bool(VARIANT_SPECS[variant]["canonical_patch"]):
        sys.argv.append("--no-canonical-patch")
    return int(implementation.main())


if __name__ == "__main__":
    raise SystemExit(main())

