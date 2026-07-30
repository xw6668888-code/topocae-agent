from __future__ import annotations

"""Synthetic evaluation with verifier-v4 face/candidate consensus."""

from typing import Any

import evaluate_p1_b_synthetic_pipeline as implementation
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction as verify_v4


_original_truth_objects = implementation._truth_objects
_original_gnn_prediction_data = implementation._gnn_prediction_data
_face_macros_by_sample: dict[str, list[str]] = {}


def _truth_objects(graph: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    records = _original_truth_objects(graph, sample_id)
    for record in records:
        record["boundary_face_indices"] = sorted(implementation._boundary_faces(graph, set(record["face_indices"])))
    return records


def _boundary_faces_from_objects(record: dict[str, Any], sample_id: str) -> set[int]:
    return set(int(index) for index in record.get("boundary_face_indices", []))


def _gnn_prediction_data(record: dict[str, Any], macros: list[str]) -> dict[str, Any]:
    values = [macros[int(value)] for value in record["macro_prediction"]]
    _face_macros_by_sample[str(record["sample_id"])] = values
    return _original_gnn_prediction_data(record, macros)


def _verify(
    graph_data: dict[str, Any], proposal: dict[str, Any], model_prediction: dict[str, Any],
    *, thresholds: dict[str, Any],
) -> dict[str, Any]:
    augmented = dict(model_prediction)
    augmented["face_macro_predictions"] = _face_macros_by_sample.get(str(graph_data.get("sample_id", "")), [])
    return verify_v4(graph_data, proposal, augmented, thresholds=thresholds)


implementation._truth_objects = _truth_objects
implementation._boundary_faces_from_objects = _boundary_faces_from_objects
implementation._gnn_prediction_data = _gnn_prediction_data
implementation.verify_candidate_prediction = _verify


if __name__ == "__main__":
    raise SystemExit(implementation.main())
