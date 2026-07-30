from __future__ import annotations

"""Final synthetic evaluation: verifier v4 plus selected global decode v2."""

import json
from pathlib import Path
import sys
from typing import Any

import evaluate_p1_b_synthetic_pipeline as implementation
from utils.p1_b_global_decode_v2 import decode_feature_objects as decode_v2
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction as verify_v4


_original_truth_objects = implementation._truth_objects
_original_gnn_prediction_data = implementation._gnn_prediction_data
_face_macros_by_sample: dict[str, list[str]] = {}


def _argument_value(name: str) -> str:
    position = sys.argv.index(name)
    return sys.argv[position + 1]


_seed = int(_argument_value("--seed"))
_root = Path(__file__).resolve().parent
_selection = json.loads((_root / "reports" / "p1_b_training_runs_v1" / f"same_instance_seed_{_seed}" / "global_decode_v2_selection.json").read_text(encoding="utf-8"))


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


def _decode(graph_data: dict[str, Any], verified_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return decode_v2(
        graph_data, verified_candidates,
        subset_confidence_tolerance=float(_selection["global_subset_confidence_tolerance"]),
        selection_sha256=str(_selection["selection_sha256"]),
    )


implementation._truth_objects = _truth_objects
implementation._boundary_faces_from_objects = _boundary_faces_from_objects
implementation._gnn_prediction_data = _gnn_prediction_data
implementation.verify_candidate_prediction = _verify
implementation.decode_feature_objects = _decode


if __name__ == "__main__":
    raise SystemExit(implementation.main())
