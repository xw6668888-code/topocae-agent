from __future__ import annotations

"""Correct boundary-exact P1-B synthetic evaluation entry point."""

from typing import Any

import evaluate_p1_b_synthetic_pipeline as implementation


_original_truth_objects = implementation._truth_objects


def _truth_objects(graph: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    records = _original_truth_objects(graph, sample_id)
    for record in records:
        record["boundary_face_indices"] = sorted(implementation._boundary_faces(graph, set(record["face_indices"])))
    return records


def _boundary_faces_from_objects(record: dict[str, Any], sample_id: str) -> set[int]:
    return set(int(index) for index in record.get("boundary_face_indices", []))


implementation._truth_objects = _truth_objects
implementation._boundary_faces_from_objects = _boundary_faces_from_objects


if __name__ == "__main__":
    raise SystemExit(implementation.main())
