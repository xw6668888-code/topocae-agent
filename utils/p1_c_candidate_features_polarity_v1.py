from __future__ import annotations

"""Candidate representation extended by universal cylinder material polarity."""

from typing import Any, Mapping

import utils.p1_b_candidate_features as base
from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME


CANDIDATE_FEATURE_NAME = "mean_oriented_cylindrical_radial_alignment"
ENGINEERED_FEATURE_SCHEMA = list(base.ENGINEERED_FEATURE_SCHEMA) + [CANDIDATE_FEATURE_NAME]
MACRO_CLASSES = base.MACRO_CLASSES
PROPOSAL_SOURCES = base.PROPOSAL_SOURCES


def build_candidate_tensor_record(
    graph_data: Mapping[str, Any], prediction_data: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    record = base.build_candidate_tensor_record(graph_data, prediction_data, proposal)
    nodes = {
        int(node.get("idx", position)): node
        for position, node in enumerate(graph_data.get("nodes", []))
    }
    cylinder_values = []
    for index in record["face_indices"]:
        node = nodes.get(int(index), {})
        if node.get("primitive_surface") != "cylinder":
            continue
        recognition = node.get("attributes", {}).get("recognition_features_v2", {})
        if FEATURE_NAME not in recognition:
            raise RuntimeError(f"candidate cylinder node lacks {FEATURE_NAME}")
        cylinder_values.append(float(recognition[FEATURE_NAME]))
    value = sum(cylinder_values) / len(cylinder_values) if cylinder_values else 0.0
    if not -1.000001 <= value <= 1.000001:
        raise RuntimeError("candidate cylinder polarity mean is out of range")
    revised = dict(record)
    revised["engineered_feature_schema"] = list(ENGINEERED_FEATURE_SCHEMA)
    revised["engineered_features"] = list(record["engineered_features"]) + [value]
    engineered_values = dict(record.get("engineered_feature_values", {}))
    engineered_values[CANDIDATE_FEATURE_NAME] = value
    revised["engineered_feature_values"] = engineered_values
    revised["trace"] = {
        **dict(record.get("trace", {})),
        "oriented_cylinder_polarity_revision": "p1_c_oriented_cylinder_polarity_v1",
        "oriented_cylinder_face_count": len(cylinder_values),
    }
    return revised

