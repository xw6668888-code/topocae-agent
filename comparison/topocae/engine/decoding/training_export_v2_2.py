from __future__ import annotations

from typing import Any

from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME
from utils import training_export_v2_1 as base


GRAPH_V2_NODE_FEATURE_SCHEMA = [*base.GRAPH_V2_NODE_FEATURE_SCHEMA, FEATURE_NAME]
GRAPH_V2_EDGE_FEATURE_SCHEMA = list(base.GRAPH_V2_EDGE_FEATURE_SCHEMA)
GRAPH_V2_NODE_FEATURE_MODE = "graph_v2_2_true_brep_polarity_v1"
GraphV2QuantileTransform = base.GraphV2QuantileTransform
MAX_MODEL_FEATURE_ABS = base.MAX_MODEL_FEATURE_ABS


def build_graph_v2_node_feature_vector(node: dict[str, Any], ndigits: int | None = None) -> list[float]:
    compatible = base.build_graph_v2_node_feature_vector(node, ndigits=None)
    polarity = float(node.get("attributes", {}).get("recognition_features_v2", {}).get(FEATURE_NAME, 0.0))
    values = [*compatible, polarity]
    if ndigits is not None:
        values = [round(value, ndigits) for value in values]
    return values


def build_graph_v2_edge_feature_vector(edge: dict[str, Any], ndigits: int | None = None) -> list[float]:
    return base.build_graph_v2_edge_feature_vector(edge, ndigits=ndigits)


def export_graph_v2_2_training_record(graph: dict[str, Any], *, ndigits: int = 8) -> dict[str, Any]:
    if graph.get("schema_version") != "topocae_graph_v2_2_true_brep_polarity_1":
        raise ValueError("Graph-v2.2 exporter requires the versioned true B-Rep graph schema")
    return {
        "schema_version": "topocae_graph_v2_2_training_record_1",
        "node_feature_mode": GRAPH_V2_NODE_FEATURE_MODE,
        "node_feature_schema": list(GRAPH_V2_NODE_FEATURE_SCHEMA),
        "edge_feature_schema": list(GRAPH_V2_EDGE_FEATURE_SCHEMA),
        "node_features": [build_graph_v2_node_feature_vector(node, ndigits) for node in graph.get("nodes", [])],
        "edge_features": [build_graph_v2_edge_feature_vector(edge, ndigits) for edge in graph.get("edge_attributes", [])],
        "edges": [list(edge) for edge in graph.get("edges", [])],
        "node_labels": [int(node.get("macro_feature_id", 0)) for node in graph.get("nodes", [])],
        "instance_ids": [int(node.get("instance_id", 0)) for node in graph.get("nodes", [])],
        "canonical_patches": graph.get("canonical_patches", []),
        "face_to_canonical_patch": graph.get("face_to_canonical_patch", {}),
    }


__all__ = [
    "GRAPH_V2_EDGE_FEATURE_SCHEMA",
    "GRAPH_V2_NODE_FEATURE_MODE",
    "GRAPH_V2_NODE_FEATURE_SCHEMA",
    "build_graph_v2_edge_feature_vector",
    "build_graph_v2_node_feature_vector",
    "export_graph_v2_2_training_record",
]

