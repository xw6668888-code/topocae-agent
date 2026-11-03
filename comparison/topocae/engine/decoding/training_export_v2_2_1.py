from __future__ import annotations

from typing import Any

from core.graph_v2_2_1 import GRAPH_V2_SCHEMA
from utils.training_export_v2_2 import (
    GRAPH_V2_EDGE_FEATURE_SCHEMA,
    GRAPH_V2_NODE_FEATURE_MODE as PREVIOUS_NODE_FEATURE_MODE,
    GRAPH_V2_NODE_FEATURE_SCHEMA,
    GraphV2QuantileTransform,
    MAX_MODEL_FEATURE_ABS,
    build_graph_v2_edge_feature_vector,
    build_graph_v2_node_feature_vector,
)


GRAPH_V2_NODE_FEATURE_MODE = "graph_v2_2_1_exact_planar_rank_policy_v1"


def export_graph_v2_2_1_training_record(graph: dict[str, Any], *, ndigits: int = 8) -> dict[str, Any]:
    if graph.get("schema_version") != GRAPH_V2_SCHEMA:
        raise ValueError("Graph-v2.2.1 exporter requires the exact-planar-rank graph schema")
    return {
        "schema_version": "topocae_graph_v2_2_1_training_record_1",
        "node_feature_mode": GRAPH_V2_NODE_FEATURE_MODE,
        "previous_node_feature_mode": PREVIOUS_NODE_FEATURE_MODE,
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
    "GraphV2QuantileTransform",
    "MAX_MODEL_FEATURE_ABS",
    "build_graph_v2_edge_feature_vector",
    "build_graph_v2_node_feature_vector",
    "export_graph_v2_2_1_training_record",
]
