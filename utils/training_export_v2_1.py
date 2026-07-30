from __future__ import annotations

from typing import Any, Iterable

import utils.training_export_v2 as base


GRAPH_V2_NODE_FEATURE_MODE = base.GRAPH_V2_NODE_FEATURE_MODE
GRAPH_V2_NODE_FEATURE_SCHEMA = base.GRAPH_V2_NODE_FEATURE_SCHEMA
GRAPH_V2_EDGE_FEATURE_SCHEMA = base.GRAPH_V2_EDGE_FEATURE_SCHEMA
GraphV2QuantileTransform = base.GraphV2QuantileTransform
MAX_MODEL_FEATURE_ABS = base.MAX_MODEL_FEATURE_ABS
build_graph_v2_node_feature_vector = base.build_graph_v2_node_feature_vector
build_graph_v2_edge_feature_vector = base.build_graph_v2_edge_feature_vector
fit_training_quantile_transform = base.fit_training_quantile_transform
apply_quantile_transform = base.apply_quantile_transform


def export_graph_v2_training_record(
    graph: dict[str, Any],
    *,
    ndigits: int = 8,
    quantile_transform: GraphV2QuantileTransform | None = None,
) -> dict[str, Any]:
    if graph.get("schema_version") != "topocae_graph_v2_relative_1_1":
        raise ValueError("Graph v2.1 training export requires topocae_graph_v2_relative_1_1 input")
    compatible = dict(graph)
    compatible["schema_version"] = "topocae_graph_v2_relative_1"
    record = base.export_graph_v2_training_record(
        compatible,
        ndigits=ndigits,
        quantile_transform=quantile_transform,
    )
    record["schema_version"] = "topocae_graph_v2_1_training_record_1"
    record["source_graph_schema"] = "topocae_graph_v2_relative_1_1"
    return record


def fit_training_quantile_transform_from_graphs(
    graphs: Iterable[dict[str, Any]],
    *,
    split_name: str,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    max_abs: float = MAX_MODEL_FEATURE_ABS,
) -> GraphV2QuantileTransform:
    node_vectors = []
    edge_vectors = []
    for graph in graphs:
        record = export_graph_v2_training_record(graph)
        node_vectors.extend(record["node_features"])
        edge_vectors.extend(record["edge_features"])
    return base.fit_training_quantile_transform(
        node_vectors,
        edge_vectors,
        split_name=split_name,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        max_abs=max_abs,
    )


__all__ = [
    "GRAPH_V2_EDGE_FEATURE_SCHEMA",
    "GRAPH_V2_NODE_FEATURE_MODE",
    "GRAPH_V2_NODE_FEATURE_SCHEMA",
    "GraphV2QuantileTransform",
    "MAX_MODEL_FEATURE_ABS",
    "apply_quantile_transform",
    "build_graph_v2_edge_feature_vector",
    "build_graph_v2_node_feature_vector",
    "export_graph_v2_training_record",
    "fit_training_quantile_transform",
    "fit_training_quantile_transform_from_graphs",
]
