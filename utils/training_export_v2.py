from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


GRAPH_V2_NODE_FEATURE_MODE = "graph_v2_relative_v1"
MAX_MODEL_FEATURE_ABS = 32.0


GRAPH_V2_NODE_FEATURE_SCHEMA = [
    *(f"type_onehot_{index}" for index in range(11)),
    "semi_angle_over_pi",
    "log1p_face_area_over_part_total_area",
    "log1p_face_area_over_part_scale_sq",
    "log1p_radius_over_part_scale",
    "log1p_major_radius_over_part_scale",
    "log1p_minor_radius_over_part_scale",
    "log1p_pca_long_over_part_scale",
    "log1p_pca_middle_over_part_scale",
    "log1p_pca_short_over_part_scale",
    "log1p_projected_long_over_part_scale",
    "log1p_projected_short_over_part_scale",
    "log1p_uv_metric_long_over_part_scale",
    "log1p_uv_metric_short_over_part_scale",
    "log1p_face_area_over_neighbor_area_mean",
    "log1p_neighbor_area_mean_over_part_total_area",
    "log1p_local_extent_long_middle_ratio",
    "log1p_local_extent_middle_short_ratio",
    "log1p_projected_aspect_ratio",
    "log1p_projected_area_fill_ratio",
    "log1p_uv_aspect_ratio",
    "vertex_count",
    "edge_count",
    "neighbor_count",
    "wire_count",
    "outer_loop_count",
    "inner_loop_count",
    "outer_loop_edge_count",
    "inner_loop_edge_count",
]


GRAPH_V2_EDGE_FEATURE_SCHEMA = [
    "curve_type_id",
    "is_line",
    "is_circle",
    "is_ellipse",
    "is_spline",
    "convexity_id",
    "dihedral_angle_over_pi",
    "adjacent_plane_loop_role_id",
    "adjacent_plane_loop_role_inner",
    "adjacent_plane_loop_role_outer",
    "adjacent_plane_loop_role_unknown",
    "cylindrical_wall_role_hint_id",
    "cylindrical_wall_role_inner_hint",
    "cylindrical_wall_role_outer_hint",
    "cylindrical_wall_role_ambiguous_hint",
    "log1p_edge_length_over_part_scale",
    "log1p_circle_radius_over_part_scale",
    "material_side_inner_evidence",
    "material_side_outer_evidence",
    "material_side_ambiguous_evidence",
]


@dataclass(frozen=True)
class GraphV2QuantileTransform:
    schema_version: str
    fitted_split: str
    lower_quantile: float
    upper_quantile: float
    node_schema: list[str]
    edge_schema: list[str]
    node_lower: list[float]
    node_upper: list[float]
    edge_lower: list[float]
    edge_upper: list[float]
    max_abs: float = MAX_MODEL_FEATURE_ABS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphV2QuantileTransform":
        return cls(**payload)


def build_graph_v2_node_feature_vector(node: dict[str, Any], ndigits: int | None = None) -> list[float]:
    attrs = node.get("attributes", {})
    type_onehot = list(attrs.get("type_onehot", [0.0] * 11))[:11]
    type_onehot.extend([0.0] * (11 - len(type_onehot)))
    specific = attrs.get("specific", {})
    recognition = attrs.get("recognition_features_v2", {})
    values = [_finite(value) for value in type_onehot]
    values.append(_finite(specific.get("semi_angle", 0.0)) / math.pi)
    for key in GRAPH_V2_NODE_FEATURE_SCHEMA[12:]:
        values.append(_finite(recognition.get(key, 0.0)))
    return _rounded_bounded(values, ndigits)


def build_graph_v2_edge_feature_vector(edge_attr: dict[str, Any], ndigits: int | None = None) -> list[float]:
    recognition = edge_attr.get("recognition_features_v2", {})
    values = []
    for key in GRAPH_V2_EDGE_FEATURE_SCHEMA:
        if key in recognition:
            values.append(_finite(recognition[key]))
        else:
            values.append(_finite(edge_attr.get(key, 0.0)))
    return _rounded_bounded(values, ndigits)


def export_graph_v2_training_record(
    graph: dict[str, Any],
    *,
    ndigits: int = 8,
    quantile_transform: GraphV2QuantileTransform | None = None,
) -> dict[str, Any]:
    if graph.get("schema_version") != "topocae_graph_v2_relative_1":
        raise ValueError("Graph v2 training export requires topocae_graph_v2_relative_1 input")
    node_vectors = [build_graph_v2_node_feature_vector(node, ndigits) for node in graph.get("nodes", [])]
    edge_vectors = [build_graph_v2_edge_feature_vector(edge, ndigits) for edge in graph.get("edge_attributes", [])]
    if quantile_transform is not None:
        node_vectors = apply_quantile_transform(node_vectors, quantile_transform.node_lower, quantile_transform.node_upper, quantile_transform.max_abs)
        edge_vectors = apply_quantile_transform(edge_vectors, quantile_transform.edge_lower, quantile_transform.edge_upper, quantile_transform.max_abs)
    return {
        "schema_version": "topocae_graph_v2_training_record_1",
        "node_feature_mode": GRAPH_V2_NODE_FEATURE_MODE,
        "node_feature_schema": list(GRAPH_V2_NODE_FEATURE_SCHEMA),
        "edge_feature_schema": list(GRAPH_V2_EDGE_FEATURE_SCHEMA),
        "node_features": node_vectors,
        "edge_features": edge_vectors,
        "edges": [list(edge) for edge in graph.get("edges", [])],
        "node_labels": [int(node.get("macro_feature_id", 0)) for node in graph.get("nodes", [])],
        "instance_ids": [int(node.get("instance_id", 0)) for node in graph.get("nodes", [])],
        "canonical_patches": graph.get("canonical_patches", []),
        "face_to_canonical_patch": graph.get("face_to_canonical_patch", {}),
        "raw_physical_metadata": graph.get("part_metadata_v2", {}).get("raw_physical", {}),
        "quantile_transform_applied": quantile_transform.to_dict() if quantile_transform is not None else None,
    }


def fit_training_quantile_transform(
    node_vectors: Iterable[Iterable[float]],
    edge_vectors: Iterable[Iterable[float]],
    *,
    split_name: str,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    max_abs: float = MAX_MODEL_FEATURE_ABS,
) -> GraphV2QuantileTransform:
    if str(split_name).strip().lower() != "train":
        raise ValueError("Graph v2 quantile bounds may only be fitted on the training split")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("invalid quantile interval")
    node_rows = _validated_rows(node_vectors, len(GRAPH_V2_NODE_FEATURE_SCHEMA))
    edge_rows = _validated_rows(edge_vectors, len(GRAPH_V2_EDGE_FEATURE_SCHEMA))
    node_lower, node_upper = _column_quantile_bounds(node_rows, lower_quantile, upper_quantile, max_abs)
    edge_lower, edge_upper = _column_quantile_bounds(edge_rows, lower_quantile, upper_quantile, max_abs)
    return GraphV2QuantileTransform(
        schema_version="topocae_graph_v2_quantile_transform_1",
        fitted_split="train",
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        node_schema=list(GRAPH_V2_NODE_FEATURE_SCHEMA),
        edge_schema=list(GRAPH_V2_EDGE_FEATURE_SCHEMA),
        node_lower=node_lower,
        node_upper=node_upper,
        edge_lower=edge_lower,
        edge_upper=edge_upper,
        max_abs=float(max_abs),
    )


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
    return fit_training_quantile_transform(
        node_vectors,
        edge_vectors,
        split_name=split_name,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        max_abs=max_abs,
    )


def apply_quantile_transform(
    rows: Iterable[Iterable[float]],
    lower: list[float],
    upper: list[float],
    max_abs: float = MAX_MODEL_FEATURE_ABS,
) -> list[list[float]]:
    result = []
    for row in rows:
        values = list(row)
        if len(values) != len(lower) or len(values) != len(upper):
            raise ValueError("quantile bounds do not match feature vector width")
        transformed = []
        for value, low, high in zip(values, lower, upper):
            number = _finite(value)
            low_value = max(-max_abs, min(max_abs, _finite(low)))
            high_value = max(low_value, min(max_abs, _finite(high)))
            transformed.append(max(low_value, min(high_value, number)))
        result.append(transformed)
    return result


def _validated_rows(rows: Iterable[Iterable[float]], width: int) -> list[list[float]]:
    result = []
    for row in rows:
        values = [_finite(value) for value in row]
        if len(values) != width:
            raise ValueError(f"expected feature width {width}, got {len(values)}")
        result.append(values)
    return result


def _column_quantile_bounds(
    rows: list[list[float]],
    lower_quantile: float,
    upper_quantile: float,
    max_abs: float,
) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    lower, upper = [], []
    for column_index in range(len(rows[0])):
        column = sorted(max(-max_abs, min(max_abs, _finite(row[column_index]))) for row in rows)
        low = _quantile(column, lower_quantile)
        high = _quantile(column, upper_quantile)
        if high < low:
            low, high = high, low
        lower.append(low)
        upper.append(high)
    return lower, upper


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _rounded_bounded(values: Iterable[Any], ndigits: int | None) -> list[float]:
    result = []
    for value in values:
        number = max(-MAX_MODEL_FEATURE_ABS, min(MAX_MODEL_FEATURE_ABS, _finite(value)))
        result.append(round(number, ndigits) if ndigits is not None else number)
    return result


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


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
