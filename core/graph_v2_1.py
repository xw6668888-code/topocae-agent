from __future__ import annotations

from typing import Any

import core.graph_v2 as base


GRAPH_V2_SCHEMA = "topocae_graph_v2_relative_1_1"
DEGENERATE_EXTENT_TOLERANCE = 1e-7


def extract_graph_v2(
    shape: Any,
    *,
    face_labels: list[dict[str, Any]] | None = None,
    primitive_surface_index: dict[str, Any] | None = None,
    build_patches: bool = True,
) -> dict[str, Any]:
    graph = base.extract_graph_v2(
        shape,
        face_labels=face_labels,
        primitive_surface_index=primitive_surface_index,
        build_patches=build_patches,
    )
    for node in graph.get("nodes", []):
        recognition = node.get("attributes", {}).get("recognition_features_v2", {})
        short_extent = abs(float(recognition.get("pca_short_over_part_scale", 0.0)))
        if short_extent < DEGENERATE_EXTENT_TOLERANCE:
            recognition["pca_short_over_part_scale"] = 0.0
            recognition["local_extent_middle_short_ratio"] = 0.0
            recognition["log1p_pca_short_over_part_scale"] = 0.0
            recognition["log1p_local_extent_middle_short_ratio"] = 0.0
    graph["schema_version"] = GRAPH_V2_SCHEMA
    graph["graph_v2_protocol"]["degenerate_extent_tolerance"] = DEGENERATE_EXTENT_TOLERANCE
    base._assert_graph_v2_finite(graph)
    return graph


def build_canonical_patches(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return base.build_canonical_patches(*args, **kwargs)


def canonical_graph_signature(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return base.canonical_graph_signature(*args, **kwargs)


MAX_STABLE_RATIO = base.MAX_STABLE_RATIO


__all__ = [
    "DEGENERATE_EXTENT_TOLERANCE",
    "GRAPH_V2_SCHEMA",
    "MAX_STABLE_RATIO",
    "build_canonical_patches",
    "canonical_graph_signature",
    "extract_graph_v2",
]
