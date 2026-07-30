"""Public P3-A1R pilot geometry API with narrow multi-instance repairs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import pilot_geometry_base as _base
from .pilot_geometry_base import (  # noqa: F401
    CANONICAL_FEATURES,
    COMPARISON_METRIC,
    COMPARISON_SEMANTICS,
    COMPLETE_DIMENSION_KEYS,
    FEATURE_FAMILY,
    build_graph_with_metric_evidence,
    metric_table,
)


def _normalize_parameters(feature_type: str, raw: dict[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_NORMALIZE(feature_type, raw)
    if raw.get("sweep_path_length") is not None:
        result["length"] = float(raw["sweep_path_length"])
        profile_width = float(raw["profile_width"])
        if feature_type == "slot_v":
            profile_depth = float(raw["profile_depth"])
            top_margin = float(raw["top_margin"])
            opening_width = profile_width * profile_depth / (profile_depth + top_margin)
        else:
            opening_width = profile_width
        result["opening_width"] = opening_width
        result["width"] = opening_width
        result["depth"] = float(raw["profile_depth"])
    return result


def _merge_rib_history_without_reassigning_shared_support_faces(*groups: Any) -> list[Any]:
    """Merge boolean-history tags while retaining an existing face owner.

    OpenCASCADE may report an already-labeled support/earlier-rib face in the
    new tool's Modified/Generated history.  The geometry is disjoint in X and
    every new rib must still retain its own exposed faces; later audits verify
    that exact instance set and face ownership.
    """

    result: list[Any] = []
    for group_index, group in enumerate(groups):
        for candidate in group:
            existing = next((tag for tag in result if candidate.face.IsSame(tag.face)), None)
            if existing is None:
                result.append(candidate)
                continue
            same_owner = (
                candidate.semantic_role == existing.semantic_role
                and candidate.macro_feature == existing.macro_feature
                and candidate.instance_id == existing.instance_id
            )
            if same_owner:
                continue
            if group_index == 0:
                raise ValueError("conflicting propagated rib owner")
            # Keep the propagated owner for a shared historical support face.
    return result


def _build_supported_multi_rib(instance_count: int, seed: int, *, tie: bool) -> Any:
    original_merge = _base.audited_core._merge_tags
    _base.audited_core._merge_tags = _merge_rib_history_without_reassigning_shared_support_faces
    try:
        result = _ORIGINAL_RIB_BUILDER(instance_count, seed, tie=tie)
    finally:
        _base.audited_core._merge_tags = original_merge
    parameters = deepcopy(result.parameters)
    parameters["rib_history_disambiguation"] = {
        "policy": "retain_propagated_owner_when_new_tool_history_reports_an_already_owned_face",
        "instance_completeness_revalidated": True,
    }
    return _base.audited_core.CleanFeatureResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


_ORIGINAL_NORMALIZE = _base._normalize_parameters
_ORIGINAL_RIB_BUILDER = _base._build_supported_multi_rib
_base._normalize_parameters = _normalize_parameters
_base._build_supported_multi_rib = _build_supported_multi_rib


def build_multi_instance_result(
    feature_type: str,
    instance_count: int,
    seed: int,
    *,
    tie: bool = False,
) -> Any:
    return _base.build_multi_instance_result(feature_type, instance_count, seed, tie=tie)


__all__ = [
    "CANONICAL_FEATURES",
    "COMPARISON_METRIC",
    "COMPARISON_SEMANTICS",
    "COMPLETE_DIMENSION_KEYS",
    "FEATURE_FAMILY",
    "build_graph_with_metric_evidence",
    "build_multi_instance_result",
    "metric_table",
]
