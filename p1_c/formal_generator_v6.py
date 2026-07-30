from __future__ import annotations

from copy import deepcopy
from typing import Any

from p1_c import all_features as feature_core
from p1_c import formal_generator as base_protocol
from p1_c import formal_generator_v5 as revision5
from p1_c.formal_plan_v5 import PLAN_REVISION, PlannedSample
from p1_c.rib_web_v3 import RIB_HOST_FAMILIES, WEB_HOST_FAMILIES
from p1_c.supported_wall_families_v2 import build_supported_wall_sample


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r6_strict_concept_hosts"
SUPPORTED_WALL_CLASSES = revision5.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision5.transformed_shape_and_audit
_ORIGINAL_SUPPORTED_FEATURE_CENTERS = base_protocol._supported_feature_centers


def _strict_supported_host_parameters(
    raw: feature_core.CleanFeatureResult,
    host_family: str,
) -> dict[str, Any]:
    source = raw.parameters["source_parameters"]
    metadata = deepcopy(source["host_metadata"])
    if host_family == "l_corner":
        length = float(metadata["length"])
        width = float(metadata["width"])
        top_z = float(metadata["base_thickness"])
        x_min = -length / 2.0 + float(metadata.get("coordinate_alignment_x", 0.0))
        x_max = x_min + length
    elif host_family == "t_junction":
        length = float(metadata["length"])
        width = float(metadata["width"])
        top_z = float(metadata["base_thickness"])
        x_min, x_max = -length / 2.0, length / 2.0
    elif host_family in WEB_HOST_FAMILIES:
        length = float(metadata["length"])
        width = float(metadata["depth"])
        top_z = float(metadata["base_thickness"])
        x_min, x_max = -length / 2.0, length / 2.0
    elif host_family in {"rect_cavity", "circular_cavity"}:
        length = float(metadata["length"])
        width = float(metadata["width"])
        top_z = float(metadata["total_height"])
        x_min, x_max = -length / 2.0, length / 2.0
    else:
        raise KeyError(host_family)
    return {
        **metadata,
        "family": host_family,
        "length": length,
        "width": width,
        "height": top_z,
        "top_z": top_z,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": -width / 2.0,
        "y_max": width / 2.0,
        "construction_method": source["construction_method"],
        "concept_host_mapping_revision": GENERATOR_REVISION,
    }


def _strict_supported_feature_centers(
    indexed_general: list[tuple[str, int]],
    host: dict[str, Any],
    host_family: str,
) -> dict[int, tuple[float, float]]:
    if host_family in WEB_HOST_FAMILIES:
        placement_family = "two_supports"
    elif host_family == "t_junction":
        placement_family = "l_corner"
    else:
        placement_family = host_family
    return _ORIGINAL_SUPPORTED_FEATURE_CENTERS(
        indexed_general,
        host,
        placement_family,
    )


base_protocol.build_supported_wall_sample = build_supported_wall_sample
base_protocol._supported_host_parameters = _strict_supported_host_parameters
base_protocol._supported_feature_centers = _strict_supported_feature_centers


def build_planned_result(plan: PlannedSample) -> feature_core.CleanFeatureResult:
    result = revision5.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["plan_revision"] = PLAN_REVISION
    parameters["strict_concept_host_contract"] = {
        "semantic_relabel_used": False,
        "rib_host_families": list(RIB_HOST_FAMILIES),
        "web_host_families": list(WEB_HOST_FAMILIES),
    }
    supported = parameters.get("supported_wall")
    if supported:
        source = supported["source_parameters"]
        macro = str(supported["macro_feature"])
        if source.get("source_engineering_macro") != macro:
            raise ValueError("formal supported-wall target differs from engineering source macro")
        allowed = RIB_HOST_FAMILIES if macro == "rib" else WEB_HOST_FAMILIES
        if plan.host_family not in allowed:
            raise ValueError(f"formal concept-incompatible host: {macro}/{plan.host_family}")
        if "semantic_relabel" in source:
            raise ValueError("legacy semantic relabel record leaked into strict formal sample")
    return feature_core.CleanFeatureResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
