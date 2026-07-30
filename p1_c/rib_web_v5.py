"""Cleanup-corrected full-height web support revision."""

from __future__ import annotations

from copy import deepcopy
import random

from p1_c import rib_web_v3 as host_dispatch
from p1_c import rib_web_v4 as revision4


GENERATOR_REVISION = "p1_c_clean_rib_web_r5_full_height_web_support_cleanup_corrected"
RIB_HOST_FAMILIES = revision4.RIB_HOST_FAMILIES
WEB_HOST_FAMILIES = revision4.WEB_HOST_FAMILIES
VARIANTS = revision4.VARIANTS
SCENARIO_SPECS = revision4.SCENARIO_SPECS
SCENARIO_BY_KEY = revision4.SCENARIO_BY_KEY


def _web_buttressed_host(rng: random.Random):
    length, depth, base_t, support_width = host_dispatch._web_dimensions(rng)
    support_h = rng.uniform(29.0, 36.0)
    buttress = rng.uniform(4.0, 6.0)
    shoulder_z = base_t + support_h * rng.uniform(0.38, 0.48)
    left, right, inner_left, inner_right = host_dispatch._support_x(length, support_width)
    top_z = base_t + support_h
    # No collinear vertex is inserted at base_t on the outer boundaries. The
    # resulting host is one clean closed profile while the two inner interfaces
    # remain vertical over the complete web height.
    profile = [
        (left, 0.0),
        (right, 0.0),
        (right, shoulder_z),
        (right - buttress, top_z),
        (inner_right, top_z),
        (inner_right, base_t),
        (inner_left, base_t),
        (inner_left, top_z),
        (left + buttress, top_z),
        (left, shoulder_z),
    ]
    return revision4._finish_full_height_web_host(
        profile,
        length,
        depth,
        base_t,
        inner_left,
        inner_right,
        support_h,
        support_h,
        support_width,
        "single_closed_outer_buttressed_support_bridge_profile_prism_r2",
        extra={"outer_buttress_run": buttress, "outer_buttress_shoulder_z": shoulder_z},
    )


host_dispatch._web_buttressed_host = _web_buttressed_host


def scenario_keys() -> list[str]:
    return revision4.scenario_keys()


def build_rib_web_sample(scenario_key: str, seed: int):
    result = revision4.build_rib_web_sample(scenario_key, int(seed))
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["host_metadata"]["generator_revision"] = GENERATOR_REVISION
    parameters["concept_contract"] = {
        **deepcopy(parameters["concept_contract"]),
        "buttressed_profile_collinear_vertices_removed": True,
    }
    return revision4.CleanRibWebResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


CleanRibWebResult = revision4.CleanRibWebResult
FaceTag = revision4.FaceTag


__all__ = [
    "CleanRibWebResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "RIB_HOST_FAMILIES",
    "SCENARIO_SPECS",
    "VARIANTS",
    "WEB_HOST_FAMILIES",
    "build_rib_web_sample",
    "scenario_keys",
]
