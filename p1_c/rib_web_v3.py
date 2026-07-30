from __future__ import annotations

import random
from typing import Any

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

from p1_c import rib_web as core
from p1_c import rib_web_v2 as aligned_revision  # activates aligned L-corner host
from p1_c.geometry import cleanup_audit, prism_xz


GENERATOR_REVISION = "p1_c_clean_rib_web_r3_strict_concept_hosts"
RIB_HOST_FAMILIES = (
    "l_corner",
    "rect_cavity",
    "circular_cavity",
    "t_junction",
)
WEB_HOST_FAMILIES = (
    "two_supports_equal",
    "two_supports_asymmetric",
    "two_supports_tapered",
    "two_supports_buttressed",
)
VARIANTS = ("plain", "hole", "transition")
SCENARIO_SPECS = tuple(
    (f"rib_{host}_{variant}", "rib", host, variant)
    for host in RIB_HOST_FAMILIES
    for variant in VARIANTS
) + tuple(
    (f"web_{host}_{variant}", "web", host, variant)
    for host in WEB_HOST_FAMILIES
    for variant in VARIANTS
)
SCENARIO_BY_KEY = {record[0]: record for record in SCENARIO_SPECS}


_ORIGINAL_TWO_SUPPORT_HOST = core._two_support_host
_ORIGINAL_FEATURE_PROFILE = core._feature_profile
_ACTIVE_HOST_FAMILY: str | None = None


def scenario_keys() -> list[str]:
    return [record[0] for record in SCENARIO_SPECS]


def build_rib_web_sample(scenario_key: str, seed: int):
    if scenario_key not in SCENARIO_BY_KEY:
        raise KeyError(f"unknown strict P1-C rib/web scenario: {scenario_key}")
    global _ACTIVE_HOST_FAMILY
    previous = _ACTIVE_HOST_FAMILY
    _ACTIVE_HOST_FAMILY = SCENARIO_BY_KEY[scenario_key][2]
    core.SCENARIO_BY_KEY[scenario_key] = SCENARIO_BY_KEY[scenario_key]
    try:
        result = core.build_rib_web_sample(scenario_key, int(seed))
    finally:
        _ACTIVE_HOST_FAMILY = previous
    parameters = dict(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["concept_contract"] = {
        "target_macro_equals_source_engineering_macro": True,
        "semantic_relabel_used": False,
        "rib_host_families": list(RIB_HOST_FAMILIES),
        "web_host_families": list(WEB_HOST_FAMILIES),
        "host_family_concept_compatible": True,
    }
    return core.CleanRibWebResult(result.shape, result.face_tags, parameters, result.cleanup)


def _strict_host_dispatch(rng: random.Random):
    host = _ACTIVE_HOST_FAMILY
    if host == "t_junction":
        return _t_junction_host(rng)
    if host == "two_supports_equal":
        return _web_equal_host(rng)
    if host == "two_supports_asymmetric":
        return _web_asymmetric_host(rng)
    if host == "two_supports_tapered":
        return _web_tapered_host(rng)
    if host == "two_supports_buttressed":
        return _web_buttressed_host(rng)
    if host == "two_supports":
        return _ORIGINAL_TWO_SUPPORT_HOST(rng)
    raise KeyError(f"unsupported strict rib/web host dispatch: {host}")


def _strict_feature_profile(
    host_family: str,
    variant: str,
    span: float,
    z0: float,
    height: float,
) -> list[tuple[float, float]]:
    if host_family == "t_junction":
        left = -span * 0.5
        right = left + span
        if variant == "transition":
            return [
                (left, z0),
                (right, z0),
                (right * 0.72 + left * 0.28, z0 + height * 0.30),
                (left + span * 0.42, z0 + height * 0.72),
                (left, z0 + height),
            ]
        return [(left, z0), (right, z0), (left, z0 + height)]
    return _ORIGINAL_FEATURE_PROFILE(host_family, variant, span, z0, height)


def _t_junction_host(rng: random.Random):
    length = rng.uniform(100.0, 116.0)
    width = rng.uniform(72.0, 88.0)
    base_t = rng.uniform(5.4, 6.6)
    wall_t = rng.uniform(5.4, 6.6)
    wall_h = rng.uniform(31.0, 39.0)
    span = rng.uniform(29.0, 38.0)
    left, right = -length / 2.0, length / 2.0
    inner_wall = -span / 2.0
    outer_wall = inner_wall - wall_t
    profile = [
        (left, 0.0), (right, 0.0), (right, base_t),
        (inner_wall, base_t), (inner_wall, base_t + wall_h),
        (outer_wall, base_t + wall_h), (outer_wall, base_t),
        (left, base_t),
    ]
    host = prism_xz(profile, y_min=-width / 2.0, depth=width)
    supports = (
        (
            "base_region",
            BRepPrimAPI_MakeBox(
                gp_Pnt(inner_wall, -10.0, base_t - 0.5), span, 20.0, 1.0
            ).Shape(),
        ),
        (
            "central_vertical_wall_region",
            BRepPrimAPI_MakeBox(
                gp_Pnt(inner_wall - 0.8, -10.0, base_t), 1.2, 20.0, wall_h
            ).Shape(),
        ),
    )
    metadata = {
        "construction_method": "single_closed_offset_T_profile_prism",
        "host_cleanup": cleanup_audit(host),
        "height_limit": wall_h,
        "length": length,
        "width": width,
        "base_thickness": base_t,
        "wall_thickness": wall_t,
        "wall_height": wall_h,
        "overlapping_support_solids": False,
    }
    return host, [], supports, metadata, span, base_t - 0.25


def _web_equal_host(rng: random.Random):
    length, depth, base_t, support_width = _web_dimensions(rng)
    support_h = rng.uniform(28.0, 35.0)
    left, right, inner_left, inner_right = _support_x(length, support_width)
    profile = [
        (left, 0.0), (right, 0.0), (right, base_t + support_h),
        (inner_right, base_t + support_h), (inner_right, base_t),
        (inner_left, base_t), (inner_left, base_t + support_h),
        (left, base_t + support_h),
    ]
    return _finish_web_host(
        rng, profile, length, depth, base_t, inner_left, inner_right,
        support_h, support_h, support_width,
        "single_closed_equal_support_bridge_profile_prism",
    )


def _web_asymmetric_host(rng: random.Random):
    length, depth, base_t, support_width = _web_dimensions(rng)
    left_h = rng.uniform(27.0, 31.0)
    right_h = rng.uniform(33.0, 38.0)
    if rng.random() < 0.5:
        left_h, right_h = right_h, left_h
    left, right, inner_left, inner_right = _support_x(length, support_width)
    profile = [
        (left, 0.0), (right, 0.0), (right, base_t + right_h),
        (inner_right, base_t + right_h), (inner_right, base_t),
        (inner_left, base_t), (inner_left, base_t + left_h),
        (left, base_t + left_h),
    ]
    return _finish_web_host(
        rng, profile, length, depth, base_t, inner_left, inner_right,
        left_h, right_h, support_width,
        "single_closed_asymmetric_support_bridge_profile_prism",
    )


def _web_tapered_host(rng: random.Random):
    length, depth, base_t, support_width = _web_dimensions(rng)
    support_h = rng.uniform(29.0, 36.0)
    taper = rng.uniform(3.0, 5.0)
    left, right, inner_left, inner_right = _support_x(length, support_width)
    profile = [
        (left, 0.0), (right, 0.0), (right, base_t + support_h),
        (inner_right + taper, base_t + support_h), (inner_right, base_t),
        (inner_left, base_t), (inner_left - taper, base_t + support_h),
        (left, base_t + support_h),
    ]
    return _finish_web_host(
        rng, profile, length, depth, base_t, inner_left, inner_right,
        support_h, support_h, support_width,
        "single_closed_tapered_support_bridge_profile_prism",
        extra={"support_taper": taper},
    )


def _web_buttressed_host(rng: random.Random):
    length, depth, base_t, support_width = _web_dimensions(rng)
    support_h = rng.uniform(29.0, 36.0)
    buttress = rng.uniform(4.0, 6.0)
    shoulder_z = base_t + support_h * rng.uniform(0.38, 0.48)
    left, right, inner_left, inner_right = _support_x(length, support_width)
    gap_left, gap_right = inner_left + buttress, inner_right - buttress
    profile = [
        (left, 0.0), (right, 0.0), (right, base_t + support_h),
        (inner_right, base_t + support_h), (inner_right, shoulder_z),
        (gap_right, base_t), (gap_left, base_t),
        (inner_left, shoulder_z), (inner_left, base_t + support_h),
        (left, base_t + support_h),
    ]
    return _finish_web_host(
        rng, profile, length, depth, base_t, gap_left, gap_right,
        support_h, support_h, support_width,
        "single_closed_buttressed_support_bridge_profile_prism",
        extra={"buttress_run": buttress, "buttress_shoulder_z": shoulder_z},
    )


def _web_dimensions(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        rng.uniform(96.0, 110.0),
        rng.uniform(30.0, 37.0),
        rng.uniform(5.0, 6.2),
        rng.uniform(17.0, 20.5),
    )


def _support_x(length: float, support_width: float) -> tuple[float, float, float, float]:
    left, right = -length / 2.0, length / 2.0
    return left, right, left + support_width, right - support_width


def _finish_web_host(
    rng: random.Random,
    profile: list[tuple[float, float]],
    length: float,
    depth: float,
    base_t: float,
    gap_left: float,
    gap_right: float,
    left_h: float,
    right_h: float,
    support_width: float,
    construction_method: str,
    *,
    extra: dict[str, float] | None = None,
):
    host = prism_xz(profile, y_min=-depth / 2.0, depth=depth)
    minimum_h = min(left_h, right_h)
    supports = (
        (
            "support_body_a",
            BRepPrimAPI_MakeBox(
                gp_Pnt(gap_left - 0.8, -8.0, base_t), 1.2, 16.0, minimum_h
            ).Shape(),
        ),
        (
            "support_body_b",
            BRepPrimAPI_MakeBox(
                gp_Pnt(gap_right - 0.4, -8.0, base_t), 1.2, 16.0, minimum_h
            ).Shape(),
        ),
        (
            "base_region",
            BRepPrimAPI_MakeBox(
                gp_Pnt(gap_left, -8.0, base_t - 0.5), gap_right - gap_left, 16.0, 1.0
            ).Shape(),
        ),
    )
    metadata = {
        "construction_method": construction_method,
        "host_cleanup": cleanup_audit(host),
        "height_limit": minimum_h,
        "length": length,
        "depth": depth,
        "base_thickness": base_t,
        "support_height_left": left_h,
        "support_height_right": right_h,
        "support_width": support_width,
        "overlapping_support_solids": False,
        **dict(extra or {}),
    }
    span = gap_right - gap_left + 0.8
    return host, [], supports, metadata, span, base_t - 0.25


core._two_support_host = _strict_host_dispatch
core._feature_profile = _strict_feature_profile


CleanRibWebResult = core.CleanRibWebResult
FaceTag = core.FaceTag


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
