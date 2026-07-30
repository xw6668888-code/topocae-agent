"""Strict rib/web revision with full-height web-to-support interfaces."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Pnt

from p1_c import rib_web_v3 as revision3
from p1_c.geometry import cleanup_audit, prism_xz


GENERATOR_REVISION = "p1_c_clean_rib_web_r4_full_height_web_support_interfaces"
RIB_HOST_FAMILIES = revision3.RIB_HOST_FAMILIES
WEB_HOST_FAMILIES = revision3.WEB_HOST_FAMILIES
VARIANTS = revision3.VARIANTS
SCENARIO_SPECS = revision3.SCENARIO_SPECS
SCENARIO_BY_KEY = revision3.SCENARIO_BY_KEY


def _web_tapered_host(rng: random.Random):
    length, depth, base_t, support_width = revision3._web_dimensions(rng)
    support_h = rng.uniform(29.0, 36.0)
    taper = rng.uniform(3.0, 5.0)
    left, right, inner_left, inner_right = revision3._support_x(length, support_width)
    top_z = base_t + support_h
    profile = [
        (left, 0.0),
        (right, 0.0),
        (right, base_t),
        (right - taper, top_z),
        (inner_right, top_z),
        (inner_right, base_t),
        (inner_left, base_t),
        (inner_left, top_z),
        (left + taper, top_z),
        (left, base_t),
    ]
    return _finish_full_height_web_host(
        profile,
        length,
        depth,
        base_t,
        inner_left,
        inner_right,
        support_h,
        support_h,
        support_width,
        "single_closed_outer_tapered_support_bridge_profile_prism",
        extra={"outer_support_taper": taper},
    )


def _web_buttressed_host(rng: random.Random):
    length, depth, base_t, support_width = revision3._web_dimensions(rng)
    support_h = rng.uniform(29.0, 36.0)
    buttress = rng.uniform(4.0, 6.0)
    shoulder_z = base_t + support_h * rng.uniform(0.38, 0.48)
    left, right, inner_left, inner_right = revision3._support_x(length, support_width)
    top_z = base_t + support_h
    profile = [
        (left, 0.0),
        (right, 0.0),
        (right, base_t),
        (right, shoulder_z),
        (right - buttress, top_z),
        (inner_right, top_z),
        (inner_right, base_t),
        (inner_left, base_t),
        (inner_left, top_z),
        (left + buttress, top_z),
        (left, shoulder_z),
        (left, base_t),
    ]
    return _finish_full_height_web_host(
        profile,
        length,
        depth,
        base_t,
        inner_left,
        inner_right,
        support_h,
        support_h,
        support_width,
        "single_closed_outer_buttressed_support_bridge_profile_prism",
        extra={"outer_buttress_run": buttress, "outer_buttress_shoulder_z": shoulder_z},
    )


def _finish_full_height_web_host(
    profile: list[tuple[float, float]],
    length: float,
    depth: float,
    base_t: float,
    inner_left: float,
    inner_right: float,
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
            "support_body_a_full_height_interface",
            BRepPrimAPI_MakeBox(
                gp_Pnt(inner_left - 0.8, -8.0, base_t - 0.5),
                1.2,
                16.0,
                minimum_h + 1.0,
            ).Shape(),
        ),
        (
            "support_body_b_full_height_interface",
            BRepPrimAPI_MakeBox(
                gp_Pnt(inner_right - 0.4, -8.0, base_t - 0.5),
                1.2,
                16.0,
                minimum_h + 1.0,
            ).Shape(),
        ),
        (
            "base_region",
            BRepPrimAPI_MakeBox(
                gp_Pnt(inner_left, -8.0, base_t - 0.5),
                inner_right - inner_left,
                16.0,
                1.0,
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
        "inner_interface_x_left": inner_left,
        "inner_interface_x_right": inner_right,
        "inner_support_interfaces_vertical": True,
        "inner_support_interfaces_full_height": True,
        "overlapping_support_solids": False,
        **dict(extra or {}),
    }
    span = inner_right - inner_left + 0.8
    return host, [], supports, metadata, span, base_t - 0.25


revision3._web_tapered_host = _web_tapered_host
revision3._web_buttressed_host = _web_buttressed_host


def scenario_keys() -> list[str]:
    return revision3.scenario_keys()


def build_rib_web_sample(scenario_key: str, seed: int):
    result = revision3.build_rib_web_sample(scenario_key, int(seed))
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["host_metadata"]["generator_revision"] = GENERATOR_REVISION
    parameters["source_engineering_macro"] = parameters["macro_feature"]
    end_faces = _exposed_feature_end_faces(result)
    parameters["exposed_feature_end_face_count"] = len(end_faces)
    parameters["full_height_two_support_connection"] = (
        parameters["macro_feature"] != "web" or len(end_faces) == 0
    )
    parameters["concept_contract"] = {
        **deepcopy(parameters["concept_contract"]),
        "source_engineering_macro": parameters["macro_feature"],
        "target_macro": parameters["macro_feature"],
        "web_exposed_end_faces_forbidden": True,
        "web_full_height_connection_required": True,
    }
    if parameters["macro_feature"] == "web" and end_faces:
        raise ValueError(
            f"web has exposed end faces and is not fully joined to both supports: "
            f"{scenario_key}, count={len(end_faces)}"
        )
    return revision3.CleanRibWebResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


def _exposed_feature_end_faces(result: Any) -> list[Any]:
    feature_instance = int(result.parameters["feature_instance_id"])
    end_faces: list[Any] = []
    for tag in result.face_tags:
        if int(tag.instance_id) != feature_instance:
            continue
        surface = BRepAdaptor_Surface(tag.face)
        if surface.GetType() != GeomAbs_Plane:
            continue
        normal = surface.Plane().Axis().Direction()
        if abs(float(normal.X())) >= 0.95:
            end_faces.append(tag.face)
    return end_faces


CleanRibWebResult = revision3.CleanRibWebResult
FaceTag = revision3.FaceTag


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
