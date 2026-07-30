from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.geometry import (
    clean_cut,
    clean_fuse,
    cleanup_audit,
    common_volume,
    face_center,
    operation_faces,
    prism_xz,
    propagated_faces,
)


GENERATOR_REVISION = "p1_c_clean_rib_web_r1"
SCENARIO_SPECS = (
    ("rib_l_corner_plain", "rib", "l_corner", "plain"),
    ("rib_l_corner_hole", "rib", "l_corner", "hole"),
    ("rib_l_corner_transition", "rib", "l_corner", "transition"),
    ("rib_rect_cavity_plain", "rib", "rect_cavity", "plain"),
    ("rib_rect_cavity_hole", "rib", "rect_cavity", "hole"),
    ("rib_rect_cavity_transition", "rib", "rect_cavity", "transition"),
    ("rib_circular_cavity_plain", "rib", "circular_cavity", "plain"),
    ("rib_circular_cavity_hole", "rib", "circular_cavity", "hole"),
    ("rib_circular_cavity_transition", "rib", "circular_cavity", "transition"),
    ("web_two_supports_plain", "web", "two_supports", "plain"),
    ("web_two_supports_hole", "web", "two_supports", "hole"),
    ("web_two_supports_transition", "web", "two_supports", "transition"),
)
SCENARIO_BY_KEY = {item[0]: item for item in SCENARIO_SPECS}


@dataclass(frozen=True)
class FaceTag:
    face: Any
    semantic_role: str
    macro_feature: str
    instance_id: int


@dataclass(frozen=True)
class CleanRibWebResult:
    shape: Any
    face_tags: tuple[FaceTag, ...]
    parameters: dict[str, Any]
    cleanup: dict[str, Any]


def scenario_keys() -> list[str]:
    return [item[0] for item in SCENARIO_SPECS]


def build_rib_web_sample(scenario_key: str, seed: int) -> CleanRibWebResult:
    if scenario_key not in SCENARIO_BY_KEY:
        raise KeyError(f"unknown P1-C rib/web scenario: {scenario_key}")
    _, macro_feature, host_family, variant = SCENARIO_BY_KEY[scenario_key]
    rng = random.Random(int(seed))
    if host_family == "l_corner":
        host, host_tags, support_regions, host_metadata, span_limit, z0 = _l_corner_host(rng)
    elif host_family in {"rect_cavity", "circular_cavity"}:
        host, host_tags, support_regions, host_metadata, span_limit, z0 = _cavity_host(host_family, rng)
    else:
        host, host_tags, support_regions, host_metadata, span_limit, z0 = _two_support_host(rng)

    thickness = rng.uniform(3.4, 4.4)
    height_limit = float(host_metadata["height_limit"])
    height = rng.uniform(height_limit * 0.62, height_limit * 0.78)
    span = float(span_limit)
    profile = _feature_profile(host_family, variant, span, z0, height)
    tool = prism_xz(profile, y_min=-thickness / 2.0, depth=thickness)
    contacts = [
        {"name": name, "intersection_volume": common_volume(tool, region)}
        for name, region in support_regions
    ]
    operation, shape = clean_fuse(host, tool)
    tags = _propagate_tags(operation, host_tags)
    feature_instance = 2 if host_tags else 1
    tags = _merge_tags(
        tags,
        _tags_from_tool(
            operation,
            tool,
            role=lambda face: _thin_wall_role(face, macro_feature),
            macro_feature=macro_feature,
            instance_id=feature_instance,
        ),
    )

    hole_parameters: dict[str, Any] | None = None
    if variant == "hole":
        radius = min(height * rng.uniform(0.13, 0.16), span * 0.065)
        hole_center_z = z0 + height * rng.uniform(0.45, 0.55)
        hole_tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0.0, -thickness / 2.0 - 2.0, hole_center_z), gp_Dir(0.0, 1.0, 0.0)),
            radius,
            thickness + 4.0,
        ).Shape()
        hole_operation, shape = clean_cut(shape, hole_tool)
        tags = _propagate_tags(hole_operation, tags)
        hole_instance = feature_instance + 1
        tags = _merge_tags(
            tags,
            _tags_from_tool(
                hole_operation,
                hole_tool,
                role=lambda _face: "through_hole_cylinder",
                macro_feature="through_hole_round",
                instance_id=hole_instance,
            ),
        )
        hole_parameters = {
            "macro_feature": "through_hole_round",
            "instance_id": hole_instance,
            "radius": radius,
            "axis": [0.0, 1.0, 0.0],
            "passes_only_through_thin_wall": True,
        }

    tags = _filter_final_tags(shape, tags)
    cleanup = cleanup_audit(shape)
    parameters = {
        "schema_version": "topocae_p1_c_clean_rib_web_parameters_1",
        "generator_revision": GENERATOR_REVISION,
        "scenario_key": scenario_key,
        "seed": int(seed),
        "macro_feature": macro_feature,
        "host_family": host_family,
        "variant": variant,
        "construction_method": host_metadata["construction_method"],
        "host_cleanup_before_feature": host_metadata["host_cleanup"],
        "feature_instance_id": feature_instance,
        "span": span,
        "thickness": thickness,
        "height": height,
        "span_thickness_ratio": span / thickness,
        "height_thickness_ratio": height / thickness,
        "support_contacts": contacts,
        "positive_support_contact_count": sum(item["intersection_volume"] > 1e-7 for item in contacts),
        "has_hole": hole_parameters is not None,
        "hole_parameters": hole_parameters,
        "transition_kind": "continuous_profile_haunch" if variant == "transition" else "none",
        "feature_face_count": sum(tag.instance_id == feature_instance for tag in tags),
        "pocket_face_count": sum(tag.macro_feature in {"pocket_rect", "pocket_circular"} for tag in tags),
        "hole_face_count": sum(tag.macro_feature == "through_hole_round" for tag in tags),
        "tagged_face_count": len(tags),
        "host_metadata": {key: value for key, value in host_metadata.items() if key not in {"host_cleanup", "height_limit"}},
    }
    result = CleanRibWebResult(shape=shape, face_tags=tuple(tags), parameters=parameters, cleanup=cleanup)
    _validate_result(result)
    return result


def _l_corner_host(rng: random.Random) -> tuple[Any, list[FaceTag], tuple[tuple[str, Any], ...], dict[str, Any], float, float]:
    length = rng.uniform(98.0, 114.0)
    width = rng.uniform(72.0, 88.0)
    base_t = rng.uniform(5.4, 6.6)
    wall_t = rng.uniform(5.4, 6.6)
    wall_h = rng.uniform(31.0, 39.0)
    left = -length / 2.0
    right = length / 2.0
    inner = left + wall_t
    profile = [
        (left, 0.0),
        (right, 0.0),
        (right, base_t),
        (inner, base_t),
        (inner, base_t + wall_h),
        (left, base_t + wall_h),
    ]
    host = prism_xz(profile, y_min=-width / 2.0, depth=width)
    host_cleanup = cleanup_audit(host)
    support_regions = (
        (
            "base_region",
            BRepPrimAPI_MakeBox(gp_Pnt(inner - 1.0, -10.0, base_t - 0.5), length * 0.45, 20.0, 1.0).Shape(),
        ),
        (
            "vertical_wall_region",
            BRepPrimAPI_MakeBox(gp_Pnt(inner - 0.8, -10.0, base_t), 1.2, 20.0, wall_h).Shape(),
        ),
    )
    metadata = {
        "construction_method": "single_closed_L_profile_prism",
        "host_cleanup": host_cleanup,
        "height_limit": wall_h,
        "length": length,
        "width": width,
        "base_thickness": base_t,
        "wall_thickness": wall_t,
        "wall_height": wall_h,
        "overlapping_support_solids": False,
    }
    span = rng.uniform(max(28.0, wall_h * 0.82), min(38.0, length * 0.36))
    return host, [], support_regions, metadata, span, base_t - 0.25


def _cavity_host(host_family: str, rng: random.Random) -> tuple[Any, list[FaceTag], tuple[tuple[str, Any], ...], dict[str, Any], float, float]:
    length = rng.uniform(94.0, 108.0)
    width = rng.uniform(74.0, 86.0)
    total_h = rng.uniform(23.0, 28.0)
    depth = rng.uniform(13.5, 17.0)
    bottom_z = total_h - depth
    stock = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, total_h).Shape()
    if host_family == "rect_cavity":
        cavity_x = rng.uniform(48.0, 56.0)
        cavity_y = rng.uniform(34.0, 42.0)
        tool = BRepPrimAPI_MakeBox(
            gp_Pnt(-cavity_x / 2.0, -cavity_y / 2.0, bottom_z), cavity_x, cavity_y, depth + 2.0
        ).Shape()
        span_half = cavity_x / 2.0
        pocket_macro = "pocket_rect"
        side_role = "rect_pocket_side_plane"
        bottom_role = "rect_pocket_bottom_plane"
        cavity_metadata: dict[str, Any] = {"cavity_x": cavity_x, "cavity_y": cavity_y}
    else:
        radius = rng.uniform(21.0, 24.0)
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0.0, 0.0, bottom_z), gp_Dir(0.0, 0.0, 1.0)), radius, depth + 2.0
        ).Shape()
        span_half = radius
        pocket_macro = "pocket_circular"
        side_role = "circular_pocket_side_cylinder"
        bottom_role = "circular_pocket_bottom_plane"
        cavity_metadata = {"cavity_radius": radius}
    operation, host = clean_cut(stock, tool)
    host_tags = _tags_from_tool(
        operation,
        tool,
        role=lambda face: _pocket_role(face, bottom_z, side_role, bottom_role),
        macro_feature=pocket_macro,
        instance_id=1,
    )
    host_cleanup = cleanup_audit(host)
    wall_width = 1.2
    support_regions = (
        (
            "opposed_cavity_wall_a",
            BRepPrimAPI_MakeBox(
                gp_Pnt(-span_half - 0.8, -8.0, bottom_z), wall_width, 16.0, depth + 0.5
            ).Shape(),
        ),
        (
            "opposed_cavity_wall_b",
            BRepPrimAPI_MakeBox(
                gp_Pnt(span_half - 0.4, -8.0, bottom_z), wall_width, 16.0, depth + 0.5
            ).Shape(),
        ),
        (
            "cavity_floor",
            BRepPrimAPI_MakeBox(
                gp_Pnt(-span_half, -8.0, bottom_z - 0.5), 2.0 * span_half, 16.0, 1.0
            ).Shape(),
        ),
    )
    metadata = {
        "construction_method": "single_block_single_cavity_cut",
        "host_cleanup": host_cleanup,
        "height_limit": depth,
        "length": length,
        "width": width,
        "total_height": total_h,
        "cavity_depth": depth,
        "cavity_macro_feature": pocket_macro,
        **cavity_metadata,
    }
    span = 2.0 * span_half + 0.8
    return host, host_tags, support_regions, metadata, span, bottom_z - 0.25


def _two_support_host(rng: random.Random) -> tuple[Any, list[FaceTag], tuple[tuple[str, Any], ...], dict[str, Any], float, float]:
    length = rng.uniform(96.0, 108.0)
    depth = rng.uniform(30.0, 36.0)
    base_t = rng.uniform(5.0, 6.2)
    support_h = rng.uniform(28.0, 35.0)
    support_width = rng.uniform(17.0, 20.0)
    left = -length / 2.0
    right = length / 2.0
    inner_left = left + support_width
    inner_right = right - support_width
    profile = [
        (left, 0.0),
        (right, 0.0),
        (right, base_t + support_h),
        (inner_right, base_t + support_h),
        (inner_right, base_t),
        (inner_left, base_t),
        (inner_left, base_t + support_h),
        (left, base_t + support_h),
    ]
    host = prism_xz(profile, y_min=-depth / 2.0, depth=depth)
    host_cleanup = cleanup_audit(host)
    support_regions = (
        (
            "support_body_a",
            BRepPrimAPI_MakeBox(gp_Pnt(inner_left - 0.8, -8.0, base_t), 1.2, 16.0, support_h).Shape(),
        ),
        (
            "support_body_b",
            BRepPrimAPI_MakeBox(gp_Pnt(inner_right - 0.4, -8.0, base_t), 1.2, 16.0, support_h).Shape(),
        ),
        (
            "base_region",
            BRepPrimAPI_MakeBox(gp_Pnt(inner_left, -8.0, base_t - 0.5), inner_right - inner_left, 16.0, 1.0).Shape(),
        ),
    )
    metadata = {
        "construction_method": "single_closed_stepped_bridge_profile_prism",
        "host_cleanup": host_cleanup,
        "height_limit": support_h,
        "length": length,
        "depth": depth,
        "base_thickness": base_t,
        "support_height": support_h,
        "support_width": support_width,
        "overlapping_support_solids": False,
    }
    span = inner_right - inner_left + 0.8
    return host, [], support_regions, metadata, span, base_t - 0.25


def _feature_profile(host_family: str, variant: str, span: float, z0: float, height: float) -> list[tuple[float, float]]:
    if host_family == "l_corner":
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
    half = span / 2.0
    if variant == "transition":
        return [
            (-half, z0),
            (half, z0),
            (half, z0 + height * 0.42),
            (half * 0.74, z0 + height),
            (-half * 0.74, z0 + height),
            (-half, z0 + height * 0.42),
        ]
    return [(-half, z0), (half, z0), (half, z0 + height), (-half, z0 + height)]


def _tags_from_tool(
    operation: Any,
    tool: Any,
    *,
    role: Callable[[Any], str],
    macro_feature: str,
    instance_id: int,
) -> list[FaceTag]:
    tags: list[FaceTag] = []
    for tool_face in TopologyExplorer(tool).faces():
        semantic_role = role(tool_face)
        for mapped in operation_faces(operation, tool_face):
            _append_tag(tags, FaceTag(mapped, semantic_role, macro_feature, int(instance_id)))
    return tags


def _propagate_tags(operation: Any, tags: Iterable[FaceTag]) -> list[FaceTag]:
    result: list[FaceTag] = []
    for tag in tags:
        for mapped in propagated_faces(operation, tag.face):
            _append_tag(result, FaceTag(mapped, tag.semantic_role, tag.macro_feature, tag.instance_id))
    return result


def _merge_tags(*groups: Iterable[FaceTag]) -> list[FaceTag]:
    result: list[FaceTag] = []
    for group in groups:
        for tag in group:
            _append_tag(result, tag)
    return result


def _filter_final_tags(shape: Any, tags: Iterable[FaceTag]) -> list[FaceTag]:
    final_faces = list(TopologyExplorer(shape).faces())
    return [tag for tag in tags if any(tag.face.IsSame(face) for face in final_faces)]


def _append_tag(tags: list[FaceTag], candidate: FaceTag) -> None:
    for existing in tags:
        if candidate.face.IsSame(existing.face):
            if (
                candidate.semantic_role != existing.semantic_role
                or candidate.macro_feature != existing.macro_feature
                or candidate.instance_id != existing.instance_id
            ):
                raise ValueError("one final face received conflicting P1-C labels")
            return
    tags.append(candidate)


def _thin_wall_role(face: Any, macro_feature: str) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Y()) >= 0.85:
            return f"{macro_feature}_side_plane"
    return f"{macro_feature}_top_plane"


def _pocket_role(face: Any, bottom_z: float, side_role: str, bottom_role: str) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and abs(face_center(face)[2] - bottom_z) < 0.5:
            return bottom_role
    return side_role


def _validate_result(result: CleanRibWebResult) -> None:
    params = result.parameters
    if not result.cleanup["passed"]:
        raise ValueError(f"final rib/web topology is not cleanup-idempotent: {result.cleanup}")
    if not params["host_cleanup_before_feature"]["passed"]:
        raise ValueError("host is not clean before feature construction")
    if params["positive_support_contact_count"] < 2:
        raise ValueError(f"thin wall does not contact two support regions: {params['support_contacts']}")
    if params["span_thickness_ratio"] < 5.0 or params["height_thickness_ratio"] < 2.0:
        raise ValueError("thin-wall proportions are not reinforcement-like")
    feature_tags = [tag for tag in result.face_tags if tag.instance_id == params["feature_instance_id"]]
    if len(feature_tags) < 3:
        raise ValueError("rib/web exposed feature faces were not preserved")
    if sum(tag.semantic_role == f"{params['macro_feature']}_side_plane" for tag in feature_tags) < 2:
        raise ValueError("rib/web must preserve two large side faces")
    if params["host_family"] in {"rect_cavity", "circular_cavity"} and params["pocket_face_count"] < 2:
        raise ValueError("cavity truth was not preserved as an independent instance")
    if params["has_hole"] and params["hole_face_count"] < 1:
        raise ValueError("hole truth was not preserved as an independent instance")


__all__ = [
    "CleanRibWebResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "SCENARIO_SPECS",
    "build_rib_web_sample",
    "scenario_keys",
]
