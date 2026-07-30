from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer


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

FEATURE_ROLE_BY_MACRO = {
    "rib": ("rib_side_plane", "rib_top_plane"),
    "web": ("web_side_plane", "web_top_plane"),
}
POCKET_ROLE_BY_CONTEXT = {
    "rect_cavity": ("pocket_rect", "rect_pocket_side_plane", "rect_pocket_bottom_plane"),
    "circular_cavity": ("pocket_circular", "circular_pocket_side_cylinder", "circular_pocket_bottom_plane"),
}


@dataclass(frozen=True)
class FaceTag:
    face: Any
    semantic_role: str
    macro_key: str
    instance_id: int


@dataclass(frozen=True)
class P1BRibWebResult:
    shape: Any
    parameters: dict[str, Any]
    face_tags: tuple[FaceTag, ...]


def scenario_keys() -> list[str]:
    return [item[0] for item in SCENARIO_SPECS]


def build_rib_web_sample(scenario_key: str, seed: int) -> P1BRibWebResult:
    if scenario_key not in SCENARIO_BY_KEY:
        raise KeyError(f"unknown P1-B rib/web scenario: {scenario_key}")
    _, macro_key, context_kind, variant = SCENARIO_BY_KEY[scenario_key]
    rng = random.Random(int(seed))
    if context_kind == "l_corner":
        result = _build_l_corner_rib(scenario_key, variant, rng, int(seed))
    elif context_kind in {"rect_cavity", "circular_cavity"}:
        result = _build_cavity_rib(scenario_key, context_kind, variant, rng, int(seed))
    else:
        result = _build_two_support_web(scenario_key, variant, rng, int(seed))
    _validate_result(result)
    return result


def _build_l_corner_rib(scenario_key: str, variant: str, rng: random.Random, seed: int) -> P1BRibWebResult:
    length = rng.uniform(98.0, 114.0)
    width = rng.uniform(72.0, 88.0)
    base_t = rng.uniform(5.2, 6.8)
    wall_t = rng.uniform(5.2, 6.8)
    wall_h = rng.uniform(31.0, 39.0)
    base = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, base_t).Shape()
    wall = BRepPrimAPI_MakeBox(
        gp_Pnt(-length / 2.0, -width / 2.0, base_t - 0.5), wall_t, width, wall_h + 0.5
    ).Shape()
    stock = _fuse_shapes(base, wall)
    inner_x = -length / 2.0 + wall_t
    anchor_y = rng.uniform(-width * 0.16, width * 0.16)
    thickness = rng.uniform(3.6, 4.8)
    span = rng.uniform(max(25.0, wall_h * 0.70), min(36.0, length * 0.34))
    height = rng.uniform(wall_h * 0.62, wall_h * 0.82)
    z0 = base_t - 0.6
    if variant == "transition":
        profile = [
            (inner_x - 0.5, z0),
            (inner_x + span, z0),
            (inner_x + span * 0.78, z0 + height * 0.28),
            (inner_x + span * 0.48, z0 + height * 0.66),
            (inner_x - 0.5, z0 + height),
        ]
        transition_kind = "multi_slope_haunch"
    else:
        profile = [
            (inner_x - 0.5, z0),
            (inner_x + span, z0),
            (inner_x - 0.5, z0 + height),
        ]
        transition_kind = "single_slope_gusset"
    tool = _vertical_wall_tool(profile, thickness, anchor_y, axis="x")
    supports = (("base_plate", base), ("vertical_wall", wall))
    contact_records = _support_contact_records(tool, supports)
    fuse = BRepAlgoAPI_Fuse(stock, tool)
    fuse.Build()
    if not fuse.IsDone():
        raise ValueError("L-corner rib fuse failed")
    feature_inst = 1
    tags = _tag_tool_result_faces(
        fuse,
        tool,
        lambda face: _feature_face_role(face, "rib", z0),
        macro_key="rib",
        instance_id=feature_inst,
    )
    shape = fuse.Shape()
    hole_parameters = None
    if variant == "hole":
        center_s = span * 0.32
        center_z = z0 + height * 0.34
        radius = min(span, height) * rng.uniform(0.105, 0.125)
        hole = _horizontal_hole_tool(
            axis="x",
            center_s=inner_x + center_s,
            center_cross=anchor_y,
            center_z=center_z,
            thickness=thickness,
            radius=radius,
        )
        shape, tags, hole_parameters = _apply_feature_hole(shape, tags, hole, feature_inst + 1, radius, axis="y")
    params = _base_parameters(
        scenario_key=scenario_key,
        seed=seed,
        macro_key="rib",
        context_kind="l_corner",
        variant=variant,
        span=span,
        thickness=thickness,
        height=height,
        feature_inst=feature_inst,
        support_contacts=contact_records,
        transition_kind=transition_kind,
        hole_parameters=hole_parameters,
    )
    params.update({
        "stock_kind": "l_bracket_base_and_vertical_wall",
        "base_dimensions": [length, width, base_t],
        "wall_dimensions": [wall_t, width, wall_h],
        "anchor_y": anchor_y,
        "support_plate_count": 2,
        "connects_main_faces": True,
    })
    return _finish_result(shape, params, tags)


def _build_cavity_rib(
    scenario_key: str,
    context_kind: str,
    variant: str,
    rng: random.Random,
    seed: int,
) -> P1BRibWebResult:
    length, width, total_h = 100.0, 80.0, 25.0
    stock = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, total_h).Shape()
    depth = rng.uniform(14.0, 17.0)
    bottom_z = total_h - depth
    axis = rng.choice(("x", "y"))
    if context_kind == "rect_cavity":
        cavity_x = rng.uniform(48.0, 56.0)
        cavity_y = rng.uniform(34.0, 42.0)
        cavity_tool = BRepPrimAPI_MakeBox(
            gp_Pnt(-cavity_x / 2.0, -cavity_y / 2.0, bottom_z), cavity_x, cavity_y, depth + 2.0
        ).Shape()
        span_half = (cavity_x if axis == "x" else cavity_y) / 2.0
        cross_half = (cavity_y if axis == "x" else cavity_x) / 2.0
        cavity_dimensions: dict[str, Any] = {"x": cavity_x, "y": cavity_y, "depth": depth}
    else:
        radius = rng.uniform(21.0, 24.0)
        cavity_tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0.0, 0.0, bottom_z), gp_Dir(0.0, 0.0, 1.0)), radius, depth + 2.0
        ).Shape()
        span_half = radius
        cross_half = radius
        cavity_dimensions = {"radius": radius, "depth": depth}
    cavity_cut = BRepAlgoAPI_Cut(stock, cavity_tool)
    cavity_cut.Build()
    if not cavity_cut.IsDone():
        raise ValueError("cavity construction failed")
    host = cavity_cut.Shape()
    pocket_macro, pocket_side_role, pocket_bottom_role = POCKET_ROLE_BY_CONTEXT[context_kind]
    pocket_tags = _tag_tool_result_faces(
        cavity_cut,
        cavity_tool,
        lambda face: _pocket_face_role(face, bottom_z, pocket_side_role, pocket_bottom_role),
        macro_key=pocket_macro,
        instance_id=1,
    )
    thickness = rng.uniform(max(3.4, cross_half * 0.14), max(4.2, cross_half * 0.20))
    span = 2.0 * span_half + rng.uniform(1.8, 2.8)
    height = depth * rng.uniform(0.62, 0.78)
    z0 = bottom_z - 0.6
    profile = _symmetric_wall_profile(span, z0, height, variant == "transition")
    tool = _vertical_wall_tool(profile, thickness, 0.0, axis=axis)
    support_shapes = _cavity_support_shapes(
        host,
        context_kind=context_kind,
        axis=axis,
        span_half=span_half,
        cross_half=cross_half,
        bottom_z=bottom_z,
        total_h=total_h,
        thickness=thickness,
        length=length,
        width=width,
    )
    contact_records = _support_contact_records(tool, support_shapes)
    fuse = BRepAlgoAPI_Fuse(host, tool)
    fuse.Build()
    if not fuse.IsDone():
        raise ValueError("cavity rib fuse failed")
    feature_inst = 2
    propagated_pocket = _propagate_tags(fuse, pocket_tags)
    feature_tags = _tag_tool_result_faces(
        fuse,
        tool,
        lambda face: _feature_face_role(face, "rib", z0),
        macro_key="rib",
        instance_id=feature_inst,
    )
    tags = _merge_tags(propagated_pocket, feature_tags)
    shape = fuse.Shape()
    hole_parameters = None
    if variant == "hole":
        radius = min(height * rng.uniform(0.13, 0.16), span * 0.075)
        hole = _horizontal_hole_tool(
            axis=axis,
            center_s=0.0,
            center_cross=0.0,
            center_z=z0 + height * 0.40,
            thickness=thickness,
            radius=radius,
        )
        shape, tags, hole_parameters = _apply_feature_hole(shape, tags, hole, feature_inst + 1, radius, axis=_cross_axis(axis))
    params = _base_parameters(
        scenario_key=scenario_key,
        seed=seed,
        macro_key="rib",
        context_kind=context_kind,
        variant=variant,
        span=span,
        thickness=thickness,
        height=height,
        feature_inst=feature_inst,
        support_contacts=contact_records,
        transition_kind="paired_sloped_wall_transitions" if variant == "transition" else "none",
        hole_parameters=hole_parameters,
    )
    params.update({
        "stock_kind": "block_with_open_cavity",
        "cavity_dimensions": cavity_dimensions,
        "cavity_bottom_z": bottom_z,
        "span_axis": axis,
        "parent_macro_feature": pocket_macro,
        "parent_instance_id": 1,
        "cavity_and_rib_instances_independent": True,
        "connects_opposed_cavity_walls": True,
    })
    return _finish_result(shape, params, tags)


def _build_two_support_web(scenario_key: str, variant: str, rng: random.Random, seed: int) -> P1BRibWebResult:
    length, width = 100.0, 64.0
    base_t = rng.uniform(4.8, 6.0)
    support_width = rng.uniform(17.0, 20.0)
    support_cross = rng.uniform(28.0, 34.0)
    support_h = rng.uniform(29.0, 35.0)
    axis = rng.choice(("x", "y"))
    base = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, base_t).Shape()
    center_offset = rng.uniform(35.0, 38.0)
    if axis == "x":
        left = BRepPrimAPI_MakeBox(
            gp_Pnt(-center_offset - support_width / 2.0, -support_cross / 2.0, base_t - 0.5),
            support_width, support_cross, support_h + 0.5,
        ).Shape()
        right = BRepPrimAPI_MakeBox(
            gp_Pnt(center_offset - support_width / 2.0, -support_cross / 2.0, base_t - 0.5),
            support_width, support_cross, support_h + 0.5,
        ).Shape()
    else:
        left = BRepPrimAPI_MakeBox(
            gp_Pnt(-support_cross / 2.0, -center_offset - support_width / 2.0, base_t - 0.5),
            support_cross, support_width, support_h + 0.5,
        ).Shape()
        right = BRepPrimAPI_MakeBox(
            gp_Pnt(-support_cross / 2.0, center_offset - support_width / 2.0, base_t - 0.5),
            support_cross, support_width, support_h + 0.5,
        ).Shape()
    stock = _fuse_shapes(base, left, right)
    inner_half_span = center_offset - support_width / 2.0
    span = 2.0 * inner_half_span + rng.uniform(3.0, 5.0)
    thickness = rng.uniform(3.4, 4.6)
    height = rng.uniform(support_h * 0.62, support_h * 0.80)
    z0 = base_t - 0.6
    profile = _symmetric_wall_profile(span, z0, height, variant == "transition")
    tool = _vertical_wall_tool(profile, thickness, 0.0, axis=axis)
    supports = (("base_region", base), ("support_body_a", left), ("support_body_b", right))
    contact_records = _support_contact_records(tool, supports)
    fuse = BRepAlgoAPI_Fuse(stock, tool)
    fuse.Build()
    if not fuse.IsDone():
        raise ValueError("two-support web fuse failed")
    feature_inst = 1
    tags = _tag_tool_result_faces(
        fuse,
        tool,
        lambda face: _feature_face_role(face, "web", z0),
        macro_key="web",
        instance_id=feature_inst,
    )
    shape = fuse.Shape()
    hole_parameters = None
    if variant == "hole":
        radius = min(height * rng.uniform(0.15, 0.18), span * 0.075)
        hole = _horizontal_hole_tool(
            axis=axis,
            center_s=0.0,
            center_cross=0.0,
            center_z=z0 + height * 0.43,
            thickness=thickness,
            radius=radius,
        )
        shape, tags, hole_parameters = _apply_feature_hole(shape, tags, hole, feature_inst + 1, radius, axis=_cross_axis(axis))
    params = _base_parameters(
        scenario_key=scenario_key,
        seed=seed,
        macro_key="web",
        context_kind="two_supports",
        variant=variant,
        span=span,
        thickness=thickness,
        height=height,
        feature_inst=feature_inst,
        support_contacts=contact_records,
        transition_kind="paired_sloped_support_transitions" if variant == "transition" else "none",
        hole_parameters=hole_parameters,
    )
    params.update({
        "stock_kind": "base_with_two_raised_support_bodies",
        "span_axis": axis,
        "support_body_count": 2,
        "connects_two_main_body_regions": True,
        "paired_large_side_faces_expected": True,
    })
    return _finish_result(shape, params, tags)


def _base_parameters(
    *,
    scenario_key: str,
    seed: int,
    macro_key: str,
    context_kind: str,
    variant: str,
    span: float,
    thickness: float,
    height: float,
    feature_inst: int,
    support_contacts: list[dict[str, Any]],
    transition_kind: str,
    hole_parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "scenario_key": scenario_key,
        "seed": int(seed),
        "macro_key": macro_key,
        "context_kind": context_kind,
        "variant": variant,
        "construction_method": "p1_b_engineering_supported_thin_wall_boolean",
        "feature_instance_id": int(feature_inst),
        "span": float(span),
        "thickness": float(thickness),
        "height": float(height),
        "span_thickness_ratio": float(span) / float(thickness),
        "height_thickness_ratio": float(height) / float(thickness),
        "support_contacts": support_contacts,
        "positive_support_contact_count": sum(item["intersection_volume"] > 1e-7 for item in support_contacts),
        "isolated_thin_plate": False,
        "transition_kind": transition_kind,
        "has_hole": hole_parameters is not None,
        "hole_parameters": hole_parameters,
    }


def _finish_result(shape: Any, parameters: dict[str, Any], tags: list[FaceTag]) -> P1BRibWebResult:
    final_faces = list(TopologyExplorer(shape).faces())
    kept_tags = [tag for tag in tags if any(tag.face.IsSame(face) for face in final_faces)]
    parameters = dict(parameters)
    parameters["feature_face_count"] = sum(
        tag.macro_key == parameters["macro_key"] and tag.instance_id == parameters["feature_instance_id"] for tag in kept_tags
    )
    parameters["pocket_face_count"] = sum(tag.macro_key in {"pocket_rect", "pocket_circular"} for tag in kept_tags)
    parameters["hole_face_count"] = sum(tag.macro_key == "through_hole" for tag in kept_tags)
    parameters["tagged_face_count"] = len(kept_tags)
    return P1BRibWebResult(shape, parameters, tuple(kept_tags))


def _validate_result(result: P1BRibWebResult) -> None:
    params = result.parameters
    if not BRepCheck_Analyzer(result.shape).IsValid():
        raise ValueError("P1-B rib/web result is not a valid B-Rep")
    if len(list(TopologyExplorer(result.shape).solids())) != 1:
        raise ValueError("P1-B rib/web result must be one solid")
    if _volume(result.shape) <= 0.0:
        raise ValueError("P1-B rib/web result must have positive volume")
    if params["positive_support_contact_count"] < 2:
        raise ValueError("thin wall does not contact at least two support regions")
    if params["span_thickness_ratio"] < 8.0 or params["height_thickness_ratio"] < 3.0:
        raise ValueError("thin-wall dimensional ratio is not reinforcement-like")
    if params["feature_face_count"] < 2:
        raise ValueError("thin-wall exposed faces were not preserved")
    feature_tags = [
        tag for tag in result.face_tags
        if tag.macro_key == params["macro_key"] and tag.instance_id == params["feature_instance_id"]
    ]
    if not feature_tags or {tag.instance_id for tag in feature_tags} != {params["feature_instance_id"]}:
        raise ValueError("thin-wall faces do not share one instance")
    if params["context_kind"] in {"rect_cavity", "circular_cavity"} and params["pocket_face_count"] < 2:
        raise ValueError("cavity/pocket labels were not preserved independently")
    if params["has_hole"] and params["hole_face_count"] < 1:
        raise ValueError("hole variant did not preserve an independent hole wall")


def _vertical_wall_tool(profile: list[tuple[float, float]], thickness: float, cross_center: float, axis: str) -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    if axis == "x":
        for span_coordinate, z_value in profile:
            polygon.Add(gp_Pnt(span_coordinate, cross_center - thickness / 2.0, z_value))
        prism_vector = gp_Vec(0.0, thickness, 0.0)
    elif axis == "y":
        for span_coordinate, z_value in profile:
            polygon.Add(gp_Pnt(cross_center - thickness / 2.0, span_coordinate, z_value))
        prism_vector = gp_Vec(thickness, 0.0, 0.0)
    else:
        raise ValueError(axis)
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, prism_vector).Shape()


def _symmetric_wall_profile(span: float, z0: float, height: float, transition: bool) -> list[tuple[float, float]]:
    half = span / 2.0
    if not transition:
        return [(-half, z0), (half, z0), (half, z0 + height), (-half, z0 + height)]
    return [
        (-half, z0),
        (half, z0),
        (half, z0 + height),
        (half * 0.72, z0 + height),
        (half * 0.52, z0 + height * 0.78),
        (-half * 0.52, z0 + height * 0.78),
        (-half * 0.72, z0 + height),
        (-half, z0 + height),
    ]


def _horizontal_hole_tool(
    *, axis: str, center_s: float, center_cross: float, center_z: float, thickness: float, radius: float
) -> Any:
    overrun = 1.5
    if axis == "x":
        origin = gp_Pnt(center_s, center_cross - thickness / 2.0 - overrun, center_z)
        direction = gp_Dir(0.0, 1.0, 0.0)
    else:
        origin = gp_Pnt(center_cross - thickness / 2.0 - overrun, center_s, center_z)
        direction = gp_Dir(1.0, 0.0, 0.0)
    return BRepPrimAPI_MakeCylinder(gp_Ax2(origin, direction), radius, thickness + 2.0 * overrun).Shape()


def _apply_feature_hole(
    shape: Any,
    tags: list[FaceTag],
    hole_tool: Any,
    hole_instance_id: int,
    radius: float,
    *,
    axis: str,
) -> tuple[Any, list[FaceTag], dict[str, Any]]:
    cut = BRepAlgoAPI_Cut(shape, hole_tool)
    cut.Build()
    if not cut.IsDone():
        raise ValueError("thin-wall hole cut failed")
    propagated = _propagate_tags(cut, tags)
    hole_tags = _tag_tool_result_faces(
        cut,
        hole_tool,
        lambda _face: "through_hole_cylinder",
        macro_key="through_hole",
        instance_id=hole_instance_id,
    )
    return cut.Shape(), _merge_tags(propagated, hole_tags), {
        "macro_key": "through_hole",
        "instance_id": int(hole_instance_id),
        "radius": float(radius),
        "axis": axis,
        "passes_only_through_thin_wall": True,
    }


def _pocket_face_role(face: Any, bottom_z: float, side_role: str, bottom_role: str) -> str:
    surface = BRepAdaptor_Surface(face)
    center = _face_center(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and abs(center[2] - bottom_z) < 0.3:
            return bottom_role
    return side_role


def _feature_face_role(face: Any, macro_key: str, z0: float) -> str:
    side_role, top_role = FEATURE_ROLE_BY_MACRO[macro_key]
    surface = BRepAdaptor_Surface(face)
    center = _face_center(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.45 and center[2] > z0 + 0.2:
            return top_role
    return side_role


def _tag_tool_result_faces(
    operation: Any,
    tool: Any,
    role_for_tool_face: Callable[[Any], str],
    *,
    macro_key: str,
    instance_id: int,
) -> list[FaceTag]:
    final_faces = list(TopologyExplorer(operation.Shape()).faces())
    result: list[FaceTag] = []
    for tool_face in TopologyExplorer(tool).faces():
        role = role_for_tool_face(tool_face)
        mapped = [face for face in operation.Modified(tool_face) if face.ShapeType() == 4]
        if not mapped:
            mapped = [face for face in final_faces if face.IsSame(tool_face)]
        for face in mapped:
            _append_tag_unique(result, FaceTag(face, role, macro_key, int(instance_id)))
    return result


def _propagate_tags(operation: Any, tags: list[FaceTag] | tuple[FaceTag, ...]) -> list[FaceTag]:
    final_faces = list(TopologyExplorer(operation.Shape()).faces())
    result: list[FaceTag] = []
    for tag in tags:
        mapped = [face for face in operation.Modified(tag.face) if face.ShapeType() == 4]
        if not mapped and any(tag.face.IsSame(face) for face in final_faces):
            mapped = [tag.face]
        for face in mapped:
            _append_tag_unique(result, FaceTag(face, tag.semantic_role, tag.macro_key, tag.instance_id))
    return result


def _merge_tags(*groups: list[FaceTag]) -> list[FaceTag]:
    result: list[FaceTag] = []
    for group in groups:
        for tag in group:
            _append_tag_unique(result, tag)
    return result


def _append_tag_unique(tags: list[FaceTag], candidate: FaceTag) -> None:
    for existing in tags:
        if candidate.face.IsSame(existing.face):
            if (
                candidate.semantic_role != existing.semantic_role
                or candidate.macro_key != existing.macro_key
                or candidate.instance_id != existing.instance_id
            ):
                raise ValueError("conflicting semantic tags for the same final face")
            return
    tags.append(candidate)


def _cavity_support_shapes(
    host: Any,
    *,
    context_kind: str,
    axis: str,
    span_half: float,
    cross_half: float,
    bottom_z: float,
    total_h: float,
    thickness: float,
    length: float,
    width: float,
) -> tuple[tuple[str, Any], ...]:
    del context_kind
    margin = 3.0
    if axis == "x":
        support_a_proxy = BRepPrimAPI_MakeBox(
            gp_Pnt(-span_half - margin, -thickness, bottom_z - 0.2), margin + 0.4, 2.0 * thickness, total_h - bottom_z + 0.4
        ).Shape()
        support_b_proxy = BRepPrimAPI_MakeBox(
            gp_Pnt(span_half - 0.4, -thickness, bottom_z - 0.2), margin + 0.4, 2.0 * thickness, total_h - bottom_z + 0.4
        ).Shape()
    else:
        support_a_proxy = BRepPrimAPI_MakeBox(
            gp_Pnt(-thickness, -span_half - margin, bottom_z - 0.2), 2.0 * thickness, margin + 0.4, total_h - bottom_z + 0.4
        ).Shape()
        support_b_proxy = BRepPrimAPI_MakeBox(
            gp_Pnt(-thickness, span_half - 0.4, bottom_z - 0.2), 2.0 * thickness, margin + 0.4, total_h - bottom_z + 0.4
        ).Shape()
    floor_proxy = BRepPrimAPI_MakeBox(
        gp_Pnt(-length / 2.0, -width / 2.0, bottom_z - 1.5), length, width, 1.7
    ).Shape()
    return (
        ("opposed_wall_a", _common_shape(host, support_a_proxy)),
        ("opposed_wall_b", _common_shape(host, support_b_proxy)),
        ("cavity_floor", _common_shape(host, floor_proxy)),
    )


def _support_contact_records(tool: Any, supports: tuple[tuple[str, Any], ...]) -> list[dict[str, Any]]:
    records = []
    for name, support in supports:
        records.append({"name": name, "intersection_volume": float(_common_volume(tool, support))})
    return records


def _fuse_shapes(*shapes: Any) -> Any:
    result = shapes[0]
    for shape in shapes[1:]:
        fuse = BRepAlgoAPI_Fuse(result, shape)
        fuse.Build()
        if not fuse.IsDone():
            raise ValueError("support stock fuse failed")
        result = fuse.Shape()
    return result


def _common_shape(first: Any, second: Any) -> Any:
    common = BRepAlgoAPI_Common(first, second)
    common.Build()
    if not common.IsDone():
        raise ValueError("support proxy intersection failed")
    return common.Shape()


def _common_volume(first: Any, second: Any) -> float:
    return _volume(_common_shape(first, second))


def _volume(shape: Any) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return float(props.Mass())


def _face_center(face: Any) -> tuple[float, float, float]:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    center = props.CentreOfMass()
    return float(center.X()), float(center.Y()), float(center.Z())


def _cross_axis(axis: str) -> str:
    return "y" if axis == "x" else "x"


__all__ = [
    "FaceTag",
    "P1BRibWebResult",
    "SCENARIO_SPECS",
    "build_rib_web_sample",
    "scenario_keys",
]
