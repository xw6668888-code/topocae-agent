from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.geometry import clean_cut, clean_fuse, cleanup_audit, face_center, operation_faces, propagated_faces
from p1_c.planar_tools import capsule_prism_xy, polygon_prism_xy, rectangle_prism_xy
from p1_c.rib_web_v2 import build_rib_web_sample


GENERATOR_REVISION = "p1_c_all_features_clean_r1"
MACRO_CLASSES = (
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rib",
    "web",
    "rectangular_boss",
    "fillet",
    "chamfer",
    "slot_v",
    "slot_o",
)
HOST_FAMILIES = ("compact_plate", "long_x_plate", "long_y_plate", "tall_block")
TRANSITION_CLASSES = {"fillet", "chamfer"}
H_C2_SECONDARY_BY_PRIMARY = {
    macro: ("rectangular_boss" if macro == "through_hole_round" else "through_hole_round")
    for macro in MACRO_CLASSES
}
H_C2_SECONDARY_BY_PRIMARY["rib"] = "through_hole_round"
H_C2_SECONDARY_BY_PRIMARY["web"] = "through_hole_round"


@dataclass(frozen=True)
class FaceTag:
    face: Any
    semantic_role: str
    macro_feature: str
    instance_id: int


@dataclass(frozen=True)
class CleanFeatureResult:
    shape: Any
    face_tags: tuple[FaceTag, ...]
    parameters: dict[str, Any]
    cleanup: dict[str, Any]


def build_feature_sample(
    primary: str,
    seed: int,
    *,
    secondary_features: Sequence[str] = (),
    host_family: str | None = None,
) -> CleanFeatureResult:
    if primary not in MACRO_CLASSES:
        raise KeyError(primary)
    if len(secondary_features) > 2:
        raise ValueError("P1-C supports target-only, pair, or triple contexts")
    if any(macro not in MACRO_CLASSES for macro in secondary_features):
        raise KeyError("unknown secondary feature")
    if len({primary, *secondary_features}) != 1 + len(secondary_features):
        raise ValueError("one sample cannot repeat a macro feature")
    if primary in {"rib", "web"}:
        return _build_rib_web_context(primary, int(seed), tuple(secondary_features))

    rng = random.Random(int(seed))
    selected_host_family = host_family or HOST_FAMILIES[int(seed) % len(HOST_FAMILIES)]
    shape, host_metadata = _make_host(selected_host_family, rng)
    host_cleanup = cleanup_audit(shape)
    if not host_cleanup["passed"]:
        raise ValueError("new P1-C host is not cleanup-idempotent")
    tags: list[FaceTag] = []
    macros = [primary, *secondary_features]
    centers = _placement_centers(len(macros), host_metadata)
    execution = sorted(
        [(macro, index + 1, centers[index]) for index, macro in enumerate(macros)],
        key=lambda item: (item[0] in TRANSITION_CLASSES, item[1]),
    )
    feature_records: dict[int, dict[str, Any]] = {}
    for macro, instance_id, center in execution:
        shape, tags, record = _apply_feature(
            shape,
            tags,
            macro=macro,
            instance_id=instance_id,
            center=center,
            host=host_metadata,
            rng=rng,
        )
        feature_records[instance_id] = record
    cleanup = cleanup_audit(shape)
    ordered_records = [feature_records[index] for index in sorted(feature_records)]
    parameters = {
        "schema_version": "topocae_p1_c_clean_feature_parameters_1",
        "generator_revision": GENERATOR_REVISION,
        "seed": int(seed),
        "primary_target": primary,
        "secondary_features": list(secondary_features),
        "context": ("target_only", "pair", "triple")[len(secondary_features)],
        "host_family": selected_host_family,
        "host_metadata": host_metadata,
        "host_cleanup_before_features": host_cleanup,
        "features": ordered_records,
        "instance_ids": {record["macro_feature"]: record["instance_id"] for record in ordered_records},
        "construction_contract": {
            "host_single_primitive": True,
            "cleanup_after_each_boolean": True,
            "final_labeling_after_cleanup": True,
            "feature_intersection_allowed": False,
        },
    }
    result = CleanFeatureResult(shape, tuple(_filter_final_tags(shape, tags)), parameters, cleanup)
    _validate_result(result)
    return result


def _make_host(host_family: str, rng: random.Random) -> tuple[Any, dict[str, Any]]:
    if host_family == "compact_plate":
        length, width, height = rng.uniform(82.0, 94.0), rng.uniform(68.0, 80.0), rng.uniform(19.0, 24.0)
    elif host_family == "long_x_plate":
        length, width, height = rng.uniform(108.0, 124.0), rng.uniform(62.0, 74.0), rng.uniform(20.0, 27.0)
    elif host_family == "long_y_plate":
        length, width, height = rng.uniform(68.0, 78.0), rng.uniform(108.0, 124.0), rng.uniform(20.0, 27.0)
    elif host_family == "tall_block":
        length, width, height = rng.uniform(82.0, 94.0), rng.uniform(72.0, 86.0), rng.uniform(32.0, 42.0)
    else:
        raise KeyError(host_family)
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, height).Shape()
    return shape, {
        "family": host_family,
        "construction_method": "single_box_primitive",
        "length": length,
        "width": width,
        "height": height,
        "top_z": height,
        "origin": [-length / 2.0, -width / 2.0, 0.0],
    }


def _placement_centers(count: int, host: dict[str, Any]) -> list[tuple[float, float]]:
    length = float(host["length"])
    width = float(host["width"])
    if count == 1:
        return [(0.0, 0.0)]
    if count == 2:
        return [(-0.23 * length, 0.0), (0.23 * length, 0.0)]
    return [(-0.24 * length, -0.14 * width), (0.24 * length, -0.14 * width), (0.0, 0.24 * width)]


def _apply_feature(
    shape: Any,
    tags: list[FaceTag],
    *,
    macro: str,
    instance_id: int,
    center: tuple[float, float],
    host: dict[str, Any],
    rng: random.Random,
) -> tuple[Any, list[FaceTag], dict[str, Any]]:
    if macro in TRANSITION_CLASSES:
        return _apply_transition(shape, tags, macro, instance_id, host, rng)
    tool, operation_kind, role_for_face, public = _feature_tool(macro, center, host, rng)
    if operation_kind == "cut":
        operation, new_shape = clean_cut(shape, tool)
    else:
        operation, new_shape = clean_fuse(shape, tool)
    propagated = _propagate_tags(operation, tags)
    created = _tags_from_tool(
        operation,
        tool,
        role=role_for_face,
        macro_feature=macro,
        instance_id=instance_id,
    )
    merged = _merge_tags(propagated, created)
    audit = cleanup_audit(new_shape)
    if not audit["passed"]:
        raise ValueError(f"{macro} boolean result is not cleanup-idempotent: {audit}")
    record = {
        "macro_feature": macro,
        "instance_id": int(instance_id),
        "operation": operation_kind,
        "center": [float(center[0]), float(center[1])],
        "parameters": public,
        "feature_face_count": sum(tag.instance_id == instance_id for tag in merged),
        "cleanup_after_operation": audit,
    }
    if record["feature_face_count"] <= 0:
        raise ValueError(f"{macro} produced no labeled final face")
    return new_shape, merged, record


def _feature_tool(
    macro: str,
    center: tuple[float, float],
    host: dict[str, Any],
    rng: random.Random,
) -> tuple[Any, str, Callable[[Any], str], dict[str, Any]]:
    cx, cy = center
    top = float(host["top_z"])
    height = float(host["height"])
    base_radius = rng.uniform(4.0, 5.6)
    if macro == "through_hole_round":
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, -1.0), gp_Dir(0.0, 0.0, 1.0)), base_radius, height + 2.0).Shape()
        return tool, "cut", lambda _face: "through_hole_cylinder", {"radius": base_radius, "through": True}
    if macro == "blind_hole_round":
        depth = min(rng.uniform(8.0, 12.0), height * 0.62)
        bottom = top - depth
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, bottom), gp_Dir(0.0, 0.0, 1.0)), base_radius, depth + 1.0).Shape()
        return tool, "cut", lambda face: _blind_hole_role(face, bottom), {"radius": base_radius, "depth": depth}
    if macro in {"stepped_hole_round", "counterbore_hole"}:
        large_radius = base_radius * rng.uniform(1.55, 1.85)
        step_depth = min(rng.uniform(3.5, 5.5), height * 0.25)
        if macro == "stepped_hole_round":
            total_depth = min(rng.uniform(10.0, 14.0), height * 0.72)
            small_origin = top - total_depth
            small_height = total_depth + 1.0
            bottom_z: float | None = small_origin
        else:
            small_origin = -1.0
            small_height = height + 2.0
            bottom_z = None
        large = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(cx, cy, top - step_depth), gp_Dir(0.0, 0.0, 1.0)), large_radius, step_depth + 1.0
        ).Shape()
        small = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(cx, cy, small_origin), gp_Dir(0.0, 0.0, 1.0)), base_radius, small_height
        ).Shape()
        _, tool = clean_fuse(large, small)
        if macro == "stepped_hole_round":
            role = lambda face: _stepped_hole_role(face, base_radius, large_radius, top - step_depth, bottom_z)
        else:
            role = lambda face: _counterbore_role(face, base_radius, large_radius, top - step_depth)
        return tool, "cut", role, {
            "small_radius": base_radius,
            "large_radius": large_radius,
            "step_depth": step_depth,
            "bottom_z": bottom_z,
        }
    if macro == "countersink_hole":
        sink_depth = min(rng.uniform(4.0, 6.0), height * 0.28)
        large_radius = base_radius * rng.uniform(1.7, 2.0)
        cylinder = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(cx, cy, -1.0), gp_Dir(0.0, 0.0, 1.0)), base_radius, height + 2.0
        ).Shape()
        cone = BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(cx, cy, top - sink_depth), gp_Dir(0.0, 0.0, 1.0)),
            base_radius,
            large_radius,
            sink_depth,
        ).Shape()
        _, tool = clean_fuse(cylinder, cone)
        return tool, "cut", _countersink_role, {
            "small_radius": base_radius,
            "large_radius": large_radius,
            "sink_depth": sink_depth,
        }
    if macro == "obround_hole":
        radius = rng.uniform(3.5, 4.5)
        distance = rng.uniform(10.0, 14.0)
        tool = capsule_prism_xy(center, distance, radius, z_min=-1.0, height=height + 2.0)
        return tool, "cut", _obround_role, {"radius": radius, "center_distance": distance, "through": True}
    if macro in {"slot_u", "slot_rect", "slot_v", "slot_o"}:
        depth = min(rng.uniform(5.0, 8.0), height * 0.38)
        bottom = top - depth
        radius = rng.uniform(3.5, 4.5)
        length = rng.uniform(15.0, 19.0)
        if macro == "slot_rect":
            tool = rectangle_prism_xy(center, length + 2.0 * radius, 2.0 * radius, z_min=bottom, height=depth + 1.0)
        elif macro == "slot_v":
            half = length / 2.0
            tool = polygon_prism_xy(
                [
                    (cx - half, cy),
                    (cx - half + radius, cy - radius),
                    (cx + half - radius, cy - radius),
                    (cx + half, cy),
                    (cx + half - radius, cy + radius),
                    (cx - half + radius, cy + radius),
                ],
                z_min=bottom,
                height=depth + 1.0,
            )
        elif macro == "slot_u":
            tool = _d_profile_tool(center, length, radius, bottom, depth + 1.0)
        else:
            tool = capsule_prism_xy(center, length - 2.0 * radius, radius, z_min=bottom, height=depth + 1.0)
        return tool, "cut", lambda face: _slot_role(face, macro, bottom), {
            "length": length,
            "width": 2.0 * radius,
            "depth": depth,
            "bottom_z": bottom,
            "profile": {"slot_u": "D_closed", "slot_rect": "rectangle_closed", "slot_v": "double_V_closed", "slot_o": "capsule_closed"}[macro],
        }
    if macro == "pocket_rect":
        depth = min(rng.uniform(6.0, 9.0), height * 0.44)
        bottom = top - depth
        size_x, size_y = rng.uniform(17.0, 22.0), rng.uniform(13.0, 18.0)
        tool = rectangle_prism_xy(center, size_x, size_y, z_min=bottom, height=depth + 1.0)
        return tool, "cut", lambda face: _bottom_or_side(face, bottom, "rect_pocket_bottom_plane", "rect_pocket_side_plane"), {
            "size_x": size_x,
            "size_y": size_y,
            "depth": depth,
        }
    if macro == "pocket_circular":
        depth = min(rng.uniform(6.0, 9.0), height * 0.44)
        bottom = top - depth
        radius = rng.uniform(7.0, 9.0)
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(cx, cy, bottom), gp_Dir(0.0, 0.0, 1.0)), radius, depth + 1.0
        ).Shape()
        return tool, "cut", lambda face: _bottom_or_side(face, bottom, "circular_pocket_bottom_plane", "circular_pocket_side_cylinder"), {
            "radius": radius,
            "depth": depth,
        }
    if macro == "cylinder_boss":
        radius = rng.uniform(6.5, 8.5)
        boss_height = rng.uniform(7.0, 11.0)
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(cx, cy, top - 0.25), gp_Dir(0.0, 0.0, 1.0)), radius, boss_height + 0.25
        ).Shape()
        return tool, "fuse", lambda face: _boss_role(face, top, "cylinder_boss_side", "cylinder_boss_top"), {
            "radius": radius,
            "height": boss_height,
        }
    if macro == "rectangular_boss":
        size_x, size_y = rng.uniform(14.0, 18.0), rng.uniform(12.0, 16.0)
        boss_height = rng.uniform(7.0, 11.0)
        tool = rectangle_prism_xy(center, size_x, size_y, z_min=top - 0.25, height=boss_height + 0.25)
        return tool, "fuse", lambda face: _boss_role(face, top, "pad_side_plane", "pad_top_plane"), {
            "size_x": size_x,
            "size_y": size_y,
            "height": boss_height,
        }
    raise KeyError(macro)


def _d_profile_tool(center: tuple[float, float], length: float, radius: float, z_min: float, height: float) -> Any:
    cx, cy = center
    half = length / 2.0
    points = [
        (cx - half, cy - radius),
        (cx + half - radius, cy - radius),
        (cx + half, cy),
        (cx + half - radius, cy + radius),
        (cx - half, cy + radius),
    ]
    return polygon_prism_xy(points, z_min=z_min, height=height)


def _apply_transition(
    shape: Any,
    tags: list[FaceTag],
    macro: str,
    instance_id: int,
    host: dict[str, Any],
    rng: random.Random,
) -> tuple[Any, list[FaceTag], dict[str, Any]]:
    edge = _select_outer_top_edge(shape, host)
    size = rng.uniform(1.2, 2.2)
    if macro == "fillet":
        builder = BRepFilletAPI_MakeFillet(shape)
        builder.Add(size, edge)
        semantic_role = "fillet_blend_surface"
        parameter_key = "radius"
    else:
        builder = BRepFilletAPI_MakeChamfer(shape)
        builder.Add(size, edge)
        semantic_role = "chamfer_bevel_plane"
        parameter_key = "distance"
    builder.Build()
    if not builder.IsDone():
        raise ValueError(f"{macro} transition build failed")
    new_shape = builder.Shape()
    propagated = _propagate_tags(builder, tags)
    created: list[FaceTag] = []
    for generated in builder.Generated(edge):
        if generated.ShapeType() == 4:
            _append_tag(created, FaceTag(generated, semantic_role, macro, int(instance_id)))
    merged = _merge_tags(propagated, created)
    audit = cleanup_audit(new_shape)
    if not audit["passed"]:
        raise ValueError(f"{macro} result is not cleanup-idempotent: {audit}")
    record = {
        "macro_feature": macro,
        "instance_id": int(instance_id),
        "operation": "edge_transition",
        "center": list(_edge_midpoint(edge)),
        "parameters": {parameter_key: size},
        "feature_face_count": sum(tag.instance_id == instance_id for tag in merged),
        "cleanup_after_operation": audit,
    }
    if record["feature_face_count"] != 1:
        raise ValueError(f"{macro} should create exactly one transition face")
    return new_shape, merged, record


def _select_outer_top_edge(shape: Any, host: dict[str, Any]) -> Any:
    top = float(host["top_z"])
    candidates: list[tuple[float, Any]] = []
    for edge in TopologyExplorer(shape).edges():
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line:
            continue
        first = curve.Value(curve.FirstParameter())
        last = curve.Value(curve.LastParameter())
        if abs(first.Z() - top) > 1e-5 or abs(last.Z() - top) > 1e-5:
            continue
        midpoint = _edge_midpoint(edge)
        outer = abs(midpoint[0]) > 0.40 * float(host["length"]) or abs(midpoint[1]) > 0.40 * float(host["width"])
        if outer:
            candidates.append((first.Distance(last), edge))
    if not candidates:
        raise ValueError("no clean outer top edge available for transition")
    return max(candidates, key=lambda item: item[0])[1]


def _edge_midpoint(edge: Any) -> tuple[float, float, float]:
    curve = BRepAdaptor_Curve(edge)
    point = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2.0)
    return float(point.X()), float(point.Y()), float(point.Z())


def _blind_hole_role(face: Any, bottom: float) -> str:
    if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
        return "blind_hole_cylinder"
    return "blind_hole_bottom" if abs(face_center(face)[2] - bottom) < 0.5 else "blind_hole_cylinder"


def _stepped_hole_role(face: Any, small_radius: float, large_radius: float, step_z: float, bottom_z: float | None) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Cylinder:
        radius = surface.Cylinder().Radius()
        return "stepped_hole_large_hole_cylinder" if abs(radius - large_radius) < abs(radius - small_radius) else "stepped_hole_small_hole_cylinder"
    z_value = face_center(face)[2]
    if bottom_z is not None and abs(z_value - bottom_z) < abs(z_value - step_z):
        return "stepped_hole_small_hole_bottom"
    return "stepped_hole_step_plane"


def _counterbore_role(face: Any, small_radius: float, large_radius: float, step_z: float) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Cylinder:
        radius = surface.Cylinder().Radius()
        return "counterbore_large_cylinder" if abs(radius - large_radius) < abs(radius - small_radius) else "counterbore_small_cylinder"
    return "counterbore_step_plane"


def _countersink_role(face: Any) -> str:
    return "countersink_cone" if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cone else "countersink_small_cylinder"


def _obround_role(face: Any) -> str:
    return "obround_hole_end_cylinder" if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder else "obround_hole_side_plane"


def _slot_role(face: Any, macro: str, bottom: float) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and abs(face_center(face)[2] - bottom) < 0.5:
            return f"{macro}_bottom_plane"
    if macro == "slot_u" and surface.GetType() == GeomAbs_Cylinder:
        return "slot_u_end_cylinder"
    return f"{macro}_side_plane"


def _bottom_or_side(face: Any, bottom: float, bottom_role: str, side_role: str) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and abs(face_center(face)[2] - bottom) < 0.5:
            return bottom_role
    return side_role


def _boss_role(face: Any, host_top: float, side_role: str, top_role: str) -> str:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and face_center(face)[2] > host_top + 0.25:
            return top_role
    return side_role


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


def _append_tag(tags: list[FaceTag], candidate: FaceTag) -> None:
    for existing in tags:
        if candidate.face.IsSame(existing.face):
            if (
                candidate.semantic_role != existing.semantic_role
                or candidate.macro_feature != existing.macro_feature
                or candidate.instance_id != existing.instance_id
            ):
                raise ValueError("one final face received conflicting feature labels")
            return
    tags.append(candidate)


def _filter_final_tags(shape: Any, tags: Iterable[FaceTag]) -> list[FaceTag]:
    final_faces = list(TopologyExplorer(shape).faces())
    return [tag for tag in tags if any(tag.face.IsSame(face) for face in final_faces)]


def _build_rib_web_context(primary: str, seed: int, secondaries: tuple[str, ...]) -> CleanFeatureResult:
    if len(secondaries) > 1 or (secondaries and secondaries != ("through_hole_round",)):
        raise ValueError("current clean rib/web composition only permits one independent through-hole secondary")
    if primary == "rib":
        scenario = "rib_l_corner_hole" if secondaries else "rib_l_corner_plain"
    else:
        scenario = "web_two_supports_hole" if secondaries else "web_two_supports_plain"
    raw = build_rib_web_sample(scenario, seed)
    tags = tuple(FaceTag(tag.face, tag.semantic_role, tag.macro_feature, tag.instance_id) for tag in raw.face_tags)
    feature_instance = int(raw.parameters["feature_instance_id"])
    if feature_instance != 1:
        raise ValueError("H-C2 rib/web target must use a host without a pocket instance")
    features = [
        {
            "macro_feature": primary,
            "instance_id": 1,
            "operation": "supported_thin_wall_fuse",
            "feature_face_count": sum(tag.instance_id == 1 for tag in tags),
        }
    ]
    if secondaries:
        features.append(
            {
                "macro_feature": "through_hole_round",
                "instance_id": 2,
                "operation": "thin_wall_hole_cut",
                "feature_face_count": sum(tag.instance_id == 2 for tag in tags),
            }
        )
    parameters = {
        "schema_version": "topocae_p1_c_clean_feature_parameters_1",
        "generator_revision": GENERATOR_REVISION,
        "seed": int(seed),
        "primary_target": primary,
        "secondary_features": list(secondaries),
        "context": "pair" if secondaries else "target_only",
        "host_family": raw.parameters["host_family"],
        "host_metadata": raw.parameters["host_metadata"],
        "host_cleanup_before_features": raw.parameters["host_cleanup_before_feature"],
        "features": features,
        "instance_ids": {record["macro_feature"]: record["instance_id"] for record in features},
        "rib_web_parameters": raw.parameters,
    }
    result = CleanFeatureResult(raw.shape, tags, parameters, raw.cleanup)
    _validate_result(result)
    return result


def _validate_result(result: CleanFeatureResult) -> None:
    if not result.cleanup["passed"]:
        raise ValueError("final P1-C feature sample is not cleanup-idempotent")
    if not result.parameters["host_cleanup_before_features"]["passed"]:
        raise ValueError("P1-C feature host is not clean")
    final_faces = list(TopologyExplorer(result.shape).faces())
    for tag in result.face_tags:
        if not any(tag.face.IsSame(face) for face in final_faces):
            raise ValueError("feature truth references a non-final face")
    instances = {tag.instance_id for tag in result.face_tags}
    expected = set(range(1, len(result.parameters["secondary_features"]) + 2))
    if instances != expected:
        raise ValueError(f"feature instances are incomplete: expected={expected}, actual={instances}")
    for record in result.parameters["features"]:
        macros = {tag.macro_feature for tag in result.face_tags if tag.instance_id == record["instance_id"]}
        if macros != {record["macro_feature"]}:
            raise ValueError(f"instance macro mismatch: {record}, {macros}")


__all__ = [
    "CleanFeatureResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "H_C2_SECONDARY_BY_PRIMARY",
    "HOST_FAMILIES",
    "MACRO_CLASSES",
    "build_feature_sample",
]
