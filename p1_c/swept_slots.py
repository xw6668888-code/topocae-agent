from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.geometry import cleanup_audit, face_center, topology_counts


SLOT_PROFILE_BY_MACRO = {
    "slot_rect": "rect",
    "slot_u": "u",
    "slot_v": "v",
    "slot_o": "o",
}
SLOT_ROLES = {
    "slot_rect": ("slot_rect_side_plane", "slot_rect_bottom_plane"),
    "slot_u": ("slot_u_side_plane", "slot_u_bottom_plane"),
    "slot_v": ("slot_v_side_plane", "slot_v_bottom_plane"),
    "slot_o": ("slot_o_side_plane", "slot_o_bottom_plane"),
}


@dataclass(frozen=True)
class SweptSlotTool:
    tool: Any
    guide_wire: Any
    profile_wire: Any
    role_for_face: Callable[[Any], str]
    parameters: dict[str, Any]


def build_swept_slot_tool(
    macro: str,
    center: tuple[float, float],
    host: dict[str, Any],
    rng: random.Random,
) -> SweptSlotTool:
    if macro not in SLOT_PROFILE_BY_MACRO:
        raise KeyError(macro)
    surface_z = float(host["top_z"])
    width = rng.uniform(5.5, 7.0)
    profile = SLOT_PROFILE_BY_MACRO[macro]
    depth = width / 2.0 if profile == "o" else width * rng.uniform(0.75, 0.95)
    top_margin = max(0.9, width * 0.18)
    if abs(float(center[0])) > float(host["length"]) * 0.08:
        tangent = (0.0, 1.0)
        path_length = min(float(host["width"]) * 0.58, 48.0)
    else:
        tangent = (1.0, 0.0)
        path_length = min(float(host["length"]) * 0.58, 56.0)
    start = (
        float(center[0]) - tangent[0] * path_length / 2.0,
        float(center[1]) - tangent[1] * path_length / 2.0,
    )
    end = (
        float(center[0]) + tangent[0] * path_length / 2.0,
        float(center[1]) + tangent[1] * path_length / 2.0,
    )
    guide_edge = BRepBuilderAPI_MakeEdge(
        gp_Pnt(start[0], start[1], surface_z),
        gp_Pnt(end[0], end[1], surface_z),
    ).Edge()
    guide_wire = BRepBuilderAPI_MakeWire(guide_edge).Wire()
    profile_wire = build_profile_wire(
        profile,
        start=start,
        tangent=tangent,
        surface_z=surface_z,
        width=width,
        depth=depth,
        top_margin=top_margin,
    )
    if not profile_wire.Closed():
        raise ValueError("swept slot profile must be closed")
    profile_face_builder = BRepBuilderAPI_MakeFace(profile_wire)
    if not profile_face_builder.IsDone():
        raise ValueError("swept slot profile face failed")
    pipe = BRepOffsetAPI_MakePipe(guide_wire, profile_face_builder.Face())
    pipe.Build()
    if not pipe.IsDone():
        raise ValueError("continuous slot pipe failed")
    tool = pipe.Shape()
    if not BRepCheck_Analyzer(tool).IsValid() or topology_counts(tool)["solid_count"] != 1:
        raise ValueError("continuous slot pipe is not one valid solid")
    cleanup = cleanup_audit(tool)
    if not cleanup["passed"]:
        raise ValueError(f"continuous slot tool is not cleanup-idempotent: {cleanup}")
    parameters = {
        "construction_method": "p1_c_continuous_closed_profile_sweep_cut",
        "macro_feature": macro,
        "profile_shape": profile,
        "guide_family": "straight_surface_guide",
        "guide_on_existing_planar_face": True,
        "guide_surface_z": surface_z,
        "guide_start": [start[0], start[1], surface_z],
        "guide_end": [end[0], end[1], surface_z],
        "guide_tangent": [tangent[0], tangent[1], 0.0],
        "guide_edge_count": len(list(TopologyExplorer(guide_wire).edges())),
        "guide_polyline": False,
        "profile_closed": True,
        "profile_edge_count": len(list(TopologyExplorer(profile_wire).edges())),
        "profile_width": width,
        "profile_depth": depth,
        "top_margin": top_margin,
        "sweep_path_length": path_length,
        "path_length_profile_width_ratio": path_length / width,
        "tool_cleanup": cleanup,
    }
    if parameters["path_length_profile_width_ratio"] < 4.0:
        raise ValueError("slot guide is too short relative to its section")
    role_for_face = lambda face: _tool_face_role(face, macro, parameters)
    return SweptSlotTool(tool, guide_wire, profile_wire, role_for_face, parameters)


def build_profile_wire(
    profile: str,
    *,
    start: tuple[float, float],
    tangent: tuple[float, float],
    surface_z: float,
    width: float,
    depth: float,
    top_margin: float,
) -> Any:
    half = float(width) / 2.0
    top_z = float(surface_z) + float(top_margin)
    bottom_z = float(surface_z) - float(depth)
    lateral = (-float(tangent[1]), float(tangent[0]))

    def point(offset: float, z_value: float) -> gp_Pnt:
        return gp_Pnt(start[0] + lateral[0] * offset, start[1] + lateral[1] * offset, z_value)

    if profile == "rect":
        top_left = point(-half, top_z)
        bottom_left = point(-half, bottom_z)
        bottom_right = point(half, bottom_z)
        top_right = point(half, top_z)
        return _closed_wire(
            BRepBuilderAPI_MakeEdge(top_left, bottom_left).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_left, bottom_right).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_right, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        )
    if profile == "u":
        radius = min(half, max(float(depth) - 1e-6, 1e-6))
        arc_center_z = bottom_z + radius
        top_left = point(-half, top_z)
        left_arc = point(-half, arc_center_z)
        bottom_mid = point(0.0, arc_center_z - radius)
        right_arc = point(half, arc_center_z)
        top_right = point(half, top_z)
        return _closed_wire(
            BRepBuilderAPI_MakeEdge(top_left, left_arc).Edge(),
            BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(left_arc, bottom_mid, right_arc).Value()).Edge(),
            BRepBuilderAPI_MakeEdge(right_arc, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        )
    if profile == "v":
        top_left = point(-half, top_z)
        bottom_mid = point(0.0, bottom_z)
        top_right = point(half, top_z)
        return _closed_wire(
            BRepBuilderAPI_MakeEdge(top_left, bottom_mid).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_mid, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        )
    if profile == "o":
        center = point(0.0, surface_z)
        axis = gp_Dir(float(tangent[0]), float(tangent[1]), 0.0)
        circle = GC_MakeCircle(gp_Ax2(center, axis), min(half, float(depth))).Value()
        return _closed_wire(BRepBuilderAPI_MakeEdge(circle).Edge())
    raise KeyError(profile)


def _closed_wire(*edges: Any) -> Any:
    builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        builder.Add(edge)
    if not builder.IsDone():
        raise ValueError("could not assemble closed slot profile")
    wire = builder.Wire()
    if not wire.Closed():
        raise ValueError("slot profile wire is open")
    return wire


def _tool_face_role(face: Any, macro: str, parameters: dict[str, Any]) -> str:
    side_role, bottom_role = SLOT_ROLES[macro]
    center = face_center(face)
    start = parameters["guide_start"]
    end = parameters["guide_end"]
    terminal_tolerance = float(parameters["profile_width"]) * 0.22
    if _xy_distance(center, start) <= terminal_tolerance or _xy_distance(center, end) <= terminal_tolerance:
        return side_role
    surface = BRepAdaptor_Surface(face)
    profile = str(parameters["profile_shape"])
    surface_z = float(parameters["guide_surface_z"])
    depth = float(parameters["profile_depth"])
    if profile == "rect" and surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and center[2] < surface_z - depth * 0.65:
            return bottom_role
    if profile == "u" and surface.GetType() == GeomAbs_Cylinder:
        return bottom_role
    if profile == "v" and center[2] < surface_z - depth * 0.25:
        return bottom_role
    if profile == "o" and surface.GetType() == GeomAbs_Cylinder:
        return bottom_role
    return side_role


def _xy_distance(center: tuple[float, float, float], point: list[float]) -> float:
    return math.hypot(center[0] - float(point[0]), center[1] - float(point[1]))


__all__ = ["SLOT_PROFILE_BY_MACRO", "SweptSlotTool", "build_profile_wire", "build_swept_slot_tool"]
