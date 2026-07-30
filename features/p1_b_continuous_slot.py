from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from features.base_feature import BaseFeatureGenerator


PROFILE_SHAPES = ("rect", "u", "v", "o")
GUIDE_END_COMBINATIONS = (
    ("straight", "open_open"),
    ("straight", "open_closed"),
    ("curved_open", "open_open"),
    ("curved_open", "open_closed"),
    ("curved_closed", "closed_loop"),
)
PROFILE_TO_MACRO = {
    "rect": "slot_rect",
    "u": "slot_u",
    "v": "slot_v",
    "o": "slot_o",
}
PROFILE_TO_ROLES = {
    "rect": ("slot_rect_side_plane", "slot_rect_bottom_plane"),
    "u": ("slot_u_side_plane", "slot_u_bottom_plane"),
    "v": ("slot_v_side_plane", "slot_v_bottom_plane"),
    "o": ("slot_o_side_plane", "slot_o_bottom_plane"),
}
TERMINAL_ROLE = "slot_terminal_face"


@dataclass(frozen=True)
class SlotBuildResult:
    shape: Any
    tool: Any
    guide_wire: Any
    profile_wire: Any
    parameters: dict[str, Any]


def all_slot_combinations() -> list[tuple[str, str, str]]:
    return [
        (profile_shape, guide_family, end_condition)
        for profile_shape in PROFILE_SHAPES
        for guide_family, end_condition in GUIDE_END_COMBINATIONS
    ]


def build_continuous_slot_cut(
    stock: Any,
    *,
    L: float,
    W: float,
    H: float,
    profile_shape: str,
    guide_family: str,
    end_condition: str,
    seed: int,
    instance_id: int = 1,
    host_profile: dict[str, Any] | None = None,
) -> SlotBuildResult:
    profile_shape = _normalize_profile_shape(profile_shape)
    _validate_combination(profile_shape, guide_family, end_condition)
    rng = random.Random(int(seed))
    host = _host_frame(L, W, H, host_profile or {})
    width = min(float(L), float(W)) * rng.uniform(0.070, 0.095)
    width = max(width, 5.0)
    if profile_shape == "o":
        depth = width / 2.0
    else:
        depth = width * rng.uniform(0.72, 0.96)
    top_margin = max(0.8, width * 0.15)
    guide_wire, guide = build_continuous_guide(
        host,
        guide_family=guide_family,
        end_condition=end_condition,
        profile_width=width,
        rng=rng,
    )
    profile_wire = build_slot_profile_wire(
        profile_shape=profile_shape,
        start_xy=guide["tool_start_xy"],
        tangent_angle=float(guide["start_tangent_angle"]),
        surface_z=float(H),
        width=width,
        depth=depth,
        top_margin=top_margin,
    )
    profile_face = BRepBuilderAPI_MakeFace(profile_wire).Face()
    pipe = BRepOffsetAPI_MakePipe(guide_wire, profile_face)
    pipe.Build()
    if not pipe.IsDone():
        raise ValueError("continuous slot pipe construction failed")
    tool = pipe.Shape()
    if not BRepCheck_Analyzer(tool).IsValid():
        raise ValueError("continuous slot tool is not a valid B-Rep")

    cut = BRepAlgoAPI_Cut(stock, tool)
    cut.Build()
    if not cut.IsDone():
        raise ValueError("continuous slot boolean cut failed")
    shape = cut.Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("continuous slot result is not a valid B-Rep")
    if len(list(TopologyExplorer(shape).solids())) != 1:
        raise ValueError("continuous slot result must contain exactly one solid")

    macro_key = PROFILE_TO_MACRO[profile_shape]
    params: dict[str, Any] = {
        "inst": int(instance_id),
        "macro_key": macro_key,
        "construction_method": "p1_b_continuous_closed_profile_sweep_cut",
        "profile_shape": profile_shape,
        "guide_family": guide_family,
        "end_condition": end_condition,
        "guide_curve_kind": guide["guide_curve_kind"],
        "guide_continuity": "C_infinity",
        "guide_edge_count": _edge_count(guide_wire),
        "profile_edge_count": _edge_count(profile_wire),
        "host_face_kind": "existing_planar_face",
        "host_local_frame": host["local_frame"],
        "host_boundary_kind": host["boundary_kind"],
        "path_start_on_boundary": bool(guide["path_start_on_boundary"]),
        "path_end_on_boundary": bool(guide["path_end_on_boundary"]),
        "path_closed": bool(guide["path_closed"]),
        "path_length": float(guide["path_length"]),
        "sweep_path_length": float(guide["path_length"]),
        "profile_width": float(width),
        "profile_depth": float(depth),
        "path_length_profile_width_ratio": float(guide["path_length"]) / width,
        "w": float(width),
        "l": float(guide["path_length"]),
        "depth": float(depth),
        "top_margin": float(top_margin),
        "seed": int(seed),
        "visible_start_xy": guide["visible_start_xy"],
        "visible_end_xy": guide["visible_end_xy"],
        "tool_start_xy": guide["tool_start_xy"],
        "tool_end_xy": guide["tool_end_xy"],
        "terminal_point_xy": guide.get("terminal_point_xy"),
        "guide_definition": guide["serializable_definition"],
    }
    face_role_by_hash = _modified_tool_face_roles(cut, tool, params, float(H))
    params["_face_role_by_hash"] = face_role_by_hash
    params["_feature_face_hashes"] = sorted(face_role_by_hash)
    params["feature_face_count"] = len(face_role_by_hash)
    params["terminal_face_count"] = sum(role == TERMINAL_ROLE for role in face_role_by_hash.values())
    params["guide_audit"] = inspect_guide_wire(guide_wire, guide["guide_curve_kind"])
    return SlotBuildResult(shape, tool, guide_wire, profile_wire, params)


def build_continuous_guide(
    host: dict[str, Any],
    *,
    guide_family: str,
    end_condition: str,
    profile_width: float,
    rng: random.Random,
) -> tuple[Any, dict[str, Any]]:
    axis = rng.choice(("local_u", "local_v"))
    primary_half, cross_half = _host_half_extents(host, axis)
    overrun = max(float(profile_width) * 2.75, 2.0)
    if host["boundary_kind"] == "circle":
        lateral_offset = 0.0
    else:
        lateral_offset = rng.uniform(-0.08, 0.08) * cross_half

    if guide_family == "straight":
        visible_end_s = primary_half if end_condition == "open_open" else primary_half * rng.uniform(0.30, 0.55)
        visible_start = _local_to_global(host, axis, -primary_half, lateral_offset)
        visible_end = _local_to_global(host, axis, visible_end_s, lateral_offset)
        tool_start = _local_to_global(host, axis, -primary_half - overrun, lateral_offset)
        tool_end_s = primary_half + overrun if end_condition == "open_open" else visible_end_s
        tool_end = _local_to_global(host, axis, tool_end_s, lateral_offset)
        edge = BRepBuilderAPI_MakeEdge(
            gp_Pnt(tool_start[0], tool_start[1], host["z"]),
            gp_Pnt(tool_end[0], tool_end[1], host["z"]),
        ).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        tangent = _vector_angle(_local_vector_to_global(axis, 1.0, 0.0))
        path_length = math.dist(tool_start, tool_end)
        definition = {
            "type": "line",
            "axis": axis,
            "lateral_offset": lateral_offset,
            "visible_start_xy": visible_start,
            "visible_end_xy": visible_end,
            "tool_start_xy": tool_start,
            "tool_end_xy": tool_end,
        }
        return wire, {
            "guide_curve_kind": "line",
            "visible_start_xy": visible_start,
            "visible_end_xy": visible_end,
            "tool_start_xy": tool_start,
            "tool_end_xy": tool_end,
            "terminal_point_xy": visible_end if end_condition == "open_closed" else None,
            "start_tangent_angle": tangent,
            "path_start_on_boundary": True,
            "path_end_on_boundary": end_condition == "open_open",
            "path_closed": False,
            "path_length": path_length,
            "serializable_definition": definition,
        }

    if guide_family == "curved_open":
        sagitta = min(cross_half * rng.uniform(0.18, 0.27), primary_half * 0.30)
        center_t = lateral_offset + (sagitta * sagitta - primary_half * primary_half) / (2.0 * sagitta)
        radius = sagitta + lateral_offset - center_t
        right_angle = math.atan2(lateral_offset - center_t, primary_half)
        left_angle = math.pi - right_angle
        end_angle = right_angle if end_condition == "open_open" else (right_angle + math.pi / 2.0) / 2.0
        delta = min(overrun / radius, 0.22)
        tool_start_angle = left_angle + delta
        tool_end_angle = right_angle - delta if end_condition == "open_open" else end_angle
        visible_start_local = _circle_point(0.0, center_t, radius, left_angle)
        visible_end_local = _circle_point(0.0, center_t, radius, end_angle)
        tool_start_local = _circle_point(0.0, center_t, radius, tool_start_angle)
        tool_end_local = _circle_point(0.0, center_t, radius, tool_end_angle)
        mid_local = _circle_point(0.0, center_t, radius, math.pi / 2.0)
        visible_start = _local_to_global(host, axis, *visible_start_local)
        visible_end = _local_to_global(host, axis, *visible_end_local)
        tool_start = _local_to_global(host, axis, *tool_start_local)
        tool_end = _local_to_global(host, axis, *tool_end_local)
        mid = _local_to_global(host, axis, *mid_local)
        arc = GC_MakeArcOfCircle(
            gp_Pnt(tool_start[0], tool_start[1], host["z"]),
            gp_Pnt(mid[0], mid[1], host["z"]),
            gp_Pnt(tool_end[0], tool_end[1], host["z"]),
        ).Value()
        wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(arc).Edge()).Wire()
        local_tangent = (math.sin(tool_start_angle), -math.cos(tool_start_angle))
        tangent = _vector_angle(_local_vector_to_global(axis, *local_tangent))
        path_length = radius * ((tool_start_angle - tool_end_angle) % (2.0 * math.pi))
        center_xy = _local_to_global(host, axis, 0.0, center_t)
        definition = {
            "type": "circular_arc",
            "axis": axis,
            "circle_center_xy": center_xy,
            "circle_radius": radius,
            "orientation": "clockwise",
            "tool_start_angle": tool_start_angle,
            "tool_end_angle": tool_end_angle,
            "visible_start_xy": visible_start,
            "visible_end_xy": visible_end,
            "tool_start_xy": tool_start,
            "tool_end_xy": tool_end,
        }
        return wire, {
            "guide_curve_kind": "circular_arc",
            "visible_start_xy": visible_start,
            "visible_end_xy": visible_end,
            "tool_start_xy": tool_start,
            "tool_end_xy": tool_end,
            "terminal_point_xy": visible_end if end_condition == "open_closed" else None,
            "start_tangent_angle": tangent,
            "path_start_on_boundary": True,
            "path_end_on_boundary": end_condition == "open_open",
            "path_closed": False,
            "path_length": path_length,
            "serializable_definition": definition,
        }

    radius = min(primary_half, cross_half) * rng.uniform(0.38, 0.50)
    max_offset = max(0.0, min(primary_half, cross_half) - radius - profile_width)
    center_s = rng.uniform(-0.15, 0.15) * max_offset
    center_t = rng.uniform(-0.15, 0.15) * max_offset
    center_xy = _local_to_global(host, axis, center_s, center_t)
    circle = GC_MakeCircle(
        gp_Ax2(gp_Pnt(center_xy[0], center_xy[1], host["z"]), gp_Dir(0.0, 0.0, 1.0)),
        radius,
    ).Value()
    wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()
    start_xy = (center_xy[0] + radius, center_xy[1])
    definition = {
        "type": "circle",
        "axis": axis,
        "circle_center_xy": center_xy,
        "circle_radius": radius,
        "orientation": "counterclockwise",
        "tool_start_xy": start_xy,
    }
    return wire, {
        "guide_curve_kind": "circle",
        "visible_start_xy": start_xy,
        "visible_end_xy": start_xy,
        "tool_start_xy": start_xy,
        "tool_end_xy": start_xy,
        "terminal_point_xy": None,
        "start_tangent_angle": math.pi / 2.0,
        "path_start_on_boundary": False,
        "path_end_on_boundary": False,
        "path_closed": True,
        "path_length": 2.0 * math.pi * radius,
        "serializable_definition": definition,
    }


def build_slot_profile_wire(
    *,
    profile_shape: str,
    start_xy: tuple[float, float],
    tangent_angle: float,
    surface_z: float,
    width: float,
    depth: float,
    top_margin: float,
) -> Any:
    profile_shape = _normalize_profile_shape(profile_shape)
    half_width = float(width) / 2.0
    top_z = float(surface_z) + float(top_margin)
    bottom_z = float(surface_z) - float(depth)
    lateral = (-math.sin(tangent_angle), math.cos(tangent_angle))

    def point(offset: float, z_value: float) -> gp_Pnt:
        return gp_Pnt(
            float(start_xy[0]) + lateral[0] * offset,
            float(start_xy[1]) + lateral[1] * offset,
            z_value,
        )

    if profile_shape == "rect":
        points = (
            point(-half_width, top_z),
            point(-half_width, bottom_z),
            point(half_width, bottom_z),
            point(half_width, top_z),
        )
        return _wire_from_edges(
            BRepBuilderAPI_MakeEdge(points[0], points[1]).Edge(),
            BRepBuilderAPI_MakeEdge(points[1], points[2]).Edge(),
            BRepBuilderAPI_MakeEdge(points[2], points[3]).Edge(),
            BRepBuilderAPI_MakeEdge(points[3], points[0]).Edge(),
        )
    if profile_shape == "u":
        radius = min(half_width, max(float(depth) - 1e-6, 1e-6))
        center_z = bottom_z + radius
        top_left = point(-half_width, top_z)
        left_arc_end = point(-half_width, center_z)
        bottom_mid = point(0.0, center_z - radius)
        right_arc_end = point(half_width, center_z)
        top_right = point(half_width, top_z)
        return _wire_from_edges(
            BRepBuilderAPI_MakeEdge(top_left, left_arc_end).Edge(),
            BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(left_arc_end, bottom_mid, right_arc_end).Value()).Edge(),
            BRepBuilderAPI_MakeEdge(right_arc_end, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        )
    if profile_shape == "v":
        top_left = point(-half_width, top_z)
        bottom_mid = point(0.0, bottom_z)
        top_right = point(half_width, top_z)
        return _wire_from_edges(
            BRepBuilderAPI_MakeEdge(top_left, bottom_mid).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_mid, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        )
    radius = min(half_width, max(float(depth), 1e-6))
    center = point(0.0, float(surface_z))
    circle = GC_MakeCircle(
        gp_Ax2(center, gp_Dir(math.cos(tangent_angle), math.sin(tangent_angle), 0.0)),
        radius,
    ).Value()
    return _wire_from_edges(BRepBuilderAPI_MakeEdge(circle).Edge())


def inspect_guide_wire(guide_wire: Any, expected_curve_kind: str) -> dict[str, Any]:
    edges = list(TopologyExplorer(guide_wire).edges())
    curve_types = []
    continuity_values = []
    for edge in edges:
        curve = BRepAdaptor_Curve(edge)
        curve_types.append(_curve_type_name(curve.GetType()))
        continuity_values.append(int(curve.Continuity()))
    expected_type = "line" if expected_curve_kind == "line" else "circle"
    return {
        "edge_count": len(edges),
        "curve_types": curve_types,
        "continuity_values": continuity_values,
        "single_edge": len(edges) == 1,
        "analytic_curve_type_ok": curve_types == [expected_type],
        "at_least_c1": bool(continuity_values) and min(continuity_values) >= 2,
        "polyline": False,
    }


def feature_face_hashes(parameters: dict[str, Any]) -> set[int]:
    return {int(value) for value in parameters.get("_feature_face_hashes", [])}


class P1BContinuousSlotGenerator(BaseFeatureGenerator):
    supported_profile_shapes = PROFILE_SHAPES
    profile_to_macro_key = PROFILE_TO_MACRO
    guide_end_combinations = GUIDE_END_COMBINATIONS

    def __init__(self, semantic_subtype_index: dict[str, Any]):
        super().__init__(semantic_subtype_index)
        self.feature_name = "swept_slot"
        aliases = ["stock_face", TERMINAL_ROLE]
        for side_role, bottom_role in PROFILE_TO_ROLES.values():
            aliases.extend((side_role, bottom_role))
        self.semantic_map = self.build_semantic_map(aliases)

    def build_one(
        self,
        stock: Any,
        L: float,
        W: float,
        H: float,
        *,
        profile_shape: str,
        guide_family: str,
        end_condition: str,
        seed: int,
        instance_id: int = 1,
    ) -> SlotBuildResult:
        return build_continuous_slot_cut(
            stock,
            L=L,
            W=W,
            H=H,
            profile_shape=profile_shape,
            guide_family=guide_family,
            end_condition=end_condition,
            seed=seed,
            instance_id=instance_id,
            host_profile=self.generation_context.get("current_stock_profile", {}),
        )

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        occupied_zones = occupied_zones if occupied_zones is not None else []
        params_list = []
        actual_added = 0
        combinations = all_slot_combinations()
        for _ in range(int(num_instances)):
            for _attempt in range(80):
                profile_shape, guide_family, end_condition = random.choice(combinations)
                seed = random.randrange(1, 2**31 - 1)
                try:
                    result = self.build_one(
                        stock,
                        L,
                        W,
                        H,
                        profile_shape=profile_shape,
                        guide_family=guide_family,
                        end_condition=end_condition,
                        seed=seed,
                        instance_id=start_inst_id + actual_added,
                    )
                except (RuntimeError, ValueError):
                    continue
                stock = result.shape
                params_list.append(result.parameters)
                occupied_zones.append({"cx": 0.0, "cy": 0.0, "radius": max(float(L), float(W))})
                actual_added += 1
                break
        return stock, actual_added, occupied_zones, params_list

    def match_face_hash(self, face_hash: int, params_list: list[dict[str, Any]]):
        for params in params_list:
            role = params.get("_face_role_by_hash", {}).get(int(face_hash))
            if role is None:
                continue
            return self.semantic_map[role], int(params["inst"]), str(params["macro_key"])
        return None

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for params in params_list:
            role = _fallback_face_role(params, surf, pnt, float(H))
            if role is not None:
                return self.semantic_map[role], int(params["inst"]), str(params["macro_key"])
        return None


def _modified_tool_face_roles(cut: Any, tool: Any, params: dict[str, Any], H: float) -> dict[int, str]:
    roles: dict[int, str] = {}
    for tool_face in TopologyExplorer(tool).faces():
        role = _tool_face_role(tool_face, params, H)
        if role is None:
            continue
        for modified in cut.Modified(tool_face):
            if modified.ShapeType() == 4:
                roles[hash(modified)] = role
    return roles


def _tool_face_role(face: Any, params: dict[str, Any], H: float) -> str | None:
    surface = BRepAdaptor_Surface(face)
    center = _face_center(face)
    profile = str(params["profile_shape"])
    terminal = params.get("terminal_point_xy")
    if terminal is not None:
        distance = math.hypot(center[0] - float(terminal[0]), center[1] - float(terminal[1]))
        if distance <= float(params["profile_width"]) * 0.20 and H - params["profile_depth"] - 0.2 <= center[2] <= H + params["top_margin"] + 0.2:
            return TERMINAL_ROLE
    if center[2] > H + 0.05:
        return None
    side_role, bottom_role = PROFILE_TO_ROLES[profile]
    if profile == "rect" and surface.GetType() == GeomAbs_Plane:
        normal = surface.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and center[2] < H - float(params["profile_depth"]) * 0.70:
            return bottom_role
    if profile == "u" and center[2] < H - float(params["profile_depth"]) * 0.45:
        return bottom_role
    return side_role


def _fallback_face_role(params: dict[str, Any], surf: Any, pnt: Any, H: float) -> str | None:
    px, py, pz = float(pnt.X()), float(pnt.Y()), float(pnt.Z())
    if surf.GetType() == GeomAbs_Plane:
        normal = surf.Plane().Axis().Direction()
        if abs(normal.Z()) > 0.9 and abs(pz - H) < 0.1:
            return None
    terminal = params.get("terminal_point_xy")
    if terminal is not None and math.hypot(px - terminal[0], py - terminal[1]) <= params["profile_width"] * 0.25:
        return TERMINAL_ROLE
    if not _point_near_guide(params["guide_definition"], px, py, params["profile_width"] / 2.0 + 0.35):
        return None
    if not H - params["profile_depth"] - 0.35 <= pz <= H + params["top_margin"] + 0.35:
        return None
    side_role, bottom_role = PROFILE_TO_ROLES[str(params["profile_shape"])]
    if params["profile_shape"] == "rect" and pz < H - params["profile_depth"] * 0.70:
        return bottom_role
    if params["profile_shape"] == "u" and pz < H - params["profile_depth"] * 0.45:
        return bottom_role
    return side_role


def _point_near_guide(definition: dict[str, Any], px: float, py: float, tolerance: float) -> bool:
    kind = definition["type"]
    if kind == "line":
        return _point_segment_distance((px, py), definition["tool_start_xy"], definition["tool_end_xy"]) <= tolerance
    center = definition["circle_center_xy"]
    radial_error = abs(math.hypot(px - center[0], py - center[1]) - definition["circle_radius"])
    if radial_error > tolerance:
        return False
    if kind == "circle":
        return True
    theta = math.atan2(py - center[1], px - center[0])
    start = float(definition["tool_start_angle"])
    end = float(definition["tool_end_angle"])
    return (start - theta) % (2.0 * math.pi) <= (start - end) % (2.0 * math.pi) + 1e-6


def _host_frame(L: float, W: float, H: float, host_profile: dict[str, Any]) -> dict[str, Any]:
    stock_type = host_profile.get("stock_type")
    dimensions = host_profile.get("dimensions", {})
    if stock_type == "cylinder" and float(dimensions.get("radius", 0.0)) > 0.0:
        radius = float(dimensions["radius"])
        return {
            "z": float(H),
            "boundary_kind": "circle",
            "radius": radius,
            "half_u": radius,
            "half_v": radius,
            "local_frame": {"origin": [0.0, 0.0, float(H)], "normal": [0.0, 0.0, 1.0], "u": [1.0, 0.0, 0.0], "v": [0.0, 1.0, 0.0]},
        }
    return {
        "z": float(H),
        "boundary_kind": "rectangle",
        "half_u": float(L) / 2.0,
        "half_v": float(W) / 2.0,
        "local_frame": {"origin": [0.0, 0.0, float(H)], "normal": [0.0, 0.0, 1.0], "u": [1.0, 0.0, 0.0], "v": [0.0, 1.0, 0.0]},
    }


def _host_half_extents(host: dict[str, Any], axis: str) -> tuple[float, float]:
    if axis == "local_u":
        return float(host["half_u"]), float(host["half_v"])
    return float(host["half_v"]), float(host["half_u"])


def _local_to_global(host: dict[str, Any], axis: str, s: float, t: float) -> tuple[float, float]:
    del host
    if axis == "local_u":
        return float(s), float(t)
    return float(-t), float(s)


def _local_vector_to_global(axis: str, s: float, t: float) -> tuple[float, float]:
    if axis == "local_u":
        return float(s), float(t)
    return float(-t), float(s)


def _circle_point(center_s: float, center_t: float, radius: float, angle: float) -> tuple[float, float]:
    return center_s + radius * math.cos(angle), center_t + radius * math.sin(angle)


def _wire_from_edges(*edges: Any) -> Any:
    return BRepBuilderAPI_MakeWire(*edges).Wire()


def _face_center(face: Any) -> tuple[float, float, float]:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    point = props.CentreOfMass()
    return float(point.X()), float(point.Y()), float(point.Z())


def _edge_count(shape: Any) -> int:
    return len(list(TopologyExplorer(shape).edges()))


def _curve_type_name(curve_type: Any) -> str:
    if curve_type == GeomAbs_Line:
        return "line"
    if curve_type == GeomAbs_Circle:
        return "circle"
    return f"curve_type_{int(curve_type)}"


def _normalize_profile_shape(profile_shape: str) -> str:
    normalized = str(profile_shape).strip().lower()
    if normalized == "rectangle":
        normalized = "rect"
    if normalized not in PROFILE_SHAPES:
        raise ValueError(f"unsupported slot profile: {profile_shape}")
    return normalized


def _validate_combination(profile_shape: str, guide_family: str, end_condition: str) -> None:
    if profile_shape not in PROFILE_SHAPES:
        raise ValueError(profile_shape)
    if (guide_family, end_condition) not in GUIDE_END_COMBINATIONS:
        raise ValueError(f"unsupported guide/end combination: {(guide_family, end_condition)}")


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.dist(point, start)
    u = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
    u = min(1.0, max(0.0, u))
    closest = (start[0] + u * dx, start[1] + u * dy)
    return math.dist(point, closest)


def _vector_angle(vector: tuple[float, float]) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


__all__ = [
    "GUIDE_END_COMBINATIONS",
    "P1BContinuousSlotGenerator",
    "PROFILE_SHAPES",
    "PROFILE_TO_MACRO",
    "PROFILE_TO_ROLES",
    "SlotBuildResult",
    "TERMINAL_ROLE",
    "all_slot_combinations",
    "build_continuous_guide",
    "build_continuous_slot_cut",
    "build_slot_profile_wire",
    "feature_face_hashes",
    "inspect_guide_wire",
]
