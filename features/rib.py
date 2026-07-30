import math
import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator


class RibGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "rib"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "rib_side_plane",
            "rib_top_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_placement_attempts = 20

        for _ in range(num_instances):
            for _placement_attempt in range(max_placement_attempts):
                rib_t = random.uniform(min_base_size * 0.015, min_base_size * 0.035)
                rib_h = random.uniform(H * 0.25, H * 0.55)
                placement_mode = _choose_rib_placement_mode(self.generation_context)
                extra_params = {}

                if placement_mode == "cavity":
                    cavity_params = _sample_cavity_rib_params(rib_t, self.generation_context)
                    if cavity_params is None:
                        continue
                    rib_t = cavity_params["t"]
                    rib_h = cavity_params["h"]
                    rib_l = cavity_params["l"]
                    anchor_x = cavity_params["anchor_x"]
                    anchor_y = cavity_params["anchor_y"]
                    angle = cavity_params["angle"]
                    feature_cx = cavity_params["cx"]
                    feature_cy = cavity_params["cy"]
                    side = "cavity"
                    footprint_points = None
                    extra_params = {
                        "placement_mode": "cavity",
                        "support_context": cavity_params["support_context"],
                        "rib_body_kind": "cavity_wall_rib",
                        "support_source": "existing_pocket_context",
                        "parent_macro_feature": cavity_params["parent_macro_feature"],
                        "parent_instance_id": cavity_params["parent_instance_id"],
                        "connects_main_faces": True,
                        "cavity_type": cavity_params["cavity_type"],
                        "bottom_z": cavity_params["bottom_z"],
                        "top_z": cavity_params["top_z"],
                        "span_axis": cavity_params["span_axis"],
                    }
                elif placement_mode == "corner":
                    corner_params = _sample_corner_rib_params(
                        L,
                        W,
                        rib_t,
                        min_base_size,
                        self.generation_context.get("current_stock_profile", {}),
                    )
                    if corner_params is None:
                        continue
                    rib_h = random.uniform(corner_params["wall_h"] * 0.42, corner_params["wall_h"] * 0.68)
                    corner_params["t"] = rib_t
                    corner_params["h"] = rib_h
                    rib_l = corner_params["l"]
                    anchor_x = corner_params["anchor_x"]
                    anchor_y = corner_params["anchor_y"]
                    angle = corner_params["angle"]
                    feature_cx = corner_params["cx"]
                    feature_cy = corner_params["cy"]
                    side = corner_params["side"]
                    footprint_points = None
                    extra_params = {
                        "placement_mode": "corner",
                        "support_context": "l_bracket_base_wall",
                        "rib_body_kind": "right_triangle_gusset",
                        "support_plate_count": 2,
                        "corner_sides": corner_params["corner_sides"],
                        "connects_main_faces": True,
                        "support_source": "existing_l_plate_stock",
                        "background_wall_side": side,
                        "wall_t": corner_params["wall_t"],
                        "wall_l": corner_params["wall_l"],
                        "wall_h": corner_params["wall_h"],
                        "wall_inner_x": corner_params["wall_inner_x"],
                        "wall_inner_y": corner_params["wall_inner_y"],
                        "inward_x": corner_params["inward_x"],
                        "inward_y": corner_params["inward_y"],
                        "anchor_span": corner_params["anchor_span"],
                    }
                else:
                    side = random.choice(["left", "right", "bottom", "top"])

                    if side in {"left", "right"}:
                        max_l = max(20.0, W * 0.65)
                        rib_l = random.uniform(max(rib_t * 8.0, 20.0), max(rib_t * 14.0, min(W * 0.82, max_l)))
                        half_span = rib_l / 2.0
                        anchor_x = -L / 2.0 if side == "left" else L / 2.0
                        anchor_y = random.uniform(-W / 2.0 + half_span + 4.0, W / 2.0 - half_span - 4.0)
                        angle = 0.0 if side == "left" else math.pi
                        feature_cx = anchor_x + (rib_l / 2.0 if side == "left" else -rib_l / 2.0)
                        feature_cy = anchor_y
                    else:
                        max_l = max(20.0, L * 0.65)
                        rib_l = random.uniform(max(rib_t * 8.0, 20.0), max(rib_t * 14.0, min(L * 0.82, max_l)))
                        half_span = rib_l / 2.0
                        anchor_x = random.uniform(-L / 2.0 + half_span + 4.0, L / 2.0 - half_span - 4.0)
                        anchor_y = -W / 2.0 if side == "bottom" else W / 2.0
                        angle = math.pi / 2.0 if side == "bottom" else -math.pi / 2.0
                        feature_cx = anchor_x
                        feature_cy = anchor_y + (rib_l / 2.0 if side == "bottom" else -rib_l / 2.0)
                    footprint_points = None
                    extra_params = {"placement_mode": "boundary", "connects_main_faces": False}

                if placement_mode == "corner":
                    bounding_radius = max(rib_l, corner_params["wall_l"] / 2.0) + rib_t
                elif placement_mode == "cavity":
                    bounding_radius = math.hypot(rib_l / 2.0, rib_t / 2.0)
                else:
                    bounding_radius = math.hypot(rib_l / 2.0, rib_t / 2.0)
                if placement_mode != "cavity" and any(
                    math.hypot(feature_cx - z["cx"], feature_cy - z["cy"]) < (bounding_radius + z["radius"] + 2.0)
                    for z in occupied_zones
                ):
                    continue

                if placement_mode == "cavity":
                    rib_tool = _build_cavity_wall_rib_tool(cavity_params)
                    fuse = BRepAlgoAPI_Fuse(stock, rib_tool)
                    fuse.Build()
                elif placement_mode == "corner":
                    gusset_tool = _build_right_triangle_gusset_tool(corner_params, H)
                    fuse = BRepAlgoAPI_Fuse(stock, gusset_tool)
                    fuse.Build()
                else:
                    poly = BRepBuilderAPI_MakePolygon()
                    local_pts = [
                        (0.0, -rib_t / 2.0),
                        (rib_l, -rib_t / 2.0),
                        (rib_l, rib_t / 2.0),
                        (0.0, rib_t / 2.0),
                    ]
                    for px, py in local_pts:
                        rx = px * math.cos(angle) - py * math.sin(angle)
                        ry = px * math.sin(angle) + py * math.cos(angle)
                        poly.Add(gp_Pnt(anchor_x + rx, anchor_y + ry, H))
                    poly.Close()

                    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
                    tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, rib_h)).Shape()
                    fuse = BRepAlgoAPI_Fuse(stock, tool)
                    fuse.Build()
                if not fuse.IsDone():
                    continue

                stock = fuse.Shape()
                inst_id = start_inst_id + actual_added
                params = {
                    "inst": inst_id,
                    "anchor_x": anchor_x,
                    "anchor_y": anchor_y,
                    "cx": feature_cx,
                    "cy": feature_cy,
                    "t": rib_t,
                    "l": rib_l,
                    "h": rib_h,
                    "angle": angle,
                    "side": side,
                    "radius": bounding_radius,
                }
                params.update(extra_params)
                params_list.append(params)
                occupied_zones.append({"cx": feature_cx, "cy": feature_cy, "radius": bounding_radius})
                actual_added += 1
                break

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["radius"] + 0.2:
                continue

            if hp.get("rib_body_kind") == "right_triangle_gusset":
                if _point_inside_right_triangle_gusset(hp, px, py, pz, H, tolerance=0.2):
                    return self.semantic_map["rib_side_plane"], hp["inst"], self.feature_name
                continue
            if hp.get("rib_body_kind") == "cavity_wall_rib":
                if not _point_inside_cavity_wall_rib_footprint(hp, px, py, tolerance=0.2):
                    continue
                if not (hp["bottom_z"] - 0.2 <= pz <= hp["top_z"] + 0.2):
                    continue
                if abs(n.Z()) > 0.99 and abs(pz - hp["top_z"]) < 0.15:
                    return self.semantic_map["rib_top_plane"], hp["inst"], self.feature_name
                if abs(n.Z()) < 1e-4:
                    return self.semantic_map["rib_side_plane"], hp["inst"], self.feature_name
                continue

            dx = px - hp["anchor_x"]
            dy = py - hp["anchor_y"]
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])

            half_t = hp["t"] / 2.0
            inside_rect = -0.1 <= loc_x <= hp["l"] + 0.1 and abs(loc_y) <= half_t + 0.1

            if abs(n.Z()) > 0.99 and inside_rect and abs(pz - (H + hp["h"])) < 0.1:
                return self.semantic_map["rib_top_plane"], hp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and (H - 0.1) <= pz <= (H + hp["h"] + 0.1):
                on_long_side = abs(abs(loc_y) - half_t) < 0.1 and 0.0 <= loc_x <= hp["l"] + 0.1
                on_tip = abs(loc_x - hp["l"]) < 0.1 and abs(loc_y) <= half_t + 0.1
                if inside_rect and (on_long_side or on_tip):
                    return self.semantic_map["rib_side_plane"], hp["inst"], self.feature_name

        return None


def _choose_rib_placement_mode(generation_context: dict) -> str:
    policy = generation_context.get("rib_features", {})
    placement_modes = list(policy.get("placement_modes", ["boundary"]))
    if placement_modes == ["cavity"]:
        return "cavity"
    if placement_modes == ["corner"]:
        return "corner"
    if (
        "cavity" in placement_modes
        and generation_context.get("cavity_rib_contexts")
        and random.random() < float(policy.get("cavity_probability", 0.0))
    ):
        return "cavity"
    if "corner" in placement_modes and random.random() < float(policy.get("corner_probability", 0.0)):
        return "corner"
    return "boundary"


def _sample_corner_rib_params(
    L: float,
    W: float,
    rib_t: float,
    min_base_size: float,
    stock_profile: dict | None = None,
) -> dict | None:
    stock_profile = stock_profile or {}
    rib_context = stock_profile.get("rib_context", {})
    if stock_profile.get("stock_type") != "l_plate" or rib_context.get("type") != "l_plate_corner":
        return None

    lower_l = max(float(rib_t) * 8.0, float(min_base_size) * 0.22, float(rib_context["wall_h"]) * 0.50)
    upper_l = min(float(rib_context["wall_h"]) * 0.86, min(float(L), float(W)) * 0.42)
    if upper_l <= lower_l:
        return None

    rib_l = random.uniform(lower_l, upper_l)
    anchor_min, anchor_max = rib_context["anchor_span_range"]
    if float(anchor_max) <= float(anchor_min):
        return None
    anchor_span = random.uniform(float(anchor_min), float(anchor_max))
    side = rib_context["wall_side"]
    side_params = _l_plate_context_to_side_params(rib_context, anchor_span)

    return {
        "l": rib_l,
        "anchor_x": side_params["anchor_x"],
        "anchor_y": side_params["anchor_y"],
        "angle": side_params["angle"],
        "cx": side_params["cx"] + side_params["inward_x"] * rib_l / 3.0,
        "cy": side_params["cy"] + side_params["inward_y"] * rib_l / 3.0,
        "side": side,
        "corner_sides": ["base_plate", "vertical_wall"],
        "wall_t": float(rib_context["wall_t"]),
        "wall_l": float(rib_context["wall_l"]),
        "wall_h": float(rib_context["wall_h"]),
        "wall_inner_x": side_params["wall_inner_x"],
        "wall_inner_y": side_params["wall_inner_y"],
        "inward_x": side_params["inward_x"],
        "inward_y": side_params["inward_y"],
        "anchor_span": side_params["anchor_span"],
    }


def _l_plate_context_to_side_params(rib_context: dict, anchor_span: float) -> dict:
    side = rib_context["wall_side"]
    if side in {"left", "right"}:
        x_inner = float(rib_context["wall_inner_x"])
        inward_x = float(rib_context["inward_x"])
        return {
            "anchor_x": x_inner,
            "anchor_y": anchor_span,
            "angle": 0.0 if side == "left" else math.pi,
            "cx": x_inner,
            "cy": anchor_span,
            "wall_inner_x": x_inner,
            "wall_inner_y": anchor_span,
            "inward_x": inward_x,
            "inward_y": 0.0,
            "anchor_span": anchor_span,
        }

    y_inner = float(rib_context["wall_inner_y"])
    inward_y = float(rib_context["inward_y"])
    return {
        "anchor_x": anchor_span,
        "anchor_y": y_inner,
        "angle": math.pi / 2.0 if side == "bottom" else -math.pi / 2.0,
        "cx": anchor_span,
        "cy": y_inner,
        "wall_inner_x": anchor_span,
        "wall_inner_y": y_inner,
        "inward_x": 0.0,
        "inward_y": inward_y,
        "anchor_span": anchor_span,
    }


def _build_right_triangle_gusset_tool(params: dict, H: float):
    side = params["side"]
    rib_l = float(params["l"])
    rib_t = float(params.get("t", 0.0) or params["wall_t"] * 0.75)
    rib_h = float(params.get("h", params["wall_h"] * 0.75))
    poly = BRepBuilderAPI_MakePolygon()

    if side in {"left", "right"}:
        y0 = float(params["anchor_span"]) - rib_t / 2.0
        x0 = float(params["wall_inner_x"])
        inward_x = float(params["inward_x"])
        poly.Add(gp_Pnt(x0, y0, H))
        poly.Add(gp_Pnt(x0 + inward_x * rib_l, y0, H))
        poly.Add(gp_Pnt(x0, y0, H + rib_h))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        return BRepPrimAPI_MakePrism(face, gp_Vec(0, rib_t, 0)).Shape()

    x0 = float(params["anchor_span"]) - rib_t / 2.0
    y0 = float(params["wall_inner_y"])
    inward_y = float(params["inward_y"])
    poly.Add(gp_Pnt(x0, y0, H))
    poly.Add(gp_Pnt(x0, y0 + inward_y * rib_l, H))
    poly.Add(gp_Pnt(x0, y0, H + rib_h))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(rib_t, 0, 0)).Shape()


def _point_inside_right_triangle_gusset(params: dict, px: float, py: float, pz: float, H: float, tolerance: float = 0.0) -> bool:
    u, q = _gusset_local_uq(params, px, py)
    z = float(pz) - float(H)
    rib_l = float(params["l"])
    rib_h = float(params["h"])
    rib_t = float(params["t"])
    tol = float(tolerance)
    if u < tol or u > rib_l + tol:
        return False
    if z < tol or z > rib_h + tol:
        return False
    if abs(q) > rib_t / 2.0 + tol:
        return False
    hypotenuse_z = rib_h * (1.0 - max(u, 0.0) / max(rib_l, 1e-9))
    return z <= hypotenuse_z + tol


def _gusset_local_uq(params: dict, px: float, py: float) -> tuple[float, float]:
    if params["side"] in {"left", "right"}:
        u = (float(px) - float(params["wall_inner_x"])) * float(params["inward_x"])
        q = float(py) - float(params["anchor_span"])
        return u, q
    u = (float(py) - float(params["wall_inner_y"])) * float(params["inward_y"])
    q = float(px) - float(params["anchor_span"])
    return u, q


def _sample_cavity_rib_params(rib_t: float, generation_context: dict) -> dict | None:
    contexts = [
        context
        for context in generation_context.get("cavity_rib_contexts", [])
        if context.get("supports_cavity_rib")
    ]
    if not contexts:
        return None
    context = random.choice(contexts)
    if context["type"] == "rect_pocket":
        return _sample_rect_cavity_rib_params(context, rib_t)
    if context["type"] == "circular_pocket":
        return _sample_circular_cavity_rib_params(context, rib_t)
    return None


def _sample_rect_cavity_rib_params(context: dict, rib_t: float) -> dict:
    width = float(context["w"])
    length = float(context["l"])
    depth = float(context["depth"])
    base_angle = float(context["angle"])
    span_axis = random.choice(["width", "length"])
    if span_axis == "width":
        rib_l = width + max(float(rib_t), 0.6)
        angle = base_angle
    else:
        rib_l = length + max(float(rib_t), 0.6)
        angle = base_angle + math.pi / 2.0
    thickness = min(width, length) * random.uniform(0.08, 0.14)
    height = depth * random.uniform(0.55, 0.85)
    return _cavity_rib_params_from_context(
        context,
        rib_l=rib_l,
        rib_t=max(thickness, float(rib_t)),
        rib_h=height,
        angle=angle,
        span_axis=span_axis,
        support_context="rectangular_cavity",
    )


def _sample_circular_cavity_rib_params(context: dict, rib_t: float) -> dict:
    radius = float(context["radius"])
    depth = float(context["depth"])
    thickness = radius * random.uniform(0.14, 0.22)
    height = depth * random.uniform(0.55, 0.85)
    return _cavity_rib_params_from_context(
        context,
        rib_l=2.0 * radius + max(float(rib_t), 0.6),
        rib_t=max(thickness, float(rib_t)),
        rib_h=height,
        angle=random.uniform(0.0, math.pi),
        span_axis="diameter_chord",
        support_context="circular_cavity",
    )


def _cavity_rib_params_from_context(
    context: dict,
    *,
    rib_l: float,
    rib_t: float,
    rib_h: float,
    angle: float,
    span_axis: str,
    support_context: str,
) -> dict:
    bottom_z = float(context["bottom_z"])
    return {
        "cx": float(context["cx"]),
        "cy": float(context["cy"]),
        "anchor_x": float(context["cx"]),
        "anchor_y": float(context["cy"]),
        "l": float(rib_l),
        "t": float(rib_t),
        "h": float(rib_h),
        "angle": float(angle),
        "bottom_z": bottom_z,
        "top_z": bottom_z + float(rib_h),
        "support_context": support_context,
        "parent_macro_feature": context["parent_macro_feature"],
        "parent_instance_id": context["parent_instance_id"],
        "cavity_type": context["type"],
        "span_axis": span_axis,
    }


def _build_cavity_wall_rib_tool(params: dict):
    poly = BRepBuilderAPI_MakePolygon()
    for px, py in _cavity_wall_rib_footprint_points(params):
        poly.Add(gp_Pnt(px, py, params["bottom_z"]))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, params["h"])).Shape()


def _cavity_wall_rib_footprint_points(params: dict) -> list[tuple[float, float]]:
    half_l = float(params["l"]) / 2.0
    half_t = float(params["t"]) / 2.0
    local_points = [
        (-half_l, -half_t),
        (half_l, -half_t),
        (half_l, half_t),
        (-half_l, half_t),
    ]
    angle = float(params["angle"])
    cx = float(params["cx"])
    cy = float(params["cy"])
    points = []
    for local_x, local_y in local_points:
        rx = local_x * math.cos(angle) - local_y * math.sin(angle)
        ry = local_x * math.sin(angle) + local_y * math.cos(angle)
        points.append((cx + rx, cy + ry))
    return points


def _point_inside_cavity_wall_rib_footprint(params: dict, px: float, py: float, tolerance: float = 0.0) -> bool:
    dx = float(px) - float(params["cx"])
    dy = float(py) - float(params["cy"])
    angle = -float(params["angle"])
    local_x = dx * math.cos(angle) - dy * math.sin(angle)
    local_y = dx * math.sin(angle) + dy * math.cos(angle)
    return (
        abs(local_x) <= float(params["l"]) / 2.0 + float(tolerance)
        and abs(local_y) <= float(params["t"]) / 2.0 + float(tolerance)
    )
