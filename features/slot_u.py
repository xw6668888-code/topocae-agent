import math
import random

from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator
from features.swept_slot import build_capsule_swept_profile_wire, swept_profile_metadata


class USlotGenerator(BaseFeatureGenerator):
    sweep_to_profile_ratio_range = (4.0, 7.0)
    radius_to_base_range = (0.025, 0.045)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "slot_u"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "slot_u_side_plane",
            "slot_u_end_cylinder",
            "slot_u_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            slot_r = random.uniform(
                min_base_size * self.radius_to_base_range[0],
                min_base_size * self.radius_to_base_range[1],
            )
            min_slot_dist = 2.0 * slot_r * self.sweep_to_profile_ratio_range[0]
            max_slot_dist = min(2.0 * slot_r * self.sweep_to_profile_ratio_range[1], min_base_size * 0.60)
            if max_slot_dist <= min_slot_dist:
                continue
            slot_dist = random.uniform(min_slot_dist, max_slot_dist)
            slot_depth = random.uniform(H * 0.10, H * 0.26)
            side = random.choice(["left", "right", "bottom", "top"])

            if side in {"left", "right"}:
                angle = 0.0 if side == "left" else math.pi
                cx = -L / 2 + slot_dist / 2.0 if side == "left" else L / 2 - slot_dist / 2.0
                cy = random.uniform(-W / 2 + slot_r + 4.0, W / 2 - slot_r - 4.0)
            else:
                angle = math.pi / 2.0 if side == "bottom" else -math.pi / 2.0
                cx = random.uniform(-L / 2 + slot_r + 4.0, L / 2 - slot_r - 4.0)
                cy = -W / 2 + slot_dist / 2.0 if side == "bottom" else W / 2 - slot_dist / 2.0

            bounding_radius = slot_dist / 2.0 + slot_r
            if any(math.hypot(cx - z["cx"], cy - z["cy"]) < (bounding_radius + z["radius"] + 2.0) for z in occupied_zones):
                continue

            boundary_kind = _path_boundary_kind(self.generation_context.get("current_stock_profile", {}))
            path_start_on_boundary = boundary_kind != "rectangular_workspace_boundary"
            inst_id = start_inst_id + actual_added
            params = {
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "r": slot_r,
                "dist": slot_dist,
                "depth": slot_depth,
                "angle": angle,
                "radius": bounding_radius,
                "path_start_xy": _boundary_path_start(side, cx, cy, L, W),
                "path_end_xy": _boundary_path_end(side, cx, cy, slot_dist),
                "path_angle": angle,
                "path_boundary_kind": boundary_kind,
                "slot_modeling_mode": "surface_anchored_open_boundary_sweep",
                "anchor_face": "current_top_planar_face",
            }
            params.update(swept_profile_metadata(
                "u",
                slot_dist,
                2.0 * slot_r,
                slot_depth,
                path_type="straight",
                path_curve_kind="line",
                path_on_face=True,
                path_start_on_boundary=path_start_on_boundary,
                path_end_on_boundary=False,
                anchor_face_kind="existing_planar_face",
            ))
            params_list.append(params)
            occupied_zones.append({"cx": cx, "cy": cy, "radius": bounding_radius})

            z0 = H - slot_depth
            wire = build_capsule_swept_profile_wire(0.0, 0.0, z0, slot_r, slot_dist, 0.0)

            prism = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire).Face(), gp_Vec(0, 0, slot_depth + 1.0)).Shape()
            t_translation = gp_Trsf()
            t_translation.SetTranslation(gp_Vec(cx, cy, 0.0))
            t_rotation = gp_Trsf()
            t_rotation.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle)
            tool = BRepBuilderAPI_Transform(prism, t_translation.Multiplied(t_rotation)).Shape()

            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def _to_local(self, hp, px, py):
        dx = px - hp["cx"]
        dy = py - hp["cy"]
        loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
        loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])
        return loc_x, loc_y

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()

        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["radius"] + 0.3:
                continue

            loc_x, loc_y = self._to_local(hp, px, py)

            if surf.GetType() == GeomAbs_Cylinder:
                if abs(math.hypot(loc_x - hp["dist"] / 2.0, loc_y) - hp["r"]) < 0.15:
                    return self.semantic_map["slot_u_end_cylinder"], hp["inst"], self.feature_name

            if surf.GetType() != GeomAbs_Plane:
                continue

            n = surf.Plane().Axis().Direction()
            inside_bottom = (
                (-hp["dist"] / 2.0 - 0.1) <= loc_x <= (hp["dist"] / 2.0 + 0.1)
                and abs(loc_y) <= hp["r"] + 0.1
            ) or math.hypot(loc_x - hp["dist"] / 2.0, loc_y) <= hp["r"] + 0.1

            if abs(n.Z()) > 0.99 and inside_bottom and abs(pz - (H - hp["depth"])) < 0.1:
                return self.semantic_map["slot_u_bottom_plane"], hp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and (-hp["dist"] / 2.0 - 0.1) <= loc_x <= (hp["dist"] / 2.0 + 0.1):
                if abs(abs(loc_y) - hp["r"]) < 0.12 and (H - hp["depth"] - 0.1) <= pz <= (H + 0.1):
                    return self.semantic_map["slot_u_side_plane"], hp["inst"], self.feature_name

        return None


def _boundary_path_start(side, cx, cy, L, W):
    if side == "left":
        return (-L / 2.0, cy)
    if side == "right":
        return (L / 2.0, cy)
    if side == "bottom":
        return (cx, -W / 2.0)
    return (cx, W / 2.0)


def _boundary_path_end(side, cx, cy, slot_dist):
    if side == "left":
        return (cx + slot_dist / 2.0, cy)
    if side == "right":
        return (cx - slot_dist / 2.0, cy)
    if side == "bottom":
        return (cx, cy + slot_dist / 2.0)
    return (cx, cy - slot_dist / 2.0)


def _path_boundary_kind(stock_profile):
    stock_type = stock_profile.get("stock_type")
    if stock_type in {"block", "plate", "beam", "composite"}:
        return "rectangular_stock_boundary"
    return "rectangular_workspace_boundary"
