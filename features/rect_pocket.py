import math
import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator
from utils.feature_footprint import circle_footprint, footprint_conflicts


class RectPocketGenerator(BaseFeatureGenerator):
    aspect_ratio_range = (1.0, 1.8)
    width_to_base_range = (0.12, 0.22)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "pocket_rect"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "rect_pocket_side_plane",
            "rect_pocket_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            pocket_w = random.uniform(
                min_base_size * self.width_to_base_range[0],
                min_base_size * self.width_to_base_range[1],
            )
            pocket_l = pocket_w * random.uniform(*self.aspect_ratio_range)
            pocket_depth = random.uniform(H * 0.12, H * 0.40)
            angle = random.uniform(0, math.pi)
            bounding_radius = math.hypot(pocket_w / 2.0, pocket_l / 2.0)

            valid = False
            cx, cy = 0.0, 0.0
            for _attempt in range(50):
                tx = random.uniform(-L / 2 + bounding_radius + 4.0, L / 2 - bounding_radius - 4.0)
                ty = random.uniform(-W / 2 + bounding_radius + 4.0, W / 2 - bounding_radius - 4.0)
                candidate = circle_footprint(tx, ty, bounding_radius, operation="cut", feature_kind="pocket_rect")
                if not footprint_conflicts(candidate, occupied_zones, clearance=4.0):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid:
                continue

            inst_id = start_inst_id + actual_added
            params = {
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "w": pocket_w,
                "l": pocket_l,
                "depth": pocket_depth,
                "angle": angle,
                "radius": bounding_radius,
            }
            params_list.append(params)
            _register_rect_cavity_rib_context(self.generation_context, params, top_z=H)
            occupied_zones.append(circle_footprint(cx, cy, bounding_radius, operation="cut", feature_kind="pocket_rect"))

            poly = BRepBuilderAPI_MakePolygon()
            local_pts = [
                (-pocket_w / 2.0, -pocket_l / 2.0),
                (pocket_w / 2.0, -pocket_l / 2.0),
                (pocket_w / 2.0, pocket_l / 2.0),
                (-pocket_w / 2.0, pocket_l / 2.0),
            ]
            for px, py in local_pts:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                poly.Add(gp_Pnt(cx + rx, cy + ry, H - pocket_depth))
            poly.Close()

            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, pocket_depth + 1.0)).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["radius"] + 0.2:
                continue

            dx = px - hp["cx"]
            dy = py - hp["cy"]
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])

            half_w = hp["w"] / 2.0
            half_l = hp["l"] / 2.0
            inside_rect = abs(loc_x) <= half_w + 0.1 and abs(loc_y) <= half_l + 0.1

            if abs(n.Z()) > 0.99 and inside_rect and abs(pz - (H - hp["depth"])) < 0.1:
                return self.semantic_map["rect_pocket_bottom_plane"], hp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and (H - hp["depth"] - 0.1) <= pz <= (H + 0.1):
                edge_dx = min(abs(abs(loc_x) - half_w), abs(abs(loc_y) - half_l))
                if edge_dx < 0.1 and inside_rect:
                    return self.semantic_map["rect_pocket_side_plane"], hp["inst"], self.feature_name

        return None


def _register_rect_cavity_rib_context(generation_context, params, top_z):
    contexts = generation_context.setdefault("cavity_rib_contexts", [])
    contexts.append({
        "type": "rect_pocket",
        "supports_cavity_rib": True,
        "parent_macro_feature": "pocket_rect",
        "parent_instance_id": params["inst"],
        "cx": params["cx"],
        "cy": params["cy"],
        "w": params["w"],
        "l": params["l"],
        "depth": params["depth"],
        "angle": params["angle"],
        "top_z": float(top_z),
        "bottom_z": float(top_z) - float(params["depth"]),
    })
