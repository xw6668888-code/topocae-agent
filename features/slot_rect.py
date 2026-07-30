import math
import random

from OCC.Core.gp import gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator
from features.swept_slot import (
    build_rectangular_swept_profile_wire,
    straight_boundary_path_on_box,
    straight_boundary_path_on_circle,
    swept_profile_metadata,
)
from utils.feature_footprint import capsule_footprint, footprint_conflicts


class SlotRectGenerator(BaseFeatureGenerator):
    aspect_ratio_range = (4.0, 8.0)
    width_to_base_range = (0.04, 0.06)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "slot_rect"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "slot_rect_side_plane",
            "slot_rect_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            slot_w = random.uniform(
                min_base_size * self.width_to_base_range[0],
                min_base_size * self.width_to_base_range[1],
            )
            slot_depth = random.uniform(H * 0.10, H * 0.28)

            valid = False
            cx, cy, angle = 0.0, 0.0, 0.0
            slot_l = 0.0
            path = None
            for _attempt in range(50):
                tx = random.uniform(-L / 2 + slot_w + 4.0, L / 2 - slot_w - 4.0)
                ty = random.uniform(-W / 2 + slot_w + 4.0, W / 2 - slot_w - 4.0)
                path_angle = random.uniform(0.0, math.pi)
                try:
                    candidate_path = _straight_path_for_current_stock(
                        tx,
                        ty,
                        path_angle,
                        L,
                        W,
                        self.generation_context.get("current_stock_profile", {}),
                    )
                except ValueError:
                    continue

                candidate_l = candidate_path["length"] + 2.0
                target_ratio = random.uniform(*self.aspect_ratio_range)
                candidate_w = min(slot_w, candidate_l / target_ratio)
                if candidate_l / max(candidate_w, 1e-9) < self.aspect_ratio_range[0]:
                    continue

                candidate_cx, candidate_cy = candidate_path["center"]
                candidate_angle = candidate_path["angle"] - math.pi / 2.0
                candidate_footprint = capsule_footprint(
                    candidate_cx,
                    candidate_cy,
                    candidate_l,
                    candidate_w,
                    candidate_angle,
                    operation="cut",
                    feature_kind="slot_rect",
                )
                if footprint_conflicts(
                    candidate_footprint,
                    occupied_zones,
                    clearance=4.0,
                ):
                    continue

                cx, cy = candidate_cx, candidate_cy
                angle = candidate_angle
                slot_w = candidate_w
                slot_l = candidate_l
                path = candidate_path
                valid = True
                break
            if not valid:
                continue

            bounding_radius = math.hypot(slot_w / 2.0, slot_l / 2.0)
            inst_id = start_inst_id + actual_added
            params = {
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "w": slot_w,
                "l": slot_l,
                "depth": slot_depth,
                "angle": angle,
                "path_angle": path["angle"],
                "path_start_xy": path["start"],
                "path_end_xy": path["end"],
                "path_seed_xy": path["seed"],
                "path_boundary_kind": path["boundary_kind"],
                "slot_modeling_mode": "surface_anchored_boundary_sweep",
                "anchor_face": "current_top_planar_face",
                "radius": bounding_radius,
            }
            params.update(swept_profile_metadata(
                "rectangle",
                slot_l,
                slot_w,
                slot_depth,
                path_type="straight",
                path_curve_kind="line",
                path_on_face=True,
                path_start_on_boundary=True,
                path_end_on_boundary=True,
                anchor_face_kind="existing_planar_face",
            ))
            params_list.append(params)
            occupied_zones.append(capsule_footprint(
                cx,
                cy,
                slot_l,
                slot_w,
                angle,
                operation="cut",
                feature_kind="slot_rect",
            ))

            wire = build_rectangular_swept_profile_wire(cx, cy, H - slot_depth, slot_w, slot_l, angle)
            face = BRepBuilderAPI_MakeFace(wire).Face()
            tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, slot_depth + 1.0)).Shape()
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
                return self.semantic_map["slot_rect_bottom_plane"], hp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and (H - hp["depth"] - 0.1) <= pz <= (H + 0.1):
                edge_dx = min(abs(abs(loc_x) - half_w), abs(abs(loc_y) - half_l))
                if edge_dx < 0.1 and inside_rect:
                    return self.semantic_map["slot_rect_side_plane"], hp["inst"], self.feature_name

        return None

def _straight_path_for_current_stock(seed_x, seed_y, path_angle, L, W, stock_profile):
    stock_type = stock_profile.get("stock_type")
    dimensions = stock_profile.get("dimensions", {})
    if stock_type == "cylinder" and "radius" in dimensions:
        path = straight_boundary_path_on_circle(seed_x, seed_y, path_angle, float(dimensions["radius"]))
        path["boundary_kind"] = "circular_stock_boundary"
        return path

    path = straight_boundary_path_on_box(seed_x, seed_y, path_angle, L, W)
    path["boundary_kind"] = "rectangular_workspace_boundary"
    return path
