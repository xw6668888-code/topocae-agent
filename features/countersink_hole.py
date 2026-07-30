import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder

from features.base_feature import BaseFeatureGenerator
from features.oriented_hole import (
    axis_cross_distance_from_cylinder,
    make_revolved_tool,
    occupied_zone,
    point_cross_distance,
    sample_axis_placement,
)


class CountersinkHoleGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "countersink_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "countersink_cone",
            "countersink_small_cylinder",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_large_radius = min_base_size * 0.18

        for _ in range(num_instances):
            large_r = random.uniform(max_large_radius * 0.45, max_large_radius)
            small_r = random.uniform(large_r * 0.25, large_r * 0.55)
            sink_depth = random.uniform(H * 0.06, H * 0.18)

            valid = False
            axis_key = self.select_feature_axis("hole")
            placement = None
            for _attempt in range(50):
                placement = sample_axis_placement(
                    axis_key,
                    L,
                    W,
                    H,
                    large_r,
                    occupied_zones,
                    through=True,
                    stock_profile=self.generation_context.get("current_stock_profile", {}),
                )
                if placement is not None:
                    valid = True
                    break
            if not valid:
                continue

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "cx": placement["cx"],
                "cy": placement["cy"],
                "cz": placement["cz"],
                "R1": large_r,
                "R2": small_r,
                "D1": sink_depth,
                "axis": axis_key,
                "axis_sign": placement["axis_sign"],
                "entry_coord": placement["entry_coord"],
            })
            occupied_zones.append(occupied_zone(placement, large_r))

            tool = make_revolved_tool(
                placement,
                [
                    (0.0, -1.0),
                    (large_r, -1.0),
                    (large_r, 0.0),
                    (small_r, sink_depth),
                    (small_r, placement["span"] + 1.0),
                    (0.0, placement["span"] + 1.0),
                ],
            )
            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            if point_cross_distance(pnt, hp) > hp["R1"] + 0.1:
                continue

            if surf.GetType() == GeomAbs_Cone:
                return self.semantic_map["countersink_cone"], hp["inst"], self.feature_name

            if surf.GetType() == GeomAbs_Cylinder:
                try:
                    cylinder = surf.Cylinder()
                    if axis_cross_distance_from_cylinder(cylinder, hp) > hp["R1"] + 0.1:
                        continue
                    radius = cylinder.Radius()
                except Exception:
                    continue
                if abs(radius - hp["R2"]) < 0.1:
                    return self.semantic_map["countersink_small_cylinder"], hp["inst"], self.feature_name

        return None
