import math
import random
from OCC.Core.gp import gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator
from features.oriented_hole import (
    axis_cross_distance_from_cylinder,
    coordinate_at,
    is_surface_axis_parallel,
    make_revolved_tool,
    occupied_zone,
    point_axis_coord,
    point_cross_distance,
    sample_axis_placement,
)

class SteppedHoleGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "stepped_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "large_hole_cylinder",
            "step_plane",
            "small_hole_cylinder",
            "small_hole_bottom",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        min_base_size = min(L, W)
        max_rad = min_base_size * 0.2
        max_depth = H * 0.8
        
        for i in range(num_instances):
            R1 = random.uniform(max_rad * 0.4, max_rad)
            R2 = random.uniform(R1 * 0.4, R1 * 0.8)
            D1 = random.uniform(max_depth * 0.2, max_depth * 0.5)
            D2 = random.uniform(max_depth * 0.2, max_depth * 0.5)
            
            valid = False
            axis_key = self.select_feature_axis("hole")
            placement = None
            for attempt in range(50):
                placement = sample_axis_placement(
                    axis_key,
                    L,
                    W,
                    H,
                    R1,
                    occupied_zones,
                    through=False,
                    stock_profile=self.generation_context.get("current_stock_profile", {}),
                )
                if placement is not None:
                    valid = True
                    break
            
            if not valid: continue

            inst_id = start_inst_id + actual_added
            holes_params.append({
                "inst": inst_id,
                "cx": placement["cx"],
                "cy": placement["cy"],
                "cz": placement["cz"],
                "R1": R1,
                "D1": D1,
                "R2": R2,
                "D2": D2,
                "axis": axis_key,
                "axis_sign": placement["axis_sign"],
                "entry_coord": placement["entry_coord"],
                "step_coord": coordinate_at(placement, D1),
                "bottom_coord": coordinate_at(placement, D1 + D2),
            })
            occupied_zones.append(occupied_zone(placement, R1))

            tool = make_revolved_tool(
                placement,
                [
                    (0.0, -1.0),
                    (R1, -1.0),
                    (R1, D1),
                    (R2, D1),
                    (R2, D1 + D2),
                    (0.0, D1 + D2),
                ],
            )
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            axis_key = hp.get("axis", "z")
            if point_cross_distance(pnt, hp) > hp["R1"] + 0.1:
                continue

            if surf.GetType() == GeomAbs_Cylinder:
                try:
                    cylinder = surf.Cylinder()
                    if not cylinder.Axis().Direction().IsParallel(gp_Dir(0, 0, 1) if axis_key == "z" else gp_Dir(1, 0, 0) if axis_key == "x" else gp_Dir(0, 1, 0), 1e-4):
                        continue
                    if axis_cross_distance_from_cylinder(cylinder, hp) > hp["R1"] + 0.1:
                        continue
                    return (self.semantic_map["large_hole_cylinder"] if cylinder.Radius() > (hp["R1"] + hp["R2"]) / 2
                            else self.semantic_map["small_hole_cylinder"]), hp["inst"], self.feature_name
                except Exception:
                    pass
            elif surf.GetType() == GeomAbs_Plane and is_surface_axis_parallel(surf.Plane().Axis().Direction(), axis_key):
                coord = point_axis_coord(pnt, axis_key)
                if abs(coord - hp["step_coord"]) < 0.1:
                    return self.semantic_map["step_plane"], hp["inst"], self.feature_name
                if abs(coord - hp["bottom_coord"]) < 0.1:
                    return self.semantic_map["small_hole_bottom"], hp["inst"], self.feature_name
        return None
