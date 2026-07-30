import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from features.base_feature import BaseFeatureGenerator


class BearingSeatGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "bearing_seat"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "bearing_seat_side_cylinder",
            "bearing_seat_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_radius = min_base_size * 0.20

        for _ in range(num_instances):
            radius = random.uniform(max_radius * 0.55, max_radius)
            depth = random.uniform(H * 0.10, H * 0.18)

            valid = False
            cx, cy = 0.0, 0.0
            for _attempt in range(50):
                tx = random.uniform(-L / 2.0 + radius + 6.0, L / 2.0 - radius - 6.0)
                ty = random.uniform(-W / 2.0 + radius + 6.0, W / 2.0 - radius - 6.0)
                if not any(
                    math.hypot(tx - zone["cx"], ty - zone["cy"]) < (radius + zone["radius"] + 2.0)
                    for zone in occupied_zones
                ):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid:
                continue

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "R": radius,
                "depth": depth,
            })
            occupied_zones.append({"cx": cx, "cy": cy, "radius": radius})

            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(cx, cy, H - depth), gp_Dir(0, 0, 1)),
                radius,
                depth + 1.0,
            ).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for bs in params_list:
            if math.hypot(face_cx - bs["cx"], face_cy - bs["cy"]) > bs["R"] + 0.1:
                continue

            if surf.GetType() == GeomAbs_Cylinder:
                return self.semantic_map["bearing_seat_side_cylinder"], bs["inst"], self.feature_name

            if surf.GetType() == GeomAbs_Plane:
                direction = surf.Plane().Axis().Direction()
                if direction.IsParallel(gp_Dir(0, 0, 1), 1e-4) and abs(pnt.Z() - (H - bs["depth"])) < 0.1:
                    return self.semantic_map["bearing_seat_bottom_plane"], bs["inst"], self.feature_name

        return None
