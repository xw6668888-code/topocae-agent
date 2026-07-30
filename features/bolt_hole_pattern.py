import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from features.base_feature import BaseFeatureGenerator


class BoltHolePatternGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "bolt_hole_pattern"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "bolt_hole_pattern_hole_cylinder",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_radius = min_base_size * 0.07

        for _ in range(num_instances):
            radius = random.uniform(max_radius * 0.55, max_radius)
            pattern_type = random.choice(["linear2", "rect2x2"])
            angle = random.uniform(0.0, math.pi)

            if pattern_type == "linear2":
                pitch = random.uniform(radius * 3.2, radius * 5.8)
                local_holes = [(-pitch / 2.0, 0.0), (pitch / 2.0, 0.0)]
                bounding_radius = pitch / 2.0 + radius
            else:
                pitch_x = random.uniform(radius * 3.2, radius * 5.6)
                pitch_y = random.uniform(radius * 3.2, radius * 5.6)
                local_holes = [
                    (-pitch_x / 2.0, -pitch_y / 2.0),
                    (pitch_x / 2.0, -pitch_y / 2.0),
                    (-pitch_x / 2.0, pitch_y / 2.0),
                    (pitch_x / 2.0, pitch_y / 2.0),
                ]
                bounding_radius = math.hypot(pitch_x / 2.0, pitch_y / 2.0) + radius

            valid = False
            cx, cy = 0.0, 0.0
            for _attempt in range(50):
                tx = random.uniform(-L / 2.0 + bounding_radius + 6.0, L / 2.0 - bounding_radius - 6.0)
                ty = random.uniform(-W / 2.0 + bounding_radius + 6.0, W / 2.0 - bounding_radius - 6.0)
                if not any(
                    math.hypot(tx - zone["cx"], ty - zone["cy"]) < (bounding_radius + zone["radius"] + 2.0)
                    for zone in occupied_zones
                ):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid:
                continue

            holes = []
            for lx, ly in local_holes:
                rx = lx * math.cos(angle) - ly * math.sin(angle)
                ry = lx * math.sin(angle) + ly * math.cos(angle)
                holes.append({"x": cx + rx, "y": cy + ry})

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "R": radius,
                "pattern_type": pattern_type,
                "holes": holes,
                "radius_bound": bounding_radius,
            })
            occupied_zones.append({"cx": cx, "cy": cy, "radius": bounding_radius})

            for hole in holes:
                tool = BRepPrimAPI_MakeCylinder(
                    gp_Ax2(gp_Pnt(hole["x"], hole["y"], -1.0), gp_Dir(0, 0, 1)),
                    radius,
                    H + 2.0,
                ).Shape()
                cut = BRepAlgoAPI_Cut(stock, tool)
                cut.Build()
                stock = cut.Shape()

            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Cylinder:
            return None

        for pattern in params_list:
            if math.hypot(face_cx - pattern["cx"], face_cy - pattern["cy"]) > pattern["radius_bound"] + 0.2:
                continue
            for hole in pattern["holes"]:
                if math.hypot(face_cx - hole["x"], face_cy - hole["y"]) < pattern["R"] + 0.1:
                    return self.semantic_map["bolt_hole_pattern_hole_cylinder"], pattern["inst"], self.feature_name

        return None
