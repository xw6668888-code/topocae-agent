import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from features.base_feature import BaseFeatureGenerator
from utils.feature_footprint import circle_footprint, footprint_conflicts


class CircularPocketGenerator(BaseFeatureGenerator):
    depth_to_diameter_ratio_range = (0.15, 0.5)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "pocket_circular"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "circular_pocket_side_cylinder",
            "circular_pocket_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_radius = min_base_size * 0.16

        for _ in range(num_instances):
            radius = random.uniform(max_radius * 0.45, max_radius)
            ratio_upper = min(self.depth_to_diameter_ratio_range[1], (H * 0.24) / (2.0 * radius))
            ratio_lower = min(self.depth_to_diameter_ratio_range[0], ratio_upper)
            depth = 2.0 * radius * random.uniform(ratio_lower, ratio_upper)

            valid = False
            cx, cy = 0.0, 0.0
            for _attempt in range(50):
                tx = random.uniform(-L / 2 + radius + 5.0, L / 2 - radius - 5.0)
                ty = random.uniform(-W / 2 + radius + 5.0, W / 2 - radius - 5.0)
                candidate = circle_footprint(tx, ty, radius, operation="cut", feature_kind="pocket_circular")
                if not footprint_conflicts(candidate, occupied_zones, clearance=4.0):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid:
                continue

            inst_id = start_inst_id + actual_added
            params = {"inst": inst_id, "cx": cx, "cy": cy, "R": radius, "Depth": depth}
            params_list.append(params)
            _register_circular_cavity_rib_context(self.generation_context, params, top_z=H)
            occupied_zones.append(circle_footprint(cx, cy, radius, operation="cut", feature_kind="pocket_circular"))

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
        for hp in params_list:
            if math.hypot(face_cx - hp["cx"], face_cy - hp["cy"]) > hp["R"] + 0.1:
                continue

            if surf.GetType() == GeomAbs_Cylinder:
                return self.semantic_map["circular_pocket_side_cylinder"], hp["inst"], self.feature_name

            if surf.GetType() == GeomAbs_Plane:
                direction = surf.Plane().Axis().Direction()
                if direction.IsParallel(gp_Dir(0, 0, 1), 1e-4) and abs(pnt.Z() - (H - hp["Depth"])) < 0.1:
                    return self.semantic_map["circular_pocket_bottom_plane"], hp["inst"], self.feature_name

        return None


def _register_circular_cavity_rib_context(generation_context, params, top_z):
    contexts = generation_context.setdefault("cavity_rib_contexts", [])
    contexts.append({
        "type": "circular_pocket",
        "supports_cavity_rib": True,
        "parent_macro_feature": "pocket_circular",
        "parent_instance_id": params["inst"],
        "cx": params["cx"],
        "cy": params["cy"],
        "radius": params["R"],
        "depth": params["Depth"],
        "top_z": float(top_z),
        "bottom_z": float(top_z) - float(params["Depth"]),
    })
