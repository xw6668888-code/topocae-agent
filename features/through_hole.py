import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator
from features.oriented_hole import is_axis_placement_entry_exposed
from utils.feature_footprint import circle_footprint, footprint_conflicts

class ThroughHoleGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "through_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "through_hole_cylinder",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        max_rad = min(L, W) * 0.15
        
        for i in range(num_instances):
            R = random.uniform(max_rad * 0.3, max_rad)
            axis_key = self.select_feature_axis("hole")
            valid = False
            placement = None
            for attempt in range(50):
                placement = self._sample_placement(axis_key, L, W, H, R)
                if (
                    placement is not None
                    and is_axis_placement_entry_exposed(placement, R, self.generation_context.get("current_stock_profile", {}))
                    and self._is_zone_free(placement, R, occupied_zones)
                ):
                    valid = True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            holes_params.append({
                "inst": inst_id,
                "cx": placement["cx"],
                "cy": placement["cy"],
                "cz": placement["cz"],
                "R": R,
                "axis": axis_key,
            })
            occupied_zones.append(circle_footprint(
                placement["zone_a"],
                placement["zone_b"],
                R,
                operation="cut",
                zone_plane=placement["zone_plane"],
                feature_kind="through_hole",
            ))

            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*placement["tool_origin"]), gp_Dir(*placement["axis_dir"])),
                R,
                placement["tool_length"],
            ).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Cylinder:
            return None

        for hp in params_list:
            axis_key = hp.get("axis", "z")
            cylinder = surf.Cylinder()
            if not cylinder.Axis().Direction().IsParallel(gp_Dir(*self._axis_dir(axis_key)), 1e-4):
                continue
            if abs(cylinder.Radius() - hp["R"]) > 0.15:
                continue

            location = cylinder.Axis().Location()
            if axis_key == "z" and math.hypot(location.X() - hp["cx"], location.Y() - hp["cy"]) < hp["R"] + 0.1:
                return self.semantic_map["through_hole_cylinder"], hp["inst"], self.feature_name
            if axis_key == "x" and math.hypot(location.Y() - hp["cy"], location.Z() - hp["cz"]) < hp["R"] + 0.1:
                return self.semantic_map["through_hole_cylinder"], hp["inst"], self.feature_name
            if axis_key == "y" and math.hypot(location.X() - hp["cx"], location.Z() - hp["cz"]) < hp["R"] + 0.1:
                return self.semantic_map["through_hole_cylinder"], hp["inst"], self.feature_name
        return None

    def _axis_dir(self, axis_key):
        if axis_key == "x":
            return (1, 0, 0)
        if axis_key == "y":
            return (0, 1, 0)
        return (0, 0, 1)

    def _sample_placement(self, axis_key, L, W, H, R):
        margin = R + 5.0
        if axis_key == "x":
            if W <= 2.0 * margin or H <= 2.0 * margin:
                return None
            cy = random.uniform(-W / 2.0 + margin, W / 2.0 - margin)
            cz = random.uniform(margin, H - margin)
            return {
                "axis": "x",
                "cx": 0.0,
                "cy": cy,
                "cz": cz,
                "axis_dir": self._axis_dir("x"),
                "tool_origin": (-L / 2.0 - 1.0, cy, cz),
                "tool_length": L + 2.0,
                "zone_plane": "yz",
                "zone_a": cy,
                "zone_b": cz,
            }
        if axis_key == "y":
            if L <= 2.0 * margin or H <= 2.0 * margin:
                return None
            cx = random.uniform(-L / 2.0 + margin, L / 2.0 - margin)
            cz = random.uniform(margin, H - margin)
            return {
                "axis": "y",
                "cx": cx,
                "cy": 0.0,
                "cz": cz,
                "axis_dir": self._axis_dir("y"),
                "tool_origin": (cx, -W / 2.0 - 1.0, cz),
                "tool_length": W + 2.0,
                "zone_plane": "xz",
                "zone_a": cx,
                "zone_b": cz,
            }

        if L <= 2.0 * margin or W <= 2.0 * margin:
            return None
        cx = random.uniform(-L / 2.0 + margin, L / 2.0 - margin)
        cy = random.uniform(-W / 2.0 + margin, W / 2.0 - margin)
        return {
            "axis": "z",
            "cx": cx,
            "cy": cy,
            "cz": H / 2.0,
            "axis_dir": self._axis_dir("z"),
            "tool_origin": (cx, cy, -1.0),
            "tool_length": H + 2.0,
            "zone_plane": "xy",
            "zone_a": cx,
            "zone_b": cy,
        }

    def _is_zone_free(self, placement, radius, occupied_zones):
        candidate = circle_footprint(
            placement["zone_a"],
            placement["zone_b"],
            radius,
            operation="cut",
            zone_plane=placement["zone_plane"],
            feature_kind="through_hole",
        )
        return not footprint_conflicts(candidate, occupied_zones, clearance=4.0)
