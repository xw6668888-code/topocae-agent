import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator
from features.oriented_hole import is_axis_placement_entry_exposed
from utils.feature_footprint import circle_footprint, footprint_conflicts

class BlindHoleGenerator(BaseFeatureGenerator):
    depth_to_diameter_ratio_range = (0.8, 1.8)

    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "blind_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "blind_hole_cylinder",
            "blind_hole_bottom",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        max_depth = H * 0.8
        max_rad = min(min(L, W) * 0.15, max_depth / (2.0 * self.depth_to_diameter_ratio_range[0]))
        
        for i in range(num_instances):
            R = random.uniform(max_rad * 0.3, max_rad)
            ratio_max = min(self.depth_to_diameter_ratio_range[1], max_depth / (2.0 * R))
            ratio_min = min(self.depth_to_diameter_ratio_range[0], ratio_max)
            Depth = 2.0 * R * random.uniform(ratio_min, ratio_max)
            axis_key = self.select_feature_axis("hole")
            valid = False
            placement = None
            for attempt in range(50):
                placement = self._sample_placement(axis_key, L, W, H, R, Depth)
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
                "Depth": Depth,
                "axis": axis_key,
                "bottom_coord": placement["bottom_coord"],
            })
            occupied_zones.append(circle_footprint(
                placement["zone_a"],
                placement["zone_b"],
                R,
                operation="cut",
                zone_plane=placement["zone_plane"],
                feature_kind="blind_hole",
            ))

            # Side-axis tools start one unit outside the stock; only the entry
            # side needs overcut, otherwise the generated bottom face no longer
            # matches the stored blind-hole depth used for annotation.
            tool_length = Depth + (2.0 if axis_key == "z" else 1.0)
            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*placement["tool_origin"]), gp_Dir(*placement["axis_dir"])),
                R,
                tool_length,
            ).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            axis_key = hp.get("axis", "z")
            if surf.GetType() == GeomAbs_Cylinder:
                cylinder = surf.Cylinder()
                if not cylinder.Axis().Direction().IsParallel(gp_Dir(*self._axis_dir(axis_key)), 1e-4):
                    continue
                if abs(cylinder.Radius() - hp["R"]) > 0.15:
                    continue
                location = cylinder.Axis().Location()
                if axis_key == "z" and math.hypot(location.X() - hp["cx"], location.Y() - hp["cy"]) < hp["R"] + 0.1:
                    return self.semantic_map["blind_hole_cylinder"], hp["inst"], self.feature_name
                if axis_key == "x" and math.hypot(location.Y() - hp["cy"], location.Z() - hp["cz"]) < hp["R"] + 0.1:
                    return self.semantic_map["blind_hole_cylinder"], hp["inst"], self.feature_name
                if axis_key == "y" and math.hypot(location.X() - hp["cx"], location.Z() - hp["cz"]) < hp["R"] + 0.1:
                    return self.semantic_map["blind_hole_cylinder"], hp["inst"], self.feature_name

            elif surf.GetType() == GeomAbs_Plane:
                direction = surf.Plane().Axis().Direction()
                if not direction.IsParallel(gp_Dir(*self._axis_dir(axis_key)), 1e-4):
                    continue
                if axis_key == "z" and math.hypot(pnt.X() - hp["cx"], pnt.Y() - hp["cy"]) < hp["R"] + 0.1:
                    if abs(pnt.Z() - hp["bottom_coord"]) < 0.1:
                        return self.semantic_map["blind_hole_bottom"], hp["inst"], self.feature_name
                if axis_key == "x" and math.hypot(pnt.Y() - hp["cy"], pnt.Z() - hp["cz"]) < hp["R"] + 0.1:
                    if abs(pnt.X() - hp["bottom_coord"]) < 0.1:
                        return self.semantic_map["blind_hole_bottom"], hp["inst"], self.feature_name
                if axis_key == "y" and math.hypot(pnt.X() - hp["cx"], pnt.Z() - hp["cz"]) < hp["R"] + 0.1:
                    if abs(pnt.Y() - hp["bottom_coord"]) < 0.1:
                        return self.semantic_map["blind_hole_bottom"], hp["inst"], self.feature_name
        return None

    def _axis_dir(self, axis_key):
        if axis_key == "x":
            return (1, 0, 0)
        if axis_key == "y":
            return (0, 1, 0)
        return (0, 0, 1)

    def _sample_placement(self, axis_key, L, W, H, R, depth):
        margin = R + 5.0
        if axis_key == "x":
            if W <= 2.0 * margin or H <= 2.0 * margin or L <= depth + 4.0:
                return None
            side = random.choice([-1.0, 1.0])
            cy = random.uniform(-W / 2.0 + margin, W / 2.0 - margin)
            cz = random.uniform(margin, H - margin)
            if side < 0:
                origin_x = -L / 2.0 - 1.0
                axis_dir = (1, 0, 0)
                bottom_coord = -L / 2.0 + depth
            else:
                origin_x = L / 2.0 + 1.0
                axis_dir = (-1, 0, 0)
                bottom_coord = L / 2.0 - depth
            return {
                "axis": "x",
                "cx": 0.0,
                "cy": cy,
                "cz": cz,
                "axis_dir": axis_dir,
                "tool_origin": (origin_x, cy, cz),
                "bottom_coord": bottom_coord,
                "zone_plane": "yz",
                "zone_a": cy,
                "zone_b": cz,
            }
        if axis_key == "y":
            if L <= 2.0 * margin or H <= 2.0 * margin or W <= depth + 4.0:
                return None
            side = random.choice([-1.0, 1.0])
            cx = random.uniform(-L / 2.0 + margin, L / 2.0 - margin)
            cz = random.uniform(margin, H - margin)
            if side < 0:
                origin_y = -W / 2.0 - 1.0
                axis_dir = (0, 1, 0)
                bottom_coord = -W / 2.0 + depth
            else:
                origin_y = W / 2.0 + 1.0
                axis_dir = (0, -1, 0)
                bottom_coord = W / 2.0 - depth
            return {
                "axis": "y",
                "cx": cx,
                "cy": 0.0,
                "cz": cz,
                "axis_dir": axis_dir,
                "tool_origin": (cx, origin_y, cz),
                "bottom_coord": bottom_coord,
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
            "cz": H - depth,
            "axis_dir": (0, 0, 1),
            "tool_origin": (cx, cy, H - depth),
            "bottom_coord": H - depth,
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
            feature_kind="blind_hole",
        )
        return not footprint_conflicts(candidate, occupied_zones, clearance=4.0)
