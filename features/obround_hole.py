import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator
from features.oriented_hole import axis_tuple, occupied_zone, sample_axis_placement

class ObroundHoleGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "obround_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "obround_side_plane",
            "obround_end_cylinder",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        params_list = []; actual_added = 0
        max_rad = min(L, W) * 0.1; max_dist = min(L, W) * 0.3
        
        for i in range(num_instances):
            R = random.uniform(max_rad * 0.4, max_rad)
            dist = random.uniform(R, max_dist)
            angle = random.uniform(0, math.pi)
            bounding_radius = R + dist/2
            
            valid = False
            axis_key = self.select_feature_axis("hole")
            placement = None
            for attempt in range(50):
                placement = sample_axis_placement(
                    axis_key,
                    L,
                    W,
                    H,
                    bounding_radius,
                    occupied_zones,
                    through=True,
                    stock_profile=self.generation_context.get("current_stock_profile", {}),
                )
                if placement is not None:
                    valid = True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "cx": placement["cx"],
                "cy": placement["cy"],
                "cz": placement["cz"],
                "R": R,
                "dist": dist,
                "angle": angle,
                "axis": axis_key,
            })
            occupied_zones.append(occupied_zone(placement, bounding_radius))

            w = self._make_obround_wire(placement, R, dist, angle)
            prism = BRepPrimAPI_MakePrism(
                BRepBuilderAPI_MakeFace(w).Face(),
                self._prism_vector(axis_key, L, W, H),
            ).Shape()
            tool = prism

            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            dx, dy = self._cross_plane_delta(hp, pnt)
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])
            
            dist_val = abs(loc_y) if abs(loc_x) <= hp["dist"]/2.0 else math.hypot(abs(loc_x)-hp["dist"]/2.0, loc_y)
            if abs(dist_val - hp["R"]) < 0.1:
                if surf.GetType() == GeomAbs_Cylinder:
                    return self.semantic_map["obround_end_cylinder"], hp["inst"], self.feature_name
                elif surf.GetType() == GeomAbs_Plane and self._normal_is_side_plane_for_axis(
                    surf.Plane().Axis().Direction(),
                    hp.get("axis", "z"),
                ):
                    return self.semantic_map["obround_side_plane"], hp["inst"], self.feature_name
        return None

    def _normal_is_side_plane_for_axis(self, normal, axis_key):
        axis = axis_tuple(axis_key)
        dot = normal.X() * axis[0] + normal.Y() * axis[1] + normal.Z() * axis[2]
        return abs(dot) < 1e-4

    def _axis_dir(self, axis_key):
        if axis_key == "x":
            return gp_Dir(1, 0, 0)
        if axis_key == "y":
            return gp_Dir(0, 1, 0)
        return gp_Dir(0, 0, 1)

    def _prism_vector(self, axis_key, L, W, H):
        if axis_key == "x":
            return gp_Vec(L + 2.0, 0, 0)
        if axis_key == "y":
            return gp_Vec(0, W + 2.0, 0)
        return gp_Vec(0, 0, H + 2.0)

    def _plane_origin_coord(self, placement):
        if placement["axis"] == "x":
            return placement["entry_coord"] - 1.0
        if placement["axis"] == "y":
            return placement["entry_coord"] - 1.0
        return -1.0

    def _cross_plane_point(self, placement, u, v, angle):
        loc_u = u * math.cos(angle) - v * math.sin(angle)
        loc_v = u * math.sin(angle) + v * math.cos(angle)
        axis_key = placement["axis"]
        if axis_key == "x":
            return gp_Pnt(self._plane_origin_coord(placement), placement["cy"] + loc_u, placement["cz"] + loc_v)
        if axis_key == "y":
            return gp_Pnt(placement["cx"] + loc_u, self._plane_origin_coord(placement), placement["cz"] + loc_v)
        return gp_Pnt(placement["cx"] + loc_u, placement["cy"] + loc_v, self._plane_origin_coord(placement))

    def _make_obround_wire(self, placement, radius, dist, angle):
        axis_dir = self._axis_dir(placement["axis"])
        right_top = self._cross_plane_point(placement, dist / 2.0, radius, angle)
        left_top = self._cross_plane_point(placement, -dist / 2.0, radius, angle)
        left_bottom = self._cross_plane_point(placement, -dist / 2.0, -radius, angle)
        right_bottom = self._cross_plane_point(placement, dist / 2.0, -radius, angle)
        right_mid = self._cross_plane_point(placement, dist / 2.0 + radius, 0.0, angle)
        left_mid = self._cross_plane_point(placement, -dist / 2.0 - radius, 0.0, angle)

        return BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(right_bottom, right_mid, right_top).Value()).Edge(),
            BRepBuilderAPI_MakeEdge(right_top, left_top).Edge(),
            BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(left_top, left_mid, left_bottom).Value()).Edge(),
            BRepBuilderAPI_MakeEdge(left_bottom, right_bottom).Edge(),
        ).Wire()

    def _cross_plane_delta(self, hp, pnt):
        axis_key = hp.get("axis", "z")
        if axis_key == "x":
            return pnt.Y() - hp["cy"], pnt.Z() - hp["cz"]
        if axis_key == "y":
            return pnt.X() - hp["cx"], pnt.Z() - hp["cz"]
        return pnt.X() - hp["cx"], pnt.Y() - hp["cy"]
