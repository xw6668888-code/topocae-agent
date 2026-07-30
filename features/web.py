import math
import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator


class WebGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "web"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "web_side_plane",
            "web_top_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)
        max_placement_attempts = 20

        for _ in range(num_instances):
            for _placement_attempt in range(max_placement_attempts):
                thickness = random.uniform(min_base_size * 0.015, min_base_size * 0.03)
                web_h = random.uniform(H * 0.35, H * 0.75)
                orientation = random.choice(["x_span", "y_span"])

                if orientation == "x_span":
                    span = L
                    cy = random.uniform(-W / 2.0 + 12.0, W / 2.0 - 12.0)
                    strip_conflict = any(
                        abs(zone["cy"] - cy) < (zone["radius"] + thickness / 2.0 + 2.0)
                        for zone in occupied_zones
                    )
                    if strip_conflict:
                        continue
                    local_pts = [
                        (-span / 2.0, -thickness / 2.0),
                        (span / 2.0, -thickness / 2.0),
                        (span / 2.0, thickness / 2.0),
                        (-span / 2.0, thickness / 2.0),
                    ]
                    anchor = (0.0, cy)
                    zone_radius = thickness / 2.0 + 4.0
                    zone_cx, zone_cy = 0.0, cy
                else:
                    span = W
                    cx = random.uniform(-L / 2.0 + 12.0, L / 2.0 - 12.0)
                    strip_conflict = any(
                        abs(zone["cx"] - cx) < (zone["radius"] + thickness / 2.0 + 2.0)
                        for zone in occupied_zones
                    )
                    if strip_conflict:
                        continue
                    local_pts = [
                        (-thickness / 2.0, -span / 2.0),
                        (thickness / 2.0, -span / 2.0),
                        (thickness / 2.0, span / 2.0),
                        (-thickness / 2.0, span / 2.0),
                    ]
                    anchor = (cx, 0.0)
                    zone_radius = thickness / 2.0 + 4.0
                    zone_cx, zone_cy = cx, 0.0

                poly = BRepBuilderAPI_MakePolygon()
                for px, py in local_pts:
                    poly.Add(gp_Pnt(anchor[0] + px, anchor[1] + py, H))
                poly.Close()

                face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
                tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, web_h)).Shape()
                fuse = BRepAlgoAPI_Fuse(stock, tool)
                fuse.Build()
                if not fuse.IsDone():
                    continue

                stock = fuse.Shape()
                inst_id = start_inst_id + actual_added
                params_list.append({
                    "inst": inst_id,
                    "orientation": orientation,
                    "cx": anchor[0],
                    "cy": anchor[1],
                    "span": span,
                    "thickness": thickness,
                    "height": web_h,
                    "radius": max(zone_radius, 6.0),
                })
                occupied_zones.append({"cx": zone_cx, "cy": zone_cy, "radius": max(zone_radius, 6.0)})
                actual_added += 1
                break

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for wp in params_list:
            if wp["orientation"] == "x_span":
                loc_x = px - wp["cx"]
                loc_y = py - wp["cy"]
                inside = abs(loc_x) <= wp["span"] / 2.0 + 0.1 and abs(loc_y) <= wp["thickness"] / 2.0 + 0.1
                on_side = abs(abs(loc_y) - wp["thickness"] / 2.0) < 0.1
            else:
                loc_x = px - wp["cx"]
                loc_y = py - wp["cy"]
                inside = abs(loc_x) <= wp["thickness"] / 2.0 + 0.1 and abs(loc_y) <= wp["span"] / 2.0 + 0.1
                on_side = abs(abs(loc_x) - wp["thickness"] / 2.0) < 0.1

            if not inside:
                continue

            if abs(n.Z()) > 0.99 and abs(pz - (H + wp["height"])) < 0.1:
                return self.semantic_map["web_top_plane"], wp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and H - 0.1 <= pz <= H + wp["height"] + 0.1 and on_side:
                return self.semantic_map["web_side_plane"], wp["inst"], self.feature_name

        return None
