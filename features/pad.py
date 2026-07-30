import math
import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator


class PadGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "pad"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "pad_side_plane",
            "pad_top_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            pad_w = random.uniform(min_base_size * 0.16, min_base_size * 0.30)
            pad_l = random.uniform(min_base_size * 0.20, min_base_size * 0.38)
            pad_h = random.uniform(H * 0.06, H * 0.16)
            angle = random.uniform(0.0, math.pi)
            bounding_radius = math.hypot(pad_w / 2.0, pad_l / 2.0)

            valid = False
            cx, cy = 0.0, 0.0
            for _attempt in range(50):
                tx = random.uniform(-L / 2.0 + bounding_radius + 4.0, L / 2.0 - bounding_radius - 4.0)
                ty = random.uniform(-W / 2.0 + bounding_radius + 4.0, W / 2.0 - bounding_radius - 4.0)
                if not any(math.hypot(tx - zone["cx"], ty - zone["cy"]) < (bounding_radius + zone["radius"] + 2.0) for zone in occupied_zones):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid:
                continue

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "w": pad_w,
                "l": pad_l,
                "h": pad_h,
                "angle": angle,
                "radius": bounding_radius,
            })
            occupied_zones.append({"cx": cx, "cy": cy, "radius": bounding_radius})

            poly = BRepBuilderAPI_MakePolygon()
            local_pts = [
                (-pad_w / 2.0, -pad_l / 2.0),
                (pad_w / 2.0, -pad_l / 2.0),
                (pad_w / 2.0, pad_l / 2.0),
                (-pad_w / 2.0, pad_l / 2.0),
            ]
            for px, py in local_pts:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                poly.Add(gp_Pnt(cx + rx, cy + ry, H))
            poly.Close()

            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, pad_h)).Shape()
            fuse = BRepAlgoAPI_Fuse(stock, tool)
            fuse.Build()
            stock = fuse.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for pd in params_list:
            if math.hypot(px - pd["cx"], py - pd["cy"]) > pd["radius"] + 0.2:
                continue

            dx = px - pd["cx"]
            dy = py - pd["cy"]
            loc_x = dx * math.cos(-pd["angle"]) - dy * math.sin(-pd["angle"])
            loc_y = dx * math.sin(-pd["angle"]) + dy * math.cos(-pd["angle"])

            inside_rect = abs(loc_x) <= pd["w"] / 2.0 + 0.1 and abs(loc_y) <= pd["l"] / 2.0 + 0.1
            if not inside_rect:
                continue

            if abs(n.Z()) > 0.99 and abs(pz - (H + pd["h"])) < 0.1:
                return self.semantic_map["pad_top_plane"], pd["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and H - 0.1 <= pz <= H + pd["h"] + 0.1:
                edge_dist = min(abs(abs(loc_x) - pd["w"] / 2.0), abs(abs(loc_y) - pd["l"] / 2.0))
                if edge_dist < 0.1:
                    return self.semantic_map["pad_side_plane"], pd["inst"], self.feature_name

        return None
