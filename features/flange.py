import math
import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator


class FlangeGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "flange"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "flange_side_plane",
            "flange_top_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            side = random.choice(["left", "right", "bottom", "top"])
            depth = random.uniform(min_base_size * 0.08, min_base_size * 0.18)
            height = random.uniform(H * 0.35, H * 0.8)

            if side in {"left", "right"}:
                span = random.uniform(W * 0.7, W * 0.95)
                half_span = span / 2.0
                anchor_y = random.uniform(-W / 2.0 + half_span, W / 2.0 - half_span)
                anchor_x = -L / 2.0 if side == "left" else L / 2.0
                angle = 0.0 if side == "left" else math.pi
                feature_cx = anchor_x + (depth / 2.0 if side == "left" else -depth / 2.0)
                feature_cy = anchor_y
                conflict = any(
                    abs(zone["cy"] - anchor_y) < (half_span + zone["radius"] + 2.0)
                    and (
                        zone["cx"] < (-L / 2.0 + depth + zone["radius"] + 2.0)
                        if side == "left"
                        else zone["cx"] > (L / 2.0 - depth - zone["radius"] - 2.0)
                    )
                    for zone in occupied_zones
                )
            else:
                span = random.uniform(L * 0.7, L * 0.95)
                half_span = span / 2.0
                anchor_x = random.uniform(-L / 2.0 + half_span, L / 2.0 - half_span)
                anchor_y = -W / 2.0 if side == "bottom" else W / 2.0
                angle = math.pi / 2.0 if side == "bottom" else -math.pi / 2.0
                feature_cx = anchor_x
                feature_cy = anchor_y + (depth / 2.0 if side == "bottom" else -depth / 2.0)
                conflict = any(
                    abs(zone["cx"] - anchor_x) < (half_span + zone["radius"] + 2.0)
                    and (
                        zone["cy"] < (-W / 2.0 + depth + zone["radius"] + 2.0)
                        if side == "bottom"
                        else zone["cy"] > (W / 2.0 - depth - zone["radius"] - 2.0)
                    )
                    for zone in occupied_zones
                )

            if conflict:
                continue

            inst_id = start_inst_id + actual_added
            params_list.append({
                "inst": inst_id,
                "side": side,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "cx": feature_cx,
                "cy": feature_cy,
                "depth": depth,
                "span": span,
                "height": height,
                "angle": angle,
                "radius": math.hypot(depth / 2.0, span / 2.0),
            })
            occupied_zones.append({"cx": feature_cx, "cy": feature_cy, "radius": max(depth, 8.0)})

            poly = BRepBuilderAPI_MakePolygon()
            local_pts = [
                (0.0, -span / 2.0),
                (depth, -span / 2.0),
                (depth, span / 2.0),
                (0.0, span / 2.0),
            ]
            for px, py in local_pts:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                poly.Add(gp_Pnt(anchor_x + rx, anchor_y + ry, H))
            poly.Close()

            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height)).Shape()
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

        for fp in params_list:
            dx = px - fp["anchor_x"]
            dy = py - fp["anchor_y"]
            loc_x = dx * math.cos(-fp["angle"]) - dy * math.sin(-fp["angle"])
            loc_y = dx * math.sin(-fp["angle"]) + dy * math.cos(-fp["angle"])

            inside = -0.1 <= loc_x <= fp["depth"] + 0.1 and abs(loc_y) <= fp["span"] / 2.0 + 0.1
            if not inside:
                continue

            if abs(n.Z()) > 0.99 and abs(pz - (H + fp["height"])) < 0.1:
                return self.semantic_map["flange_top_plane"], fp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and H - 0.1 <= pz <= H + fp["height"] + 0.1:
                on_inner = abs(loc_x - fp["depth"]) < 0.1
                on_end = abs(abs(loc_y) - fp["span"] / 2.0) < 0.1
                if on_inner or on_end:
                    return self.semantic_map["flange_side_plane"], fp["inst"], self.feature_name

        return None
