import random

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane

from features.base_feature import BaseFeatureGenerator


class GussetGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "gusset"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "gusset_side_plane",
            "gusset_slope_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            side = random.choice(["left", "right", "bottom", "top"])
            base = random.uniform(min_base_size * 0.12, min_base_size * 0.26)
            height = random.uniform(H * 0.25, H * 0.65)
            thickness = random.uniform(min_base_size * 0.02, min_base_size * 0.05)

            if side in {"left", "right"}:
                cy = random.uniform(-W / 2.0 + thickness / 2.0 + 4.0, W / 2.0 - thickness / 2.0 - 4.0)
                zone_cx = -L / 2.0 + base / 2.0 if side == "left" else L / 2.0 - base / 2.0
                zone_cy = cy
                conflict = any(
                    abs(zone["cy"] - cy) < (zone["radius"] + thickness + 2.0)
                    and abs(zone["cx"] - zone_cx) < (zone["radius"] + base + 2.0)
                    for zone in occupied_zones
                )
            else:
                cx = random.uniform(-L / 2.0 + thickness / 2.0 + 4.0, L / 2.0 - thickness / 2.0 - 4.0)
                zone_cx = cx
                zone_cy = -W / 2.0 + base / 2.0 if side == "bottom" else W / 2.0 - base / 2.0
                conflict = any(
                    abs(zone["cx"] - cx) < (zone["radius"] + thickness + 2.0)
                    and abs(zone["cy"] - zone_cy) < (zone["radius"] + base + 2.0)
                    for zone in occupied_zones
                )

            if conflict:
                continue

            inst_id = start_inst_id + actual_added
            params = {
                "inst": inst_id,
                "side": side,
                "base": base,
                "height": height,
                "thickness": thickness,
                "radius": max(base, thickness),
            }

            poly = BRepBuilderAPI_MakePolygon()
            if side == "left":
                y0 = cy - thickness / 2.0
                y1 = cy + thickness / 2.0
                params.update({"y0": y0, "y1": y1, "wall_x": -L / 2.0})
                poly.Add(gp_Pnt(-L / 2.0, y0, H))
                poly.Add(gp_Pnt(-L / 2.0, y0, H + height))
                poly.Add(gp_Pnt(-L / 2.0 + base, y0, H))
                prism_vec = gp_Vec(0, thickness, 0)
            elif side == "right":
                y0 = cy - thickness / 2.0
                y1 = cy + thickness / 2.0
                params.update({"y0": y0, "y1": y1, "wall_x": L / 2.0})
                poly.Add(gp_Pnt(L / 2.0, y0, H))
                poly.Add(gp_Pnt(L / 2.0, y0, H + height))
                poly.Add(gp_Pnt(L / 2.0 - base, y0, H))
                prism_vec = gp_Vec(0, thickness, 0)
            elif side == "bottom":
                x0 = cx - thickness / 2.0
                x1 = cx + thickness / 2.0
                params.update({"x0": x0, "x1": x1, "wall_y": -W / 2.0})
                poly.Add(gp_Pnt(x0, -W / 2.0, H))
                poly.Add(gp_Pnt(x0, -W / 2.0, H + height))
                poly.Add(gp_Pnt(x0, -W / 2.0 + base, H))
                prism_vec = gp_Vec(thickness, 0, 0)
            else:
                x0 = cx - thickness / 2.0
                x1 = cx + thickness / 2.0
                params.update({"x0": x0, "x1": x1, "wall_y": W / 2.0})
                poly.Add(gp_Pnt(x0, W / 2.0, H))
                poly.Add(gp_Pnt(x0, W / 2.0, H + height))
                poly.Add(gp_Pnt(x0, W / 2.0 - base, H))
                prism_vec = gp_Vec(thickness, 0, 0)

            poly.Close()
            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            tool = BRepPrimAPI_MakePrism(face, prism_vec).Shape()

            fuse = BRepAlgoAPI_Fuse(stock, tool)
            fuse.Build()
            stock = fuse.Shape()
            actual_added += 1

            if side in {"left", "right"}:
                occupied_zones.append({"cx": zone_cx, "cy": zone_cy, "radius": max(base, thickness)})
            else:
                occupied_zones.append({"cx": zone_cx, "cy": zone_cy, "radius": max(base, thickness)})
            params_list.append(params)

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for gp in params_list:
            if not (H - 0.1 <= pz <= H + gp["height"] + 0.1):
                continue

            if gp["side"] == "left":
                if not (gp["y0"] - 0.1 <= py <= gp["y1"] + 0.1):
                    continue
                on_end = abs(py - gp["y0"]) < 0.1 or abs(py - gp["y1"]) < 0.1
                if on_end and abs(n.Y()) > 0.99:
                    return self.semantic_map["gusset_side_plane"], gp["inst"], self.feature_name

                if abs(n.Z()) < 0.99 and abs(n.Y()) < 0.99:
                    slope_x = gp["wall_x"] + gp["base"] * (1.0 - (pz - H) / gp["height"])
                    if abs(px - slope_x) < 0.12:
                        return self.semantic_map["gusset_slope_plane"], gp["inst"], self.feature_name
            elif gp["side"] == "right":
                if not (gp["y0"] - 0.1 <= py <= gp["y1"] + 0.1):
                    continue
                on_end = abs(py - gp["y0"]) < 0.1 or abs(py - gp["y1"]) < 0.1
                if on_end and abs(n.Y()) > 0.99:
                    return self.semantic_map["gusset_side_plane"], gp["inst"], self.feature_name

                if abs(n.Z()) < 0.99 and abs(n.Y()) < 0.99:
                    slope_x = gp["wall_x"] - gp["base"] * (1.0 - (pz - H) / gp["height"])
                    if abs(px - slope_x) < 0.12:
                        return self.semantic_map["gusset_slope_plane"], gp["inst"], self.feature_name
            elif gp["side"] == "bottom":
                if not (gp["x0"] - 0.1 <= px <= gp["x1"] + 0.1):
                    continue
                on_end = abs(px - gp["x0"]) < 0.1 or abs(px - gp["x1"]) < 0.1
                if on_end and abs(n.X()) > 0.99:
                    return self.semantic_map["gusset_side_plane"], gp["inst"], self.feature_name

                if abs(n.Z()) < 0.99 and abs(n.X()) < 0.99:
                    slope_y = gp["wall_y"] + gp["base"] * (1.0 - (pz - H) / gp["height"])
                    if abs(py - slope_y) < 0.12:
                        return self.semantic_map["gusset_slope_plane"], gp["inst"], self.feature_name
            else:
                if not (gp["x0"] - 0.1 <= px <= gp["x1"] + 0.1):
                    continue
                on_end = abs(px - gp["x0"]) < 0.1 or abs(px - gp["x1"]) < 0.1
                if on_end and abs(n.X()) > 0.99:
                    return self.semantic_map["gusset_side_plane"], gp["inst"], self.feature_name

                if abs(n.Z()) < 0.99 and abs(n.X()) < 0.99:
                    slope_y = gp["wall_y"] - gp["base"] * (1.0 - (pz - H) / gp["height"])
                    if abs(py - slope_y) < 0.12:
                        return self.semantic_map["gusset_slope_plane"], gp["inst"], self.feature_name

        return None
