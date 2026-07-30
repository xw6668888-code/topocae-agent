import math
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Pnt

from features.base_feature import BaseFeatureGenerator


class NotchReliefGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "notch_relief"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "notch_relief_side_plane",
            "notch_relief_bottom_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None:
            occupied_zones = []

        params_list = []
        actual_added = 0
        min_base_size = min(L, W)

        for _ in range(num_instances):
            span = random.uniform(min_base_size * 0.10, min_base_size * 0.22)
            depth = random.uniform(min_base_size * 0.06, min_base_size * 0.14)
            height = random.uniform(H * 0.10, H * 0.26)
            side = random.choice(["left", "right", "bottom", "top"])

            if side in {"left", "right"}:
                cy = random.uniform(-W / 2 + span / 2.0 + 4.0, W / 2 - span / 2.0 - 4.0)
                cx = -L / 2.0 + depth / 2.0 if side == "left" else L / 2.0 - depth / 2.0
            else:
                cx = random.uniform(-L / 2 + span / 2.0 + 4.0, L / 2 - span / 2.0 - 4.0)
                cy = -W / 2.0 + depth / 2.0 if side == "bottom" else W / 2.0 - depth / 2.0

            bounding_radius = math.hypot(span / 2.0, depth / 2.0)
            if any(
                math.hypot(cx - z["cx"], cy - z["cy"]) < (bounding_radius + z["radius"] + 2.0)
                for z in occupied_zones
            ):
                continue

            inst_id = start_inst_id + actual_added
            z0 = H - height
            if side == "left":
                x0 = -L / 2.0 - 1.0
                x_back = -L / 2.0 + depth
                y0 = cy - span / 2.0
                y1 = cy + span / 2.0
                corner = gp_Pnt(x0, y0, z0)
                tool = BRepPrimAPI_MakeBox(corner, depth + 1.0, span, height + 1.0).Shape()
            elif side == "right":
                x0 = L / 2.0 - depth
                x_back = L / 2.0 - depth
                y0 = cy - span / 2.0
                y1 = cy + span / 2.0
                corner = gp_Pnt(x0, y0, z0)
                tool = BRepPrimAPI_MakeBox(corner, depth + 1.0, span, height + 1.0).Shape()
            elif side == "bottom":
                x0 = cx - span / 2.0
                x1 = cx + span / 2.0
                y0 = -W / 2.0 - 1.0
                y_back = -W / 2.0 + depth
                corner = gp_Pnt(x0, y0, z0)
                tool = BRepPrimAPI_MakeBox(corner, span, depth + 1.0, height + 1.0).Shape()
            else:
                x0 = cx - span / 2.0
                x1 = cx + span / 2.0
                y0 = W / 2.0 - depth
                y_back = W / 2.0 - depth
                corner = gp_Pnt(x0, y0, z0)
                tool = BRepPrimAPI_MakeBox(corner, span, depth + 1.0, height + 1.0).Shape()

            params = {
                "inst": inst_id,
                "cx": cx,
                "cy": cy,
                "span": span,
                "depth": depth,
                "height": height,
                "side": side,
                "radius": bounding_radius,
                "z0": z0,
            }
            if side in {"left", "right"}:
                params.update({"y0": y0, "y1": y1, "x_back": x_back})
            else:
                params.update({"x0": x0, "x1": x1, "y_back": y_back})

            params_list.append(params)
            occupied_zones.append({"cx": cx, "cy": cy, "radius": bounding_radius})

            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None

        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        n = surf.Plane().Axis().Direction()

        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["radius"] + 0.4:
                continue

            if hp["side"] in {"left", "right"}:
                inside_patch = hp["y0"] - 0.1 <= py <= hp["y1"] + 0.1 and hp["z0"] - 0.1 <= pz <= H + 0.1
            else:
                inside_patch = hp["x0"] - 0.1 <= px <= hp["x1"] + 0.1 and hp["z0"] - 0.1 <= pz <= H + 0.1

            if abs(n.Z()) > 0.99 and inside_patch and abs(pz - (H - hp["height"])) < 0.1:
                return self.semantic_map["notch_relief_bottom_plane"], hp["inst"], self.feature_name

            if abs(n.Z()) < 1e-4 and (H - hp["height"] - 0.1) <= pz <= (H + 0.1) and inside_patch:
                if hp["side"] in {"left", "right"}:
                    side_match = (
                        abs(py - hp["y0"]) < 0.1
                        or abs(py - hp["y1"]) < 0.1
                        or abs(px - hp["x_back"]) < 0.1
                    )
                else:
                    side_match = (
                        abs(px - hp["x0"]) < 0.1
                        or abs(px - hp["x1"]) < 0.1
                        or abs(py - hp["y_back"]) < 0.1
                    )
                if side_match:
                    return self.semantic_map["notch_relief_side_plane"], hp["inst"], self.feature_name

        return None
