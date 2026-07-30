import math
import random
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Plane
from features.base_feature import BaseFeatureGenerator

class PolyhedronBossGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "polyhedron_boss"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "triangle_prism_side",
            "triangle_prism_top",
            "rectangle_prism_side",
            "rectangle_prism_top",
            "triangle_frustum_side",
            "triangle_frustum_top",
            "rectangle_frustum_side",
            "rectangle_frustum_top",
        ])

    def get_wire(self, pts, cx, cy, z, angle, scale):
        poly = BRepBuilderAPI_MakePolygon()
        for px, py in pts:
            # 缩放
            sx = px * scale
            sy = py * scale
            # 旋转
            rx = sx * math.cos(angle) - sy * math.sin(angle)
            ry = sx * math.sin(angle) + sy * math.cos(angle)
            # 平移
            poly.Add(gp_Pnt(cx + rx, cy + ry, z))
        poly.Close()
        return poly.Wire()

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        params_list = []
        actual_added = 0
        
        min_base_size = min(L, W)
        max_rad = min_base_size * 0.15
        
        for i in range(num_instances):
            # 随机选择多边形和凸台类型
            shape_type = random.choice(["triangle", "rectangle"])
            feat_type = random.choice(["prism", "frustum"]) # prism=棱柱, frustum=棱台
            
            R = random.uniform(max_rad * 0.4, max_rad) # 外包安全检测半径
            boss_h = random.uniform(H * 0.1, H * 0.4) # 高度
            angle = random.uniform(0, math.pi)
            top_scale = 1.0 if feat_type == "prism" else random.uniform(0.3, 0.7) # 棱台的顶部缩放比例
            
            valid = False
            cx, cy = 0, 0
            for attempt in range(50):
                tx = random.uniform(-L/2 + R + 5.0, L/2 - R - 5.0)
                ty = random.uniform(-W/2 + R + 5.0, W/2 - R - 5.0)
                if not any(math.hypot(tx-z['cx'], ty-z['cy']) < (R+z['radius']+2.0) for z in occupied_zones):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            
            pts_2d = []
            if shape_type == "rectangle":
                w = random.uniform(R, 1.8*R)
                hl = random.uniform(R, 1.8*R)
                pts_2d = [(-w/2, -hl/2), (w/2, -hl/2), (w/2, hl/2), (-w/2, hl/2)]
            else: # triangle
                pts_2d = [
                    (0, R),
                    (-R * math.cos(math.pi/6), -R * math.sin(math.pi/6)),
                    (R * math.cos(math.pi/6), -R * math.sin(math.pi/6))
                ]

            params_list.append({
                "inst": inst_id, "shape": shape_type, "ftype": feat_type,
                "cx": cx, "cy": cy, "angle": angle, "pts": pts_2d, 
                "R": R, "h": boss_h, "top_scale": top_scale
            })
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R})

            if feat_type == "prism":
                # 拉伸生成棱柱
                wire = self.get_wire(pts_2d, cx, cy, H, angle, 1.0)
                face = BRepBuilderAPI_MakeFace(wire).Face()
                tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, boss_h)).Shape()
            else: 
                # 扫掠生成棱台 (放样)
                wire1 = self.get_wire(pts_2d, cx, cy, H, angle, 1.0)
                wire2 = self.get_wire(pts_2d, cx, cy, H + boss_h, angle, top_scale)
                
                # isSolid=True, isRuled=True -> 确保生成的侧面为平面而不是样条曲面
                generator = BRepOffsetAPI_ThruSections(True, True)
                generator.AddWire(wire1)
                generator.AddWire(wire2)
                generator.Build()
                tool = generator.Shape()

            fuse = BRepAlgoAPI_Fuse(stock, tool)
            fuse.Build()
            stock = fuse.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        
        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["R"] + 0.1:
                continue
            if not ((H - 0.1) <= pz <= (H + hp["h"] + 0.1)):
                continue

            # 1. 匹配顶面
            if surf.GetType() == GeomAbs_Plane:
                n = surf.Plane().Axis().Direction()
                if abs(n.X()) < 1e-4 and abs(n.Y()) < 1e-4 and abs(n.Z()) > 0.99:
                    if abs(pz - (H + hp["h"])) < 1e-4:
                        lbl = self.semantic_map[f"{hp['shape']}_{hp['ftype']}_top"]
                        return lbl, hp["inst"], f"{hp['shape']}_{hp['ftype']}"
            
            # 2. 匹配侧面
            Z_rel = pz - H
            if Z_rel < -0.1 or Z_rel > hp["h"] + 0.1:
                continue
                
            # 根据当前点高度计算截面缩放比例
            if hp["ftype"] == "prism":
                current_scale = 1.0
            else: # frustum
                current_scale = 1.0 - (1.0 - hp["top_scale"]) * (Z_rel / hp["h"])
                
            # 逆向变换判断是否在多边形边上
            dx = px - hp["cx"]
            dy = py - hp["cy"]
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])
            
            spts = [(u*current_scale, v*current_scale) for u, v in hp["pts"]]
            pts_len = len(spts)
            
            dist = float('inf')
            for i in range(pts_len):
                x1, y1 = spts[i]
                x2, y2 = spts[(i+1)%pts_len]
                l2 = (x1-x2)**2 + (y1-y2)**2
                if l2 == 0:
                    d = math.hypot(loc_x-x1, loc_y-y1)
                else:
                    t = max(0, min(1, ((loc_x-x1)*(x2-x1) + (loc_y-y1)*(y2-y1)) / l2))
                    proj_x = x1 + t*(x2-x1)
                    proj_y = y1 + t*(y2-y1)
                    d = math.hypot(loc_x-proj_x, loc_y-proj_y)
                dist = min(dist, d)
                
            if dist < 0.1:
                if surf.GetType() == GeomAbs_Plane:
                    n = surf.Plane().Axis().Direction()
                    if abs(n.Z()) > 0.95:
                        continue
                lbl = self.semantic_map[f"{hp['shape']}_{hp['ftype']}_side"]
                return lbl, hp["inst"], f"{hp['shape']}_{hp['ftype']}"

        return None
