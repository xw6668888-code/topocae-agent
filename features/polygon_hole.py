import math
import random
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane
from features.base_feature import BaseFeatureGenerator

class PolygonHoleGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "polygon_hole"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "triangle_hole_side_plane",
            "rectangular_hole_side_plane",
        ])

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        params_list = []
        actual_added = 0
        
        min_base_size = min(L, W)
        max_rad = min_base_size * 0.15
        
        for i in range(num_instances):
            # 随机选择是三角形还是矩形
            poly_type = random.choice(["triangle", "rectangle"])
            R = random.uniform(max_rad * 0.4, max_rad) # 外包安全检测半径
            angle = random.uniform(0, math.pi)
            
            valid = False
            cx, cy = 0, 0
            for attempt in range(50):
                tx = random.uniform(-L/2 + R + 5.0, L/2 - R - 5.0)
                ty = random.uniform(-W/2 + R + 5.0, W/2 - R - 5.0)
                # 使用外接圆半径和其它特征进行碰撞检测
                if not any(math.hypot(tx-z['cx'], ty-z['cy']) < (R+z['radius']+2.0) for z in occupied_zones):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            
            # 在局部原点生成 2D 坐标轮廓点
            pts_2d = []
            if poly_type == "rectangle":
                # 随机生成长宽
                w = random.uniform(R, 1.8*R)
                h = random.uniform(R, 1.8*R)
                pts_2d = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            else: # triangle
                # 生成等边三角形（内接圆倒推控制）
                pts_2d = [
                    (0, R),
                    (-R * math.cos(math.pi/6), -R * math.sin(math.pi/6)),
                    (R * math.cos(math.pi/6), -R * math.sin(math.pi/6))
                ]
            
            params_list.append({
                "inst": inst_id, "type": poly_type, 
                "cx": cx, "cy": cy, "angle": angle, "pts": pts_2d, "R": R
            })
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R})

            # 通过线段闭合构造草图
            poly = BRepBuilderAPI_MakePolygon()
            for pt in pts_2d:
                px, py = pt
                # 矩阵旋转与平移
                rot_x = px * math.cos(angle) - py * math.sin(angle)
                rot_y = px * math.sin(angle) + py * math.cos(angle)
                poly.Add(gp_Pnt(cx + rot_x, cy + rot_y, -1.0))
            poly.Close()
            
            # 成面与拉伸 (贯穿毛坯)
            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, H + 2.0)).Shape()

            # 布尔切除
            cut = BRepAlgoAPI_Cut(stock, tool)
            cut.Build()
            stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        if surf.GetType() != GeomAbs_Plane:
            return None
        
        # 必须是垂直于工作平面的面 (法向的Z分量为0)
        n = surf.Plane().Axis().Direction()
        if abs(n.Z()) > 1e-4:
            return None
            
        px, py = pnt.X(), pnt.Y()
        
        for hp in params_list:
            # 1. 外接圆快速排除
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["R"] + 0.1:
                continue
                
            # 2. 变换探测点回局部坐标系
            dx = px - hp["cx"]
            dy = py - hp["cy"]
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])
            
            # 3. 求点到各个边线段的最短距离
            pts = hp["pts"]
            pts_len = len(pts)
            
            def point_to_segment_dist(px, py, x1, y1, x2, y2):
                l2 = (x1-x2)**2 + (y1-y2)**2
                if l2 == 0: return math.hypot(px-x1, py-y1)
                t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / l2))
                proj_x = x1 + t*(x2-x1)
                proj_y = y1 + t*(y2-y1)
                return math.hypot(px-proj_x, py-proj_y)

            for i in range(pts_len):
                p1 = pts[i]
                p2 = pts[(i+1)%pts_len]
                dist = point_to_segment_dist(loc_x, loc_y, p1[0], p1[1], p2[0], p2[1])
                # 如果点属于这个多边形截面的边界线上
                if dist < 0.1:
                    lbl = self.semantic_map["triangle_hole_side_plane"] if hp["type"] == "triangle" else self.semantic_map["rectangular_hole_side_plane"]
                    macro_name = "triangle_hole" if hp["type"] == "triangle" else "rectangular_hole"
                    return lbl, hp["inst"], macro_name
        return None
