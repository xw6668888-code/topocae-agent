import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone
from features.base_feature import BaseFeatureGenerator

class BossFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, semantic_subtype_index):
        super().__init__(semantic_subtype_index)
        self.feature_name = "boss_feature"
        self.semantic_map = self.build_semantic_map([
            "stock_face",
            "cylinder_boss_side",
            "cylinder_boss_top",
            "cone_boss_side",
            "cone_boss_top",
        ])
        self.allowed_boss_types = ["cylinder"]

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        params_list = []
        actual_added = 0
        
        min_base_size = min(L, W)
        max_rad = min_base_size * 0.15
        
        for i in range(num_instances):
            # V1 static-structural scope keeps cylindrical bosses as the default.
            boss_type = random.choice(self.allowed_boss_types)
            R1 = random.uniform(max_rad * 0.4, max_rad) # 底部半径
            boss_h = random.uniform(H * 0.1, H * 0.4) # 凸台高度
            
            valid = False
            cx, cy = 0, 0
            for attempt in range(50):
                tx = random.uniform(-L/2 + R1 + 5.0, L/2 - R1 - 5.0)
                ty = random.uniform(-W/2 + R1 + 5.0, W/2 - R1 - 5.0)
                if not any(math.hypot(tx-z['cx'], ty-z['cy']) < (R1+z['radius']+2.0) for z in occupied_zones):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            
            if boss_type == "cylinder":
                # 圆柱凸台
                R2 = R1
                tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, H), gp_Dir(0, 0, 1)), R1, boss_h).Shape()
            else:
                # 圆台凸台 (这里直接使用 MakeCone，等价于通过梯形旋转生成的圆锥几何实体)
                R2 = random.uniform(R1 * 0.3, R1 * 0.8) # 顶部半径小于底部半径
                tool = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(cx, cy, H), gp_Dir(0, 0, 1)), R1, R2, boss_h).Shape()
            
            params_list.append({
                "inst": inst_id, "type": boss_type, 
                "cx": cx, "cy": cy, "R1": R1, "R2": R2, "h": boss_h
            })
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R1})

            # 布尔求和 (Fuse) 而不是切除 (Cut)
            fuse = BRepAlgoAPI_Fuse(stock, tool)
            fuse.Build()
            stock = fuse.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        px, py, pz = pnt.X(), pnt.Y(), pnt.Z()
        
        for hp in params_list:
            if math.hypot(px - hp["cx"], py - hp["cy"]) > hp["R1"] + 0.1:
                continue
                
            # 匹配顶面 (Plane)
            if surf.GetType() == GeomAbs_Plane:
                n = surf.Plane().Axis().Direction()
                # 顶面法向应该沿 Z 轴正向
                if abs(n.X()) < 1e-4 and abs(n.Y()) < 1e-4 and n.Z() > 0.99:
                    # 顶面Z高度等于毛坯高度 H + 凸台高度
                    if abs(pz - (H + hp["h"])) < 1e-4:
                        lbl = self.semantic_map["cylinder_boss_top"] if hp["type"] == "cylinder" else self.semantic_map["cone_boss_top"]
                        return lbl, hp["inst"], f"{hp['type']}_boss"
            
            # 匹配圆柱侧面 (Cylinder)
            if hp["type"] == "cylinder" and surf.GetType() == GeomAbs_Cylinder:
                if abs(surf.Cylinder().Radius() - hp["R1"]) < 1e-4:
                    if H - 0.1 <= pz <= H + hp["h"] + 0.1:
                        return self.semantic_map["cylinder_boss_side"], hp["inst"], "cylinder_boss"
                        
            # 匹配圆台侧面 (Cone)
            if hp["type"] == "cone" and surf.GetType() == GeomAbs_Cone:
                # 判断中心轴方向
                dir_z = surf.Cone().Axis().Direction().Z()
                if abs(abs(dir_z) - 1.0) < 1e-4: # 轴应该垂直于XY面
                    if H - 0.1 <= pz <= H + hp["h"] + 0.1:
                        return self.semantic_map["cone_boss_side"], hp["inst"], "cone_boss"

        return None
