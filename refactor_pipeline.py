import os

base_feature = """class BaseFeatureGenerator:
    def __init__(self):
        self.semantic_map = {"stock_face": 0}
        self.feature_name = "base"
        
    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id, occupied_zones):
        raise NotImplementedError()
        
    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        return None
"""

stepped_hole = """import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator

class SteppedHoleGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_name = "stepped_hole"
        self.semantic_map = {
            "stock_face": 0, "large_hole_cylinder": 1, "step_plane": 2, 
            "small_hole_cylinder": 3, "small_hole_bottom": 4     
        }

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        min_base_size = min(L, W)
        max_rad = min_base_size * 0.2
        max_depth = H * 0.8
        
        for i in range(num_instances):
            R1 = random.uniform(max_rad * 0.4, max_rad)
            R2 = random.uniform(R1 * 0.4, R1 * 0.8)
            D1 = random.uniform(max_depth * 0.2, max_depth * 0.5)
            D2 = random.uniform(max_depth * 0.2, max_depth * 0.5)
            
            valid = False
            cx, cy = 0, 0
            for attempt in range(50): 
                tx = random.uniform(-L/2 + R1 + 5.0, L/2 - R1 - 5.0)
                ty = random.uniform(-W/2 + R1 + 5.0, W/2 - R1 - 5.0)
                col = any(math.hypot(tx-z['cx'], ty-z['cy']) < (R1+z['radius']+2.0) for z in occupied_zones)
                if not col:
                    cx, cy, valid = tx, ty, True
                    break
            
            if not valid: continue

            inst_id = start_inst_id + actual_added
            holes_params.append({"inst": inst_id, "cx": cx, "cy": cy, "R1": R1, "D1": D1, "R2": R2, "D2": D2})
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R1})

            poly = BRepBuilderAPI_MakePolygon()
            poly.Add(gp_Pnt(cx, cy, H + 1.0)); poly.Add(gp_Pnt(cx+R1, cy, H + 1.0)); poly.Add(gp_Pnt(cx+R1, cy, H - D1))          
            poly.Add(gp_Pnt(cx+R2, cy, H - D1)); poly.Add(gp_Pnt(cx+R2, cy, H - D1 - D2)); poly.Add(gp_Pnt(cx, cy, H - D1 - D2))     
            poly.Close()                             
            
            tool = BRepPrimAPI_MakeRevol(BRepBuilderAPI_MakeFace(poly.Wire()).Face(), gp_Ax1(gp_Pnt(cx, cy, 0), gp_Dir(0, 0, 1))).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1

        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            if math.hypot(face_cx - hp["cx"], face_cy - hp["cy"]) < hp["R1"] + 0.1:
                if surf.GetType() == GeomAbs_Cylinder:
                    try:
                        return (self.semantic_map["large_hole_cylinder"] if surf.Cylinder().Radius() > (hp["R1"]+hp["R2"])/2 
                                else self.semantic_map["small_hole_cylinder"]), hp["inst"], self.feature_name
                    except: pass
                elif surf.GetType() == GeomAbs_Plane and surf.Plane().Axis().Direction().IsParallel(gp_Dir(0,0,1), 1e-4):
                    if abs(pnt.Z() - (H - hp["D1"])) < 0.1: return self.semantic_map["step_plane"], hp["inst"], self.feature_name
                    if abs(pnt.Z() - (H - hp["D1"] - hp["D2"])) < 0.1: return self.semantic_map["small_hole_bottom"], hp["inst"], self.feature_name
        return None
"""

through_hole = """import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator

class ThroughHoleGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_name = "through_hole"
        self.semantic_map = {"stock_face": 0, "through_hole_cylinder": 5}

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        max_rad = min(L, W) * 0.15
        
        for i in range(num_instances):
            R = random.uniform(max_rad * 0.3, max_rad)
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
            holes_params.append({"inst": inst_id, "cx": cx, "cy": cy, "R": R})
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R})

            tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, -1.0), gp_Dir(0, 0, 1)), R, H + 2.0).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            if math.hypot(face_cx - hp["cx"], face_cy - hp["cy"]) < hp["R"] + 0.1:
                if surf.GetType() == GeomAbs_Cylinder:
                    return self.semantic_map["through_hole_cylinder"], hp["inst"], self.feature_name
        return None
"""

blind_hole = """import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator

class BlindHoleGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_name = "blind_hole"
        self.semantic_map = {"stock_face": 0, "blind_hole_cylinder": 6, "blind_hole_bottom": 7}

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id=1, occupied_zones=None):
        if occupied_zones is None: occupied_zones = []
        holes_params = []
        actual_added = 0
        max_rad = min(L, W) * 0.15
        max_depth = H * 0.8
        
        for i in range(num_instances):
            R = random.uniform(max_rad * 0.3, max_rad)
            Depth = random.uniform(max_depth * 0.3, max_depth)
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
            holes_params.append({"inst": inst_id, "cx": cx, "cy": cy, "R": R, "Depth": Depth})
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': R})

            tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, H - Depth), gp_Dir(0, 0, 1)), R, Depth + 2.0).Shape()
            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, holes_params

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        for hp in params_list:
            if math.hypot(face_cx - hp["cx"], face_cy - hp["cy"]) < hp["R"] + 0.1:
                if surf.GetType() == GeomAbs_Cylinder:
                    return self.semantic_map["blind_hole_cylinder"], hp["inst"], self.feature_name
                elif surf.GetType() == GeomAbs_Plane and surf.Plane().Axis().Direction().IsParallel(gp_Dir(0,0,1), 1e-4):
                    if abs(pnt.Z() - (H - hp["Depth"])) < 0.1:
                        return self.semantic_map["blind_hole_bottom"], hp["inst"], self.feature_name
        return None
"""

obround_hole = """import math
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Ax3, gp_Trsf, gp_Vec, gp_Circ
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from features.base_feature import BaseFeatureGenerator

class ObroundHoleGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.feature_name = "obround_hole"
        self.semantic_map = {"stock_face": 0, "obround_side_plane": 8, "obround_end_cylinder": 9}

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
            cx, cy = 0, 0
            for attempt in range(50): 
                tx = random.uniform(-L/2 + bounding_radius + 3.0, L/2 - bounding_radius - 3.0)
                ty = random.uniform(-W/2 + bounding_radius + 3.0, W/2 - bounding_radius - 3.0)
                if not any(math.hypot(tx-z['cx'], ty-z['cy']) < (bounding_radius+z['radius']+3.0) for z in occupied_zones):
                    cx, cy, valid = tx, ty, True
                    break
            if not valid: continue

            inst_id = start_inst_id + actual_added
            params_list.append({"inst": inst_id, "cx": cx, "cy": cy, "R": R, "dist": dist, "angle": angle})
            occupied_zones.append({'cx': cx, 'cy': cy, 'radius': bounding_radius})

            w = BRepBuilderAPI_MakeWire(
                BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(dist/2, 0, 0), gp_Dir(0, 0, 1)), R), -math.pi/2, math.pi/2).Edge(),
                BRepBuilderAPI_MakeEdge(gp_Pnt(dist/2, R, 0), gp_Pnt(-dist/2, R, 0)).Edge(),
                BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(-dist/2, 0, 0), gp_Dir(0, 0, 1)), R), math.pi/2, 3*math.pi/2).Edge(),
                BRepBuilderAPI_MakeEdge(gp_Pnt(-dist/2, -R, 0), gp_Pnt(dist/2, -R, 0)).Edge()
            ).Wire()
            prism = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(w).Face(), gp_Vec(0, 0, H + 2.0)).Shape()
            
            t1 = gp_Trsf(); t1.SetTranslation(gp_Vec(cx, cy, -1.0))
            t2 = gp_Trsf(); t2.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle)
            tool = BRepBuilderAPI_Transform(prism, t1.Multiplied(t2)).Shape()

            cut = BRepAlgoAPI_Cut(stock, tool); cut.Build(); stock = cut.Shape()
            actual_added += 1
        return stock, actual_added, occupied_zones, params_list

    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        px, py = pnt.X(), pnt.Y()
        for hp in params_list:
            dx, dy = px - hp["cx"], py - hp["cy"]
            loc_x = dx * math.cos(-hp["angle"]) - dy * math.sin(-hp["angle"])
            loc_y = dx * math.sin(-hp["angle"]) + dy * math.cos(-hp["angle"])
            
            dist_val = abs(loc_y) if abs(loc_x) <= hp["dist"]/2.0 else math.hypot(abs(loc_x)-hp["dist"]/2.0, loc_y)
            if abs(dist_val - hp["R"]) < 0.1:
                if surf.GetType() == GeomAbs_Cylinder:
                    return self.semantic_map["obround_end_cylinder"], hp["inst"], self.feature_name
                elif surf.GetType() == GeomAbs_Plane and abs(surf.Plane().Axis().Direction().Z()) < 1e-4:
                    return self.semantic_map["obround_side_plane"], hp["inst"], self.feature_name
        return None
"""

main_py = """import os
import random
import json
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

from core.AAGExtractor import extract_aag_graph
from features.stepped_hole import SteppedHoleGenerator
from features.through_hole import ThroughHoleGenerator
from features.blind_hole import BlindHoleGenerator
from features.obround_hole import ObroundHoleGenerator

def build_single_dataset_sample(L, W, H, output_step, output_json):
    stock = BRepPrimAPI_MakeBox(gp_Pnt(-L/2, -W/2, 0), gp_Pnt(L/2, W/2, H)).Shape()
    
    current_inst_id = 1
    occupied_zones = []
    feature_registry = [] 
    
    generators = [SteppedHoleGenerator(), ThroughHoleGenerator(), BlindHoleGenerator(), ObroundHoleGenerator()]
    
    for gen in generators:
        stock, added, occupied_zones, params = gen.apply_feature(
            stock, L, W, H, random.randint(0, 2), current_inst_id, occupied_zones
        )
        if added > 0:
            feature_registry.append((gen, params))
            current_inst_id += added

    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

    final_face_infos = {}
    for face in TopologyExplorer(stock).faces():
        surf = BRepAdaptor_Surface(face)
        u = (surf.FirstUParameter() + surf.LastUParameter()) / 2.0
        v = (surf.FirstVParameter() + surf.LastVParameter()) / 2.0
        try: pnt = surf.Value(u, v)
        except: continue
        
        face_cx, face_cy = pnt.X(), pnt.Y()
        if surf.GetType() == GeomAbs_Cylinder:
            try: face_cx, face_cy = surf.Cylinder().Location().X(), surf.Cylinder().Location().Y()
            except: pass
        
        matched_res = None
        for gen, params in feature_registry:
            matched_res = gen.match_face(surf, pnt, face_cx, face_cy, H, params)
            if matched_res is not None: break

        if matched_res:
            final_face_infos[face.__hash__()] = {"semantic_label": matched_res[0], "instance_id": matched_res[1], "macro_feature": matched_res[2]}
        else:
            final_face_infos[face.__hash__()] = {"semantic_label": 0, "instance_id": 0, "macro_feature": "stock"}

    graph_data = extract_aag_graph(stock, final_face_infos)
    
    writer = STEPControl_Writer()
    writer.Transfer(stock, STEPControl_AsIs)
    writer.Write(output_step)
    with open(output_json, 'w', encoding='utf-8') as f: json.dump(graph_data, f, indent=4)
    print(f"✅ 成功生成样本，特征数: {current_inst_id-1} -> {output_step}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "output", "steps"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "output", "graphs"), exist_ok=True)
    for i in range(5):
        build_single_dataset_sample(random.uniform(100, 150), random.uniform(100, 150), random.uniform(40, 60),
            os.path.join(current_dir, "output", "steps", f"composite_feature_{i:03d}.step"),
            os.path.join(current_dir, "output", "graphs", f"composite_feature_{i:03d}.json"))
"""

base_dir = "/home/phl/ywx/FEA/dataset_generator/"
with open(base_dir + "features/base_feature.py", "w") as f: f.write(base_feature)
with open(base_dir + "features/stepped_hole.py", "w") as f: f.write(stepped_hole)
with open(base_dir + "features/through_hole.py", "w") as f: f.write(through_hole)
with open(base_dir + "features/blind_hole.py", "w") as f: f.write(blind_hole)
with open(base_dir + "features/obround_hole.py", "w") as f: f.write(obround_hole)
with open(base_dir + "main.py", "w") as f: f.write(main_py)
print("架构重构与部署完毕!")
