import json
import os
from collections import Counter

def check_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"=== 分析文件: {os.path.basename(json_path)} ===")
    
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    print(f"面(节点)总数: {len(nodes)}")
    print(f"边(连接)总数: {len(edges)}")
    
    labels = [n.get("semantic_label", -1) for n in nodes]
    counter = Counter(labels)
    
    label_names = {
        0: "毛坯面 (Stock Face)",
        1: "阶梯孔-大孔圆柱面",
        2: "阶梯孔-过渡面",
        3: "阶梯孔-小孔圆柱面",
        4: "阶梯孔-小孔底面",
        5: "圆通孔-圆柱面",
        6: "圆盲孔-圆柱面",
        7: "圆盲孔-底面",
        8: "长圆孔-侧平面",
        9: "长圆孔-端部半圆柱面",
        10: "三角形通孔-侧平面",
        11: "矩形通孔-侧平面",
        12: "圆柱凸台-侧圆柱面",
        13: "圆柱凸台-顶平面",
        14: "圆台凸台-侧圆柱锥面",
        15: "圆台凸台-顶平面",
        16: "三角形棱柱-侧平面",
        17: "三角形棱柱-顶平面",
        18: "矩形棱柱-侧平面",
        19: "矩形棱柱-顶平面",
        20: "三角形棱台-侧斜平面",
        21: "三角形棱台-顶平面",
        22: "矩形棱台-侧斜平面",
        23: "矩形棱台-顶平面"
    }
    
    print("\n特征详情 (带物理与几何特定参数):")
    if "feature_sets" in data:
        for fs in data["feature_sets"]:
            inst_id = fs.get("instance_id")
            macro = fs.get("macro_feature")
            node_ids = fs.get("node_indices", [])
            print(f"\n  [ 实例集 ID = {inst_id} ] 宏特征类型: {macro}")
            
            for nid in node_ids:
                if nid < len(nodes):
                    n = nodes[nid]
                    lbl = n["semantic_label"]
                    name = label_names.get(lbl, "未知")
                    attrs = n["attributes"]
                    
                    surf_type = attrs.get('surface_type_str', 'Unknown')
                    phys = attrs.get('physical', {})
                    spec = attrs.get('specific', {})
                    
                    area = phys.get('area', 0.0)
                    
                    # 动态拼接特定后缀
                    spec_str = []
                    for k, v in spec.items():
                        if isinstance(v, list):
                            spec_str.append(f"{k}: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
                        else:
                            spec_str.append(f"{k}: {v:.2f}")
                    spec_out = " | ".join(spec_str)

                    print(f"      -> 节点 {n['idx']:02d}: {name}")
                    print(f"         └─ 类别: {surf_type} | 面积: {area:.2f}")
                    if spec_out:
                        print(f"         └─ 特定参数: {spec_out}")

    print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.normpath(os.path.join(current_dir, "../output/graphs"))
    
    # 取前两个文件打印看效果
    files = [f for f in os.listdir(graphs_dir) if f.endswith(".json")]
    for filename in files[:2]:
        json_file = os.path.join(graphs_dir, filename)
        check_json(json_file)
