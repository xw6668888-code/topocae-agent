"""Resolve roles and measurements only within frozen recognized feature objects."""
from __future__ import annotations

import itertools
import json
import math
from preflight import OUT, ROOT, load_case, sha
from plan_contract import PlanError
from parameter_policy import SELECTOR_RADIUS_MM


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def sub(a,b):
    return [x-y for x,y in zip(a,b)]


def add(a,b):
    return [x+y for x,y in zip(a,b)]


def mul(a,s):
    return [x*s for x in a]


def unit(a):
    return mul(a,1/math.hypot(*a))


def load(part):
    source = ROOT / "test_part_0901" / part
    prediction = OUT / "recognition" / (source.stem + "_recognition.json")
    data = json.loads((OUT / "geometry" / (source.stem + ".json")).read_text(encoding="utf-8"))
    if data["source_sha256"] != sha(source) or data["prediction_sha256"] != sha(prediction):
        raise PlanError("stale_geometry")
    return load_case(source=source, prediction_path=prediction), data


def select(data, selector):
    matches = [o for o in data["objects"] if o["class"] in selector["feature_classes"]]
    if selector["radius_mm"] is not None:
        radius = selector["radius_mm"]
        matches = [o for o in matches if any(abs(r-radius)<SELECTOR_RADIUS_MM for r in o["geometry_summary"]["positive_radii"])]
    if not matches:
        raise PlanError("no_target")
    if any(not o["topology_valid"] for o in matches):
        raise PlanError("invalid_predicted_topology")
    if selector["scope"] == "single" and len(matches) != 1:
        raise PlanError("needs_clarification", "Multiple recognized instances match")
    return matches


def roles(data, objects, role):
    records = {r["face_number"]: r for r in data["faces"]}
    groups = []
    for obj in objects:
        rows = [records[n] for n in obj["original_face_numbers"]]
        evidence = {"method":"within_recognized_instance", "instance_faces":obj["original_face_numbers"]}
        if role == "top_face":
            # IDs are fixed by the checkpoint's frozen face semantic ontology.
            semantic = {"cylinder_boss":13, "rectangular_boss":51}.get(obj["class"])
            if semantic is None:
                raise PlanError("unsupported_role")
            rows = [r for r in rows if r["semantic_label"] == semantic and "normal" in r]
            if not rows and obj["class"] == "cylinder_boss":
                rows, path = complete_cylinder_top(data,obj,semantic)
                evidence.update(method="predicted_top_semantics_with_transition_adjacency", transition_path=path)
            if len(rows) != 1:
                raise PlanError("ambiguous_top_face")
        elif role == "inner_walls":
            if obj["class"] not in {"through_hole_round","obround_hole"}:
                raise PlanError("unsupported_role")
        elif role == "cavity_interior":
            if obj["class"] != "pocket_circular" or not any("normal" in r for r in rows):
                raise PlanError("unsupported_cavity")
        elif role != "whole_feature":
            raise PlanError("unsupported_role")
        groups.append({"object_index":obj["object_index"],"class":obj["class"],"role":role,
                       "face_numbers":[r["face_number"] for r in rows],
                       "area_mm2":sum(r["area_mm2"] for r in rows), "role_evidence":evidence})
    return groups


def complete_cylinder_top(data,obj,semantic):
    graph_path = OUT / "recognition" / (data["part"][:-4]+"_graph.json")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if len(graph["nodes"]) != len(data["faces"]):
        raise PlanError("graph_geometry_size_mismatch")
    wall = [data["faces"][n-1] for n in obj["original_face_numbers"] if "radius_mm" in data["faces"][n-1]]
    if len(wall)!=1:
        return [],[]
    wall=wall[0]
    neighbors={i+1:set() for i in range(len(data["faces"]))}
    for a,b in graph["edges"]:
        neighbors[a+1].add(b+1)
    transitions={n for o in data["objects"] if o["class"] in {"chamfer","fillet"} for n in o["original_face_numbers"]}
    paths={n:[n] for n in obj["original_face_numbers"]}
    for _ in range(3):
        for n,path in list(paths.items()):
            if n not in obj["original_face_numbers"] and n not in transitions:
                continue
            for other in neighbors[n]:
                if other not in paths:
                    paths[other]=path+[other]
    candidates=[]
    for row in data["faces"]:
        if row["semantic_label"]!=semantic or "normal" not in row or row["face_number"] not in paths:
            continue
        offset=sub(row["center_mm"],wall["origin_mm"])
        radial=sub(offset,mul(wall["axis"],dot(offset,wall["axis"])))
        if math.hypot(*radial)<1e-5 and abs(abs(dot(row["normal"],wall["axis"]))-1)<1e-6:
            candidates.append(row)
    return (candidates,paths[candidates[0]["face_number"]]) if len(candidates)==1 else ([],[])


def cylinder(row):
    if "radius_mm" not in row:
        raise PlanError("noncylindrical_face")
    radius = row["radius_mm"]
    depth = row["v_range"][1]-row["v_range"][0]
    if abs(row["u_range"][1]-row["u_range"][0]-2*math.pi)>1e-6 or abs(row["area_mm2"]-2*math.pi*radius*depth)>1e-4:
        raise PlanError("trimmed_cylinder_not_supported_for_edit")
    return {"face_number":row["face_number"],"radius_mm":radius,"depth_mm":depth,"axis":row["axis"],
            "start_mm":add(row["origin_mm"],mul(row["axis"],row["v_range"][0])),"center_mm":row["center_mm"]}


def thickness(data,obj):
    rows = [data["faces"][n-1] for n in obj["original_face_numbers"]]
    pairs = []
    for a,b in itertools.combinations(rows,2):
        if "normal" not in a or "normal" not in b or dot(a["normal"],b["normal"])>-1+1e-7:
            continue
        displacement = sub(b["center_mm"],a["center_mm"])
        distance = abs(dot(displacement,a["normal"]))
        if distance < 1e-6 or abs(a["area_mm2"]-b["area_mm2"])>1e-4:
            continue
        av, bv = a["vertices_mm"], b["vertices_mm"]
        if len(av)!=len(bv) or any(min(math.dist(add(v,displacement),w) for w in bv)>1e-5 for v in av):
            continue
        pairs.append({"object_index":obj["object_index"],"side_faces":[a["face_number"],b["face_number"]],
                      "thickness_mm":distance,"sweep_vector_mm":displacement,"side_area_mm2":a["area_mm2"],
                      "midpoint_mm":mul(add(a["center_mm"],b["center_mm"]),.5),"normal":a["normal"]})
    if not pairs:
        raise PlanError("parallel_congruent_sides_not_found")
    return max(pairs,key=lambda row:row["side_area_mm2"])


def public_catalog(data):
    records = []
    for o in data["objects"]:
        params = {}
        radii = o["geometry_summary"]["positive_radii"]
        if radii:
            params["radii_mm"] = radii
        if o["class"] in {"rib","web"}:
            try:
                params["thickness_mm"] = thickness(data,o)["thickness_mm"]
            except PlanError:
                pass
        available_roles = ["whole_feature"]
        for role in ("inner_walls","top_face","cavity_interior"):
            try:
                roles(data,[o],role)
                available_roles.append(role)
            except PlanError:
                pass
        records.append({"instance_id":"f%04d"%o["object_index"],"class":o["class"],
                        "parameters":params,"available_roles":available_roles})
    return {"schema_version":"feature_catalog/2","geometry_version":data["source_sha256"],"instances":records}
