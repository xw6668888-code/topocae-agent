"""Local B-Rep edits parameterized by recognized objects and validated plans."""
from __future__ import annotations

import math
from geometry import load, select, cylinder, thickness, add, sub, mul, unit, dot
from preflight import OUT, write_json, properties, xyz
from build_cases import clean, read_step, sha, build_cad, hole_geometry
from contracts import validate
from plan_contract import PlanError
from parameter_policy import SOURCE_THICKNESS_MM, SOURCE_SPACING_MM
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from preflight import measured

BOOLEAN_EVENTS=[]


def boolean(kind,first,second):
    for tolerance in (1e-7,1e-6,1e-5):
        op=kind(first,second)
        op.SetFuzzyValue(tolerance)
        op.Build()
        if op.IsDone() and BRepCheck_Analyzer(op.Shape()).IsValid():
            if tolerance>1e-7:
                BOOLEAN_EVENTS.append({"operation":kind.__name__,"fuzzy_tolerance_mm":tolerance})
            return op.Shape()
    raise PlanError("boolean_invalid_at_allowed_tolerances")


def move(shape,vector):
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(*vector))
    return BRepBuilderAPI_Transform(shape,transform,True).Shape()


def tube(row,radius=None,shift=None):
    start = add(row["start_mm"],shift or [0,0,0])
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*start),gp_Dir(*row["axis"])),
                                   radius if radius is not None else row["radius_mm"],row["depth_mm"]).Shape()


def volume(shape):
    return abs(properties(shape,True)[0])


def join(shapes):
    result = shapes[0]
    for shape in shapes[1:]:
        result = boolean(BRepAlgoAPI_Fuse,result,shape)
    return result


def execute(part,plan,output,*,execution_mode="local_reference_plan"):
    BOOLEAN_EVENTS.clear()
    validate(plan)
    if plan["status"] != "ready" or plan["domain"] != "cad":
        raise PlanError("nonready_cad_execution")
    case,data = load(part)
    source,original,faces,prediction,predpath = case
    op = plan["operations"][0]
    objects = select(data,op["target"])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output / "semantic_plan.json",plan)
    if op["action"] == "set_hole_radius" and {o["class"] for o in objects} == {"blind_hole_round"}:
        old_radius = op["target"]["radius_mm"]
        if old_radius is None:
            radii = [hole_geometry(o, faces)['radius_before_mm'] for o in objects]
            if not radii or any(abs(r - radii[0]) > 1e-6 for r in radii):
                raise PlanError('mixed_blind_hole_radii_unsupported')
            old_radius = radii[0]
        destination = build_cad(case=case,output=output,selected=objects,
                                old_radius=old_radius,new_radius=op["new_radius_mm"])
        import json
        audit = json.loads((output / 'edit_audit.json').read_text(encoding='utf-8'))
        audit.update(execution_mode=execution_mode, selector_radius_mm=op['target']['radius_mm'],
                     measured_execution_radius_mm=old_radius)
        write_json(output / 'edit_audit.json', audit)
        return destination
    changes, tools_add, tools_remove, checks = [], [], [], []
    changed = original
    expected_delta = 0.
    if op["action"] == "set_hole_radius":
        if {o["class"] for o in objects}!={"through_hole_round"}:
            raise PlanError("unsupported_radius_feature")
        for obj in objects:
            if len(obj["original_face_numbers"])!=1:
                raise PlanError("multiwall_through_hole_edit_unsupported")
            h = cylinder(data["faces"][obj["original_face_numbers"][0]-1])
            old,new = h["radius_mm"],op["new_radius_mm"]
            if volume(boolean(BRepAlgoAPI_Common,original,tube(h)))>1e-4:
                raise PlanError("hole_tool_intersects_original_material")
            if new < old:
                ring = boolean(BRepAlgoAPI_Cut,tube(h),tube(h,new))
                changed = clean(boolean(BRepAlgoAPI_Fuse,changed,ring))
                tools_add.append(ring)
            elif new > old:
                ring = boolean(BRepAlgoAPI_Cut,tube(h,new),tube(h))
                changed = clean(boolean(BRepAlgoAPI_Cut,changed,ring))
                tools_remove.append(ring)
            else:
                raise PlanError("no_geometry_change")
            expected_delta += math.pi*(old**2-new**2)*h["depth_mm"]
            changes.append({**h,"new_radius_mm":new,"object_index":obj["object_index"]})
    elif op["action"] == "set_hole_spacing":
        if len(objects)!=3 or {o["class"] for o in objects}!={"counterbore_hole"}:
            raise PlanError("spacing_requires_three_counterbores")
        holes = []
        for obj in objects:
            cylinders = [cylinder(data["faces"][n-1]) for n in obj["original_face_numbers"] if "radius_mm" in data["faces"][n-1]]
            if len(cylinders)!=2:
                raise PlanError("unsupported_counterbore_geometry")
            holes.append({"object_index":obj["object_index"],"cylinders":cylinders,"center":max(cylinders,key=lambda c:c["radius_mm"])["center_mm"]})
        center = min(holes,key=lambda h:sum(math.dist(h["center"],other["center"]) for other in holes))
        sides = [h for h in holes if h is not center]
        direction = unit(sub(sides[1]["center"],sides[0]["center"]))
        for hole in sides:
            relative = sub(hole["center"],center["center"])
            position = dot(relative,direction)
            if math.hypot(*sub(relative,mul(direction,position)))>1e-5 or abs(abs(position)-op["old_spacing_mm"])>SOURCE_SPACING_MM:
                raise PlanError("spacing_pattern_mismatch")
            shift = mul(direction,math.copysign(op["new_spacing_mm"],position)-position)
            old_tool = join([tube(c) for c in hole["cylinders"]])
            new_tool = move(old_tool,shift)
            if volume(boolean(BRepAlgoAPI_Common,original,old_tool))>1e-4:
                raise PlanError("counterbore_tool_intersects_material")
            changed = clean(boolean(BRepAlgoAPI_Fuse,changed,old_tool))
            tools_add.append(old_tool)
            tools_remove.append(new_tool)
            changes.append({"object_index":hole["object_index"],"translation_mm":shift,
                            "center_before_mm":hole["center"],"center_after_mm":add(hole["center"],shift),
                            "cylinders":hole["cylinders"]})
        for tool in tools_remove:
            changed = clean(boolean(BRepAlgoAPI_Cut,changed,tool))
        checks.append({"central_hole_unchanged":True,"central_object_index":center["object_index"],"central_center_mm":center["center"]})
    elif op["action"] == "set_thickness":
        if {o["class"] for o in objects} - {"rib","web"}:
            raise PlanError("unsupported_thickness_feature")
        for obj in objects:
            t = thickness(data,obj)
            if abs(t["thickness_mm"]-op["old_thickness_mm"])>SOURCE_THICKNESS_MM:
                raise PlanError("source_thickness_mismatch")
            ratio = op["new_thickness_mm"]/t["thickness_mm"]
            side = faces[t["side_faces"][0]-1]
            original_prism = BRepPrimAPI_MakePrism(side,gp_Vec(*t["sweep_vector_mm"])).Shape()
            if volume(boolean(BRepAlgoAPI_Cut,original_prism,original))>1e-4:
                raise PlanError("recognized_slab_contains_void_or_unsupported_trim")
            new_side = move(side,mul(t["sweep_vector_mm"],(1-ratio)/2))
            new_prism = BRepPrimAPI_MakePrism(new_side,gp_Vec(*mul(t["sweep_vector_mm"],ratio))).Shape()
            if ratio>1:
                delta = boolean(BRepAlgoAPI_Cut,new_prism,original_prism)
                changed = clean(boolean(BRepAlgoAPI_Fuse,changed,delta))
                tools_add.append(delta)
            else:
                delta = boolean(BRepAlgoAPI_Cut,original_prism,new_prism)
                changed = clean(boolean(BRepAlgoAPI_Cut,changed,delta))
                tools_remove.append(delta)
            expected_delta += t["side_area_mm2"]*(op["new_thickness_mm"]-t["thickness_mm"])
            changes.append({**t,"new_thickness_mm":op["new_thickness_mm"],"anchor":"midplane"})
    else:
        raise PlanError("unsupported_cad_action")
    path = output / (source.stem + "_edited.stp")
    writer = STEPControl_Writer()
    if writer.Transfer(changed,STEPControl_AsIs)!=IFSelect_RetDone or writer.Write(str(path))!=IFSelect_RetDone:
        raise PlanError("step_export_failed")
    final = read_step(path)
    # Use solid containment to avoid invalid empty solids from coincident STEP differences.
    # Any changed volume outside the allowed tools remains uncovered by these unions.
    allowed_final = join([original]+tools_add)
    allowed_original = join([final]+tools_remove)
    covered_final = volume(boolean(BRepAlgoAPI_Common,final,allowed_final))
    covered_original = volume(boolean(BRepAlgoAPI_Common,allowed_original,original))
    outside_add = abs(volume(final)-covered_final)
    outside_remove = abs(volume(original)-covered_original)
    actual_delta = volume(final)-volume(original)
    post = [measured(f,i+1) for i,f in enumerate(TopologyExplorer(final).faces())]
    for c in changes:
        if "new_radius_mm" in c:
            rows = [r for r in post if "radius_mm" in r and abs(r["radius_mm"]-c["new_radius_mm"])<1e-6 and math.dist(r["center_mm"],c["center_mm"])<1e-5]
            checks.append({"radius_and_axis_extent_verified":len(rows)==1 and abs(cylinder(rows[0])["depth_mm"]-c["depth_mm"])<1e-5})
        elif "new_thickness_mm" in c:
            half = mul(c["sweep_vector_mm"],c["new_thickness_mm"]/c["thickness_mm"]/2)
            centers = [add(c["midpoint_mm"],half),sub(c["midpoint_mm"],half)]
            matching = [r for r in post if "normal" in r and abs(abs(dot(r["normal"],c["normal"]))-1)<1e-6
                        and abs(abs(dot(sub(r["center_mm"],c["midpoint_mm"]),c["normal"]))-c["new_thickness_mm"]/2)<1e-5
                        and abs(r["area_mm2"]-c["side_area_mm2"])<1e-4
                        and min(math.dist(r["center_mm"],center) for center in centers)<1e-5]
            checks.append({"symmetric_thickness_verified":len(matching)==2})
        else:
            valid = []
            for h in c["cylinders"]:
                center = add(h["center_mm"],c["translation_mm"])
                valid.append(any("radius_mm" in r and abs(r["radius_mm"]-h["radius_mm"])<1e-6 and math.dist(r["center_mm"],center)<1e-5 for r in post))
            checks.append({"translated_hole_geometry_verified":all(valid)})
    if op["action"]=="set_hole_spacing":
        central = checks[0]["central_center_mm"]
        checks[0]["central_hole_unchanged"] = any("radius_mm" in r and math.dist(r["center_mm"],central)<1e-5 for r in post)
    passed = outside_add<1e-4 and outside_remove<1e-4 and abs(actual_delta-expected_delta)<max(1e-3,abs(expected_delta)*1e-7)
    passed = passed and all(v for row in checks for key,v in row.items() if key.endswith("verified") or key=="central_hole_unchanged")
    audit = {"status":"PASS" if passed else "FAIL","truth_used":False,"input_step":str(source),"output_step":str(path),
             "source_sha256":sha(source),"output_sha256":sha(path),"prediction_sha256":sha(predpath),
             "changes":changes,"checks":checks,"volume_before_mm3":volume(original),"volume_after_mm3":volume(final),
             "expected_volume_change_mm3":expected_delta,"actual_volume_change_mm3":actual_delta,
             "added_outside_allowed_region_mm3":outside_add,"removed_outside_allowed_region_mm3":outside_remove,
             "valid_solid_count":1,"execution_mode":execution_mode}
    audit["non_target_verification"]="valid-solid union/intersection volumetric containment"
    audit["boolean_tolerance_adjustments"]=list(BOOLEAN_EVENTS)
    write_json(output / "edit_audit.json",audit)
    if not passed:
        raise PlanError("cad_postcondition_failed")
    return path
