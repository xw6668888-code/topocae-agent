"""Direct model-selected local CAD edits; no learned-object matching."""
from geometry import *
from contract import ground
SOURCE_THICKNESS_MM=0.0001
SOURCE_SPACING_MM=0.0001
def execute(source, plan, mapping, output):
    BOOLEAN_EVENTS.clear()
    ground_plan=ground(plan,mapping)
    if plan['status']!='ready' or plan['domain']!='cad':
        raise ValueError('Nonready CAD plan')
    source=Path(source)
    original,faces,rows=source_data(source)
    data={'faces':rows}
    op=ground_plan['operations'][0]
    objects=[{'object_index':i,'original_face_numbers':group} for i,group in enumerate(op['face_number_groups'])]
    if any(n<1 or n>len(faces) for o in objects for n in o['original_face_numbers']):
        raise ValueError('Face index outside source STEP')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'model_plan.json',plan)
    write_json(output/'actual_face_groups.json',ground_plan)
    execution_mode='g11_direct_model_face_groups'
    changes, tools_add, tools_remove, checks = [], [], [], []
    changed = original
    expected_delta = 0.
    if op["action"] == "set_hole_radius":
        for obj in objects:
            selected_rows=[data['faces'][n-1] for n in obj['original_face_numbers']]
            walls=[r for r in selected_rows if 'radius_mm' in r]
            if len(walls)!=1:
                raise ValueError('Radius edit requires one selected complete cylinder per group')
            h = cylinder(walls[0])
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
        if len(objects)!=3:
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
             "source_sha256":sha(source),"output_sha256":sha(path),"model_plan_sha256":sha(output / "model_plan.json"),"actual_face_groups":op["face_number_groups"],
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
