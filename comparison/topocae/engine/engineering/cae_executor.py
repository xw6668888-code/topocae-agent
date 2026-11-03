"""Compile recognized feature roles into an auditable Abaqus program."""
from __future__ import annotations

import json
import math
from pathlib import Path
from geometry import load, select, roles, add, sub, mul, dot, unit, cylinder
from preflight import OUT, write_json, sha
from build_cases import face_record
from contracts import validate
from plan_contract import PlanError
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer
from preflight import xyz


def import_face_record(face, number):
    record=face_record(face,number)
    record['boundary_vertices_mm']=[xyz(BRep_Tool.Pnt(v)) for v in TopologyExplorer(face).vertices()]
    return record


def compile_case(part,plan,output,*,execution_mode="local_reference_plan"):
    validate(plan)
    if plan["status"]!="ready" or plan["domain"]!="cae":
        raise PlanError("nonready_cae_execution")
    case,data=load(part)
    source,shape,faces,prediction,predpath=case
    ops={op["action"]:op for op in plan["operations"]}
    material=json.loads((OUT / "materials/selected_materials.json").read_text(encoding="utf-8"))[ops["set_material"]["material_id"]]
    fixed=roles(data,select(data,ops["fix_features"]["target"]),ops["fix_features"]["role"])
    local=[]
    if "set_local_mesh" in ops:
        op=ops["set_local_mesh"]
        if op["target"]["feature_classes"]!=["chamfer"]:
            raise PlanError("local_refinement_requires_chamfer")
        local=[{"size_mm":op["size_mm"],"regions":roles(data,select(data,op["target"]),"whole_feature")}]
    loads=[]
    resultant=[0.,0.,0.]
    scale=0.
    for action in ("apply_force","apply_traction","apply_pressure","apply_centrifugal"):
        if action not in ops:
            continue
        op=ops[action]
        if action=="apply_centrifugal":
            axis=unit(op["axis_direction"])
            r=sub(data["center_of_mass_mm"],op["axis_point_mm"])
            radial=sub(r,mul(axis,dot(r,axis)))
            resultant=mul(radial,material["density_tonne_mm3"]*data["volume_mm3"]*op["angular_velocity_rad_s"]**2)
            scale=math.hypot(*resultant)
            loads.append({"kind":"centrifugal","angular_velocity_rad_s":op["angular_velocity_rad_s"],
                          "axis_point_mm":op["axis_point_mm"],"axis_direction":axis})
            continue
        groups=roles(data,select(data,op["target"]),op["role"])
        totalarea=sum(g["area_mm2"] for g in groups)
        for group in groups:
            if set(group["face_numbers"]) & {n for g in fixed for n in g["face_numbers"]}:
                raise PlanError("overlapping_fixed_and_loaded_regions")
            if action=="apply_force":
                force=op["magnitude_N"]*(1 if op["basis"]=="per_instance" else group["area_mm2"]/totalarea)
                direction=unit(op["direction"])
                result=mul(direction,force)
                loads.append({**group,"kind":"force","force_N":force,"direction":direction})
                scale+=force
            elif action=="apply_traction":
                direction=unit(op["direction"])
                force=op["magnitude_MPa"]*group["area_mm2"]
                result=mul(direction,force)
                loads.append({**group,"kind":"traction","magnitude_MPa":op["magnitude_MPa"],"direction":direction})
                scale+=force
            else:
                result=[0.,0.,0.]
                for n in group["face_numbers"]:
                    row=data["faces"][n-1]
                    if "normal" in row:
                        result=add(result,mul(row["normal"],-op["magnitude_MPa"]*row["area_mm2"]))
                    elif "radius_mm" in row:
                        cylinder(row)
                    else:
                        raise PlanError("pressure_resultant_geometry_unsupported")
                loads.append({**group,"kind":"pressure","magnitude_MPa":op["magnitude_MPa"]})
                scale+=math.hypot(*result)
            resultant=add(resultant,result)
    output.mkdir(parents=True,exist_ok=False)
    payload={"case_id":source.stem.lower().replace(" ","_").replace("-","_"),"part":part,"source_step":str(source),
             "source_sha256":sha(source),"prediction_sha256":sha(predpath),"model":prediction["model"],
             "script_directory":str(output.resolve()),"execution_mode":execution_mode,"truth_used":False,
             "material":material,"global_mesh_mm":ops["set_global_mesh"]["size_mm"],"local_mesh":local,
             "fixed_regions":fixed,"loads":loads,"expected_force_N":resultant,"force_balance_scale_N":scale,
             "mapping_tolerances":{"vertex_distance_mm":1e-4,"relative_area":.005},
             "source_faces":[import_face_record(face,i+1) for i,face in enumerate(faces)]}
    write_json(output / "execution_plan.json",payload)
    write_json(output / "semantic_plan.json",plan)
    template=Path(__file__).with_name("abaqus_generic.template.py").read_text(encoding="ascii")
    verification=Path(__file__).with_name('abaqus_verification.py').read_text(encoding='ascii')
    template=template.replace('# __VERIFICATION_FUNCTION__',verification)
    script=output / (source.stem.replace(" ","_").replace("-","_")+"_Abaqus.py")
    script.write_text(template.replace("__EMBEDDED_PLAN_JSON__",repr(json.dumps(payload,ensure_ascii=True))),encoding="ascii")
    return script
