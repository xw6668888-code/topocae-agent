"""Compile only the source STEP, direct model faces, and physical parameters."""
from __future__ import annotations
import math
from common import *
from contract import ground
from geometry import source_data, properties, import_face_record, unit, sub, mul, dot, add

def compile_plan(source,plan,mapping,materials,output,config):
    grounded=ground(plan,mapping)
    if plan['status']!='ready' or plan['domain']!='cae': raise ValueError('Nonready CAE plan')
    source=Path(source);output=Path(output)
    shape,faces,rows=source_data(source)
    ops={o['action']:o for o in grounded['operations']}
    material=materials[ops['set_material']['material_id']]
    def regions(op):
        result=[]
        for i,ns in enumerate(op['face_number_groups']):
            if any(n<1 or n>len(faces) for n in ns): raise ValueError('Invalid face index')
            result.append(dict(group_index=i,face_numbers=list(ns),area_mm2=sum(rows[n-1]['area_mm2'] for n in ns)))
        return result
    fixed=regions(ops['fix_features'])
    local=[dict(size_mm=ops['set_local_mesh']['size_mm'],regions=regions(ops['set_local_mesh']))] if 'set_local_mesh' in ops else []
    loads=[];resultant=[0.,0.,0.];scale=0.
    for action in ('apply_force','apply_traction','apply_pressure','apply_centrifugal'):
        if action not in ops: continue
        op=ops[action]
        if action=='apply_centrifugal':
            axis=unit(op['axis_direction']);vol,center=properties(shape,True)
            relative=sub(center,op['axis_point_mm']);radial=sub(relative,mul(axis,dot(relative,axis)))
            resultant=mul(radial,material['density_tonne_mm3']*vol*op['angular_velocity_rad_s']**2)
            scale=math.hypot(*resultant)
            loads.append(dict(kind='centrifugal',angular_velocity_rad_s=op['angular_velocity_rad_s'],axis_point_mm=op['axis_point_mm'],axis_direction=axis))
            continue
        groups=regions(op);total_area=sum(g['area_mm2'] for g in groups)
        for group in groups:
            # Overlapping boundary/load faces remain model-selected physics and are executed.
            if action=='apply_force':
                amount=op['magnitude_N'] if op['basis']=='per_instance' else op['magnitude_N']*group['area_mm2']/total_area
                direction=unit(op['direction']);contribution=mul(direction,amount);scale+=amount
                loads.append(dict(group,kind='force',force_N=amount,direction=direction))
            elif action=='apply_traction':
                direction=unit(op['direction']);amount=op['magnitude_MPa']*group['area_mm2']
                contribution=mul(direction,amount);scale+=amount
                loads.append(dict(group,kind='traction',magnitude_MPa=op['magnitude_MPa'],direction=direction))
            else:
                contribution=[0.,0.,0.]
                for n in group['face_numbers']:
                    row=rows[n-1]
                    if 'normal' in row: contribution=add(contribution,mul(row['normal'],-op['magnitude_MPa']*row['area_mm2']))
                    elif 'radius_mm' in row:
                        # Normal integral of a full circular cylinder is zero. Trimmed cylinders need integration.
                        if abs(row['u_range'][1]-row['u_range'][0]-2*math.pi)>1e-6:
                            raise ValueError('Pressure normal integral of trimmed curved surface unsupported')
                    else: raise ValueError('Pressure normal integral unsupported for this selected surface')
                scale+=math.hypot(*contribution)
                loads.append(dict(group,kind='pressure',magnitude_MPa=op['magnitude_MPa']))
            resultant=add(resultant,contribution)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'model_plan.json',plan);write(output/'actual_face_groups.json',grounded)
    payload=dict(schema='g11_direct_cae_execution_plan_v1',case_id='model_plan',part=source.name,
       source_step=str(source.resolve()),source_sha256=sha(source),geometry_version=sha(source),
       model_plan_sha256=sha(output/'model_plan.json'),script_directory=str(output.resolve()),
       run_directory=str((output/'run').resolve()),execution_mode='g11_direct_model_face_groups',truth_used=False,
       material=material,global_mesh_mm=ops['set_global_mesh']['size_mm'],local_mesh=local,fixed_regions=fixed,loads=loads,
       expected_force_N=resultant,force_balance_scale_N=scale,
       mapping_tolerances=dict(vertex_distance_mm=1e-4,relative_area=.005),
       source_faces=[import_face_record(f,i+1) for i,f in enumerate(faces)],
       cpus=config['cpus'],memory_mb=config['memory_mb'],
       binding_files=[descriptor(source),descriptor(output/'model_plan.json'),descriptor(CODE/'abaqus_verification.py')])
    write(output/'execution_plan.json',payload)
    template=(CODE/'abaqus_direct.template.py').read_text(encoding='ascii')
    template=template.replace('# __VERIFICATION_FUNCTION__',(CODE/'abaqus_verification.py').read_text(encoding='ascii'))
    template=template.replace('__EMBEDDED_PLAN_JSON__',repr(json.dumps(payload,ensure_ascii=True,separators=(',',':'))))
    text(output/'final.py',template)
    return payload
