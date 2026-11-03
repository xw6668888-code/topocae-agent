# -*- coding: ascii -*-
"""Generated model-backed CAE operation. Run from Abaqus/CAE or cae noGUI."""
from abaqus import mdb, session
from abaqusConstants import *
from mesh import ElemType
import part, material, section, assembly, step, load, mesh, job, visualization
import regionToolset
import hashlib, json, math, os, sys, tempfile, time, traceback
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
sum, min, max, abs = builtins.sum, builtins.min, builtins.max, builtins.abs


def native(value):
    if sys.version_info[0] < 3 and isinstance(value, unicode):
        return value.encode('ascii')
    if isinstance(value, dict):
        return dict((native(k),native(v)) for k,v in value.items())
    if isinstance(value,list):
        return [native(v) for v in value]
    return value


PLAN = native(json.loads(__EMBEDDED_PLAN_JSON__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else PLAN['script_directory']
RUN_DIR = tempfile.mkdtemp(prefix='run_',dir=SCRIPT_DIR)
AUDIT_PATH = os.path.join(RUN_DIR,'execution_audit.json')
SUBMIT_JOB = os.environ.get('AEI_PREPARE_ONLY','0') != '1'
SOLVER_KIND = os.environ.get('AEI_MATRIX_SOLVER','AUTO').upper()
if SOLVER_KIND not in ('AUTO','DIRECT','ITERATIVE'):
    raise ValueError('AEI_MATRIX_SOLVER must be AUTO, DIRECT or ITERATIVE')


def dump(data):
    with open(AUDIT_PATH,'w') as handle:
        if sys.version_info[0] < 3:
            json.dump(data,handle,indent=2,encoding='mbcs')
        else:
            json.dump(data,handle,indent=2)


# __VERIFICATION_FUNCTION__


def sequence(array,indices):
    indices = sorted(set(indices))
    if not indices:
        raise RuntimeError('Empty geometric selection')
    result = array[indices[0]:indices[0]+1]
    for index in indices[1:]:
        result += array[index:index+1]
    return result


def mapping_for(part_object):
    faces=part_object.faces
    if len(faces)!=len(PLAN['source_faces']):
        raise RuntimeError('Imported face count changed')
    mapping,records = {},[]
    for record in PLAN['source_faces']:
        matches = [faces.findAt(coordinates=tuple(p),printWarning=False).index for p in record['interior_points_mm']]
        if len(set(matches))!=1 or matches[0] in mapping.values():
            raise RuntimeError('Face witnesses or bijection failed')
        index=matches[0]
        error=abs(faces[index].getSize(printResults=False)-record['area_mm2'])/record['area_mm2']
        source_vertices=record['boundary_vertices_mm']
        distances=[]
        if source_vertices:
            edges=sequence(part_object.edges,faces[index].getEdges())
            # Periodic seam vertices need not be topological vertices in ACIS.
            closest=edges.getClosest(coordinates=tuple(tuple(v) for v in source_vertices),
                                     searchTolerance=PLAN['mapping_tolerances']['vertex_distance_mm'])
            if len(closest)!=len(source_vertices):
                raise RuntimeError('Source boundary point is absent from imported face edges')
            distances=[math.sqrt(sum((a[k]-closest[i][1][k])**2 for k in range(3))) for i,a in enumerate(source_vertices)]
        vertex_error=max(distances+[0.])
        if vertex_error>PLAN['mapping_tolerances']['vertex_distance_mm']:
            raise RuntimeError('Imported face boundary vertices moved')
        # ACIS area queries are approximate on trimmed/spline faces. Witnesses,
        # boundary vertices and a bijection provide independent mapping checks.
        if error>PLAN['mapping_tolerances']['relative_area']:
            raise RuntimeError('Imported face area mismatch')
        mapping[record['occ_face_number']]=index
        records.append({'source_face':record['occ_face_number'],'abaqus_index':index,'relative_area_error':error,
                        'max_vertex_distance_mm':vertex_error,'boundary_vertex_count':len(source_vertices)})
    return mapping,records


def main(audit):
    with open(PLAN['source_step'],'rb') as handle:
        digest=hashlib.sha256(handle.read()).hexdigest()
    if digest!=PLAN['source_sha256']:
        raise RuntimeError('STEP version differs from recognition')
    os.chdir(RUN_DIR)
    name='AEI-'+PLAN['case_id']
    suffix=1
    while name in mdb.models:
        name='AEI-'+PLAN['case_id']+'-%d'%suffix
        suffix+=1
    model=mdb.Model(name=name,modelType=STANDARD_EXPLICIT)
    geometry=mdb.openStep(fileName=PLAN['source_step'],scaleFromFile=OFF)
    p=model.PartFromGeometryFile(name='Part-1',geometryFile=geometry,combine=False,dimensionality=THREE_D,type=DEFORMABLE_BODY)
    if len(p.cells)!=1:
        raise RuntimeError('Expected one solid')
    mapping,records=mapping_for(p)
    mat=model.Material(name=PLAN['material']['name'])
    mat.Elastic(table=((PLAN['material']['youngs_modulus_mpa'],PLAN['material']['poisson_ratio']),))
    mat.Density(table=((PLAN['material']['density_tonne_mm3'],),))
    model.HomogeneousSolidSection(name='Solid-Section',material=mat.name,thickness=None)
    p.SectionAssignment(region=p.Set(name='All-Cells',cells=p.cells[:]),sectionName='Solid-Section')
    a=model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst=a.Instance(name='Part-1-1',part=p,dependent=ON)
    model.StaticStep(name='Static-Load',previous='Initial',nlgeom=OFF,initialInc=1.,minInc=1e-6,maxInc=1.,maxNumInc=100,
                     matrixSolver=ITERATIVE if SOLVER_KIND=='ITERATIVE' else DIRECT)
    audit['matrix_solver']=SOLVER_KIND
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S','U','RF'))
    fixed_ids=sorted(set(mapping[n] for group in PLAN['fixed_regions'] for n in group['face_numbers']))
    fixed=a.Set(name='FIXED-FEATURES',faces=sequence(inst.faces,fixed_ids))
    model.EncastreBC(name='Fixed-Features',createStepName='Initial',region=fixed)
    audit.update(face_mapping=records,face_mapping_bijective=True,fixed_abaqus_indices=fixed_ids,
                 fixed_regions=PLAN['fixed_regions'],source_sha256=digest,material=PLAN['material'])
    audit['loads']=[]
    for ordinal,record in enumerate(PLAN['loads'],1):
        kind=record['kind']
        if kind=='centrifugal':
            start=tuple(record['axis_point_mm'])
            end=tuple(start[k]+record['axis_direction'][k] for k in range(3))
            model.RotationalBodyForce(name='Centrifugal',createStepName='Static-Load',
                region=regionToolset.Region(cells=inst.cells[:]),magnitude=record['angular_velocity_rad_s'],
                centrifugal=ON,rotaryAcceleration=OFF,point1=start,point2=end)
            audit['loads'].append(record)
            continue
        indices=[mapping[n] for n in record['face_numbers']]
        area=sum(inst.faces[n].getSize(printResults=False) for n in indices)
        surface=a.Surface(name='LOAD-%02d'%ordinal,side1Faces=sequence(inst.faces,indices))
        if kind=='pressure':
            model.Pressure(name='Pressure-%02d'%ordinal,createStepName='Static-Load',region=surface,
                           distributionType=UNIFORM,magnitude=record['magnitude_MPa'])
            magnitude=record['magnitude_MPa']
        else:
            magnitude=record['force_N']/area if kind=='force' else record['magnitude_MPa']
            model.SurfaceTraction(name='Traction-%02d'%ordinal,createStepName='Static-Load',region=surface,
                distributionType=UNIFORM,magnitude=magnitude,traction=GENERAL,
                directionVector=((0.,0.,0.),tuple(record['direction'])),localCsys=None,follower=OFF,resultant=ON)
        audit['loads'].append(dict(record,abaqus_indices=indices,area_mm2=area,traction_or_pressure_MPa=magnitude))
    p.seedPart(size=PLAN['global_mesh_mm'],deviationFactor=.1,minSizeFactor=.1)
    audit['local_mesh']=[]
    for local in PLAN['local_mesh']:
        face_ids=sorted(set(mapping[n] for g in local['regions'] for n in g['face_numbers']))
        edge_ids=sorted(set(e for n in face_ids for e in p.faces[n].getEdges()))
        p.seedEdgeBySize(edges=sequence(p.edges,edge_ids),size=local['size_mm'],deviationFactor=.1,minSizeFactor=.1,constraint=FINER)
        sizes=[float(p.getEdgeSeeds(edge=p.edges[e],attribute=AVERAGE_SIZE)) for e in edge_ids]
        numbers=[int(p.getEdgeSeeds(edge=p.edges[e],attribute=NUMBER)) for e in edge_ids]
        methods=[str(p.getEdgeSeeds(edge=p.edges[e],attribute=EDGE_SEEDING_METHOD)) for e in edge_ids]
        constraints=[str(p.getEdgeSeeds(edge=p.edges[e],attribute=CONSTRAINT)) for e in edge_ids]
        if any(method not in ('CURVATURE_BASED_BY_SIZE','UNIFORM_BY_SIZE') for method in methods) or any(c!='FINER' for c in constraints):
            raise RuntimeError('Local size seeding or FINER constraint missing')
        if any(size<=0 or size>local['size_mm']*(1.+1./max(1,number))+1e-8 for size,number in zip(sizes,numbers)):
            raise RuntimeError('Local average edge spacing exceeds integer-seed rounding allowance')
        audit['local_mesh'].append({'abaqus_face_indices':face_ids,'edge_indices':edge_ids,'size_mm':local['size_mm'],
                                   'readback_average_sizes_mm':sizes,'seed_numbers':numbers,'methods':methods,'constraints':constraints})
    p.setMeshControls(regions=p.cells[:],elemShape=TET,technique=FREE)
    p.setElementType(regions=(p.cells[:],),elemTypes=(ElemType(elemCode=C3D10,elemLibrary=STANDARD),))
    if os.environ.get('AEI_SKIP_MESH','0')=='1':
        cae_path=os.path.join(RUN_DIR,'AEI_'+PLAN['case_id']+'_unmeshed.cae')
        mdb.saveAs(pathName=cae_path)
        audit.update(status='MODEL_READY',cae_path=cae_path,mesh_seed_mm=PLAN['global_mesh_mm'],
                     mesh_skipped=True,submission_skipped=True)
        return
    p.generateMesh()
    a.regenerate()
    if not len(p.elements) or not len(p.nodes):
        raise RuntimeError('Empty mesh')
    counts={}
    for e in p.elements:
        counts[str(e.type)]=counts.get(str(e.type),0)+1
    if set(counts)!=set(['C3D10']):
        raise RuntimeError('Unexpected element types')
    if SOLVER_KIND=='AUTO':
        chosen='ITERATIVE' if len(p.nodes)>500000 else 'DIRECT'
        model.steps['Static-Load'].setValues(matrixSolver=ITERATIVE if chosen=='ITERATIVE' else DIRECT)
        audit['matrix_solver']=chosen
    jobname='AEI_'+PLAN['case_id']
    suffix=1
    while jobname in mdb.jobs:
        jobname='AEI_'+PLAN['case_id']+'_%d'%suffix
        suffix+=1
    memory_mb=int(os.environ.get('AEI_MEMORY_MB','20480' if len(p.nodes)>500000 else '4096'))
    task=mdb.Job(name=jobname,model=name,type=ANALYSIS,numCpus=2,numDomains=2,memory=memory_mb,memoryUnits=MEGA_BYTES,
                 scratch=RUN_DIR)
    audit['requested_solver_memory_mb']=memory_mb
    task.writeInput(consistencyChecking=ON)
    input_path=os.path.join(RUN_DIR,jobname+'.inp')
    with open(input_path,'r') as handle:
        lines=handle.read().splitlines()
    rotational=[r for r in PLAN['loads'] if r['kind']=='centrifugal']
    if rotational:
        centrifugal=[line for line in lines if ', CENTRIF,' in line.upper()]
        expected=rotational[0]['angular_velocity_rad_s']**2
        if len(centrifugal)!=1 or abs(float(centrifugal[0].split(',')[2])-expected)>1e-6:
            raise RuntimeError('CENTRIF input must contain squared angular velocity')
        audit['centrifugal_input_record']=centrifugal[0]
    cae_path=os.path.join(RUN_DIR,jobname+'.cae')
    mdb.saveAs(pathName=cae_path)
    audit.update(status='PREPARED',node_count=len(p.nodes),element_count=len(p.elements),element_types=counts,
                 mesh_seed_mm=PLAN['global_mesh_mm'],cae_path=cae_path,input_path=input_path)
    dump(audit)
    if not SUBMIT_JOB:
        return
    task.submit(consistencyChecking=ON)
    task.waitForCompletion()
    audit['abaqus_job_status']=str(task.status)
    odb_path=os.path.join(RUN_DIR,jobname+'.odb')
    odb=session.openOdb(name=odb_path,readOnly=True)
    try:
        audit.update(verify_odb(odb,PLAN))
        audit['odb_path']=odb_path
        if audit['status']!='PASS':
            raise RuntimeError('Solver or force-balance verification failed')
    finally:
        odb.close()


started=time.time()
audit={'case':PLAN['part'],'status':'FAIL','truth_used':False,'execution_mode':PLAN['execution_mode'],
       'model_checkpoint_sha256':PLAN['model']['checkpoint_sha256'],'prediction_sha256':PLAN['prediction_sha256']}
if '__file__' in globals():
    with open(__file__,'rb') as program:
        audit['script_sha256']=hashlib.sha256(program.read()).hexdigest()
try:
    main(audit)
except Exception:
    audit.update(status='FAIL',error=traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    audit['elapsed_seconds']=time.time()-started
    dump(audit)
    print('AEI audit: '+AUDIT_PATH)
