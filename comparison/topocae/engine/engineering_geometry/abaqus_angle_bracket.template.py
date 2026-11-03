# -*- coding: ascii -*-
"""Angle Bracket: run using Abaqus/CAE File > Run Script, or cae noGUI.

The generated script embeds frozen-model selections and geometric witnesses.
It requires Abaqus only; PyTorch/pythonOCC are not needed at execution time.
Units: mm, N, MPa, tonne, s. Load magnitudes and directions come from PLAN.
"""
from abaqus import mdb, session
from abaqusConstants import *
from mesh import ElemType
import part
import material
import section
import assembly
import step
import interaction
import load
import mesh
import job
import sketch
import visualization
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# Abaqus/CAE injects array-oriented names into a script's global namespace.
sum, min, max, abs = builtins.sum, builtins.min, builtins.max, builtins.abs

def native_strings(value):
    if sys.version_info[0] < 3 and isinstance(value, unicode):
        return value.encode('ascii')
    if isinstance(value, dict):
        return dict((native_strings(k), native_strings(v)) for k, v in value.items())
    if isinstance(value, list):
        return [native_strings(v) for v in value]
    return value


PLAN = native_strings(json.loads(__EMBEDDED_PLAN_JSON__))
STEP_PATH = PLAN['source_step']
SUBMIT_JOB = True
NUM_CPUS = 2
MODEL_NAME = 'AEI-Angle-Bracket'
JOB_NAME = 'AEI_Angle_Bracket'
STEP_NAME = 'Static-Load'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else PLAN['script_directory']
RUN_DIR = tempfile.mkdtemp(prefix='run_', dir=SCRIPT_DIR)
AUDIT_PATH = os.path.join(RUN_DIR, 'execution_audit.json')


def dump(path, payload):
    with open(path, 'w') as handle:
        if sys.version_info[0] < 3:
            json.dump(payload, handle, indent=2, sort_keys=True, encoding='mbcs')
        else:
            json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def face_sequence(face_array, indices):
    indices = sorted(set(indices))
    if not indices:
        raise RuntimeError('Empty face region')
    result = face_array[indices[0]:indices[0] + 1]
    for index in indices[1:]:
        result += face_array[index:index + 1]
    return result


def map_faces(abaqus_faces):
    mapping, records = {}, []
    if len(abaqus_faces) != len(PLAN['source_faces']):
        raise RuntimeError('STEP import changed the face count; remapping needs review')
    for source in PLAN['source_faces']:
        matches = []
        for point in source['interior_points_mm']:
            found = abaqus_faces.findAt(coordinates=tuple(point), printWarning=False)
            if found is None:
                raise RuntimeError('Cannot map OCC face %s' % source['occ_face_number'])
            matches.append(found.index)
        if len(set(matches)) != 1:
            raise RuntimeError('Interior witnesses disagree on a face mapping')
        index = matches[0]
        area = float(abaqus_faces[index].getSize(printResults=False))
        relative_error = abs(area - source['area_mm2']) / source['area_mm2']
        if relative_error > 1e-5 or index in mapping.values():
            raise RuntimeError('Face mapping area/bijection check failed')
        mapping[source['occ_face_number']] = index
        records.append({'occ_face_number': source['occ_face_number'], 'abaqus_index': index,
                        'area_mm2': area, 'relative_area_error': relative_error,
                        'interior_witnesses_agree': True})
    return mapping, records


def main(audit):
    with open(STEP_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest.lower() != PLAN['source_sha256'].lower():
        raise RuntimeError('Input STEP differs from the recognized file; rerun recognition')
    os.chdir(RUN_DIR)
    unique_model = MODEL_NAME
    serial = 1
    while unique_model in mdb.models:
        unique_model = MODEL_NAME + '-%d' % serial
        serial += 1
    model = mdb.Model(name=unique_model, modelType=STANDARD_EXPLICIT)
    geometry = mdb.openStep(fileName=STEP_PATH, scaleFromFile=OFF)
    p = model.PartFromGeometryFile(name='Angle-Bracket', geometryFile=geometry,
                                  combine=False, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    if len(p.cells) != 1:
        raise RuntimeError('Expected one imported solid cell')
    mapping, records = map_faces(p.faces)
    audit.update(source_sha256=digest, model_name=unique_model, face_mapping=records,
                 face_mapping_bijective=True, imported_face_count=len(p.faces))
    material = model.Material(name=PLAN['material']['name'])
    material.Elastic(table=((PLAN['material']['youngs_modulus_mpa'], PLAN['material']['poisson_ratio']),))
    material.Density(table=((PLAN['material']['density_tonne_mm3'],),))
    model.HomogeneousSolidSection(name='Steel45-Solid', material=material.name, thickness=None)
    p.SectionAssignment(region=p.Set(name='All-Cells', cells=p.cells[:]), sectionName='Steel45-Solid')
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    instance = assembly.Instance(name='Angle-Bracket-1', part=p, dependent=ON)
    model.StaticStep(name=STEP_NAME, previous='Initial', nlgeom=OFF,
                     initialInc=1.0, minInc=1e-6, maxInc=1.0, maxNumInc=100)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF'))
    fixed_indices = []
    for obj in PLAN['fixed_objects']:
        fixed_indices.extend(mapping[n] for n in obj['original_face_numbers'])
    fixed = assembly.Set(name='FIXED-COUNTERBORES', faces=face_sequence(instance.faces, fixed_indices))
    model.EncastreBC(name='Fix-All-Counterbores', createStepName='Initial', region=fixed)
    audit['fixed_abaqus_face_indices'] = sorted(set(fixed_indices))
    audit['loads'] = []
    for ordinal, obj in enumerate(PLAN['loaded_objects'], 1):
        indices = [mapping[n] for n in obj['original_face_numbers']]
        area = sum(float(instance.faces[n].getSize(printResults=False)) for n in indices)
        surface_name = 'OBROUND-%02d' % ordinal
        surface = assembly.Surface(name=surface_name,
                                   side1Faces=face_sequence(instance.faces, indices))
        force = PLAN['object_forces_N'][ordinal - 1]
        direction = tuple(PLAN['force_direction'])
        traction = force / area
        model.SurfaceTraction(name='Force-Obround-%02d' % ordinal, createStepName=STEP_NAME,
                              region=surface, distributionType=UNIFORM, magnitude=traction,
                              traction=GENERAL, directionVector=((0.0, 0.0, 0.0), direction),
                              localCsys=None, follower=OFF, resultant=ON)
        audit['loads'].append({'object_index': obj['object_index'], 'surface': surface_name,
                               'occ_face_numbers': obj['original_face_numbers'], 'abaqus_indices': indices,
                               'area_mm2': area, 'traction_N_mm2': traction,
                               'prescribed_force_N': [v * force for v in direction]})
    p.seedPart(size=PLAN['global_mesh_mm'], deviationFactor=0.1, minSizeFactor=0.1)
    p.setMeshControls(regions=p.cells[:], elemShape=TET, technique=FREE)
    p.setElementType(regions=(p.cells[:],), elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),))
    p.generateMesh()
    assembly.regenerate()
    if not len(p.elements) or not len(p.nodes):
        raise RuntimeError('Mesh generation produced no elements/nodes')
    counts = {}
    for element in p.elements:
        counts[str(element.type)] = counts.get(str(element.type), 0) + 1
    if set(counts) != set(['C3D10']):
        raise RuntimeError('Unexpected element type: ' + repr(counts))
    audit.update(mesh_seed_mm=PLAN['global_mesh_mm'], node_count=len(p.nodes),
                 element_count=len(p.elements), element_types=counts, material=PLAN['material'],
                 expected_applied_force_N=PLAN['expected_force_N'])
    job_name = JOB_NAME
    serial = 1
    while job_name in mdb.jobs:
        job_name = JOB_NAME + '_%d' % serial
        serial += 1
    job = mdb.Job(name=job_name, model=unique_model, type=ANALYSIS,
                  numCpus=NUM_CPUS, numDomains=NUM_CPUS)
    job.writeInput(consistencyChecking=ON)
    cae_path = os.path.join(RUN_DIR, job_name + '.cae')
    mdb.saveAs(pathName=cae_path)
    audit.update(cae_path=cae_path, input_path=os.path.join(RUN_DIR, job_name + '.inp'), status='PREPARED')
    dump(AUDIT_PATH, audit)
    if not SUBMIT_JOB:
        return
    job.submit(consistencyChecking=ON)
    job.waitForCompletion()
    odb_path = os.path.join(RUN_DIR, job_name + '.odb')
    odb = session.openOdb(name=odb_path, readOnly=True)
    try:
        diagnostic = odb.diagnosticData
        frame = odb.steps[STEP_NAME].frames[-1]
        fields = frame.fieldOutputs
        if not all(name in fields.keys() for name in ('U', 'S', 'RF')):
            raise RuntimeError('Missing U, S, or RF results')
        fixed_nodes = odb.rootAssembly.nodeSets['FIXED-COUNTERBORES']
        reactions = fields['RF'].getSubset(region=fixed_nodes).values
        reaction = [sum(float(value.data[k]) for value in reactions) for k in range(3)]
        applied = PLAN['expected_force_N']
        residual = [reaction[k] + applied[k] for k in range(3)]
        error = math.sqrt(sum(v * v for v in residual)) / math.sqrt(sum(v * v for v in applied))
        status = str(diagnostic.jobStatus)
        completed = status == 'JOB_STATUS_COMPLETED_SUCCESSFULLY' and diagnostic.numberOfAnalysisErrors == 0
        final_time_ok = abs(float(frame.frameValue) - 1.0) < 1e-8
        audit.update(job_status=str(job.status), diagnostic_status=status,
                     analysis_errors=int(diagnostic.numberOfAnalysisErrors),
                     analysis_warnings=int(diagnostic.numberOfAnalysisWarnings), odb_path=odb_path,
                     final_frame_time=float(frame.frameValue), reaction_force_N=reaction,
                     equilibrium_residual_N=residual, equilibrium_relative_error=error,
                     max_displacement_mm=max(float(value.magnitude) for value in fields['U'].values),
                     max_von_mises_mpa=max(float(value.mises) for value in fields['S'].values),
                     fixed_node_count=len(reactions), field_outputs=['U', 'S', 'RF'])
        if not completed or not final_time_ok or error > PLAN['mesh_force_balance_relative_tolerance']:
            raise RuntimeError('Solver completion or force balance validation failed')
        audit['status'] = 'PASS'
    finally:
        odb.close()


started = time.time()
audit = {'case': 'Angle Bracket', 'status': 'FAIL', 'truth_used': False, 'run_dir': RUN_DIR,
         'model_checkpoint_sha256': PLAN['model']['checkpoint_sha256'],
         'prediction_sha256': PLAN['prediction_sha256'], 'abaqus_python': sys.version}
try:
    main(audit)
except Exception:
    audit.update(status='FAIL', error=traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    audit['elapsed_seconds'] = time.time() - started
    dump(AUDIT_PATH, audit)
    print('AEI result: ' + AUDIT_PATH)
