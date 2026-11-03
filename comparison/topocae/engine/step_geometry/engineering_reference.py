"""Evaluate arbitrary engineering predictions without CAD/CAE or model execution.

Existing geometric helper function bodies are loaded by AST into an isolated
namespace. Cylinder-top completion receives the supplied graph rather than
opening the historical model graph. Expected regions are read only by scoring.
"""
from __future__ import annotations
import argparse
import ast
import collections
import copy
import itertools
import json
import math
import sys
from pathlib import Path

from g07_common import ROOT, FROZEN, REPORT, ARTIFACT, asset, read, sha, write

ENG = REPORT / 'engineering'
REFERENCE = ROOT / 'reports/aei_case_suite_v2/controlled_v1/recognition_contribution'
HELPER = ROOT / 'tools/aei_case_suite/geometry.py'
TOLERANCES = {'radius_mm': 1e-5, 'thickness_mm': 1e-4, 'spacing_mm': 1e-4, 'cylinder_depth_mm': 1e-5}


class PlanError(ValueError):
    def __init__(self, code, detail=''):
        self.code = code
        super().__init__(code + (': ' + detail if detail else ''))


def helpers():
    source = ast.parse(HELPER.read_text(encoding='utf-8-sig'))
    names = {'dot', 'sub', 'add', 'mul', 'unit', 'select', 'roles', 'complete_cylinder_top', 'cylinder', 'thickness'}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    completion = next(node for node in functions if node.name == 'complete_cylinder_top')
    # The two removed statements constructed and opened OUT/recognition/<part>_graph.json.
    if [type(x).__name__ for x in completion.body[:2]] != ['Assign', 'Assign']:
        raise ValueError('Historical completion helper shape changed; re-review adapter')
    completion.body[:2] = ast.parse("graph = data['_graph']").body
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {'math': math, 'itertools': itertools, 'json': json,
                 'PlanError': PlanError, 'SELECTOR_RADIUS_MM': TOLERANCES['radius_mm']}
    exec(compile(module, str(HELPER) + ':g07_geometry_adapter', 'exec'), namespace)
    return namespace


H = helpers()


def measured_faces(step):
    """Fresh OCC measurements, no recognition/target fields copied from old cache."""
    if str(FROZEN) not in sys.path:
        sys.path.insert(0, str(FROZEN))
    from utils.step_solid_validation import read_step_shape
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_REVERSED
    xyz = lambda value: [float(value.X()), float(value.Y()), float(value.Z())]
    shape = read_step_shape(Path(step))
    rows = []
    for i, face in enumerate(TopologyExplorer(shape).faces()):
        surface = BRepAdaptor_Surface(face)
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        row = {'face_number': i + 1, 'area_mm2': float(props.Mass()), 'center_mm': xyz(props.CentreOfMass()),
            'type': str(surface.GetType()), 'vertices_mm': [xyz(BRep_Tool.Pnt(v)) for v in TopologyExplorer(face).vertices()]}
        if surface.GetType() == GeomAbs_Plane:
            normal = xyz(surface.Plane().Axis().Direction())
            row['normal'] = [-v for v in normal] if face.Orientation() == TopAbs_REVERSED else normal
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            row.update(radius_mm=cylinder.Radius(), axis=xyz(cylinder.Axis().Direction()), origin_mm=xyz(cylinder.Axis().Location()),
                v_range=[surface.FirstVParameter(), surface.LastVParameter()],
                u_range=[surface.FirstUParameter(), surface.LastUParameter()])
        rows.append(row)
    return rows


def adapt_objects(objects, graph):
    n = len(graph['nodes'])
    result = []
    for index, obj in enumerate(objects):
        cls = obj.get('class', obj.get('macro_feature'))
        if 'original_face_numbers' in obj:
            numbers = [int(f) for f in obj['original_face_numbers']]
        else:
            numbers = [int(f) + 1 for f in obj.get('faces', obj.get('face_indices', obj.get('graph_node_indices', [])))]
        if not cls or not numbers or len(numbers) != len(set(numbers)) or any(f < 1 or f > n for f in numbers):
            raise PlanError('invalid_prediction_objects')
        radii = sorted({float(graph['nodes'][f-1]['attributes']['specific'].get('radius', 0)) for f in numbers
                        if float(graph['nodes'][f-1]['attributes']['specific'].get('radius', 0)) > 0})
        result.append({'object_index': index, 'class': cls, 'original_face_numbers': sorted(numbers),
                       'topology_valid': bool(obj.get('topology_valid', True)),
                       'geometry_summary': {'positive_radii': radii}})
    return result


def prediction_data(part, feature_objects, face_predictions):
    geometry = read(ENG / 'geometry' / (Path(part).stem + '.json'))
    if sha(geometry['source_step']['path']) != geometry['source_step']['sha256']:
        raise ValueError('Engineering STEP changed')
    graph = read(geometry['input_graph']['path'])
    if sha(geometry['input_graph']['path']) != geometry['input_graph']['sha256']:
        raise ValueError('Engineering geometry graph changed')
    faces = copy.deepcopy(geometry['faces'])
    if face_predictions is not None and len(face_predictions) != len(faces):
        raise PlanError('semantic_face_count_mismatch')
    for i, row in enumerate(faces):
        pred = face_predictions[i] if face_predictions is not None else {}
        row['semantic_label'] = int(pred.get('semantic_label', -1))
        row['predicted_class'] = pred.get('class', pred.get('macro_feature', 'unavailable'))
    return {'part': part, 'source_sha256': geometry['source_step']['sha256'], 'faces': faces,
            'objects': adapt_objects(feature_objects, graph), '_graph': graph}


def parameter_values(data, obj):
    values = {}
    radii = obj['geometry_summary']['positive_radii']
    if radii:
        values['radii_mm'] = radii
    if obj['class'] in ('rib', 'web'):
        try:
            values['thickness_mm'] = H['thickness'](data, obj)['thickness_mm']
        except PlanError as exc:
            values['thickness_unavailable'] = exc.code
    depths = []
    for number in obj['original_face_numbers']:
        face = data['faces'][number - 1]
        if 'radius_mm' in face:
            try:
                depths.append({'face_number': number, 'radius_mm': face['radius_mm'],
                               'depth_mm': H['cylinder'](face)['depth_mm']})
            except PlanError:
                pass
    if depths:
        values['full_cylinder_depths'] = depths
    return values


def region_comparison(actual, expected, data):
    key = lambda groups: collections.Counter((g['class'], tuple(sorted(g['face_numbers']))) for g in groups)
    aa = {f for g in actual for f in g['face_numbers']}
    bb = {f for g in expected for f in g['face_numbers']}
    area = lambda faces: sum(data['faces'][f-1]['area_mm2'] for f in faces)
    return {'group_exact': key(actual) == key(expected), 'face_exact': aa == bb,
            'area_iou': area(aa & bb) / area(aa | bb) if aa | bb else 1.0,
            'extra_faces': sorted(aa - bb), 'missing_faces': sorted(bb - aa)}


def spacing_values(data, objects):
    """Same three-counterbore center construction as cad_executor, without editing."""
    if len(objects) != 3 or {o['class'] for o in objects} != {'counterbore_hole'}:
        raise PlanError('spacing_requires_three_counterbores')
    holes = []
    for obj in objects:
        cylinders = [H['cylinder'](data['faces'][n-1]) for n in obj['original_face_numbers'] if 'radius_mm' in data['faces'][n-1]]
        if len(cylinders) != 2:
            raise PlanError('unsupported_counterbore_geometry')
        holes.append({'faces': obj['original_face_numbers'], 'center': max(cylinders, key=lambda c: c['radius_mm'])['center_mm']})
    center = min(holes, key=lambda h: sum(math.dist(h['center'], other['center']) for other in holes))
    sides = [h for h in holes if h is not center]
    direction = H['unit'](H['sub'](sides[1]['center'], sides[0]['center']))
    distances = []
    for hole in sides:
        relative = H['sub'](hole['center'], center['center'])
        position = H['dot'](relative, direction)
        if math.hypot(*H['sub'](relative, H['mul'](direction, position))) > 1e-5:
            raise PlanError('spacing_pattern_not_collinear')
        distances.append(abs(position))
    return {'adjacent_center_distances_mm': sorted(distances), 'central_instance_faces': sorted(center['faces']),
            'central_center_mm': center['center']}


def generate_contract():
    from input_audit_followup import parse_manual_truth
    truth_path = ROOT / 'test_part_0901/真值.txt'
    truth = parse_manual_truth(truth_path.read_text(encoding='utf-8-sig'))
    jobs = [j for j in read(REPORT / 'input_audit/initial_protocol.json')['jobs'] if j['split'] == 'engineering']
    parts = []
    for job in jobs:
        part = Path(job['step_path']).name
        graph_path = ARTIFACT / 'input_audit' / job['audit_id'] / 'step_input_graph.json'
        faces = measured_faces(job['step_path'])
        prior = read(ROOT / 'reports/aei_case_suite_v2/geometry' / (Path(part).stem + '.json'))
        deltas = []
        for new, old in zip(faces, prior['faces']):
            for key in new:
                if new[key] != old[key]:
                    deltas.append({'face_number': new['face_number'], 'field': key, 'new': new[key], 'old': old[key]})
        if len(faces) != len(prior['faces']) or deltas:
            raise ValueError('Fresh engineering geometry differs from historical measured cache: ' + part)
        geometry = {'part': part, 'source_step': asset(job['step_path']), 'input_graph': asset(graph_path),
                    'faces': faces, 'fresh_OCC_measurements_equal_previous': True}
        write(ENG / 'geometry' / (Path(part).stem + '.json'), geometry)
        alias = {'Stiffened Panel.stp': 'Stiffened Panel_stp.stp'}.get(part, part)
        expected = truth[alias]['objects']
        # Parameter references are read from fresh geometric measurements of manually specified instances.
        objects = [{'class': x['class'], 'original_face_numbers': x['faces']} for x in expected]
        data = prediction_data(part, objects, None)
        references = [{'class': obj['class'], 'faces': obj['original_face_numbers'], 'parameters': parameter_values(data, obj)} for obj in data['objects']]
        parts.append({'part': part, 'truth_name': alias, 'source_step': asset(job['step_path']),
            'geometry': asset(ENG / 'geometry' / (Path(part).stem + '.json')), 'parameter_reference_instances': references})
    operations = []
    top = {(r['part'], r['operation_index']): r for r in read(REFERENCE / 'top_face_review.json')}
    for row in read(REFERENCE / 'operation_results.json'):
        independent = row['independent_role_truth']
        expected = row['expected_instance_groups'] if independent else top[(row['part'], row['operation_index'])]['candidate_top_faces']
        record = {key: row[key] for key in ('part', 'operation_index', 'action', 'target', 'role')} | \
            {'reference_kind': 'independent_instance_face_sets' if independent else 'reviewed_top_face_reference_not_independent_truth',
             'expected_groups': [{'class': r['class'], 'face_numbers': sorted(r['face_numbers'])} for r in expected]}
        if row['action'] == 'set_hole_spacing':
            gt_objects = [{'class': g['class'], 'original_face_numbers': g['face_numbers']} for g in expected]
            data = prediction_data(row['part'], gt_objects, None)
            record['source_spacing_reference'] = spacing_values(data, data['objects'])
        operations.append(record)
    result = {'schema': 'g07_engineering_reference_contract_v1', 'parts': parts, 'operations': operations,
        'counts': {'parts': len(parts), 'instances': sum(len(p['parameter_reference_instances']) for p in parts),
            'independent_target_operations': sum(o['reference_kind'] == 'independent_instance_face_sets' for o in operations),
            'reviewed_top_face_operations': sum(o['reference_kind'] != 'independent_instance_face_sets' for o in operations)},
        'parameter_tolerances': TOLERANCES,
        'parameter_reference_policy': 'Geometry measured from current sourceSTEP on complete independently manually specified instance faces. This is geometric measurement agreement, not an independent metrology experiment.',
        'role_reference_policy': '18 instance-face target operations have manual-instance references; three topface groups are previously reviewed engineering references, not blinded independent role labels.',
        'helper_implementation': asset(HELPER), 'helper_adapter': asset(__file__),
        'completion_change': 'Only file loading is replaced with supplied geometrygraph; original transition-adjacency, semantic and axial tests retained.',
        'source_assets': [asset(truth_path), asset(REFERENCE / 'operation_results.json'), asset(REFERENCE / 'top_face_review.json'),
                          asset(ROOT / 'tools/aei_case_suite/parameter_policy.py')],
        'no_CAD_CAE_solver_execution': True, 'no_external_API': True}
    write(ENG / 'REFERENCE_CONTRACT.json', result)
    return result


def evaluate(part, feature_objects, face_predictions, *, role_completion=True):
    contract = read(ENG / 'REFERENCE_CONTRACT.json')
    reference = next(p for p in contract['parts'] if p['part'] == part)
    result = {'part': part, 'role_completion': role_completion, 'target_operations': [], 'parameters': []}
    try:
        data = prediction_data(part, feature_objects, face_predictions)
    except Exception as exc:
        return {**result, 'prediction_error': repr(exc), 'passed': False}
    for op in (o for o in contract['operations'] if o['part'] == part):
        record = {key: op[key] for key in ('operation_index', 'action', 'role', 'reference_kind')}
        try:
            objects = H['select'](data, op['target'])
            if not role_completion and op['role'] == 'top_face':
                for obj in objects:
                    semantic = {'cylinder_boss': 13, 'rectangular_boss': 51}.get(obj['class'])
                    if len([f for f in obj['original_face_numbers'] if data['faces'][f-1]['semantic_label'] == semantic and 'normal' in data['faces'][f-1]]) != 1:
                        raise PlanError('role_completion_disabled')
            groups = H['roles'](data, objects, op['role'])
            record.update(resolved=True, predicted_groups=groups, **region_comparison(groups, op['expected_groups'], data))
            if 'source_spacing_reference' in op:
                actual, expected = spacing_values(data, objects), op['source_spacing_reference']
                record['spacing_parameter'] = {'actual': actual, 'expected': expected,
                    'passed': actual['central_instance_faces'] == expected['central_instance_faces'] and
                    all(abs(a-b) <= TOLERANCES['spacing_mm'] for a, b in zip(actual['adjacent_center_distances_mm'], expected['adjacent_center_distances_mm']))}
        except Exception as exc:
            record.update(resolved=False, group_exact=False, face_exact=False, area_iou=0.0, error=str(exc))
        result['target_operations'].append(record)
    predicted = collections.defaultdict(list)
    for obj in data['objects']:
        predicted[(obj['class'], tuple(obj['original_face_numbers']))].append(obj)
    for ref in reference['parameter_reference_instances']:
        expected = ref['parameters']
        numeric = {k: v for k, v in expected.items() if k in ('radii_mm', 'thickness_mm', 'full_cylinder_depths')}
        if not numeric:
            continue
        key = (ref['class'], tuple(ref['faces']))
        record = {'class': ref['class'], 'faces': ref['faces'], 'expected': numeric}
        if not predicted[key]:
            record.update(matched_instance=False, passed=False, error='complete_instance_missing')
        else:
            obj = predicted[key].pop(0)
            actual = parameter_values(data, obj)
            checks = {}
            for parameter, value in numeric.items():
                other = actual.get(parameter)
                if parameter == 'radii_mm':
                    checks[parameter] = other is not None and len(value) == len(other) and all(abs(x-y) <= TOLERANCES['radius_mm'] for x, y in zip(value, other))
                elif parameter == 'thickness_mm':
                    checks[parameter] = other is not None and abs(value-other) <= TOLERANCES['thickness_mm']
                else:
                    checks[parameter] = other is not None and len(value) == len(other) and all(x['face_number'] == y['face_number'] and abs(x['depth_mm']-y['depth_mm']) <= TOLERANCES['cylinder_depth_mm'] for x, y in zip(value, other))
            record.update(matched_instance=True, actual=actual, checks=checks, passed=all(checks.values()))
        result['parameters'].append(record)
    result['passed'] = all(r['group_exact'] and r.get('spacing_parameter', {}).get('passed', True) for r in result['target_operations']) and all(r['passed'] for r in result['parameters'])
    result['limits'] = 'Target-region and geometric-parameter agreement do not replace full182-instance classification scoring or CAD/CAE execution.'
    return result


def verify_baseline():
    contract = generate_contract()
    outputs = []
    jobs = [j for j in read(REPORT / 'input_audit/initial_protocol.json')['jobs'] if j['split'] == 'engineering']
    for job in jobs:
        directory = ARTIFACT / 'input_audit' / job['audit_id']
        objects = read(directory / 'stages.json')['guard']['service_objects']
        faces = read(directory / 'probabilities.json')['guard']['faces']
        part = Path(job['step_path']).name
        outputs.extend(evaluate(part, objects, faces, role_completion=complete) for complete in (False, True))
    summary = {}
    for complete in (False, True):
        rows = [r for r in outputs if r['role_completion'] == complete]
        levels = {}
        for kind in ('independent_instance_face_sets', 'reviewed_top_face_reference_not_independent_truth'):
            operations = [op for r in rows for op in r['target_operations'] if op['reference_kind'] == kind]
            levels[kind] = {'correct': sum(o['group_exact'] for o in operations), 'total': len(operations)}
        params = [p for r in rows for p in r['parameters']]
        levels['parameter_reference_instances'] = {'correct': sum(p['passed'] for p in params), 'total': len(params)}
        levels['parameter_fields'] = dict(collections.Counter(k for p in params for k in p['expected']))
        spacing = [o['spacing_parameter'] for r in rows for o in r['target_operations'] if 'spacing_parameter' in o]
        levels['spacing_operations'] = {'correct': sum(s['passed'] for s in spacing), 'total': len(spacing)}
        summary['R_plus' if complete else 'R'] = levels
    result = {'reference_contract': asset(ENG / 'REFERENCE_CONTRACT.json'), 'summary': summary,
              'per_part': outputs, 'model_seed': 20260719, 'actual_new_model_forward_count': 0,
              'actual_fresh_geometry_faces_measured': sum(len(read(p['geometry']['path'])['faces']) for p in contract['parts'])}
    write(ENG / 'BASELINE_REFERENCE_EVALUATION.json', result)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify-baseline'])
    parser.parse_args()
    verify_baseline()
