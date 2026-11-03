"""G07 decoder experiments. Geometry and model probabilities are the only inputs.

This is a new decoder policy around the existing G03 relation head; it is not a
new neural architecture. No truth, filename, source ID, or fixed face ID rule.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'paper_release/topocae_p1_legacy_baseline_v1'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from p1_e6r.candidate_decision import resolve_candidate_macro
from p1_e6r.common import TARGET_CLASSES
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_global_decode import _candidate_relation, _adjacency

CONFIGS = {
    'sparse_old_full': {'model': 'full', 'policy': 'old_full'},
    'sparse_minimal': {'model': 'full', 'policy': 'minimal'},
    'dense_old_full': {'model': 'dense_instance', 'policy': 'old_full'},
    'dense_minimal': {'model': 'dense_instance', 'policy': 'minimal'},
    'dense_geometry_minimal': {'model': 'dense_instance', 'policy': 'minimal', 'domain': 'geometry'},
    'dense_decoded': {'model': 'dense_instance', 'policy': 'decoded'},
    'dense_equivariant': {'model': 'dense_instance', 'policy': 'equivariant'},
    'dense_equivariant_full': {'model': 'dense_instance', 'policy': 'equivariant_full'},
}


def canonical(objects):
    return sorted((o['macro_feature'], tuple(sorted(o['face_indices']))) for o in objects)


def clean_graph(graph):
    """Explicit decoder allowlist: geometry primitive/attributes and B-Rep edges."""
    return {'schema_version': graph.get('schema_version'),
        'nodes': [{'idx': i, 'primitive_surface': n.get('primitive_surface', ''),
                   'primitive_surface_id': n.get('primitive_surface_id', 0),
                   'attributes': copy.deepcopy(n.get('attributes', {}))}
                  for i, n in enumerate(graph['nodes'])],
        'edges': copy.deepcopy(graph['edges']),
        'edge_attributes': copy.deepcopy(graph.get('edge_attributes', [])),
        'part_metadata_v2': copy.deepcopy(graph.get('part_metadata_v2', {}))}


def clean_prediction(prediction):
    result = {k: copy.deepcopy(prediction[k]) for k in
              ('faces', 'edge_probabilities', 'relation_edges', 'relation_probabilities') if k in prediction}
    return result


def validate_objects(objects, node_count):
    result, rejected, seen = [], [], set()
    for obj in objects:
        faces = obj.get('face_indices', [])
        reason = None
        if not faces:
            reason = 'empty_object'
        elif any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= node_count for i in faces):
            reason = 'invalid_face_index'
        elif len(faces) != len(set(faces)):
            reason = 'duplicate_member'
        elif obj.get('macro_feature') not in TARGET_CLASSES:
            reason = 'invalid_class'
        else:
            key = (obj['macro_feature'], tuple(sorted(faces)))
            if key in seen:
                reason = 'duplicate_object'
            else:
                seen.add(key)
                result.append({'macro_feature': key[0], 'face_indices': list(key[1])})
        if reason:
            rejected.append({'reason': reason, 'object': obj})
    return result, rejected


def geometry_relation_allowed(graph, left, right):
    """Physical neighbors or overlapping centroid support radii; no union by size.

    Support radius = half the supplied orientation-stable projected long extent.
    All admitted pairs still require the trained relation probability threshold.
    Missing geometry conservatively excludes a nonadjacent pair.
    """
    a, b = graph['nodes'][left].get('attributes', {}), graph['nodes'][right].get('attributes', {})
    try:
        ca, cb = a['physical']['center_of_mass'], b['physical']['center_of_mass']
        ra = a['shape_descriptors']['projected_long_extent'] / 2
        rb = b['shape_descriptors']['projected_long_extent'] / 2
        return math.dist(ca, cb) <= ra + rb + 1e-9 * max(ra + rb, 1.0)
    except (KeyError, TypeError, ValueError):
        return False


def relation_domain(graph, prediction, geometry=False):
    edges = prediction.get('relation_edges', graph['edges'])
    probabilities = prediction.get('relation_probabilities', prediction['edge_probabilities'])
    if len(edges) != len(probabilities):
        raise ValueError('relation probability count mismatch')
    physical = {tuple(sorted(e)) for e in graph['edges']}
    keep = [i for i, (a, b) in enumerate(edges) if not geometry or tuple(sorted((a, b))) in physical
            or geometry_relation_allowed(graph, a, b)]
    return [edges[i] for i in keep], [probabilities[i] for i in keep]


def candidate_decode(graph, prediction, same, geometry=False):
    edges, probabilities = relation_domain(graph, prediction, geometry)
    dense = 'relation_edges' in prediction
    rg = {**graph, 'edges': edges, 'edge_attributes': [{} for _ in edges]} if dense else graph
    fp = {'faces': prediction['faces'], 'edge_same_instance_probabilities': probabilities}
    assembly = assemble_same_instance_candidates(rg, fp,
        global_threshold=float(same['global_threshold']), threshold_selection_sha256=same['selection_sha256'])
    objects, decisions = [], []
    for candidate in assembly['candidates']:
        decision = resolve_candidate_macro({'face_indices': candidate['face_indices'],
            'macro_hints': list(TARGET_CLASSES), 'topology_evidence': {'source': 'same_instance_candidate_assembly'}}, fp)
        decisions.append(decision.to_dict())
        if decision.status == 'selected' and decision.selected_macro in TARGET_CLASSES:
            objects.append({'macro_feature': decision.selected_macro, 'face_indices': candidate['face_indices']})
    output, rejected = validate_objects(objects, len(graph['nodes']))
    return {'feature_objects': output, 'trace': {'assembly': assembly, 'decisions': decisions,
            'legality_rejections': rejected, 'relation_pair_count': len(edges)}, 'truth_used': False}


def equivariant_conflicts(graph, verified, tolerance):
    """Score/size priority, simultaneous subset filtering and exact-tie abstention.

    Identical class/member objects are collapsed; genuinely competing equal-score
    equal-size peers all abstain. IDs and coordinates never break decision ties.
    This covers conflict selection only; legacy proposal/verification can still
    have their own ordering dependencies. It claims no full rotation invariance.
    """
    unique = {}
    for row in verified:
        if row.get('status') != 'accepted':
            continue
        key = (str(row['predicted_macro']), tuple(sorted(set(row['refined_face_indices']))))
        score = float(row.get('final_confidence', 0) or 0)
        if key not in unique or score > unique[key]['final_confidence']:
            unique[key] = {'predicted_macro': key[0], 'refined_face_indices': list(key[1]),
                           'final_confidence': score, 'candidate_id': 'metadata_only'}
    accepted = list(unique.values())
    dominated = [a for a in accepted if any(a['predicted_macro'] == b['predicted_macro']
        and set(a['refined_face_indices']) < set(b['refined_face_indices'])
        and b['final_confidence'] + tolerance >= a['final_confidence'] for b in accepted)]
    ordered = sorted((a for a in accepted if a not in dominated),
                     key=lambda a: (-a['final_confidence'], -len(a['refined_face_indices'])))
    selected, rejected = [], []
    adjacency = _adjacency(graph)
    for _, peers_iter in itertools.groupby(ordered, key=lambda a: (a['final_confidence'], len(a['refined_face_indices']))):
        peers = list(peers_iter)
        available = []
        for a in peers:
            if any(_candidate_relation(a, b, adjacency)['kind'] == 'conflict' for b in selected):
                rejected.append({'reason': 'higher_priority_conflict', 'candidate': a})
            else:
                available.append(a)
        tied = {i for i, a in enumerate(available) if any(i != j and
            _candidate_relation(a, b, adjacency)['kind'] == 'conflict' for j, b in enumerate(available))}
        rejected.extend({'reason': 'indistinguishable_priority_conflict_abstention', 'candidate': available[i]} for i in sorted(tied))
        selected.extend(a for i, a in enumerate(available) if i not in tied)
    return {'feature_objects': [{'macro_feature': a['predicted_macro'],
            'face_indices': a['refined_face_indices']} for a in selected],
        'rejections': rejected, 'subset_dominated_count': len(dominated)}


_STAGES = None
def legacy_stages():
    global _STAGES
    if _STAGES is None:
        folder = ROOT / 'tools/aei_paper_goals_v1/g03'
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            result = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result)
            return result
        old_common = sys.modules.get('common')
        try:
            sys.modules['common'] = module('_g07_g03_common', folder / 'common.py')
            _STAGES = module('_g07_g03_stages', folder / 'stages.py')
        finally:
            if old_common is None:
                sys.modules.pop('common', None)
            else:
                sys.modules['common'] = old_common
    return _STAGES


def frozen_decoder_assets():
    base = SOURCE / 'reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718'
    return [json.loads(p.read_text(encoding='utf-8-sig')) for p in
            (base / 'candidate_model/candidate_threshold_selection.json', base / 'global_decode_selection.json')]


def decode(graph, prediction, same, config='dense_minimal', *, cached_stages=None,
           candidate_thresholds=None, global_decode=None):
    """Public inference interface. Cached stages are an evaluation-only speedup.

    Return feature_objects, trace and truth_used=False. Callers should pass only
    STEP-derived geometry and predictions; any record/feature_sets is dropped.
    """
    cfg = CONFIGS[config]
    graph, prediction = clean_graph(graph), clean_prediction(prediction)
    if cfg['model'] == 'dense_instance' and 'relation_edges' not in prediction:
        raise ValueError('dense decoder requires explicit dense relation probabilities')
    if cfg['policy'] == 'minimal':
        return candidate_decode(graph, prediction, same, cfg.get('domain') == 'geometry')
    if candidate_thresholds is None or global_decode is None:
        candidate_thresholds, global_decode = frozen_decoder_assets()
    stages = legacy_stages()
    recovered = cached_stages if cached_stages is not None else stages.recover_stages(
        graph, prediction, same, candidate_thresholds, global_decode, 'unlabelled_input')
    policy = cfg['policy']
    if policy in ('old_full', 'decoded'):
        objects = recovered['objects']['full' if policy == 'old_full' else 'decoded']
        trace = {'source': 'frozen_G03_recovery', 'policy': policy}
    else:
        trace = equivariant_conflicts(graph, recovered['trace']['verified'],
            float(global_decode['global_subset_confidence_tolerance']))
        objects = trace['feature_objects']
        if policy == 'equivariant_full':
            trace['boundary'] = stages.boundary_objects(graph, objects,
                prediction.get('relation_probabilities', prediction['edge_probabilities']),
                float(same['global_threshold']), prediction.get('relation_edges'))
            objects = trace['boundary']['feature_objects']
    objects, rejected = validate_objects(objects, len(graph['nodes']))
    return {'feature_objects': objects, 'trace': {**trace, 'legality_rejections': rejected}, 'truth_used': False}


def permute_inputs(graph, prediction, old_to_new):
    graph, prediction = clean_graph(graph), clean_prediction(prediction)
    nodes, faces = [None] * len(old_to_new), [None] * len(old_to_new)
    for old, new in enumerate(old_to_new):
        nodes[new] = {**graph['nodes'][old], 'idx': new}
    for face in prediction['faces']:
        new = old_to_new[face['idx']]
        faces[new] = {**face, 'idx': new}
    graph['nodes'] = nodes
    graph['edges'] = [[old_to_new[a], old_to_new[b]] for a, b in graph['edges']]
    prediction['faces'] = faces
    if 'relation_edges' in prediction:
        prediction['relation_edges'] = [[old_to_new[a], old_to_new[b]] for a, b in prediction['relation_edges']]
    return graph, prediction
