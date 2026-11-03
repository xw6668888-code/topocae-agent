from __future__ import annotations

import copy
import time
from collections import Counter

from common import *
from p1_e6.common import object_counts, aggregate_object_counts, truth_objects
from p1_e6r import evaluate as legacy_evaluate
from p1_e6r.common import TARGET_CLASSES
from p1_e6r.candidate_decision import resolve_candidate_macro
from p1_m2.boundary import enforce_base_candidate_boundaries
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction
from utils.p1_b_global_decode_v2 import decode_feature_objects
from p1_e6r2_r3.instance_boundary import (
    constrain_feature_objects_to_instance_components, predicted_same_instance_components, _topology_components,
)


def canonical(objects):
    return sorted((obj['macro_feature'], tuple(sorted(obj['face_indices']))) for obj in objects)


def without_truth(graph):
    result = copy.deepcopy(graph)
    for field in ('canonical_patches', 'face_to_canonical_patch', 'feature_sets', 'labels', 'node_labels', 'instance_ids'):
        result.pop(field, None)
    for node in result.get('nodes', []):
        for field in ('macro_feature', 'macro_feature_id', 'instance_id', 'semantic_label', 'semantic_role', 'label', 'face_class_label'):
            node.pop(field, None)
    return result


def boundary_objects(graph, objects, probabilities, threshold, relation_edges=None):
    if relation_edges is None:
        return constrain_feature_objects_to_instance_components(graph, objects, probabilities, threshold)
    _, by_node = predicted_same_instance_components(len(graph['nodes']), relation_edges, probabilities, threshold)
    outputs, splits, rejections = [], [], []
    for index, obj in enumerate(objects):
        groups = {}
        for face in obj['face_indices']:
            if face not in by_node:
                rejections.append({'candidate_index': index, 'reason': 'invalid_face_index'})
                continue
            groups.setdefault(by_node[face], set()).add(face)
        parts = [part for faces in groups.values() for part in _topology_components(faces, graph['edges'])]
        if len(parts) > 1:
            splits.append({'candidate_index': index, 'output_part_count': len(parts)})
        if not parts:
            rejections.append({'candidate_index': index, 'reason': 'empty_after_validation'})
        outputs.extend({**obj, 'face_indices': sorted(part)} for part in parts)
    return {'feature_objects': outputs, 'split_events': splits, 'rejections': rejections}


def recover_stages(graph, prediction, same, candidate_thresholds, global_decode, sample_id):
    graph = without_truth(graph)
    relations = prediction.get('relation_edges')
    probabilities = prediction.get('relation_probabilities', prediction['edge_probabilities'])
    relation_graph = graph if relations is None else {**graph, 'edges': relations, 'edge_attributes': [{} for _ in relations]}
    face_prediction = {'faces': prediction['faces'], 'node_embeddings': [],
        'edge_same_instance_probabilities': probabilities, 'contracts': {'truth_used': False, 'latent_candidate_classifier_enabled': False}}
    costs = {}
    start = time.perf_counter()
    assembly = assemble_same_instance_candidates(relation_graph, face_prediction,
        global_threshold=float(same['global_threshold']), threshold_selection_sha256=same['selection_sha256'])
    candidates, candidate_decisions = legacy_evaluate.candidate_objects(assembly, face_prediction, sample_id)
    costs['candidate'] = time.perf_counter() - start
    start = time.perf_counter()
    # Geometry-based proposal generation receives original physical adjacency.
    physical_prediction = {**face_prediction, 'edge_same_instance_probabilities': prediction['edge_probabilities']}
    raw = generate_candidate_proposals(graph, physical_prediction, assembly)['proposals']
    boundary = enforce_base_candidate_boundaries(raw, assembly)
    proposals, verified, decisions = [], [], []
    face_macros = {int(face['idx']): str(face['macro_feature']) for face in prediction['faces']}
    for proposal in boundary['proposals']:
        decision = resolve_candidate_macro(proposal, physical_prediction)
        decisions.append({'candidate_id': proposal.get('candidate_id'), **decision.to_dict()})
        if decision.status != 'selected' or decision.selected_macro not in TARGET_CLASSES:
            continue
        faces = sorted(set(int(i) for i in proposal.get('face_indices', [])))
        macro = str(decision.selected_macro)
        proposals.append({'sample_id': sample_id, 'macro_feature': macro, 'face_indices': faces})
        synthetic = {'candidate_id': proposal.get('candidate_id'), 'face_indices': faces,
            'macro_classes': list(TARGET_CLASSES), 'macro_logits': [20.0 if name == macro else -20.0 for name in TARGET_CLASSES],
            'macro_probabilities': [1.0 if name == macro else 0.0 for name in TARGET_CLASSES],
            'validity_score': 1.0, 'membership_probabilities': [1.0] * len(faces),
            'boundary_probabilities': [1.0 if face in legacy_evaluate._boundary_faces(graph, set(faces)) else 0.0 for face in faces],
            'face_macro_predictions': face_macros}
        verified.append((proposal, synthetic, decision.to_dict()))
    costs['proposals'] = time.perf_counter() - start
    start = time.perf_counter()
    verified_rows = []
    for proposal, synthetic, decision in verified:
        result = verify_candidate_prediction(graph, proposal, synthetic, thresholds=candidate_thresholds)
        result['candidate_macro_decision'] = decision
        verified_rows.append(result)
    accepted = [{'sample_id': sample_id, 'macro_feature': v['predicted_macro'], 'face_indices': list(v['refined_face_indices'])}
                for v in verified_rows if v.get('status') == 'accepted']
    costs['verified'] = time.perf_counter() - start
    start = time.perf_counter()
    decoded = decode_feature_objects(graph, verified_rows,
        subset_confidence_tolerance=float(global_decode['global_subset_confidence_tolerance']), selection_sha256=global_decode['selection_sha256'])
    costs['decoded'] = time.perf_counter() - start
    start = time.perf_counter()
    full = boundary_objects(graph, decoded['feature_objects'], probabilities, float(same['global_threshold']), relations)
    costs['full'] = time.perf_counter() - start
    start = time.perf_counter()
    constrained_candidate = boundary_objects(graph, candidates, probabilities, float(same['global_threshold']), relations)
    costs['candidate_boundary'] = time.perf_counter() - start
    return {
        'objects': {'candidate': candidates, 'proposals': proposals, 'verified': accepted,
            'decoded': decoded['feature_objects'], 'full': full['feature_objects'],
            'candidate_boundary': constrained_candidate['feature_objects']},
        'incremental_seconds': costs,
        'rejections': {
            'candidate': sum(x['status'] != 'selected' for x in candidate_decisions),
            'proposals': sum(x['status'] != 'selected' for x in decisions),
            'verified': sum(v.get('status') != 'accepted' for v in verified_rows),
            'decoded': decoded['global_conflict_reject_count'] + decoded['subset_dominated_count'],
            'full': len(full['rejections']), 'candidate_boundary': len(constrained_candidate['rejections'])},
        'trace': {'assembly': assembly, 'proposal_boundary': boundary, 'macro_decisions': decisions,
            'verified': verified_rows, 'decoded': decoded, 'instance_boundary': full, 'candidate_boundary': constrained_candidate},
    }


def counts(truth, objects, node_count, edges):
    legacy = object_counts(truth, objects)
    true_sets = Counter(canonical(truth))
    pred_sets = Counter(canonical(objects))
    matches = true_sets & pred_sets
    per_class = {name: {'tp': sum(v for (label, _), v in matches.items() if label == name),
        'truth': sum(v for (label, _), v in true_sets.items() if label == name),
        'prediction': sum(v for (label, _), v in pred_sets.items() if label == name)} for name in TARGET_CLASSES}
    exact = {**legacy, 'tp': sum(matches.values()), 'fp': len(objects) - sum(matches.values()),
        'fn': len(truth) - sum(matches.values()), 'exact': sum(matches.values()), 'per_class': per_class}
    invalid_objects = 0
    for obj in objects:
        faces = obj['face_indices']
        invalid_objects += int(not faces or len(faces) != len(set(faces)) or any(f < 0 or f >= node_count for f in faces))
    disconnected = sum(len(_topology_components(set(o['face_indices']), edges)) > 1 for o in objects)
    return {'iou_0_5': legacy, 'exact': exact, 'invalid_objects': invalid_objects, 'disconnected_objects': disconnected}


def summarize(rows):
    names = list(rows[0]['stages'])
    results = {}
    for name in names:
        results[name] = {metric: aggregate_object_counts([{'count': row['stages'][name][metric]} for row in rows], 'count') for metric in ('exact', 'iou_0_5')}
        results[name].update({
            'invalid_objects': sum(row['stages'][name]['invalid_objects'] for row in rows),
            'disconnected_objects': sum(row['stages'][name]['disconnected_objects'] for row in rows),
            'rejection_count': sum(row['rejections'][name] for row in rows),
            'incremental_seconds': sum(row['incremental_seconds'][name] for row in rows),
        })
    return results


def replay(seed):
    import torch
    from p1_e6r2_r3.training_common import build_model
    torch.set_num_threads(4)
    folder = ARTIFACT / 'baseline_replay' / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    records = load_records('val')
    cache = folder / 'predictions.pt'
    if cache.exists():
        predictions = torch.load(cache, map_location='cpu', weights_only=False)
    else:
        model = build_model().eval()
        model.load_state_dict(torch.load(run_dir(seed) / 'best.pt', map_location='cpu', weights_only=False)['model_state_dict'])
        started = time.perf_counter()
        predictions = legacy_evaluate.infer_records(records, model, torch.device('cpu'))
        torch.save(predictions, cache)
        write_json(folder / 'prediction_manifest.json', {'checkpoint': hashed(run_dir(seed) / 'best.pt'), 'val': hashed(DATA / 'val.pt'),
            'predictions': hashed(cache), 'device': 'cpu', 'seconds': time.perf_counter() - started, 'created_at': now()})
    same = read_json(run_dir(seed) / 'same_instance_threshold_selection.json')
    candidate, decoder = [read_json(p) for p in decoder_assets()]
    rows, equivalence = [], []
    from p1_e6r2_r3.validate import evaluate_constrained_sample
    trace_path = folder / 'stages.jsonl'
    with trace_path.open('w', encoding='utf-8') as stream:
        for index, prediction in enumerate(predictions):
            record = prediction['record']
            graph = read_json(graph_path(record['sample_id']))
            result = recover_stages(graph, prediction, same, candidate, decoder, record['sample_id'])
            old = evaluate_constrained_sample(prediction, graph, same, candidate, decoder)
            same_objects = canonical(old['full_objects']) == canonical(result['objects']['full']) and canonical(old['candidate_objects']) == canonical(result['objects']['candidate_boundary'])
            equivalence.append(same_objects)
            truth = truth_objects(record['feature_sets'])
            row = {'sample_id': record['sample_id'], 'truth_objects': truth,
                'stages': {name: counts(truth, objects, len(graph['nodes']), graph['edges']) for name, objects in result['objects'].items()},
                'rejections': result['rejections'], 'incremental_seconds': result['incremental_seconds']}
            rows.append(row)
            stream.write(json.dumps({**row, **result}, ensure_ascii=False, allow_nan=False) + '\n')
            if (index + 1) % 100 == 0:
                print(f'seed={seed} replay={index + 1}/{len(predictions)}', flush=True)
    summary = summarize(rows)
    historical = read_json(OLD_REPORT / 'validation_metrics.json')['seed_results'][str(seed)]
    differences = {stage: summary[stage]['iou_0_5']['instance_class_macro_f1'] - historical[key]['instance_class_macro_f1']
        for stage, key in (('full', 'full_pipeline_objects'), ('candidate_boundary', 'candidate_objects'))}
    output = {'seed': seed, 'sample_count': len(rows), 'device': 'cpu', 'stages': summary,
        'legacy_object_equivalence': all(equivalence), 'legacy_equivalence_count': sum(equivalence),
        'historical_macro_f1_deltas': differences, 'created_at': now(), 'prediction_cache': hashed(cache), 'traces': hashed(trace_path)}
    write_json(folder / 'summary.json', output)
    print(json.dumps({'seed': seed, 'equivalent': all(equivalence), 'historical_deltas': differences,
        'scores': {name: {'exact': val['exact']['instance_class_macro_f1'], 'iou': val['iou_0_5']['instance_class_macro_f1'], 'overmerge': val['exact']['overmerge_count'], 'fn': val['exact']['fn']} for name, val in summary.items()}}))
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    replay(parser.parse_args().seed)
