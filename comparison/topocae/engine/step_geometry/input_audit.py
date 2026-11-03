"""Resumable G07 STEP audit. The inference worker accepts only geometry records.

Targets are deliberately absent; selected classes are used by the parent only for
stratification. No original artifact is changed.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'paper_release/topocae_p1_legacy_baseline_v1'
REPORT = ROOT / 'reports/aei_paper_goals_v1/g07'
OUT = ROOT / 'artifacts/aei_paper_goals_v1/g07/input_audit'
DATA = SOURCE / 'output_training_p1_e6r2_r3_graph_v2_2_1_v1'
GEN = SOURCE / 'output_generation_p1_e6r2_r1_3_r1_complete_r2_mixed_9300_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path).resolve()
    if not any(path.is_relative_to(p) for p in (OUT, REPORT)):
        raise ValueError('outside audit ownership')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def asset(path):
    p = Path(path).resolve()
    return {'path': p.as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}


def frozen(path):
    p = Path(path)
    if p.is_relative_to(SOURCE):
        return p
    if p.is_relative_to(ROOT):
        p = SOURCE / p.relative_to(ROOT)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def selection(inventory):
    chosen = []
    for split, target, per_class in [('train', 72, 4), ('val', 36, 2)]:
        pool = [r for r in inventory if r['split'] == split]
        picked = {}
        classes = sorted({c for r in pool for c in r['classes']})
        for c in classes:
            candidates = sorted([r for r in pool if c in r['classes']], key=lambda r: (r['node_count'], r['edge_count'], r['sample_id']))
            # Each class has both extreme low/high topology complexity, with central
            # and hash-selected representatives for train. Duplicates are filled.
            indices = [0, len(candidates) - 1]
            if per_class == 4:
                indices += [len(candidates) // 2, min(range(len(candidates)), key=lambda i: digest(candidates[i]['sample_id']))]
            for i in indices:
                r = candidates[i]
                picked.setdefault(r['sample_id'], {**r, 'selection_reason': []})['selection_reason'].append(c)
        remaining = sorted([r for r in pool if r['sample_id'] not in picked], key=lambda r: digest('G07:108:' + r['sample_id']))
        for r in remaining:
            if len(picked) >= target:
                break
            picked[r['sample_id']] = {**r, 'selection_reason': ['deterministic_hash_fill']}
        if len(picked) != target:
            raise ValueError(f'sampling target mismatch {split}: {len(picked)}')
        chosen.extend(sorted(picked.values(), key=lambda r: r['sample_id']))
    return chosen


def clean_graph(graph):
    allowed = ('schema_version', 'nodes', 'edges', 'edge_attributes', 'part_metadata_v2', 'graph_v2_protocol')
    result = {key: copy.deepcopy(graph[key]) for key in allowed}
    result['nodes'] = [{key: node[key] for key in ('idx', 'primitive_surface', 'primitive_surface_id', 'attributes')} for node in result['nodes']]
    return result


def initialize():
    global torch, np, mapper, extractor, service, exporter, metadata, model, same, recover_stages, candidate, decoder
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(ROOT / 'tools/aei_paper_goals_v1/g03'))
    import torch
    import numpy as np
    from p1_e6r2_r1_2.face_mapper import map_current_graph_nodes_to_final_faces as mapper
    from core.graph_v2_2_1 import extract_graph_v2_2_1 as extractor
    from scripts import run_inference as service
    from utils.training_export_v2_2_1 import export_graph_v2_2_1_training_record as exporter
    from stages import recover_stages
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    metadata = read(DATA / 'metadata.json')
    model = service.build_model().eval()
    checkpoint = SOURCE / 'model_packages/p1_e12_reviewed_legacy_v1/model.pt'
    if sha(checkpoint) != '491655bfec5591c5a6a1b7f6300294f4818c29bf60b4c1b8dd5ff6078a057054':
        raise ValueError('service model checksum')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
    same = read(SOURCE / 'model_packages/p1_e12_reviewed_legacy_v1/same_instance_threshold.json')
    candidate, decoder = read(service.CANDIDATE_PATH), read(service.GLOBAL_PATH)


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_comparison(a, b):
    rows = {}
    for name, left, right in zip(('node40', 'edge_index', 'primitive', 'edge20'), a, b):
        shape_equal = left.shape == right.shape
        rows[name] = {'shape': list(left.shape), 'other_shape': list(right.shape),
            'byte_equal': bool(shape_equal and torch.equal(left, right)),
            'within_2e_5': bool(shape_equal and torch.allclose(left, right, rtol=2e-5, atol=2e-5)),
            'max_abs_error': float((left.double() - right.double()).abs().max()) if shape_equal and left.numel() else None,
            'left_sha256': tensor_sha(left), 'right_sha256': tensor_sha(right)}
    return rows


def align_bundle(source, rebuilt, face_map):
    node_map = [int(face_map[i]) for i in range(len(source[0]))]
    inverse = {v: i for i, v in enumerate(node_map)}
    groups_a, groups_b = collections.defaultdict(list), collections.defaultdict(list)
    for i, pair in enumerate(source[1].t().tolist()):
        groups_a[tuple(pair)].append(i)
    for i, pair in enumerate(rebuilt[1].t().tolist()):
        groups_b[tuple(inverse[v] for v in pair)].append(i)
    if set(groups_a) != set(groups_b):
        raise ValueError('directed endpoint sets differ')
    from scipy.optimize import linear_sum_assignment
    edge_map, ambiguities = {}, []
    for pair, indices in groups_a.items():
        others = groups_b[pair]
        if len(indices) != len(others):
            raise ValueError('parallel B-Rep edge multiplicity differs')
        cost = torch.cdist(source[3][indices].double(), rebuilt[3][others].double()).numpy()
        aa, bb = linear_sum_assignment(cost)
        optimum = float(cost[aa, bb].sum())
        alternate_equal = False
        if len(indices) > 1:
            for x, y in zip(aa, bb):
                other = cost.copy()
                other[x, y] = 1e99
                x2, y2 = linear_sum_assignment(other)
                if abs(float(other[x2, y2].sum()) - optimum) <= 1e-12:
                    alternate_equal = True
                    break
        if alternate_equal:
            ambiguities.append({'endpoints': list(pair), 'multiplicity': len(indices), 'minimum_attribute_cost': optimum,
                'interpretation': 'nonunique attribute assignment; tensors compared as multiset, no unique geometric edge identity claimed'})
        edge_map.update({indices[x]: others[y] for x, y in zip(aa, bb)})
    perm = [edge_map[i] for i in range(source[1].shape[1])]
    adjacency = torch.tensor([[inverse[v] for v in rebuilt[1][:, i].tolist()] for i in perm], dtype=torch.long).t()
    return (rebuilt[0][node_map], adjacency, rebuilt[2][node_map], rebuilt[3][perm]), perm, ambiguities


def raw_bundle(graph):
    exported = exporter(graph)
    return (torch.tensor(exported['node_features'], dtype=torch.float64).reshape(-1, 40),
            torch.tensor(exported['edges'], dtype=torch.long).reshape(-1, 2).t(),
            torch.tensor([n['primitive_surface_id'] for n in graph['nodes']], dtype=torch.long),
            torch.tensor(exported['edge_features'], dtype=torch.float64).reshape(-1, 20))


def attribute_differences(a, b, pointer=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield {'path': pointer + '/' + key, 'left': a.get(key), 'right': b.get(key)}
            else:
                yield from attribute_differences(a[key], b[key], pointer + '/' + key)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (left, right) in enumerate(zip(a, b)):
            yield from attribute_differences(left, right, pointer + '/' + str(i))
    elif a != b:
        yield {'path': pointer, 'left': a, 'right': b}


def brep_inventory(shape):
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRep import BRep_Tool
    explorer = TopologyExplorer(shape)
    faces = list(explorer.faces())
    edges = list(explorer.edges())
    seams = 0
    edge_face_counts = collections.Counter()
    for edge in edges:
        adjacent = list(explorer.faces_from_edge(edge))
        edge_face_counts[len(adjacent)] += 1
        seams += sum(bool(BRep_Tool.IsClosed(edge, face)) for face in adjacent)
    return {'face_count': len(faces), 'brep_unique_edge_count': len(edges),
            'edge_incident_face_multiplicities': dict(edge_face_counts), 'seam_edge_face_pairs': seams,
            'face_orientation_counts': dict(collections.Counter(str(face.Orientation()) for face in faces))}


def audit_one(job):
    start = time.perf_counter()
    directory = OUT / job['audit_id']
    try:
        from utils.step_solid_validation import read_step_shape
        archived = clean_graph(read(job['graph_path']))
        t0 = time.perf_counter()
        shape = read_step_shape(Path(job['step_path']))
        rebuilt = clean_graph(extractor(shape, build_patches=False))
        extraction_seconds = time.perf_counter() - t0
        guarded = clean_graph(service.guard_degenerate_projection(rebuilt))
        guard_changes = service.guard_degenerate_projection(rebuilt)['projection_numeric_guard']
        mapping = mapper(shape, archived)
        if not mapping['passed']:
            raise ValueError('face mapping failed: ' + json.dumps(mapping['audit']))
        face_map = mapping['face_indices_by_node']
        aa, bb, cc = [service.graph_tensors(g, metadata) for g in (archived, rebuilt, guarded)]
        aligned, edge_map, edge_ambiguities = align_bundle(aa, bb, face_map)
        aligned_guard, _, _ = align_bundle(aa, cc, face_map)
        raw_a, raw_b, raw_c = [raw_bundle(g) for g in (archived, rebuilt, guarded)]
        raw_aligned, _, _ = align_bundle(raw_a, raw_b, face_map)
        raw_guard, _, _ = align_bundle(raw_a, raw_c, face_map)
        comparisons = {'archive_vs_step_normalized': tensor_comparison(aa, aligned),
            'step_vs_guard_normalized': tensor_comparison(bb, cc),
            'archive_vs_guard_normalized': tensor_comparison(aa, aligned_guard),
            'archive_vs_step_raw40_20': tensor_comparison(raw_a, raw_aligned),
            'archive_vs_guard_raw40_20': tensor_comparison(raw_a, raw_guard)}
        disturbed = copy.deepcopy(read(job['graph_path']))
        for i, node in enumerate(disturbed['nodes']):
            node.update(macro_feature='sentinel_label', macro_feature_id=(i * 11) % 18,
                        semantic_label=(i * 7) % 64, instance_id=100000 + i, source_node_index=100000 - i)
        disturbed.update(feature_sets=[{'sentinel': True}], provenance={'source_id': 'changed'},
                         sample_id='renamed_no_identity', template_id='changed')
        clean_changed = clean_graph(disturbed)
        changed_tensors = service.graph_tensors(clean_changed, metadata)
        comparisons['label_source_id_perturbation'] = tensor_comparison(aa, changed_tensors)
        graphs = {'archive': archived, 'step': rebuilt, 'guard': guarded}
        probability_outputs, stage_outputs, timings = {}, {}, {}
        for label, graph in graphs.items():
            t0 = time.perf_counter()
            pred = service.predict(graph, model, metadata, torch.device('cpu'))
            forward_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            stages = recover_stages(graph, pred, same, candidate, decoder, 'geometry_only')
            service_objects = service.decode(graph, pred, same)
            timings[label] = {'forward_seconds': forward_seconds, 'decode_seconds': time.perf_counter() - t0}
            probability_outputs[label] = pred
            stage_outputs[label] = {'objects': stages['objects'], 'service_objects': service_objects}
            write(directory / (label + '_input_graph.json'), graph)
        # Exact equality of normalized tensors plus deterministic eval establishes
        # prediction independence; additionally run the perturbed input explicitly.
        changed_pred = service.predict(clean_changed, model, metadata, torch.device('cpu'))
        def probability_array(pred):
            return np.asarray([[v for v in face['macro_probabilities'].values()] for face in pred['faces']], dtype=np.float64)
        pa, pb, pc = [probability_array(probability_outputs[name]) for name in ('archive', 'step', 'guard')]
        idx = [int(face_map[i]) for i in range(len(pa))]
        probability_comparison = {'archive_step_max_abs': float(np.max(np.abs(pa - pb[idx]))),
            'archive_guard_max_abs': float(np.max(np.abs(pa - pc[idx]))),
            'step_guard_max_abs': float(np.max(np.abs(pb - pc))),
            'label_source_perturbation_prediction_exact': changed_pred == probability_outputs['archive']}
        diffs = []
        for i, j in enumerate(idx):
            diffs.extend(attribute_differences(archived['nodes'][i]['attributes'], rebuilt['nodes'][j]['attributes'], '/nodes/' + str(i) + '/attributes'))
        attr_categories = collections.Counter('/'.join(d['path'].split('/')[:5]) for d in diffs)
        result = {'sample_id': job['sample_id'], 'audit_id': job['audit_id'], 'split': job['split'],
            'archive_role': job['archive_role'], 'source_graph': asset(job['graph_path']), 'source_step': asset(job['step_path']),
            'passed_execution': True, 'face_mapping': mapping['audit'],
            'node_to_step_face': face_map, 'edge_permutation': edge_map,
            'edge_unique_assignment_passed': not edge_ambiguities, 'edge_mapping_ambiguities': edge_ambiguities,
            'brep': brep_inventory(shape), 'graph_edge_count': len(rebuilt['edges']),
            'parallel_directed_endpoint_groups': sum(v > 1 for v in collections.Counter(tuple(e) for e in rebuilt['edges']).values()),
            'tensor_comparisons': comparisons, 'probability_comparison': probability_comparison,
            'projection_guard_changes': guard_changes, 'raw_attribute_difference_count': len(diffs),
            'raw_attribute_difference_categories': dict(attr_categories), 'raw_attribute_differences': diffs,
            'timings': {'step_read_and_extract_seconds': extraction_seconds, 'paths': timings,
                        'total_seconds': time.perf_counter() - start},
            'full_seed': 20260719, 'role_completion_evaluated': False,
            'truth_read_by_worker': False}
        result['archive_step_tensor_bytes_equal'] = all(r['byte_equal'] for r in comparisons['archive_vs_step_normalized'].values())
        result['archive_step_tensor_within_tolerance'] = all(r['within_2e_5'] for r in comparisons['archive_vs_step_normalized'].values())
        write(directory / 'probabilities.json', probability_outputs)
        write(directory / 'stages.json', stage_outputs)
    except Exception as exc:
        result = {'sample_id': job['sample_id'], 'audit_id': job['audit_id'], 'split': job['split'],
            'passed_execution': False, 'error': str(exc), 'traceback': traceback.format_exc(),
            'elapsed_seconds': time.perf_counter() - start}
    write(directory / 'result.json', result)
    return result


def prepare(mode):
    inventory = read(ROOT / 'reports/aei_paper_goals_v1/g01/source_sample_inventory.json')
    chosen = selection(inventory) if mode == 'initial' else [r for r in inventory if r['split'] == 'val']
    manifest = {r['sample_id']: r for r in (json.loads(line) for line in (GEN / 'manifests/sample_manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip())}
    jobs = []
    for r in chosen:
        m = manifest[r['sample_id']]
        jobs.append({'audit_id': r['sample_id'], 'sample_id': r['sample_id'], 'split': r['split'],
            'step_path': frozen(m['final_step_path']).as_posix(),
            'graph_path': (GEN / 'graphs' / (r['sample_id'] + '.json')).as_posix(),
            'archive_role': 'original_unprotected_training_graph'})
    if mode == 'initial':
        for r in read(ROOT / 'reports/aei_paper_goals_v1/g01/engineering_versions.json')['parts']:
            stem = Path(r['step']['path']).stem
            jobs.append({'audit_id': 'engineering_' + stem.replace(' ', '_'), 'sample_id': stem,
                'split': 'engineering', 'step_path': r['step']['path'],
                'graph_path': (ROOT / 'reports/aei_case_suite_v2/recognition' / (stem + '_graph.json')).as_posix(),
                'archive_role': 'historical_service_graph_already_projection_guarded'})
    protocol = {'mode': mode, 'created_unix': time.time(), 'selection_rule': 'per-class low/high face-count extremes; train adds median and SHA256 selector; unique IDs deterministically hash-filled to 72/36',
        'class_support': {split: dict(collections.Counter(c for r in chosen if r['split'] == split for c in r['classes'])) for split in ('train', 'val')},
        'selected_strata': chosen, 'jobs': jobs, 'workers': 2, 'cpu_threads_per_worker': 2,
        'execution_seed': 20260719, 'threshold_from': 'unchanged service package',
        'mapping_policy': 'Unique geometric+adjacency face optimum required. Parallel edge attribute assignments exhaustively reject equivalent optima for identity claims; tensor multiset comparisons retained.',
        'atol': 2e-5, 'rtol': 2e-5, 'truth_not_passed_to_worker': True}
    path = REPORT / 'input_audit' / (mode + '_protocol.json')
    if path.exists():
        old = read(path)
        if old['jobs'] != jobs:
            raise ValueError('existing protocol job list changed')
    else:
        write(path, protocol)
    return jobs


def summarize(jobs, mode):
    results = [read(OUT / r['audit_id'] / 'result.json') for r in jobs if (OUT / r['audit_id'] / 'result.json').exists()]
    succeeded = [r for r in results if r['passed_execution']]
    return {'schema': 'g07_input_path_audit_v1', 'mode': mode, 'planned_count': len(jobs), 'completed_count': len(results),
        'success_count': len(succeeded), 'failure_count': len(results) - len(succeeded),
        'split_counts': dict(collections.Counter(r['split'] for r in results)),
        'archive_step_tensor_byte_difference_ids': [r['sample_id'] for r in succeeded if not r['archive_step_tensor_bytes_equal']],
        'archive_step_tensor_tolerance_failure_ids': [r['sample_id'] for r in succeeded if not r['archive_step_tensor_within_tolerance']],
        'guard_changed_ids': [r['sample_id'] for r in succeeded if r['projection_guard_changes']],
        'ambiguous_edge_identity_ids': [r['sample_id'] for r in succeeded if not r['edge_unique_assignment_passed']],
        'raw_attribute_difference_ids': [r['sample_id'] for r in succeeded if r['raw_attribute_difference_count']],
        'result_assets': [asset(OUT / r['audit_id'] / 'result.json') for r in results],
        'limitations': ['No whole-8370 STEP re-extraction claim.', 'Only historic validation-selected full seed 20260719 probabilities in Stage A.',
            'Engineering archive is already numerically guarded, unlike D0 archive.', 'No role or geometric-parameter scoring in this input audit.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('initial', 'validation_extension'), default='initial')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    jobs = prepare(args.mode)
    pending = [r for r in jobs if not (OUT / r['audit_id'] / 'result.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        future_to_job = {pool.submit(audit_one, job): job for job in pending}
        for future in as_completed(future_to_job):
            result = future.result()
            print(result['audit_id'], result['passed_execution'], result.get('error', '')[:160], flush=True)
            write(REPORT / 'input_audit' / (args.mode + '_progress.json'), summarize(jobs, args.mode))
    summary = summarize(jobs, args.mode)
    write(REPORT / 'INPUT_PATH_AUDIT.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('result_assets', 'limitations')}), flush=True)


if __name__ == '__main__':
    main()
