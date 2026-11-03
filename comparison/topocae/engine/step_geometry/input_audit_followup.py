"""CPU-only independent entry check, cached Stage-A summary, and service scoring."""
from __future__ import annotations

import collections
import copy
import json
import re
import sys
import time

import input_audit as audit


def canonical(objects, one_based=False):
    return sorted((obj['macro_feature'], tuple(sorted(int(i) + int(one_based) for i in obj['face_indices']))) for obj in objects)


def parse_manual_truth(text):
    """Independent narrow parser; require declared count and complete rows."""
    cases = {}
    current = None
    for line in text.splitlines():
        if line.startswith('输入文件'):
            current = re.split(r'[\\/]', line)[-1].strip()
            if current in cases:
                raise ValueError('duplicate part')
            cases[current] = {'objects': []}
        elif line.startswith(('特征数量', '识别出的特征实例数量')):
            cases[current]['declared_count'] = int(re.split('[：:]', line)[-1])
        elif re.match(r'^\s*\[\d+\]', line):
            m = re.match(r'^\s*\[(\d+)\]\s*(\w+)\s*\|\s*面编号\s*=\s*(\[[\d,\s]+\])\s*\|\s*总面积\s*=\s*([\d.]+)\s*$', line)
            if not m or current is None:
                raise ValueError('unparsed target row: ' + line)
            faces = json.loads(m[3])
            if not faces or len(set(faces)) != len(faces) or min(faces) < 1:
                raise ValueError('invalid truth face group')
            cases[current]['objects'].append({'class': m[2], 'faces': sorted(faces), 'area': float(m[4])})
    for part, row in cases.items():
        if row['declared_count'] != len(row['objects']):
            raise ValueError('truth count mismatch: ' + part)
    return cases


def matching(pred, truth):
    # Explicit one-to-one matching, independent from the old Counter evaluator.
    available = set(range(len(truth)))
    matched, false_positives = [], []
    for cls, faces in pred:
        matches = [i for i in available if truth[i]['class'] == cls and tuple(truth[i]['faces']) == tuple(faces)]
        if matches:
            index = min(matches)
            available.remove(index)
            matched.append(index)
        else:
            false_positives.append({'class': cls, 'faces': list(faces)})
    false_negatives = [truth[i] for i in sorted(available)]
    per_class = {}
    for cls in sorted({x[0] for x in pred} | {x['class'] for x in truth}):
        tp = sum(truth[i]['class'] == cls for i in matched)
        fp = sum(x['class'] == cls for x in false_positives)
        fn = sum(x['class'] == cls for x in false_negatives)
        per_class[cls] = {'tp': tp, 'fp': fp, 'fn': fn}
    return {'tp': len(matched), 'fp': len(false_positives), 'fn': len(false_negatives),
            'per_class': per_class, 'false_positives': false_positives, 'false_negatives': false_negatives}


def mutate_nested(value):
    if isinstance(value, dict):
        result = {key: mutate_nested(item) for key, item in value.items()}
        result.update(label='not_a_geometry_feature', macro_feature='sentinel', macro_feature_id=99,
            semantic_label=777, instance_id=9999, source_id='random_source', template_id='hidden_template',
            source_node_index=765, provenance={'false_ground_truth': True})
        return result
    if isinstance(value, list):
        return [mutate_nested(item) for item in value]
    return copy.deepcopy(value)


def verify_entry(jobs):
    import step_entry
    previous_path = audit.REPORT / 'input_audit/UNIFIED_ENTRY_VALIDATION.json'
    if previous_path.exists():
        previous = audit.read(previous_path)
        audit.write(previous_path.with_name('UNIFIED_ENTRY_VALIDATION_' + previous['entry']['sha256'][:16] + '.json'), previous)
    rows = []
    start = time.perf_counter()
    for job in jobs:
        directory = audit.OUT / job['audit_id']
        result = audit.read(directory / 'result.json')
        probabilities = audit.read(directory / 'probabilities.json')
        stages = audit.read(directory / 'stages.json')
        for label in ('archive', 'step', 'guard'):
            graph = audit.read(directory / (label + '_input_graph.json'))
            old_tensors = audit.service.graph_tensors(graph, audit.metadata)
            new_tensors = step_entry.graph_tensors(graph, audit.metadata)
            altered = mutate_nested(graph)
            altered_tensors = step_entry.graph_tensors(altered, audit.metadata)
            clean = step_entry.geometry_only(graph)
            clean_altered = step_entry.geometry_only(altered)
            pred = audit.service.predict(clean, audit.model, audit.metadata, audit.torch.device('cpu'))
            changed_pred = audit.service.predict(clean_altered, audit.model, audit.metadata, audit.torch.device('cpu'))
            objects = audit.service.decode(clean, pred, audit.same)
            row = {'audit_id': job['audit_id'], 'path': label,
                'old_new_tensor_byte_equal': all(audit.torch.equal(a, b) for a, b in zip(old_tensors, new_tensors)),
                'recursive_injected_fields_tensor_byte_equal': all(audit.torch.equal(a, b) for a, b in zip(new_tensors, altered_tensors)),
                'recursive_whitelist_graph_equal': clean == clean_altered,
                'new_entry_prediction_equal': pred == probabilities[label],
                'recursive_injected_fields_prediction_equal': changed_pred == pred,
                'new_entry_service_objects_equal': canonical(objects) == canonical(stages[label]['service_objects'])}
            row['passed'] = all(v for k, v in row.items() if k not in ('audit_id', 'path'))
            rows.append(row)
        if len(rows) % 60 == 0:
            print('entry_checks', len(rows), flush=True)
    bad = {**audit.read(audit.OUT / jobs[0]['audit_id'] / 'step_input_graph.json'), 'schema_version': 'legacy_aag'}
    try:
        step_entry.graph_tensors(bad, audit.metadata)
    except ValueError as exc:
        old_schema_rejected = {'passed': True, 'message': str(exc)}
    else:
        old_schema_rejected = {'passed': False}
    summary = {'entry': audit.asset(step_entry.__file__), 'checked_jobs': len(jobs), 'checked_paths': len(rows),
        'actual_network_forwards': len(rows) * 2, 'rows': rows, 'old_schema_rejection': old_schema_rejected,
        'elapsed_seconds': time.perf_counter() - start, 'all_passed': all(r['passed'] for r in rows) and old_schema_rejected['passed']}
    audit.write(audit.REPORT / 'input_audit/UNIFIED_ENTRY_VALIDATION.json', summary)
    return summary


def score_engineering(jobs):
    # Score only after the audited13 predictions already exist on disk.
    truth_path = audit.ROOT / 'test_part_0901/真值.txt'
    truths = parse_manual_truth(truth_path.read_text(encoding='utf-8-sig'))
    geometry_audit = {r['part']: r for r in audit.read(audit.ROOT / 'reports/aei_case_suite_v2/controlled_v1/recognition_contribution/truth_geometry_audit.json')}
    rows = []
    for job in jobs:
        if job['split'] != 'engineering':
            continue
        part = audit.Path(job['step_path']).name
        alias = {'Stiffened Panel.stp': 'Stiffened Panel_stp.stp'}.get(part, part)
        truth = truths[alias]['objects']
        audited = geometry_audit[part]
        if audited['source_sha256'] != audit.sha(job['step_path']):
            raise ValueError('truth geometry source checksum mismatch')
        if [(x['class'], x['faces']) for x in truth] != [(x['class'], sorted(x['faces'])) for x in audited['objects']]:
            raise ValueError('independent parser disagrees with prior truth audit')
        directory = audit.OUT / job['audit_id']
        stages = audit.read(directory / 'stages.json')
        levels = {}
        for path in ('archive', 'step', 'guard'):
            levels[path] = {stage: matching(canonical(objects, True), truth) for stage, objects in stages[path]['objects'].items()}
            levels[path]['service_R'] = matching(canonical(stages[path]['service_objects'], True), truth)
        rows.append({'part': part, 'truth_name': alias, 'truth_instances': len(truth),
            'source_step': audit.asset(job['step_path']), 'prediction': audit.asset(directory / 'stages.json'),
            'levels': levels, 'all_correct_guarded_R': levels['guard']['service_R']['fp'] == 0 and levels['guard']['service_R']['fn'] == 0})
    aggregate = {}
    for path in ('archive', 'step', 'guard'):
        aggregate[path] = {}
        for level in rows[0]['levels'][path]:
            counts = {k: sum(r['levels'][path][level][k] for r in rows) for k in ('tp', 'fp', 'fn')}
            counts['micro_f1'] = 2 * counts['tp'] / (2 * counts['tp'] + counts['fp'] + counts['fn'])
            classes = sorted({c for r in rows for c in r['levels'][path][level]['per_class']})
            per_class = {}
            for cls in classes:
                c = {k: sum(r['levels'][path][level]['per_class'].get(cls, {}).get(k, 0) for r in rows) for k in ('tp', 'fp', 'fn')}
                c['f1'] = 2 * c['tp'] / max(2 * c['tp'] + c['fp'] + c['fn'], 1)
                per_class[cls] = c
            counts.update(per_class=per_class, macro_f1=sum(x['f1'] for x in per_class.values()) / len(per_class), class_count=len(classes))
            aggregate[path][level] = counts
    result = {'schema': 'g07_engineering_baseline_R_v1', 'selected_seed': 20260719,
        'truth': audit.asset(truth_path), 'truth_source': 'independent manual face annotation confirmed2026-09-09; narrow parser independently reproduced all182 records',
        'parts': 13, 'instances': sum(r['truth_instances'] for r in rows),
        'all_correct_parts': sum(r['all_correct_guarded_R'] for r in rows), 'aggregate': aggregate, 'per_part': rows,
        'role_completion': 'not evaluated', 'parameters': 'not evaluated',
        'scope': 'Historic fixed service seed, current13 engineering development cases; fresh CPU rerun of STEP input, no new blind test claim'}
    audit.write(audit.REPORT / 'input_audit/ENGINEERING_BASELINE_R.json', result)
    return result


def main():
    audit.initialize()
    jobs = audit.read(audit.REPORT / 'input_audit/initial_protocol.json')['jobs']
    entry_path = audit.REPORT / 'input_audit/UNIFIED_ENTRY_VALIDATION.json'
    import step_entry
    entry = audit.read(entry_path) if entry_path.exists() else None
    if not entry or entry['entry']['sha256'] != audit.sha(step_entry.__file__):
        entry = verify_entry(jobs)
    engineering = score_engineering(jobs)
    print(json.dumps({'entry_all_passed': entry['all_passed'], 'engineering_all_correct': engineering['all_correct_parts'],
        'guard_service_counts': {k: engineering['aggregate']['guard']['service_R'][k] for k in ('tp', 'fp', 'fn', 'micro_f1', 'macro_f1')}}), flush=True)


if __name__ == '__main__':
    main()
