"""Recompute the paper engineering recognition and role/parameter observations."""
from collections import Counter
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def objects(rows, faces='faces'):
    return Counter((r['class'], tuple(sorted(r[faces]))) for r in rows)


def close(actual, expected, tolerance):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and close(actual[k], v, tolerance) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(close(a, b, tolerance) for a, b in zip(actual, expected))
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and abs(actual - expected) <= tolerance
    return actual == expected


def engineering_tables():
    truth = read('data/engineering/recognition_truth.json')['cases']
    refs = read('data/engineering/parameter_role_references.json')
    parts = {r['part']: r for r in refs['parts']}
    observations = {r['part']: r['operations'] for r in refs['measured_role_selections']}
    tally = Counter()
    per_part = []
    for case in read('data/engineering/cases.json'):
        cid, name = case['case_id'], case['part']
        prediction = read(f'data/engineering/{cid}/prediction.json')
        actual = objects(prediction['feature_objects'], 'original_face_numbers')
        expected = objects(truth[cid]['objects'])
        tp, fp, fn = sum((actual & expected).values()), sum((actual - expected).values()), sum((expected - actual).values())
        tally.update(tp=tp, fp=fp, fn=fn, whole=int(actual == expected))
        per_part.append({'case_id': cid, 'tp': tp, 'fp': fp, 'fn': fn})
        by_object = {(o['class'], tuple(sorted(o['original_face_numbers']))): o for o in prediction['feature_objects']}
        for item in parts[name]['parameter_reference_instances']:
            if not item['parameters']:
                continue
            emitted = by_object[item['class'], tuple(sorted(item['faces']))]['parameters']
            passed = True
            for field, reference in item['parameters'].items():
                tolerance = 1e-4 if field == 'thickness_mm' else 1e-5
                ok = close(emitted.get(field), reference, tolerance)
                tally[field + '_total'] += 1
                tally[field + '_pass'] += int(ok)
                passed = passed and ok
            tally['parameter_total'] += 1
            tally['parameter_pass'] += int(passed)
        for ref in (r for r in refs['operations'] if r['part'] == name):
            measured = next(r for r in observations[name] if r['operation_index'] == ref['operation_index'])
            selected = []
            for group in measured['predicted_groups']:
                obj = prediction['feature_objects'][group['object_index']]
                role = obj['roles'][ref['role']]
                assert role['face_numbers'] == group['face_numbers']
                selected.append({'class': obj['class'], 'faces': role['face_numbers']})
            wanted = [{'class': r['class'], 'faces': r['face_numbers']} for r in ref['expected_groups']]
            group = 'independent_targets' if ref['reference_kind'] == 'independent_instance_face_sets' else 'reviewed_top'
            tally[group + '_total'] += 1
            tally[group + '_pass'] += int(objects(selected) == objects(wanted))
            if 'spacing_parameter' in measured:
                tally['spacing_total'] += 1
                tally['spacing_pass'] += int(close(prediction['spacing_if_three_counterbores'], measured['spacing_parameter']['expected'], 1e-4))
    assert [tally[k] for k in ('tp', 'fp', 'fn', 'whole')] == [182, 0, 0, 13]
    for key, n in [('parameter',126),('radii_mm',106),('thickness_mm',20),('full_cylinder_depths',66),('independent_targets',18),('reviewed_top',3),('spacing',1)]:
        assert tally[key + '_pass'] == tally[key + '_total'] == n, (key, tally)
    return {'counts': dict(tally), 'parts': per_part, 'parameter_reference_is_geometric_not_independent_metrology': True, 'engineering_is_development_evidence': True}

