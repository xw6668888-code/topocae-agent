"""Replay the six reference-plan CAD edits from packaged STEP and E3 objects."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import math
import sys
from runtime_loader import prepare_runtime

ROOT = Path(__file__).resolve().parent
RUNTIME = prepare_runtime()
sys.path.insert(0, str(RUNTIME / 'tools/aei_paper_goals_v1/g10'))
import m6_runner as runner
runner.CASE_SUITE = RUNTIME / 'tools/aei_case_suite'
runner.PILOT = RUNTIME / 'tools/aei_two_case_pilot'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def replay(row, output):
    base = ROOT / 'data/engineering' / row['case_id']
    source, predpath = base / 'input.step', base / 'prediction.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['geometry_version']
    assert hashlib.sha256(predpath.read_bytes()).hexdigest() == row['prediction_sha256']
    prediction, plan = read(predpath), read(base / 'plan.json')
    assert prediction['truth_used'] is False
    with runner._case_suite_imports():
        import OCC
        assert OCC.VERSION == '7.9.0'
        from build_cases import read_step, properties
        from preflight import measured
        from OCC.Extend.TopologyUtils import TopologyExplorer
        shape = read_step(source)
        faces = list(TopologyExplorer(shape).faces())
        assert len(faces) == len(prediction['face_macro_classes'])
        face_rows = [measured(face, i + 1) for i, face in enumerate(faces)]
        objects = deepcopy(prediction['feature_objects'])
        for obj in objects:
            obj['geometry_summary'] = runner._geometry_summary(obj, face_rows)
        compatible = deepcopy(prediction)
        compatible.update(input=prediction['input_step'], feature_objects=objects,
                          model={'name': prediction['system'], 'checkpoint_sha256': prediction['source_identity']['R']['checkpoint']['sha256']})
        data = {'faces': face_rows, 'objects': objects, 'source_sha256': row['geometry_version'],
                'prediction_sha256': row['prediction_sha256'], 'truth_used': False}
        adapter = {'case_tuple': (source, shape, faces, compatible, predpath), 'data': data}
        with runner._patched_cad_adapter(adapter) as executor:
            destination = executor.execute(row['part'], plan, output, execution_mode='packaged_reference_plan')
        final = read_step(destination)
        volume = properties(final, True)[0]
        solids = len(list(TopologyExplorer(final).solids()))
        assert solids == row['roundtrip_solid_count'] == 1
        assert math.isclose(volume, row['roundtrip_volume_mm3'], rel_tol=1e-8, abs_tol=1e-5), (volume, row['roundtrip_volume_mm3'])
        return {'case_id': row['case_id'], 'status': 'PASS', 'volume_mm3': volume, 'solid_count': solids}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Use a new output directory.')
    a.output.mkdir(parents=True)
    results = []
    for row in read(ROOT / 'data/engineering/cases.json'):
        if row['domain'] != 'cad': continue
        results.append(replay(row, a.output / row['case_id']))
        print(row['case_id'] + ': PASS', flush=True)
    (a.output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
