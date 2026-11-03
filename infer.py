"""Fresh STEP inference with the frozen deployed E3 weights; no truth is read."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import prepare_runtime

REPO = Path(__file__).resolve().parent


def child(runtime, step, output):
    sys.path.insert(0, str(runtime / 'tools/aei_paper_goals_v1/g10'))
    from m8_e3_runtime_adapter_v4 import relocated_e3_runtime
    plan = json.loads((runtime / 'PROTECTED_E3_CLOSURE.json').read_text(encoding='utf-8'))
    mapping = json.loads((runtime / 'E3_SHA_MAPPING.json').read_text(encoding='utf-8'))
    with relocated_e3_runtime(runtime, plan, mapping):
        from infer_step_e3 import E3StepService
        service = E3StepService(device='cpu')
        result = service.infer(step)
    assert result['truth_used'] is False
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'objects': len(result['objects']), 'truth_used': False, 'output': str(output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-child', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    step, output = a.step.resolve(strict=True), a.output.resolve()
    if output.exists():
        p.error('Choose a new output file; existing results are never overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.runtime_child:
        return child(a.runtime_child.resolve(), step, output)
    runtime = prepare_runtime()
    sys.path.insert(0, str(REPO / 'comparison/topocae/engine/deployment'))
    from g10_common import digest
    from m8_portable_rebase_v4 import materialize
    plan = json.loads((runtime / 'PROTECTED_E3_CLOSURE.json').read_text(encoding='utf-8'))
    def write_json(path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(runtime.resolve()) or path.exists():
            raise ValueError('Runtime materialization must create a new file inside its temporary directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    mapping = materialize(runtime, plan, expected_plan_digest=digest(plan), write=write_json)
    write_json(runtime / 'E3_SHA_MAPPING.json', mapping)
    env = {**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    # Temporary model descriptors are created under the new runtime only.
    r = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), str(step),
                        '--output', str(output), '--runtime-child', str(runtime)],
                       cwd=runtime, env=env)
    if r.returncode:
        print('Runtime retained for diagnosis:', runtime, file=sys.stderr)
        raise SystemExit(r.returncode)
    print('Fresh inference completed. Relocated runtime:', runtime)


if __name__ == '__main__':
    main()
