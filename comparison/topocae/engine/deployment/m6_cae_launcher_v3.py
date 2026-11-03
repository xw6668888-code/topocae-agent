"""Run frozen M6 CAE programs with direct argv and a Python-2-safe ODB audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from g10_common import (
    ARTIFACT, checked, descriptor, now, read_json, resource_lock, write_json,
)
from m6_cae_launcher_v2 import direct_abaqus_argv, _validated_prior_failure
from m6_runner import ABAQUS, _abaqus_release, _run_process, _verify_job


ODB_ADAPTER = Path(__file__).resolve().with_name('m6_odb_reaudit_v2.py')


def _safe_smoke_id(value: str) -> str:
    if not value or any(character not in
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-'
                        for character in value):
        raise ValueError('unsafe smoke id')
    return value


def abaqus_python_smoke(smoke_id: str, timeout: int = 120) -> dict[str, Any]:
    smoke_id = _safe_smoke_id(smoke_id)
    root = ARTIFACT / 'm6_engineering' / 'compatibility_smoke' / smoke_id
    root.mkdir(parents=True, exist_ok=False)
    output = root / 'abaqus_python_string_adapter.json'
    log = root / 'launch.log'
    adapter = descriptor(ODB_ADAPTER)
    command = direct_abaqus_argv('python', ODB_ADAPTER) + [
        '--self-test', str(output.resolve()),
    ]
    started = time.perf_counter()
    release = None
    try:
        with resource_lock('abaqus', f'g10-m6-odb-adapter-smoke-{smoke_id}'):
            release = _abaqus_release()
            code, timed_out, elapsed = _run_process(
                command, root, dict(os.environ), log, timeout,
            )
        result = read_json(output) if output.exists() else {}
        passed = not timed_out and result.get('status') == 'PASS' \
            and result.get('python_major') == 2 \
            and result.get('after_native_str') is True \
            and result.get('after_unicode') is False \
            and descriptor(ODB_ADAPTER) == adapter
        receipt = {
            'schema': 'g10_m6_abaqus_python_compatibility_receipt_v1',
            'completed_utc': now(),
            'smoke_id': smoke_id,
            'status': 'PASS' if passed else 'FAIL',
            'launcher_mode': 'direct_batch_argv_v3',
            'command': command,
            'return_code': code,
            'timed_out': timed_out,
            'elapsed_seconds': elapsed,
            'abaqus_release': release,
            'odb_adapter': adapter,
            'smoke_output': descriptor(output) if output.exists() else None,
            'log': descriptor(log),
            'truth_used': False,
        }
    except BaseException as error:
        receipt = {
            'schema': 'g10_m6_abaqus_python_compatibility_receipt_v1',
            'completed_utc': now(),
            'smoke_id': smoke_id,
            'status': 'FAIL',
            'launcher_mode': 'direct_batch_argv_v3',
            'command': command,
            'abaqus_release': release,
            'odb_adapter': adapter,
            'error_type': type(error).__name__,
            'error': str(error),
            'elapsed_seconds': time.perf_counter() - started,
            'truth_used': False,
        }
        write_json(root / 'compatibility_receipt.json', receipt, overwrite=False)
        raise
    write_json(root / 'compatibility_receipt.json', receipt, overwrite=False)
    return receipt


def run_cae_v3(run_id: str, case_id: str, *, timeout: int = 7200,
               memory_mb: int = 20480, solver: str = 'AUTO',
               prior_failure_receipt: Path | None = None) -> dict[str, Any]:
    if solver not in {'AUTO', 'DIRECT', 'ITERATIVE'}:
        raise ValueError('solver must be AUTO, DIRECT, or ITERATIVE')
    if timeout < 60 or memory_mb < 1024:
        raise ValueError('unsafe timeout or Abaqus memory allowance')
    case, job, case_root = _verify_job(run_id, case_id, 'cae')
    attempt = Path(job['attempt_directory'])
    script = checked(job['program'])
    execution_plan = case_root / 'execution_plan.json'
    checked(descriptor(execution_plan))
    launcher_path = Path(__file__).resolve()
    launcher_before = descriptor(launcher_path)
    odb_adapter_before = descriptor(ODB_ADAPTER)
    recovery_of = _validated_prior_failure(prior_failure_receipt, case, job)
    launch_command = direct_abaqus_argv('cae', script)
    independent_path = attempt / 'independent_odb_audit_v2.json'
    reaudit_command = direct_abaqus_argv('python', ODB_ADAPTER) + [
        str(execution_plan.resolve()), str(independent_path.resolve()),
    ]
    write_json(case_root / 'launch_state.json', {
        'schema': 'g10_m6_cae_launcher_v3_launch_state',
        'at_utc': now(),
        'run_id': run_id,
        'case_id': case_id,
        'state': 'started_receipt_not_yet_written',
        'attempt_directory': attempt.as_posix(),
        'launcher_adapter': launcher_before,
        'odb_adapter': odb_adapter_before,
        'recovery_of': recovery_of,
        'truth_used': False,
    }, overwrite=False)

    started = time.perf_counter()
    release = None
    code = None
    reaudit_code = None
    timed_out = False
    elapsed = 0.0
    audit = {}
    independent = {}
    try:
        environment = dict(
            os.environ,
            AEI_MATRIX_SOLVER=solver,
            AEI_PREPARE_ONLY='0',
            AEI_MEMORY_MB=str(memory_mb),
            AEI_SKIP_MESH='0',
        )
        with resource_lock('abaqus', f'g10-m6-cae-v3-{run_id}-{case_id}'):
            release = _abaqus_release()
            code, timed_out, elapsed = _run_process(
                launch_command, attempt, environment,
                case_root / 'launch_v3.log', timeout,
            )
            audit_path = attempt / 'execution_audit.json'
            audit = read_json(audit_path) if audit_path.exists() else {}
            if code == 0 and not timed_out and audit.get('status') == 'PASS':
                reaudit_code, reaudit_timeout, reaudit_elapsed = _run_process(
                    reaudit_command, attempt, environment,
                    case_root / 'odb_reaudit_v3.log', min(timeout, 1800),
                )
                timed_out |= reaudit_timeout
                elapsed += reaudit_elapsed
            independent = read_json(independent_path) if independent_path.exists() else {}

        launcher_unchanged = descriptor(launcher_path) == launcher_before
        odb_adapter_unchanged = descriptor(ODB_ADAPTER) == odb_adapter_before
        plan_descriptor = descriptor(execution_plan)
        passed = code == 0 and not timed_out \
            and audit.get('status') == 'PASS' \
            and independent.get('status') == 'PASS' \
            and independent.get('execution_plan_sha256') == plan_descriptor['sha256'] \
            and independent.get('python_major') == 2 \
            and launcher_unchanged and odb_adapter_unchanged
        artifacts = {}
        for name in ('final.cae', 'final.odb', 'final.inp', 'execution_audit.json',
                     'independent_odb_audit_v2.json'):
            artifact = attempt / name
            if artifact.exists():
                artifacts[name] = descriptor(artifact)
        receipt = {
            'schema': 'g10_m6_execution_receipt_v1',
            'completed_utc': now(),
            'run_id': run_id,
            'case_id': case_id,
            'domain': 'cae',
            'status': 'PASS' if passed else 'FAIL',
            'actually_executed': True,
            'prepare_only': False,
            'skip_mesh': False,
            'manifest': job['manifest'],
            'job_spec': descriptor(case_root / 'job_spec.json'),
            'source_step': case['source_step'],
            'prediction': case['prediction'],
            'semantic_plan': job['semantic_plan'],
            'execution_plan': plan_descriptor,
            'program': job['program'],
            'frozen_binding': case['frozen_binding'],
            'launcher_adapter': launcher_before,
            'launcher_adapter_unchanged': launcher_unchanged,
            'odb_adapter': odb_adapter_before,
            'odb_adapter_unchanged': odb_adapter_unchanged,
            'launcher_mode': 'direct_batch_argv_v3',
            'launch_argv': launch_command,
            'reaudit_argv': reaudit_command,
            'recovery_of': recovery_of,
            'return_code': code,
            'independent_reaudit_return_code': reaudit_code,
            'independent_status': independent.get('status'),
            'timed_out': timed_out,
            'solver': solver,
            'memory_mb': memory_mb,
            'abaqus_release': release,
            'artifacts': artifacts,
            'elapsed_seconds': elapsed,
            'attempt_count': 1,
            'truth_used': False,
        }
    except BaseException as error:
        receipt = {
            'schema': 'g10_m6_execution_receipt_v1',
            'completed_utc': now(),
            'run_id': run_id,
            'case_id': case_id,
            'domain': 'cae',
            'status': 'FAIL',
            'actually_executed': True,
            'prepare_only': False,
            'skip_mesh': False,
            'manifest': job['manifest'],
            'job_spec': descriptor(case_root / 'job_spec.json'),
            'source_step': case['source_step'],
            'prediction': case['prediction'],
            'semantic_plan': job['semantic_plan'],
            'program': job['program'],
            'frozen_binding': case['frozen_binding'],
            'launcher_adapter': launcher_before,
            'odb_adapter': odb_adapter_before,
            'launcher_mode': 'direct_batch_argv_v3',
            'launch_argv': launch_command,
            'reaudit_argv': reaudit_command,
            'recovery_of': recovery_of,
            'abaqus_release': release,
            'return_code': code,
            'independent_reaudit_return_code': reaudit_code,
            'timed_out': timed_out,
            'error_type': type(error).__name__,
            'error': str(error),
            'elapsed_seconds': time.perf_counter() - started,
            'attempt_count': 1,
            'truth_used': False,
        }
        write_json(case_root / 'execution_receipt.json', receipt, overwrite=False)
        raise
    write_json(case_root / 'execution_receipt.json', receipt, overwrite=False)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('self-test-abaqus-python')
    smoke.add_argument('--smoke-id', required=True)
    smoke.add_argument('--timeout', type=int, default=120)
    run = sub.add_parser('run')
    run.add_argument('--run-id', required=True)
    run.add_argument('--case', required=True)
    run.add_argument('--timeout', type=int, default=7200)
    run.add_argument('--memory-mb', type=int, default=20480)
    run.add_argument('--solver', choices=('AUTO', 'DIRECT', 'ITERATIVE'), default='AUTO')
    run.add_argument('--prior-failure-receipt', type=Path)
    args = parser.parse_args()
    if args.command == 'self-test-abaqus-python':
        result = abaqus_python_smoke(args.smoke_id, args.timeout)
    else:
        result = run_cae_v3(
            args.run_id, args.case, timeout=args.timeout,
            memory_mb=args.memory_mb, solver=args.solver,
            prior_failure_receipt=args.prior_failure_receipt,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
