"""Versioned launcher-only recovery for frozen G10 M6 Abaqus programs.

The v1 execution receipt exposed a Windows ``cmd /s /c`` quoting defect before
Abaqus entered the generated program.  This adapter preserves the frozen
manifest, generated physics program, mesh, material, loads, and ODB acceptance
rules.  Its only behavioral change is launching the verified Abaqus batch file
as a direct argument vector, matching the already successful release probe.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from g10_common import (
    checked, descriptor, now, read_json, resource_lock, write_json,
)
from m6_runner import (
    ABAQUS, _abaqus_release, _run_process, _verify_job,
)


ADAPTER_SCHEMA = "g10_m6_cae_direct_batch_launcher_v2"


def direct_abaqus_argv(mode: str, program: Path,
                       launcher: Path = ABAQUS) -> list[str]:
    """Return shell-free argv for a CAE program or an ODB Python re-audit."""
    if mode not in {"cae", "python"}:
        raise ValueError("mode must be cae or python")
    launcher_text = str(Path(launcher).resolve())
    program_text = str(Path(program).resolve())
    if any(character in launcher_text + program_text for character in '"\r\n'):
        raise ValueError("launcher and program paths cannot contain quotes or newlines")
    if mode == "cae":
        return [launcher_text, "cae", f"noGUI={program_text}"]
    return [launcher_text, "python", program_text]


def _validated_prior_failure(path: Path | None, case: dict[str, Any],
                             job: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    prior_path = path.resolve()
    prior = read_json(prior_path)
    if prior.get("schema") != "g10_m6_execution_receipt_v1" \
            or prior.get("domain") != "cae" \
            or prior.get("case_id") != case["case_id"] \
            or prior.get("status") != "FAIL" \
            or prior.get("actually_executed") is not True:
        raise ValueError("prior receipt is not the matching preserved CAE failure")
    if prior.get("manifest") != job["manifest"] \
            or prior.get("source_step") != job["source_step"] \
            or prior.get("prediction") != job["prediction"] \
            or prior.get("frozen_binding") != job["frozen_binding"]:
        raise ValueError("prior failure does not share the frozen M6 identity")
    return descriptor(prior_path)


def run_cae_direct(run_id: str, case_id: str, *, timeout: int = 7200,
                   memory_mb: int = 20480, solver: str = "AUTO",
                   prior_failure_receipt: Path | None = None) -> dict[str, Any]:
    """Execute one immutable prepared CAE job with the direct batch launcher."""
    if solver not in {"AUTO", "DIRECT", "ITERATIVE"}:
        raise ValueError("solver must be AUTO, DIRECT, or ITERATIVE")
    if timeout < 60 or memory_mb < 1024:
        raise ValueError("unsafe timeout or Abaqus memory allowance")
    case, job, case_root = _verify_job(run_id, case_id, "cae")
    attempt = Path(job["attempt_directory"])
    script = checked(job["program"])
    adapter_path = Path(__file__).resolve()
    adapter_before = descriptor(adapter_path)
    recovery_of = _validated_prior_failure(prior_failure_receipt, case, job)
    launch_command = direct_abaqus_argv("cae", script)
    reaudit_program = case_root / "odb_reaudit.py"
    reaudit_command = direct_abaqus_argv("python", reaudit_program)
    launch_state = case_root / "launch_state.json"
    write_json(launch_state, {
        "schema": "g10_m6_cae_launcher_v2_launch_state",
        "at_utc": now(),
        "run_id": run_id,
        "case_id": case_id,
        "state": "started_receipt_not_yet_written",
        "attempt_directory": attempt.as_posix(),
        "launcher_adapter": adapter_before,
        "recovery_of": recovery_of,
        "truth_used": False,
    }, overwrite=False)

    started = time.perf_counter()
    release: dict[str, Any] | None = None
    code: int | None = None
    reaudit_code: int | None = None
    timed_out = False
    elapsed = 0.0
    audit: dict[str, Any] = {}
    independent: dict[str, Any] = {}
    try:
        environment = dict(
            os.environ,
            AEI_MATRIX_SOLVER=solver,
            AEI_PREPARE_ONLY="0",
            AEI_MEMORY_MB=str(memory_mb),
            AEI_SKIP_MESH="0",
        )
        with resource_lock("abaqus", f"g10-m6-cae-v2-{run_id}-{case_id}"):
            release = _abaqus_release()
            code, timed_out, elapsed = _run_process(
                launch_command, attempt, environment,
                case_root / "launch_v2.log", timeout,
            )
            audit_path = attempt / "execution_audit.json"
            audit = read_json(audit_path) if audit_path.exists() else {}
            independent_path = attempt / "independent_odb_audit.json"
            if code == 0 and not timed_out and audit.get("status") == "PASS":
                reaudit_code, reaudit_timeout, reaudit_elapsed = _run_process(
                    reaudit_command, attempt, environment,
                    case_root / "odb_reaudit_v2.log", min(timeout, 1800),
                )
                timed_out |= reaudit_timeout
                elapsed += reaudit_elapsed
            independent = read_json(independent_path) if independent_path.exists() else {}

        adapter_after = descriptor(adapter_path)
        adapter_unchanged = adapter_after == adapter_before
        passed = code == 0 and not timed_out \
            and audit.get("status") == "PASS" \
            and reaudit_code == 0 \
            and independent.get("status") == "PASS" \
            and adapter_unchanged
        artifacts = {}
        for name in ("final.cae", "final.odb", "final.inp", "execution_audit.json",
                     "independent_odb_audit.json"):
            artifact = attempt / name
            if artifact.exists():
                artifacts[name] = descriptor(artifact)
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(),
            "run_id": run_id,
            "case_id": case_id,
            "domain": "cae",
            "status": "PASS" if passed else "FAIL",
            "actually_executed": True,
            "prepare_only": False,
            "skip_mesh": False,
            "manifest": job["manifest"],
            "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"],
            "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"],
            "program": job["program"],
            "frozen_binding": case["frozen_binding"],
            "launcher_adapter": adapter_before,
            "launcher_adapter_unchanged": adapter_unchanged,
            "launcher_mode": "direct_batch_argv_v2",
            "launch_argv": launch_command,
            "reaudit_argv": reaudit_command,
            "recovery_of": recovery_of,
            "return_code": code,
            "independent_reaudit_return_code": reaudit_code,
            "timed_out": timed_out,
            "solver": solver,
            "memory_mb": memory_mb,
            "abaqus_release": release,
            "artifacts": artifacts,
            "elapsed_seconds": elapsed,
            "attempt_count": 1,
            "truth_used": False,
        }
    except BaseException as error:
        receipt = {
            "schema": "g10_m6_execution_receipt_v1",
            "completed_utc": now(),
            "run_id": run_id,
            "case_id": case_id,
            "domain": "cae",
            "status": "FAIL",
            "actually_executed": True,
            "prepare_only": False,
            "skip_mesh": False,
            "manifest": job["manifest"],
            "job_spec": descriptor(case_root / "job_spec.json"),
            "source_step": case["source_step"],
            "prediction": case["prediction"],
            "semantic_plan": job["semantic_plan"],
            "program": job["program"],
            "frozen_binding": case["frozen_binding"],
            "launcher_adapter": adapter_before,
            "launcher_mode": "direct_batch_argv_v2",
            "launch_argv": launch_command,
            "reaudit_argv": reaudit_command,
            "recovery_of": recovery_of,
            "abaqus_release": release,
            "return_code": code,
            "independent_reaudit_return_code": reaudit_code,
            "timed_out": timed_out,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "attempt_count": 1,
            "truth_used": False,
        }
        write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
        raise
    write_json(case_root / "execution_receipt.json", receipt, overwrite=False)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--memory-mb", type=int, default=20480)
    parser.add_argument("--solver", choices=("AUTO", "DIRECT", "ITERATIVE"),
                        default="AUTO")
    parser.add_argument("--prior-failure-receipt", type=Path)
    args = parser.parse_args()
    receipt = run_cae_direct(
        args.run_id,
        args.case,
        timeout=args.timeout,
        memory_mb=args.memory_mb,
        solver=args.solver,
        prior_failure_receipt=args.prior_failure_receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
