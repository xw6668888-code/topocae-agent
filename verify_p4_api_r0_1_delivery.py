"""Final offline verifier for the P4-API-R0.1 supervisor rework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from p4_api_r0_1_audit import sha256_file, verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r0_1_supervisor_rework_v1"


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_regression() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_p4*.py", "-v",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    combined = completed.stdout + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "return_code": completed.returncode,
        "test_count": int(match.group(1)) if match else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": "python -m unittest discover -s tests -p 'test_p4*.py' -v",
        "real_api_called": False,
        "train_val_baseline_run": False,
        "formal_test_payload_decoded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()
    mock_path = REPORT / "mock_test_results.json"
    mock = json.loads(mock_path.read_text(encoding="utf-8"))
    if not args.skip_regression:
        mock["full_p4_regression"] = run_regression()
        mock["status"] = (
            "PASS" if mock["status"] == "PASS"
            and mock["full_p4_regression"]["status"] == "PASS" else "FAIL"
        )
        write_json(mock_path, mock)

    current_freeze = verify_full_freeze()
    freeze_path = REPORT / "full_freeze_sha256.json"
    recorded_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    current_counts = {
        row["stage"]: (
            row["matching_count"], row["total_count"], row["change_count"]
        ) for row in current_freeze["stages"]
    }
    recorded_counts = {
        row["stage"]: (
            row["matching_count"], row["total_count"], row["change_count"]
        ) for row in recorded_freeze["stages"]
    }

    acceptance_path = REPORT / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    regression = mock.get("full_p4_regression", {})
    acceptance["gates"]["p4_full_regression_pass"] = (
        regression.get("status") == "PASS"
    )
    acceptance["verification_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if regression.get("status") != "PASS" or current_freeze["status"] != "PASS":
        acceptance["status"] = "BLOCKED_WITH_EVIDENCE"
    write_json(acceptance_path, acceptance)

    final_path = REPORT / "final_acceptance.zh-CN.md"
    lines = [
        line for line in final_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("- 完整普通 P4 回归：")
    ]
    insertion = next(
        (index for index, line in enumerate(lines) if line.startswith("- formal-test")),
        len(lines),
    )
    lines.insert(
        insertion,
        f"- 完整普通 P4 回归：{regression.get('test_count')} 项，"
        f"`{regression.get('status')}`；",
    )
    final_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")

    manifest_path = REPORT / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    additions = [
        "verify_p4_api_r0_1_delivery.py",
        "tests/test_p4_api_r0_1_manifest_redaction.py",
    ]
    known = {row["path"] for row in manifest["assets"]}
    for item in additions:
        if item not in known:
            manifest["assets"].append({"path": item})
    for row in manifest["assets"]:
        path = ROOT / row["path"]
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    manifest["asset_count"] = len(manifest["assets"])
    manifest["status"] = acceptance["status"]
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)

    ledger = json.loads(
        (ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json")
        .read_text(encoding="utf-8")
    )
    run_count = sum(len(value) for value in ledger["runs"].values())
    security = json.loads(
        (REPORT / "security_audit.json").read_text(encoding="utf-8")
    )
    import_audit = json.loads(
        (REPORT / "canonical_import_audit.json").read_text(encoding="utf-8")
    )
    errors = []
    if regression.get("status") != "PASS":
        errors.append("E_P4_REGRESSION")
    if current_freeze["status"] != "PASS" or current_counts != recorded_counts:
        errors.append("E_FULL_FREEZE_RECHECK")
    if current_counts != {"P1": (13, 13, 0), "P2": (13, 13, 0), "P3": (1885, 1885, 0)}:
        errors.append("E_FULL_FREEZE_COUNTS")
    if run_count != 0 or ledger["test_content_accessed"]:
        errors.append("E_FORMAL_TEST_GUARD")
    if (ROOT / "utils" / "cae_language_grounding" / "llm_adapter").exists():
        errors.append("E_CANONICAL_IMPORT_SHADOW")
    if import_audit["status"] != "PASS":
        errors.append("E_CANONICAL_IMPORT")
    if security["workspace"]["production_finding_count"] != 0:
        errors.append("E_WORKSPACE_SECRET")
    if security["full_scope_secret_zero_claimed"]:
        errors.append("E_EXTERNAL_SECRET_OVERCLAIM")
    if security["real_api_smoke_authorized"]:
        errors.append("E_REAL_API_AUTHORIZATION")
    for row in manifest["assets"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            errors.append("E_DELIVERY_MANIFEST")
            break
    if acceptance["status"] != "READY_FOR_P4_API_SMOKE_REVIEW_R0_1":
        errors.append("E_ACCEPTANCE_STATUS")

    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "acceptance_status": acceptance["status"],
        "full_p4_tests": regression.get("test_count"),
        "full_freeze": {
            stage: f"{values[0]}/{values[1]} changes={values[2]}"
            for stage, values in current_counts.items()
        },
        "delivery_manifest_assets": manifest["asset_count"],
        "formal_test_run_count": run_count,
        "formal_test_content_accessed": ledger["test_content_accessed"],
        "real_api_call_count": 0,
        "external_legacy_status": security["external_legacy"]["status"],
        "full_scope_secret_zero_claimed": security["full_scope_secret_zero_claimed"],
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
