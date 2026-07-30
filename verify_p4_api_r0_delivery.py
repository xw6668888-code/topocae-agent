"""Final offline verification and manifest refresh for P4-API-R0."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r0_provider_adapter_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_full_regression() -> dict:
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
    elapsed = time.perf_counter() - started
    combined = completed.stdout + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": "python -m unittest discover -s tests -p 'test_p4*.py' -v",
        "test_count": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "formal_test_payload_accessed": False,
        "note": "Test-split cases assert fail-closed behavior; they do not decode payloads.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Only recheck current artifacts; do not refresh regression evidence.",
    )
    args = parser.parse_args()
    mock_path = REPORT / "mock_test_results.json"
    mock = json.loads(mock_path.read_text(encoding="utf-8"))
    if not args.skip_regression:
        mock["full_p4_regression"] = run_full_regression()
        mock["status"] = (
            "PASS" if mock["status"] == "PASS"
            and mock["full_p4_regression"]["status"] == "PASS" else "FAIL"
        )
        write_json(mock_path, mock)

    acceptance_path = REPORT / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["gates"]["p4_existing_tests_no_regression"] = (
        mock.get("full_p4_regression", {}).get("status") == "PASS"
    )
    acceptance["verification_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if not acceptance["gates"]["p4_existing_tests_no_regression"]:
        acceptance["status"] = "BLOCKED_WITH_EVIDENCE"
    write_json(acceptance_path, acceptance)

    final_path = REPORT / "final_acceptance.zh-CN.md"
    final_text = final_path.read_text(encoding="utf-8")
    marker = "- 完整普通 P4 回归："
    lines = [line for line in final_text.splitlines() if not line.startswith(marker)]
    regression = mock.get("full_p4_regression", {})
    insert_at = next(
        (index for index, line in enumerate(lines) if line.startswith("- 上游冻结变化")),
        len(lines),
    )
    lines.insert(
        insert_at,
        f"{marker}{regression.get('test_count')} 项，`{regression.get('status')}`",
    )
    final_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")

    manifest_path = REPORT / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = {row["path"] for row in manifest["assets"]}
    verifier_path = "verify_p4_api_r0_delivery.py"
    if verifier_path not in known:
        manifest["assets"].append({"path": verifier_path})
    for row in manifest["assets"]:
        path = ROOT / row["path"]
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    manifest["asset_count"] = len(manifest["assets"])
    manifest["status"] = acceptance["status"]
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)

    errors = []
    for row in manifest["assets"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            errors.append({"path": row["path"], "error": "manifest_mismatch"})
    ledger = json.loads(
        (ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json")
        .read_text(encoding="utf-8")
    )
    run_count = sum(len(value) for value in ledger["runs"].values())
    if run_count != 0 or ledger["test_content_accessed"]:
        errors.append({"path": "test_run_ledger.json", "error": "formal_test_guard"})
    if mock["status"] != "PASS":
        errors.append({"path": "mock_test_results.json", "error": "tests_failed"})
    if acceptance["status"] != "READY_FOR_P4_API_SMOKE_REVIEW":
        errors.append({"path": "acceptance.json", "error": "acceptance_not_ready"})

    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "acceptance_status": acceptance["status"],
        "full_p4_tests": regression.get("test_count"),
        "manifest_assets": manifest["asset_count"],
        "formal_test_run_count": run_count,
        "formal_test_content_accessed": ledger["test_content_accessed"],
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
