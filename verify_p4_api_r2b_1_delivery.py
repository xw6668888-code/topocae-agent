"""Offline and final verification for P4-API-R2B.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

import execute_p4_api_r2b_1_prompt_revision_smoke as delivery
import run_p4_api_r2a_1_runtime_prompt as r2a1
import run_p4_api_r2b_1_prompt_revision_smoke as diagnosis
from utils import cae_query_frame_runtime_r2b_1 as runtime
from utils.cae_language_grounding.dataset import DecodeAudit, iter_rendering_records


ROOT = Path(__file__).resolve().parent
FORMAL_LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
EXPECTED_FROZEN_HASHES = {
    "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json": "41067a5b8916bb3ad222c9e873d8576a4ec7aaa3634c79d89d0eebdcba63468a",
    "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json": "7a9e8f184013eeeff0db794a95defc73d19438c89e7cbbf6609abdea1a6a2845",
    "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json": "44b4be432a432bce02158ab3fddc1bcdc569cb3d86587286c5ab866b193349b9",
    "reports/p4_api_r2a_compact_query_frame_v1/query_frame.schema.json": "5f1913121e750493520d818b1d377d90add70a72e90c6612f6ec47cd7179da43",
    "reports/p4_api_r2a_compact_query_frame_v1/query_frame_prompt.zh-CN.md": "aae89eb07898edcf98962e9eabce052bd878fd5b96f2185b9104b19b3f5c7142",
    "reports/p4_api_r2a_1_runtime_prompt_v1/smoke_sample_manifest.json": "83ae43ad9bb17a61a723113f56b874ab4912dd147c0254f87057152985375f07",
    "reports/p2_a1_cae_ir_v1/freeze/configs/cae_feature_role_policy.yaml": runtime.P2_ROLE_POLICY_SHA256,
    "reports/p2_a1_cae_ir_v1/freeze/schemas/cae_intent_result.schema.json": runtime.P2_RESULT_SCHEMA_SHA256,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_state() -> dict[str, Any]:
    value = json.loads(FORMAL_LEDGER.read_text(encoding="utf-8"))
    return {
        "sha256": _sha256(FORMAL_LEDGER),
        "run_count": sum(len(rows) for rows in value["runs"].values()),
        "test_content_accessed": bool(value["test_content_accessed"]),
        "test_unlock_authorized": bool(value["test_unlock_authorized"]),
    }


def _policy_exact() -> bool:
    policy = yaml.safe_load(runtime.P2_ROLE_POLICY_PATH.read_text(encoding="utf-8"))
    expected = []
    for feature, feature_value in sorted(policy["features"].items()):
        for role in sorted(feature_value["selectable_roles"], key=lambda item: item["role"]):
            expected.append({
                "feature": feature,
                "face_role": role["role"],
                "compatible_operations": sorted(
                    operation
                    for operation, review in role["operation_review"].items()
                    if review["compatible"] is True
                ),
            })
    return expected == runtime.feature_role_compatibility_table()


def _mock_gate() -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    for example in runtime.runtime_examples():
        def transport(endpoint, payload, frame=example["frame"]):
            return {
                "model": "deepseek-v4-pro",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(frame, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        outcome = runtime.execute_mock_query_frame_request(example["input"], transport)
        rows.append({
            "name": example["name"],
            "frame_valid": outcome["frame_valid"],
            "compiler_valid": outcome["compiler_valid"],
            "p2_valid": outcome["p2_valid"],
            "network_request_count": outcome["network_request_count"],
        })
    return all(
        row["frame_valid"]
        and row["compiler_valid"]
        and row["p2_valid"]
        and row["network_request_count"] == 0
        for row in rows
    ), rows


def _leakage_gate() -> dict[str, Any]:
    hashes = {
        hashlib.sha256(item["input"]["text"].encode("utf-8")).hexdigest()
        for item in runtime.runtime_examples()
    }
    overlaps = set()
    audits = {}
    for split in ("train", "val"):
        audit = DecodeAudit("renderings", split)
        for record in iter_rendering_records(split, audit=audit):
            text = record["query"]["text"] if "query" in record else record["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in hashes:
                overlaps.add(digest)
        audits[split] = audit.as_dict()
    return {
        "train_val_exact_overlap_count": len(overlaps),
        "decoded_formal_test_lines": sum(row["decoded_test_lines"] for row in audits.values()),
        "opaque_formal_test_lines_crossed_without_decode": sum(row["opaque_test_lines_crossed"] for row in audits.values()),
        "decode_audits": audits,
    }


def _run_regression() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p4*.py", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout[-12000:]


def build_offline_gate(*, run_regression: bool) -> dict[str, Any]:
    formal_before = _formal_state()
    first = runtime.build_query_frame_runtime_prompt().encode("utf-8")
    second = runtime.build_query_frame_runtime_prompt().encode("utf-8")
    mock_pass, mock_rows = _mock_gate()
    leakage = _leakage_gate()
    actual_hashes = {relative: _sha256(ROOT / relative) for relative in EXPECTED_FROZEN_HASHES}
    frozen_changes = [
        relative
        for relative, expected in EXPECTED_FROZEN_HASHES.items()
        if actual_hashes[relative] != expected
    ]
    regression_code = regression_output = None
    if run_regression:
        regression_code, regression_output = _run_regression()
    formal_after = _formal_state()
    historical = delivery.historical_tree_hashes()
    result = {
        "schema_version": "topocae-p4-api-r2b-1-offline-gate/1",
        "prompt_byte_deterministic": first == second,
        "runtime_prompt_sha256": hashlib.sha256(first).hexdigest(),
        "feature_role_policy_exact": _policy_exact(),
        "compatibility_row_count": len(runtime.feature_role_compatibility_table()),
        "all_mock_frame_compiler_p2_valid": mock_pass,
        "mock_results": mock_rows,
        **leakage,
        "formal_ledger_before": formal_before,
        "formal_ledger_after": formal_after,
        "formal_ledger_unchanged": formal_before == formal_after,
        "frozen_asset_hashes": actual_hashes,
        "frozen_asset_changes": frozen_changes,
        "frozen_asset_change_count": len(frozen_changes),
        "historical_tree_sha256_baseline": historical,
        "regression_command": "python -m unittest discover -s tests -p test_p4*.py -v",
        "regression_return_code": regression_code,
        "regression_output_tail": regression_output,
        "api_request_count": 0,
        "network_request_count": 0,
    }
    result["gate_pass"] = bool(
        result["prompt_byte_deterministic"]
        and result["feature_role_policy_exact"]
        and result["all_mock_frame_compiler_p2_valid"]
        and result["train_val_exact_overlap_count"] == 0
        and result["decoded_formal_test_lines"] == 0
        and result["formal_ledger_unchanged"]
        and result["frozen_asset_change_count"] == 0
        and result["regression_return_code"] == 0
    )
    return result


def verify_delivery() -> dict[str, Any]:
    acceptance = json.loads((diagnosis.OUTPUT / "acceptance.json").read_text(encoding="utf-8"))
    manifest = json.loads((diagnosis.OUTPUT / "delivery_manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((diagnosis.OUTPUT / "freeze_audit.json").read_text(encoding="utf-8"))
    ledger = json.loads((diagnosis.OUTPUT / "request_ledger.json").read_text(encoding="utf-8"))
    status = acceptance["status"]
    return {
        "schema_version": "topocae-p4-api-r2b-1-final-verification/1",
        "status": status,
        "allowed_stop_state": status in {
            "READY_FOR_P4_API_R2C_DISJOINT_100_VAL_REVIEW",
            "DEEPSEEK_COMPACT_FRAME_INSUFFICIENT_WITH_EVIDENCE",
            "BLOCKED_WITH_EVIDENCE",
        },
        "request_caps_respected": (
            ledger["counters"]["first_round_requests"] <= 20
            and ledger["counters"]["repair_requests"] <= 20
            and ledger["counters"]["backend_requests"] <= 40
            and ledger["counters"]["http_retries"] == 0
        ),
        "historical_assets_unchanged": freeze["change_count"] == 0 and freeze["gate_pass"],
        "manifest_assets_present": all((diagnosis.OUTPUT / row["path"]).is_file() for row in manifest["assets"]),
        "second_revision_absent": acceptance["second_prompt_revision_allowed"] is False,
    }


def main() -> int:
    if diagnosis.OUTPUT.exists():
        result = verify_delivery()
        path = diagnosis.OUTPUT / "final_verification.json"
    else:
        result = build_offline_gate(run_regression=True)
        diagnosis.WORK.mkdir(parents=True, exist_ok=True)
        path = diagnosis.WORK / "offline_gate.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"path": str(path), **{key: value for key, value in result.items() if key in {"gate_pass", "status", "regression_return_code"}}}, ensure_ascii=False, indent=2))
    if "gate_pass" in result:
        return 0 if result["gate_pass"] else 1
    return 0 if all(value is True for key, value in result.items() if key not in {"schema_version", "status"}) else 1


if __name__ == "__main__":
    raise SystemExit(main())
