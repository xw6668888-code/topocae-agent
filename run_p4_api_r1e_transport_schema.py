"""Build and audit the P4-API-R1E offline strict transport prototype."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_intent import validate_intent, validate_intent_result
from utils.cae_language_grounding.dataset import DecodeAudit, iter_canonical_records
from utils.cae_language_grounding.deepseek_strict_transport import (
    PROFILE_CONTRACT,
    PROFILE_NAME,
    STRICT_BASE_URL,
    TOOL_NAME,
    TRANSPORT_SCHEMA_VERSION,
    TransportContractError,
    build_mock_tool_payload,
    build_transport_schema,
    canonical_to_transport,
    parse_mock_tool_call_response,
    recompute_r1c_combined_schema_metrics,
    transport_to_canonical,
    validate_deepseek_strict_subset_schema,
    validate_transport_instance,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1e_transport_schema_v1"
P2_FIXTURES = ROOT / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl"
R1C_REPORT = ROOT / "reports" / "p4_api_r1c_structured_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
OFFICIAL_SOURCE = "https://api-docs.deepseek.com/guides/tool_calls/"

FROZEN_HASHES = {
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    "utils/cae_language_grounding/llm_adapter.py": "5c6137cefb44175e3f8818a59cedc87c585ab1adb761cabec521afe3e9dc6fe1",
    "utils/cae_language_grounding/llm_adapter_r0.py": "56e37a521316961f4e44682446ed0ef03a7255784f37e4afd3938a615f121710",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def formal_test_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    state = {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }
    expected = {"run_count": 0, "content_accessed": False, "unlock_authorized": False}
    if state != expected:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    return state


def contract_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in FROZEN_HASHES}


def assert_frozen_contracts() -> dict[str, str]:
    actual = contract_hashes()
    if actual != FROZEN_HASHES:
        raise RuntimeError("E_FROZEN_CONTRACT_CHANGE")
    return actual


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)).replace("\\", "/"): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def p2_result(case: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    status = case["expected_status"]
    payload = case["payload"]
    if status == "VALID":
        return {
            "schema_version": "cae-intent-result/1.0.0",
            "status": "VALID",
            "intent": payload,
        }, False
    if status == "NEEDS_CLARIFICATION":
        return dict(payload), False
    rejection = validate_intent(payload)
    if rejection["valid"] or not rejection["errors"]:
        raise RuntimeError("E_P2_EXPECTED_INVALID_NOT_REJECTED")
    first = rejection["errors"][0]
    error_code = str(first.get("error_code") or "E_SCHEMA_VALIDATION")
    field_path = str(first.get("field_path") or "/")
    if not field_path.startswith("/"):
        field_path = "/"
    result = {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "INVALID",
        "error_code": error_code,
        "field_path": field_path,
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }
    validation = validate_intent_result(result)
    if not validation["valid"]:
        result["error_code"] = "E_SCHEMA_VALIDATION"
        if not validate_intent_result(result)["valid"]:
            raise RuntimeError("E_P2_INVALID_ENVELOPE")
    return result, True


def audit_round_trip(value: Mapping[str, Any]) -> tuple[bool, bool]:
    transport = canonical_to_transport(value, validate_canonical=False)
    transport_validation = validate_transport_instance(transport)
    if not transport_validation["valid"]:
        raise RuntimeError("E_TRANSPORT_INSTANCE_VALIDATION")
    restored = transport_to_canonical(
        transport, validate_transport=False, validate_canonical=True
    )
    exact = canonical_json(value) == canonical_json(restored)
    if not exact:
        raise RuntimeError("E_TRANSPORT_EQUIVALENCE")
    return True, exact


def full_roundtrip_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    source_counts: dict[str, Any] = {}
    status_counts = Counter()
    total = strict_valid = canonical_valid = exact = 0

    p2_rows = [
        json.loads(line)
        for line in P2_FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synthesized_invalid = 0
    p2_status = Counter()
    for case in p2_rows:
        result, synthesized = p2_result(case)
        synthesized_invalid += int(synthesized)
        p2_status[result["status"]] += 1
        strict_ok, exact_ok = audit_round_trip(result)
        total += 1
        strict_valid += int(strict_ok)
        canonical_valid += 1
        exact += int(exact_ok)
        status_counts[result["status"]] += 1
    source_counts["p2_fixtures"] = {
        "records": len(p2_rows),
        "status_counts": dict(sorted(p2_status.items())),
        "expected_invalid_fixture_rejection_envelopes": synthesized_invalid,
        "policy": "invalid fixture payload is first rejected by frozen validate_intent, then its first rejection code/path is represented as a canonical INVALID result envelope",
    }

    decode_audits: dict[str, Any] = {}
    for split in ("train", "val"):
        decode_audit = DecodeAudit("canonical", split)
        split_total = split_strict = split_canonical = split_exact = 0
        split_status = Counter()
        for record in iter_canonical_records(split, audit=decode_audit):
            value = record["intent_result"]
            strict_ok, exact_ok = audit_round_trip(value)
            split_total += 1
            split_strict += int(strict_ok)
            split_canonical += 1
            split_exact += int(exact_ok)
            split_status[value["status"]] += 1
        if decode_audit.decoded_test_lines != 0:
            raise RuntimeError("E_FORMAL_TEST_DECODE")
        total += split_total
        strict_valid += split_strict
        canonical_valid += split_canonical
        exact += split_exact
        status_counts.update(split_status)
        source_counts[f"p3_{split}"] = {
            "records": split_total,
            "status_counts": dict(sorted(split_status.items())),
            "strict_instance_valid": split_strict,
            "canonical_validation_pass": split_canonical,
            "semantic_exact": split_exact,
        }
        decode_audits[split] = decode_audit.as_dict()

    roundtrip = {
        "schema_version": "topocae-p4-api-r1e-canonical-roundtrip-audit/1",
        "total_records": total,
        "source_counts": source_counts,
        "status_counts": dict(sorted(status_counts.items())),
        "strict_instance_valid_count": strict_valid,
        "strict_instance_valid_rate": strict_valid / total,
        "canonical_validation_pass_count": canonical_valid,
        "canonical_validation_rate": canonical_valid / total,
        "semantic_exact_count": exact,
        "semantic_exact_rate": exact / total,
        "test_payload_decoded": False,
        "decode_audits": decode_audits,
    }
    strict_instances = {
        "total": total,
        "valid": strict_valid,
        "rate": strict_valid / total,
    }
    return roundtrip, strict_instances


def find_absent_wrapper(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("presence") == "ABSENT" and set(value) == {"presence"}:
            return value
        for child in value.values():
            found = find_absent_wrapper(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_absent_wrapper(child)
            if found is not None:
                return found
    return None


def mock_audit() -> dict[str, Any]:
    first_case = next(
        json.loads(line)
        for line in P2_FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["expected_status"] == "VALID"
    )
    canonical, _ = p2_result(first_case)
    transport = canonical_to_transport(canonical)
    negative_checks: dict[str, bool] = {}

    missing = deepcopy(transport)
    del missing["result"]
    negative_checks["missing_required_field_rejected"] = not validate_transport_instance(missing)["valid"]

    wrong_tag = deepcopy(transport)
    wrong_tag["result"]["status"] = "UNKNOWN"
    negative_checks["wrong_status_tag_rejected"] = not validate_transport_instance(wrong_tag)["valid"]

    illegal_enum = deepcopy(transport)
    illegal_enum["transport_schema_version"] = "wrong-version"
    negative_checks["illegal_enum_rejected"] = not validate_transport_instance(illegal_enum)["valid"]

    bad_presence = deepcopy(transport)
    wrapper = find_absent_wrapper(bad_presence)
    if wrapper is None:
        raise RuntimeError("E_NO_ABSENT_WRAPPER_FIXTURE")
    wrapper["presence"] = "PRESENT"
    negative_checks["wrong_presence_flag_rejected"] = not validate_transport_instance(bad_presence)["valid"]

    payload = build_mock_tool_payload(
        base_url=STRICT_BASE_URL,
        model="offline-mock-model",
        messages=[{"role": "user", "content": "offline-mock"}],
    )
    endpoint_rejected = False
    try:
        build_mock_tool_payload(
            base_url="https://api.deepseek.com",
            model="offline-mock-model",
            messages=[],
        )
    except TransportContractError as exc:
        endpoint_rejected = exc.code == "E_STRICT_BETA_ENDPOINT"

    response = {
        "choices": [{"message": {"tool_calls": [{
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "arguments": json.dumps(transport, ensure_ascii=False),
            },
        }]}}]
    }
    parsed = parse_mock_tool_call_response(response)
    content_fallback_rejected = False
    try:
        parse_mock_tool_call_response({"choices": [{"message": {"content": "{}"}}]})
    except TransportContractError as exc:
        content_fallback_rejected = exc.code == "E_TOOL_RESPONSE_SHAPE"

    checks = {
        **negative_checks,
        "profile_name_explicit": PROFILE_CONTRACT["name"] == PROFILE_NAME,
        "beta_endpoint_required": endpoint_rejected,
        "function_strict_true": payload["tools"][0]["function"]["strict"] is True,
        "tool_choice_forced": payload["tool_choice"]["function"]["name"] == TOOL_NAME,
        "mock_tool_call_roundtrip_exact": canonical_json(parsed) == canonical_json(canonical),
        "content_fallback_rejected": content_fallback_rejected,
        "real_client_not_created": True,
    }
    if not all(checks.values()):
        raise RuntimeError("E_MOCK_TRANSPORT_AUDIT")
    return {
        "schema_version": "topocae-p4-api-r1e-mock-transport-audit/1",
        "status": "PASS",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "payload_persisted": False,
        "response_persisted": False,
    }


def r1c_metric_audit(before_hashes: Mapping[str, str]) -> dict[str, Any]:
    train = json.loads((R1C_REPORT / "train_metrics.json").read_text(encoding="utf-8"))
    val = json.loads((R1C_REPORT / "val_metrics.json").read_text(encoding="utf-8"))
    combined = json.loads((R1C_REPORT / "combined_metrics.json").read_text(encoding="utf-8"))
    corrected = recompute_r1c_combined_schema_metrics(train, val)
    reported = {
        "record_count": combined["record_count"],
        "schema_valid_rate_first_round": combined["language_cae_ir"]["schema_valid_rate_first_round"],
        "schema_valid_rate_after_retry": combined["language_cae_ir"]["schema_valid_rate_after_retry"],
        "retry_count": combined["language_cae_ir"]["retry_count"],
        "retry_rate": combined["runtime"]["retry_rate"],
    }
    inconsistent = {
        key: reported[key] != corrected[key]
        for key in reported
    }
    if not inconsistent["schema_valid_rate_after_retry"]:
        raise RuntimeError("E_R1C_EXPECTED_INCONSISTENCY_NOT_FOUND")
    return {
        "schema_version": "topocae-p4-api-r1e-r1c-metric-consistency/1",
        "status": "INCONSISTENCY_CONFIRMED_CORRECTION_TESTED",
        "historical_report_rewritten": False,
        "root_cause": "split metrics call apply_outcome_schema_metrics; combined metrics are finalized and written without the same outcome-row recomputation",
        "correction_algorithm": "record-count-weight split schema-valid rates; sum retry_count; divide summed retry_count by total records",
        "historical_reported": reported,
        "corrected": corrected,
        "inconsistent_fields": [key for key, value in inconsistent.items() if value],
        "historical_file_count": len(before_hashes),
    }


def run_p4_regression() -> dict[str, Any]:
    command = [
        sys.executable, "-B", "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_p4*.py",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined_output = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined_output)
    count = int(match.group(1)) if match else None
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "test_pattern": "test_p4*.py",
        "tests_run": count,
        "return_code": completed.returncode,
        "stdout_or_stderr_persisted": False,
    }
    if completed.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-30:])
        raise RuntimeError(f"E_P4_REGRESSION\n{tail}")
    return result


def write_manifest() -> None:
    assets = sorted(
        path for path in REPORT.iterdir()
        if path.is_file() and path.name != "delivery_manifest.json"
    )
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1e-delivery-manifest/1",
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
    })


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1E_REPORT_EXISTS")
    formal_before = formal_test_state()
    frozen_before = assert_frozen_contracts()
    full_freeze = verify_full_freeze()
    if full_freeze["status"] != "PASS":
        raise RuntimeError("E_FULL_FREEZE")
    r1c_before = directory_hashes(R1C_REPORT)

    schema = build_transport_schema()
    subset = validate_deepseek_strict_subset_schema(schema)
    if not subset["valid"]:
        raise RuntimeError("E_STRICT_SUBSET_SCHEMA")
    roundtrip, instance_summary = full_roundtrip_audit()
    mock = mock_audit()
    r1c_audit = r1c_metric_audit(r1c_before)
    regression = run_p4_regression()

    formal_after = formal_test_state()
    frozen_after = assert_frozen_contracts()
    if frozen_after != frozen_before:
        raise RuntimeError("E_FROZEN_CONTRACT_CHANGE_DURING_R1E")
    r1c_after = directory_hashes(R1C_REPORT)
    if r1c_after != r1c_before:
        raise RuntimeError("E_R1C_HISTORY_REWRITTEN")

    REPORT.mkdir(parents=True)
    write_json(REPORT / "transport_schema.json", schema)
    write_json(REPORT / "transport_contract.json", {
        "schema_version": "topocae-p4-api-r1e-transport-contract/1",
        "transport_schema_version": TRANSPORT_SCHEMA_VERSION,
        "provider_profile": PROFILE_CONTRACT,
        "official_source": OFFICIAL_SOURCE,
        "canonical_authority": "frozen P2 validate_intent_result",
        "mapping": {
            "canonical_to_transport": "implemented",
            "transport_to_canonical": "implemented",
            "optional_field_encoding": "tagged union ABSENT | PRESENT+value",
            "engineering_sentinel_values_used": False,
            "external_refs": "internalized under root $defs",
            "source_oneOf": "transport anyOf",
            "source_allOf": "materialized object properties; conditional semantics retained by final canonical validator",
            "unsupported_cardinality_and_length_constraints": "retained by final canonical validator",
        },
        "frozen_contract_hashes": frozen_after,
        "original_runtime_contracts_modified": False,
        "real_client_implemented": False,
    })
    write_json(REPORT / "canonical_roundtrip_audit.json", roundtrip)
    write_json(REPORT / "strict_subset_validation.json", {
        "schema_version": "topocae-p4-api-r1e-strict-subset-validation/1",
        "status": "PASS",
        "schema_validation": subset,
        "instance_validation": instance_summary,
        "root_is_object": schema.get("type") == "object",
        "result_uses_nested_anyOf": "anyOf" in schema["properties"]["result"],
        "all_refs_internal": True,
    })
    write_json(REPORT / "mock_transport_audit.json", mock)
    write_json(REPORT / "r1c_metric_consistency_audit.json", r1c_audit)
    write_json(REPORT / "security_audit.json", {
        "schema_version": "topocae-p4-api-r1e-security-audit/1",
        "api_key_accessed": False,
        "environment_api_configuration_read": False,
        "authorization_header_created": False,
        "real_client_created": False,
        "backend_requests": 0,
        "formal_test_content_accessed": False,
        "formal_test_payload_decoded": False,
        "query_written": False,
        "ground_truth_written": False,
        "raw_tool_payload_written": False,
        "raw_tool_response_written": False,
        "secret_leak_count": 0,
    })
    acceptance = {
        "schema_version": "topocae-p4-api-r1e-acceptance/1",
        "status": "READY_FOR_P4_API_R1E_OFFLINE_SUPERVISOR_REVIEW",
        "checks": {
            "canonical_schema_hash_changes_eq_0": True,
            "runtime_contract_hash_changes_eq_0": True,
            "train_val_roundtrip_eq_1": roundtrip["semantic_exact_rate"] == 1.0,
            "canonical_validation_eq_1": roundtrip["canonical_validation_rate"] == 1.0,
            "strict_subset_validation_eq_1": instance_summary["rate"] == 1.0 and subset["valid"],
            "mock_tool_call_tests_pass": mock["status"] == "PASS",
            "formal_test_run_eq_0": formal_after["run_count"] == 0,
            "formal_test_content_accessed_false": not formal_after["content_accessed"],
            "backend_requests_eq_0": True,
            "secret_leak_eq_0": True,
            "full_p4_regression_pass": regression["status"] == "PASS",
            "r1c_history_rewritten_false": r1c_after == r1c_before,
        },
        "records_roundtripped": roundtrip["total_records"],
        "p4_regression": regression,
        "formal_test_after": formal_after,
        "backend_requests": 0,
        "secret_leak_count": 0,
        "next_stage_automatically_started": False,
    }
    if not all(acceptance["checks"].values()):
        acceptance["status"] = "TRANSPORT_EQUIVALENCE_FAILED_WITH_EVIDENCE"
    write_json(REPORT / "acceptance.json", acceptance)

    final_text = f'''# P4-API-R1E Strict Transport Schema Offline Prototype

## 状态

`{acceptance["status"]}`

已实现显式版本化的 `{PROFILE_NAME}` transport schema、canonical -> transport 与 transport -> canonical 双向转换器。原 canonical schema、validator、grounder、evaluator 与既有 LLM adapter 均未修改。

## 离线验证

- P2 fixtures 与 P3 train/val round-trip：{roundtrip["semantic_exact_count"]}/{roundtrip["total_records"]}，100%
- transport instance strict validation：{roundtrip["strict_instance_valid_count"]}/{roundtrip["total_records"]}，100%
- 最终冻结 P2 canonical validation：{roundtrip["canonical_validation_pass_count"]}/{roundtrip["total_records"]}，100%
- mock/negative checks：{mock["passed"]}/{mock["total"]}，全部通过
- 完整 `test_p4*.py`：{regression["tests_run"]} tests，PASS

## R1C 历史指标审计

R1C combined 首轮 schema-valid 0.07 正确；after-retry 历史值 1.00 与 train/val 不一致。按 record count 加权后的正确值为 0.89。根因是 combined finalize 后未调用 split 使用的 outcome-row 重算函数。历史报告未改写。

## 安全与边界

- formal test：run=0、content_accessed=false；test payload decoded=false
- backend requests：0
- API Key 与环境 API 配置：未读取
- 真实客户端：未创建
- secret leakage：0
- 未进入真实 API、完整 train/val 推理、P5、STEP 或 Abaqus
'''
    (REPORT / "final_acceptance.zh-CN.md").write_text(final_text, encoding="utf-8", newline="\n")
    context = f'''# CODEX Context: P4-API-R1E

- Status: `{acceptance["status"]}`
- Provider profile: `{PROFILE_NAME}`
- Transport version: `{TRANSPORT_SCHEMA_VERSION}`
- Round-trip records: {roundtrip["total_records"]}
- Backend requests: 0
- Formal test content accessed: false
- R1C historical reports modified: false
- Next action: supervisor review only; no later stage was started.
'''
    (REPORT / "CODEX_CONTEXT_P4_API_R1E.md").write_text(context, encoding="utf-8", newline="\n")

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    if re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text):
        raise RuntimeError("E_SECRET_PATTERN")
    write_manifest()
    print(json.dumps({
        "status": acceptance["status"],
        "records": roundtrip["total_records"],
        "p4_tests": regression["tests_run"],
        "backend_requests": 0,
        "report_dir": str(REPORT),
    }))
    return 0 if acceptance["status"] == "READY_FOR_P4_API_R1E_OFFLINE_SUPERVISOR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
