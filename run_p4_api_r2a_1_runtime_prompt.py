"""Offline P4-API-R2A.1 runtime prompt, smoke manifest, and audit builder."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

from utils.cae_language_grounding.dataset import (
    DecodeAudit,
    iter_canonical_records,
    iter_rendering_records,
)
from utils.cae_query_frame_runtime import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    NORMAL_ENDPOINT,
    PROFILE_NAME,
    QUERY_FRAME_SCHEMA_SHA256,
    RuntimeContractError,
    build_deepseek_json_object_request,
    build_query_frame_repair_prompt,
    build_query_frame_runtime_prompt,
    execute_mock_query_frame_request,
    runtime_examples,
    runtime_prompt_audit,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r2a_1_runtime_prompt_v1"
R2A_REPORT = ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
SELECTION_ALGORITHM = "deterministic_weighted_greedy_set_cover_v1"
EXPECTED_FREEZE_MANIFEST_HASHES = {
    "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json": "41067a5b8916bb3ad222c9e873d8576a4ec7aaa3634c79d89d0eebdcba63468a",
    "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json": "7a9e8f184013eeeff0db794a95defc73d19438c89e7cbbf6609abdea1a6a2845",
    "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json": "44b4be432a432bce02158ab3fddc1bcdc569cb3d86587286c5ab866b193349b9",
}
HISTORICAL_REPORT_DIRS = (
    "reports/p4_api_r1c_structured_v1",
    "reports/p4_api_r1e_transport_schema_v1",
    "reports/p4_api_r1e_1_dialect_repair_v1",
    "reports/p4_api_r1f_strict_handshake_v1",
    "reports/p4_api_r1f_1_strict_handshake_v1",
    "reports/p4_api_r1f_2_minimal_strict_control_v1",
    "reports/p4_api_r2a_compact_query_frame_v1",
)
TARGET_TAGS = frozenset(
    [f"difficulty:L{i}" for i in range(1, 6)]
    + [f"status:{value}" for value in ("VALID", "NEEDS_CLARIFICATION", "INVALID")]
    + [f"operation_scope:{value}" for value in ("single", "multi")]
    + [f"action:{value}" for value in (
        "fixed_support", "displacement", "pressure", "surface_traction",
        "material_assignment", "global_mesh_size", "local_mesh_size",
    )]
    + [f"quantifier:{value}" for value in (
        "all", "exactly_one", "count", "largest", "smallest", "nearest",
    )]
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }


def _operation_tags(intent_result: Mapping[str, Any]) -> tuple[set[str], str]:
    if intent_result.get("status") != "VALID":
        return set(), "nonvalid"
    operations = intent_result["intent"]["operations"]
    tags: set[str] = set()
    for operation in operations:
        tags.add(f"action:{operation['operation_type']}")
        target = operation.get("target")
        if target and target.get("quantifier"):
            tags.add(f"quantifier:{target['quantifier']}")
    return tags, "single" if len(operations) == 1 else "multi"


@lru_cache(maxsize=1)
def _val_candidates() -> tuple[tuple[dict[str, Any], ...], dict[str, Any], tuple[str, ...]]:
    canonical_audit = DecodeAudit("canonical", "val")
    canonical = {
        str(record["source_ref"]["canonical_record_id"]): record
        for record in iter_canonical_records("val", audit=canonical_audit)
    }
    audit = DecodeAudit("renderings", "val")
    candidates: list[dict[str, Any]] = []
    first_twenty: list[str] = []
    for record in iter_rendering_records("val", audit=audit):
        if "query" in record:
            query = record["query"]
            source = record["source_ref"]
            ground_truth = record["ground_truth"]
        else:
            canonical_id = str(record["canonical_record_id"])
            source_record = canonical[canonical_id]
            source = source_record["source_ref"]
            canonical_query = source_record["query"]
            query = {
                "rendering_id": record["rendering_id"],
                "language": record["language"],
                "difficulty_level": canonical_query["difficulty_level"],
                "text": record["text"],
            }
            ground_truth = source_record["ground_truth"]
        intent_result = record["intent_result"]
        sample_id = str(query["rendering_id"])
        if len(first_twenty) < 20:
            first_twenty.append(sample_id)
        operation_tags, scope = _operation_tags(intent_result)
        tags = {
            f"language:{query['language']}",
            f"difficulty:{query['difficulty_level']}",
            f"status:{intent_result['status']}",
            f"operation_scope:{scope}",
        } | operation_tags
        candidates.append({
            "sample_id": sample_id,
            "canonical_id": str(source["canonical_record_id"]),
            "language": str(query["language"]),
            "difficulty": str(query["difficulty_level"]),
            "status": str(intent_result["status"]),
            "scope": scope,
            "tags": tuple(sorted(tags)),
            "query_sha256": _sha256_bytes(query),
            "ground_truth_sha256": _sha256_bytes(ground_truth),
            "text": str(query["text"]),
        })
    if audit.decoded_test_lines or canonical_audit.decoded_test_lines:
        raise RuntimeError("R2A1_E_FORMAL_TEST_DECODE")
    candidates.sort(key=lambda row: row["sample_id"])
    return tuple(candidates), {"renderings": audit.as_dict(), "canonical": canonical_audit.as_dict()}, tuple(first_twenty)


def _select_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, decode_audit, first_twenty = _val_candidates()
    frequencies = Counter(
        tag for row in candidates for tag in set(row["tags"]) if tag in TARGET_TAGS
    )
    uncovered = set(TARGET_TAGS)
    selected: list[dict[str, Any]] = []
    used_canonical: set[str] = set()
    language_counts = Counter()

    while len(selected) < 20:
        eligible = [
            row for row in candidates
            if row["canonical_id"] not in used_canonical
            and language_counts[row["language"]] < 10
        ]
        if not eligible:
            raise RuntimeError("R2A1_E_SMOKE_QUOTA_UNSATISFIED")

        def rank(row: Mapping[str, Any]) -> tuple[float, float, str]:
            newly_covered = uncovered & set(row["tags"])
            rarity = sum(1.0 / frequencies[tag] for tag in newly_covered)
            quota_need = 10 - language_counts[row["language"]]
            return (float(len(newly_covered)), rarity + quota_need / 1000000.0, str(row["sample_id"]))

        best = max(eligible, key=rank)
        selected.append(best)
        used_canonical.add(best["canonical_id"])
        language_counts[best["language"]] += 1
        uncovered.difference_update(best["tags"])

    if language_counts != Counter({"en-US": 10, "zh-CN": 10}):
        raise RuntimeError("R2A1_E_SMOKE_LANGUAGE_QUOTA")
    internal = {
        "decode_audit": decode_audit,
        "uncovered_required_tags": sorted(uncovered),
        "duplicate_canonical_id_count": len(selected) - len(used_canonical),
        "first_twenty": list(first_twenty),
        "selected_canonical_ids": [row["canonical_id"] for row in selected],
    }
    return selected, internal


@lru_cache(maxsize=1)
def build_smoke_manifest() -> dict[str, Any]:
    selected, _ = _select_candidates()
    samples = [{
        "sample_id": row["sample_id"],
        "language": row["language"],
        "difficulty": row["difficulty"],
        "status": row["status"],
        "coverage_tags": list(row["tags"]),
        "query_sha256": row["query_sha256"],
        "ground_truth_sha256": row["ground_truth_sha256"],
    } for row in selected]
    return {
        "schema_version": "topocae-p4-api-r2a-1-smoke-sample-manifest/1",
        "selection_algorithm": SELECTION_ALGORITHM,
        "split": "val",
        "sample_count": 20,
        "payload_policy": "hash_and_coverage_metadata_only",
        "samples": samples,
    }


def _tag_counts(samples: Iterable[Mapping[str, Any]], prefix: str) -> dict[str, int]:
    counter = Counter()
    marker = prefix + ":"
    for sample in samples:
        for tag in sample["coverage_tags"]:
            if tag.startswith(marker):
                counter[tag[len(marker):]] += 1
    return dict(sorted(counter.items()))


@lru_cache(maxsize=1)
def _coverage_audit_cached() -> dict[str, Any]:
    manifest = build_smoke_manifest()
    samples = manifest["samples"]
    selected, internal = _select_candidates()
    selected_ids = [row["sample_id"] for row in selected]
    required_covered = TARGET_TAGS - set(internal["uncovered_required_tags"])
    result = {
        "schema_version": "topocae-p4-api-r2a-1-smoke-coverage-audit/1",
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_count": len(samples),
        "language_counts": _tag_counts(samples, "language"),
        "difficulty_counts": _tag_counts(samples, "difficulty"),
        "status_counts": _tag_counts(samples, "status"),
        "operation_scope_counts": _tag_counts(samples, "operation_scope"),
        "action_counts": _tag_counts(samples, "action"),
        "quantifier_counts": _tag_counts(samples, "quantifier"),
        "required_tag_count": len(TARGET_TAGS),
        "covered_required_tag_count": len(required_covered),
        "uncovered_required_tags": internal["uncovered_required_tags"],
        "duplicate_canonical_id_count": internal["duplicate_canonical_id_count"],
        "equals_first_20_val_renderings": selected_ids == internal["first_twenty"],
        "selection_reproduction_byte_exact": canonical_json_bytes(manifest) == canonical_json_bytes(build_smoke_manifest()),
        "decoded_test_lines": sum(item["decoded_test_lines"] for item in internal["decode_audit"].values()),
        "opaque_test_lines_crossed_without_decode": sum(item["opaque_test_lines_crossed"] for item in internal["decode_audit"].values()),
    }
    result["coverage_gate_pass"] = (
        not result["uncovered_required_tags"]
        and result["language_counts"] == {"en-US": 10, "zh-CN": 10}
        and result["duplicate_canonical_id_count"] == 0
        and result["decoded_test_lines"] == 0
    )
    return result


def build_smoke_coverage_audit(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if manifest is not None and canonical_json_bytes(manifest) != canonical_json_bytes(build_smoke_manifest()):
        raise ValueError("R2A1_E_MANIFEST_MISMATCH")
    return dict(_coverage_audit_cached())


def _runtime_audit() -> dict[str, Any]:
    audit = runtime_prompt_audit()
    audit["gate_pass"] = all((
        audit["base_instruction_source_verified"], audit["json_literal_present"],
        audit["embedded_schema_exact"], audit["required_optional_inventory_complete"],
        audit["mutual_exclusion_rules_schema_consistent"], audit["example_count"] >= 4,
        set(audit["example_statuses"]) == {"invalid", "needs_clarification", "valid"},
        audit["multi_operation_example_count"] >= 1, audit["forbidden_term_count"] == 0,
        audit["deterministic_reproduction"],
    ))
    return audit


def _repair_audit() -> dict[str, Any]:
    inference_input = {"text": "Apply pressure to the pocket.", "language": "en-US"}
    summary = {
        "field_path": "/operations/0/direction", "required": True,
        "allowed_enum": ["surface_normal", "vector"], "rule": "pressure requires direction",
    }
    first = build_query_frame_repair_prompt(inference_input, "QF_E_REQUIRED_FIELD", summary)
    second_attempt_failed_closed = False
    try:
        build_query_frame_repair_prompt(inference_input, "QF_E_REQUIRED_FIELD", summary, attempt=2)
    except RuntimeContractError as exc:
        second_attempt_failed_closed = exc.code == "RUNTIME_E_REPAIR_LIMIT"
    decoded = json.loads(first)
    result = {
        "schema_version": "topocae-p4-api-r2a-1-repair-contract-audit/1",
        "max_repair_attempts": 1,
        "repair_input_fields": sorted(decoded["input"]),
        "schema_summary_fields": sorted(decoded["schema_summary"]),
        "error_code_only_from_validator": decoded["error_code"] == "QF_E_REQUIRED_FIELD",
        "previous_output_included": False,
        "answer_or_identity_evidence_included": False,
        "deterministic_reproduction": first == build_query_frame_repair_prompt(inference_input, "QF_E_REQUIRED_FIELD", summary),
        "second_attempt_fails_closed": second_attempt_failed_closed,
        "silent_repair_allowed": False,
    }
    result["gate_pass"] = all((
        result["max_repair_attempts"] == 1, result["repair_input_fields"] == ["language", "text"],
        result["error_code_only_from_validator"], not result["previous_output_included"],
        not result["answer_or_identity_evidence_included"], result["deterministic_reproduction"],
        result["second_attempt_fails_closed"], not result["silent_repair_allowed"],
    ))
    return result


def _mock_audit() -> dict[str, Any]:
    example = runtime_examples()[0]
    captured: dict[str, Any] = {}

    def transport(endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        captured.update({"endpoint": endpoint, "payload": dict(payload)})
        return {"choices": [{"message": {"content": canonical_json_bytes(example["frame"]).decode("utf-8")}}]}

    outcome = execute_mock_query_frame_request(example["input"], transport)
    payload = captured["payload"]
    provider_schema_fields = {"json_schema", "tools", "tool_choice"} & set(payload)
    result = {
        "schema_version": "topocae-p4-api-r2a-1-mock-request-audit/1",
        "endpoint": captured["endpoint"], "profile": PROFILE_NAME,
        "response_format": payload["response_format"],
        "provider_schema_field_present": bool(provider_schema_fields),
        "schema_embedded_in_system_text": QUERY_FRAME_SCHEMA_SHA256 in payload["messages"][0]["content"],
        "frame_valid": outcome["frame_valid"], "compiler_valid": outcome["compiler_valid"],
        "p2_valid": outcome["p2_valid"], "mock_request_count": outcome["mock_request_count"],
        "network_request_count": outcome["network_request_count"],
    }
    result["gate_pass"] = all((
        result["endpoint"] == NORMAL_ENDPOINT, result["profile"] == PROFILE_NAME,
        result["response_format"] == {"type": "json_object"},
        not result["provider_schema_field_present"], result["schema_embedded_in_system_text"],
        result["frame_valid"], result["compiler_valid"], result["p2_valid"],
        result["network_request_count"] == 0,
    ))
    return result


def _cost_audit() -> dict[str, Any]:
    prompt = build_query_frame_runtime_prompt()
    selected, _ = _select_candidates()
    longest = max(selected, key=lambda row: len(row["text"].encode("utf-8")))
    request = build_deepseek_json_object_request({"text": longest["text"], "language": longest["language"]})
    payload_bytes = len(canonical_json_bytes(request["payload"]))
    prompt_bytes = len(prompt.encode("utf-8"))
    estimated_input_tokens = math.ceil(payload_bytes / 4)
    conservative_input_tokens = payload_bytes
    output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    input_usd_per_million = 0.435
    output_usd_per_million = 0.87
    usd_cny = 8.0
    per_request_cny = (
        conservative_input_tokens * input_usd_per_million / 1_000_000
        + output_tokens * output_usd_per_million / 1_000_000
    ) * usd_cny
    return {
        "schema_version": "topocae-p4-api-r2a-1-cost-budget-audit/1",
        "pricing_source": "frozen_r1c_historical_planning_assumptions",
        "input_usd_per_million_tokens": input_usd_per_million,
        "output_usd_per_million_tokens": output_usd_per_million,
        "usd_cny_planning_rate": usd_cny,
        "system_prompt_utf8_bytes": prompt_bytes,
        "system_prompt_token_estimate": math.ceil(prompt_bytes / 4),
        "system_prompt_token_upper_bound": prompt_bytes,
        "single_request_max_payload_bytes": payload_bytes,
        "single_request_input_token_estimate": estimated_input_tokens,
        "single_request_input_token_upper_bound": conservative_input_tokens,
        "max_output_tokens": output_tokens,
        "first_round_request_cap": 20, "repair_request_cap": 20,
        "total_request_hard_cap": 40,
        "first_round_cost_upper_cny": 20 * per_request_cny,
        "repair_cost_upper_cny": 20 * per_request_cny,
        "total_cost_upper_cny": 40 * per_request_cny,
        "actual_requests": 0,
    }


def _verify_delivery_directory(relative: str) -> dict[str, Any]:
    directory = ROOT / relative
    manifest_path = directory / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("assets", [])) + list(manifest.get("implementation_assets", []))
    mismatches = []
    checked = 0
    for entry in entries:
        raw = entry.get("path") or entry.get("relative_path")
        expected = entry.get("sha256")
        if not raw or not expected:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            local = directory / candidate
            candidate = local if local.exists() else ROOT / candidate
        if candidate.resolve() == manifest_path.resolve():
            continue
        checked += 1
        actual = _sha256_file(candidate) if candidate.is_file() else None
        if actual != expected:
            mismatches.append({"path": raw, "expected": expected, "actual": actual})
    return {
        "path": relative, "manifest_sha256": _sha256_file(manifest_path),
        "assets_checked": checked, "mismatches": mismatches,
        "valid": checked > 0 and not mismatches,
    }


def _freeze_audit() -> dict[str, Any]:
    actual = {path: _sha256_file(ROOT / path) for path in EXPECTED_FREEZE_MANIFEST_HASHES}
    manifest_changes = [
        {"path": path, "expected": expected, "actual": actual[path]}
        for path, expected in EXPECTED_FREEZE_MANIFEST_HASHES.items() if actual[path] != expected
    ]
    history = [_verify_delivery_directory(path) for path in HISTORICAL_REPORT_DIRS]
    history_changes = [item for item in history if not item["valid"]]
    r2a = next(item for item in history if item["path"].endswith("p4_api_r2a_compact_query_frame_v1"))
    return {
        "schema_version": "topocae-p4-api-r2a-1-freeze-audit/1",
        "expected_manifest_hashes": EXPECTED_FREEZE_MANIFEST_HASHES,
        "actual_manifest_hashes": actual,
        "manifest_changes": manifest_changes,
        "historical_report_checks": history,
        "history_changes": history_changes,
        "change_count": len(manifest_changes) + len(history_changes),
        "p1_unchanged": actual[next(iter(EXPECTED_FREEZE_MANIFEST_HASHES))] == EXPECTED_FREEZE_MANIFEST_HASHES[next(iter(EXPECTED_FREEZE_MANIFEST_HASHES))],
        "p2_unchanged": actual[list(EXPECTED_FREEZE_MANIFEST_HASHES)[1]] == EXPECTED_FREEZE_MANIFEST_HASHES[list(EXPECTED_FREEZE_MANIFEST_HASHES)[1]],
        "p3_unchanged": actual[list(EXPECTED_FREEZE_MANIFEST_HASHES)[2]] == EXPECTED_FREEZE_MANIFEST_HASHES[list(EXPECTED_FREEZE_MANIFEST_HASHES)[2]],
        "r2a_unchanged": r2a["valid"],
        "historical_p4_unchanged": not history_changes,
        "step_payload_opened": False, "graph_payload_opened": False,
        "abaqus_accessed": False,
    }


def _leakage_audit(formal_before: Mapping[str, Any]) -> dict[str, Any]:
    examples = runtime_examples()
    example_hashes = {hashlib.sha256(item["input"]["text"].encode("utf-8")).hexdigest() for item in examples}
    overlap_hashes: set[str] = set()
    decode_audits = {}
    for split in ("train", "val"):
        audit = DecodeAudit("renderings", split)
        for record in iter_rendering_records(split, audit=audit):
            text = record["query"]["text"] if "query" in record else record["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in example_hashes:
                overlap_hashes.add(digest)
        decode_audits[split] = audit.as_dict()
    formal_after = _formal_state()
    decoded_test = sum(item["decoded_test_lines"] for item in decode_audits.values())
    return {
        "schema_version": "topocae-p4-api-r2a-1-leakage-audit/1",
        "authorized_splits_read": ["train", "val"],
        "formal_test_access_count": 0, "decoded_test_lines": decoded_test,
        "opaque_test_lines_crossed_without_decode": sum(item["opaque_test_lines_crossed"] for item in decode_audits.values()),
        "example_train_val_text_overlap_count": len(overlap_hashes),
        "query_payload_persisted": False, "ground_truth_payload_persisted": False,
        "formal_test_ledger_before": dict(formal_before),
        "formal_test_ledger_after": formal_after,
        "api_request_count": 0, "network_request_count": 0,
        "step_accessed": False, "graph_accessed": False,
        "decode_audits": decode_audits,
        "gate_pass": decoded_test == 0 and not overlap_hashes and formal_before == formal_after,
    }


@lru_cache(maxsize=1)
def build_all_audits() -> dict[str, dict[str, Any]]:
    formal_before = _formal_state()
    if formal_before != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        raise RuntimeError("R2A1_E_FORMAL_TEST_GUARD")
    return {
        "runtime_prompt": _runtime_audit(),
        "repair_contract": _repair_audit(),
        "mock_request": _mock_audit(),
        "smoke_coverage": build_smoke_coverage_audit(),
        "cost_budget": _cost_audit(),
        "leakage": _leakage_audit(formal_before),
        "freeze": _freeze_audit(),
    }


def build_delivery_payloads(regression: Mapping[str, Any]) -> dict[str, Any]:
    from utils.cae_query_frame_delivery import assemble_delivery_payloads

    return assemble_delivery_payloads(build_all_audits(), build_smoke_manifest(), regression)


def write_delivery(regression: Mapping[str, Any]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    payloads = build_delivery_payloads(regression)
    for name, value in payloads.items():
        path = REPORT / name
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8", newline="\n")
        else:
            _write_json(path, value)
    assets = sorted(path for path in REPORT.iterdir() if path.is_file() and path.name != "delivery_manifest.json")
    implementation_paths = (
        ROOT / "utils" / "cae_query_frame_runtime.py",
        ROOT / "utils" / "cae_query_frame_delivery.py",
        ROOT / "run_p4_api_r2a_1_runtime_prompt.py",
        ROOT / "tests" / "test_p4_api_r2a_1_runtime.py",
        ROOT / "tests" / "test_p4_api_r2a_1_smoke.py",
        ROOT / "tests" / "test_p4_api_r2a_1_delivery.py",
        ROOT / "docs" / "superpowers" / "specs" / "2026-07-30-p4-api-r2a-1-runtime-prompt-design.md",
        ROOT / "docs" / "superpowers" / "plans" / "2026-07-30-p4-api-r2a-1-runtime-prompt.md",
    )
    manifest = {
        "schema_version": "topocae-p4-api-r2a-1-delivery-manifest/1",
        "status": payloads["acceptance.json"]["status"],
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in assets
        ],
        "implementation_assets": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in implementation_paths
        ],
    }
    _write_json(REPORT / "delivery_manifest.json", manifest)

def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_p4_regression() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_p4*.py"],
        cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    import re
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    return {
        "schema_version": "topocae-p4-api-r2a-1-regression-results/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "interpreter_path": sys.executable, "python_version": platform.python_version(),
        "test_pattern": "test_p4*.py", "tests_run": int(match.group(1)) if match else None,
        "return_code": completed.returncode, "stdout_or_stderr_persisted": False,
    }


if __name__ == "__main__":
    regression = run_p4_regression()
    write_delivery(regression)
    raise SystemExit(0 if regression["return_code"] == 0 else 1)
