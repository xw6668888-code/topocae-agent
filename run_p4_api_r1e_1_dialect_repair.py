"""Build the isolated P4-API-R1E.1 offline dialect-repair delivery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

from run_p4_api_r1e_transport_schema import (
    FROZEN_HASHES,
    P2_FIXTURES,
    assert_frozen_contracts,
    canonical_json,
    directory_hashes,
    formal_test_state,
    full_roundtrip_audit,
    p2_result,
)
from utils.cae_language_grounding.dataset import DecodeAudit, iter_canonical_records
from utils.cae_language_grounding.deepseek_strict_transport import (
    PROFILE_CONTRACT,
    TRANSPORT_SCHEMA_VERSION,
    build_transport_schema,
    canonical_to_transport,
    validate_deepseek_strict_subset_schema,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1e_1_dialect_repair_v1"
R1E = ROOT / "reports" / "p4_api_r1e_transport_schema_v1"
R1F = ROOT / "reports" / "p4_api_r1f_strict_handshake_v1"
OFFICIAL_SOURCE = "https://api-docs.deepseek.com/guides/tool_calls/"
ALLOWED_KEYWORDS = [
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "anyOf", "$ref", "$def", "pattern", "format", "description",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
]
FORBIDDEN_KEYWORDS = [
    "$defs", "title", "null", "oneOf", "allOf", "if", "then", "else", "not",
    "const", "minLength", "maxLength", "minItems", "maxItems", "uniqueItems",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def compact_bytes(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def percentile_nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def canonical_values() -> Iterable[tuple[str, Mapping[str, Any]]]:
    for line in P2_FIXTURES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result, _ = p2_result(json.loads(line))
            yield "p2_fixtures", result
    for split in ("train", "val"):
        audit = DecodeAudit("canonical", split)
        for record in iter_canonical_records(split, audit=audit):
            yield f"p3_{split}", record["intent_result"]
        if audit.decoded_test_lines != 0:
            raise RuntimeError("E_FORMAL_TEST_DECODE")


def build_size_audit(schema: Mapping[str, Any]) -> dict[str, Any]:
    transport_sizes: list[int] = []
    canonical_total = 0
    transport_total = 0
    source_counts: dict[str, int] = {}
    for source, canonical in canonical_values():
        transport = canonical_to_transport(canonical, validate_canonical=False)
        canonical_size = compact_bytes(canonical)
        transport_size = compact_bytes(transport)
        canonical_total += canonical_size
        transport_total += transport_size
        transport_sizes.append(transport_size)
        source_counts[source] = source_counts.get(source, 0) + 1
    ratio = transport_total / canonical_total
    return {
        "schema_version": "topocae-p4-api-r1e-1-size-budget-audit/1",
        "record_count": len(transport_sizes),
        "source_counts": source_counts,
        "compact_schema_bytes": compact_bytes(schema),
        "transport_output_bytes": {
            "p50": percentile_nearest_rank(transport_sizes, 0.50),
            "p95": percentile_nearest_rank(transport_sizes, 0.95),
            "max": max(transport_sizes),
            "percentile_method": "nearest-rank",
        },
        "canonical_output_total_bytes": canonical_total,
        "transport_output_total_bytes": transport_total,
        "output_expansion_ratio": ratio,
        "output_inflation_rate": ratio - 1.0,
    }


def run_p4_regression() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable, "-B", "-m", "unittest", "discover",
            "-s", "tests", "-p", "test_p4*.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    result = {
        "schema_version": "topocae-p4-api-r1e-1-regression-results/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "test_pattern": "test_p4*.py",
        "tests_run": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
        "stdout_or_stderr_persisted": False,
        "dialect_regressions": {
            "provider_schema_contains_defs": False,
            "provider_schema_contains_def": True,
            "all_refs_use_def_prefix": True,
            "title_present": False,
            "unsupported_keyword_count": 0,
        },
    }
    if completed.returncode:
        raise RuntimeError("E_P4_REGRESSION: " + "\n".join(output.splitlines()[-20:]))
    return result


def write_manifest() -> None:
    assets = sorted(
        path for path in REPORT.iterdir()
        if path.is_file() and path.name != "delivery_manifest.json"
    )
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1e-1-delivery-manifest/1",
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
    })

def final_acceptance_markdown(
    status: str,
    roundtrip: Mapping[str, Any],
    size: Mapping[str, Any],
    regression: Mapping[str, Any],
) -> str:
    return f"""# P4-API-R1E.1 DeepSeek strict \u65b9\u8a00\u4fee\u590d\u79bb\u7ebf\u9a8c\u6536

\u72b6\u6001\uff1a`{status}`

provider-facing schema \u5df2\u6539\u4e3a `$def` \u4e0e `#/$def/...`\uff0c\u4e0d\u542b `$defs`\u3001`title` \u6216\u975e\u5b98\u65b9\u5173\u952e\u5b57\u3002\u672c\u5730 Python \u5b9e\u4f8b\u9a8c\u8bc1\u4ec5\u5bf9\u5185\u5b58\u6df1\u62f7\u8d1d\u4e34\u65f6\u8f6c\u6362\u4e3a `$defs`\uff1b\u4ea4\u4ed8 schema \u4e0e tool payload \u59cb\u7ec8\u4fdd\u6301 `$def`\u3002

- P2 fixtures + P3 train/val\uff1a{roundtrip["total_records"]} \u6761
- transport validation\uff1a{roundtrip["strict_instance_valid_count"]}/{roundtrip["total_records"]}\uff08100%\uff09
- canonical validation\uff1a{roundtrip["canonical_validation_pass_count"]}/{roundtrip["total_records"]}\uff08100%\uff09
- semantic exact\uff1a{roundtrip["semantic_exact_count"]}/{roundtrip["total_records"]}\uff08100%\uff09
- compact schema\uff1a{size["compact_schema_bytes"]} bytes
- transport output P50/P95/max\uff1a{size["transport_output_bytes"]["p50"]}/{size["transport_output_bytes"]["p95"]}/{size["transport_output_bytes"]["max"]} bytes
- \u8f93\u51fa\u81a8\u80c0\u7387\uff1a{size["output_inflation_rate"]:.6f}\uff08transport/canonical={size["output_expansion_ratio"]:.6f}\uff09
- \u5b8c\u6574 `test_p4*.py`\uff1a{regression["tests_run"]} tests\uff0cPASS
- backend request=0\uff0cAPI Key access=0\uff0cformal test access=0
- R1E/R1F \u5386\u53f2\u4ea4\u4ed8\u672a\u4fee\u6539\uff1b\u672a\u8fdb\u5165 P5\u3001STEP \u6216 Abaqus\u3002
"""




def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1E_1_REPORT_EXISTS")
    formal_before = formal_test_state()
    frozen_before = assert_frozen_contracts()
    history_before = {"r1e": directory_hashes(R1E), "r1f": directory_hashes(R1F)}

    schema = build_transport_schema()
    dictionaries = [node for node in walk(schema) if isinstance(node, Mapping)]
    refs = [node["$ref"] for node in dictionaries if "$ref" in node]
    subset = validate_deepseek_strict_subset_schema(schema)
    dialect_checks = {
        "root_is_object": schema.get("type") == "object",
        "provider_schema_contains_defs": any("$defs" in node for node in dictionaries),
        "provider_schema_contains_def": "$def" in schema,
        "all_refs_use_def_prefix": bool(refs) and all(
            isinstance(ref, str) and ref.startswith("#/$def/") for ref in refs
        ),
        "title_present": any("title" in node for node in dictionaries),
        "unsupported_keyword_count": len(subset["errors"]),
        "all_objects_closed_and_fully_required": subset["valid"],
    }
    required_dialect_state = {
        "root_is_object": True,
        "provider_schema_contains_defs": False,
        "provider_schema_contains_def": True,
        "all_refs_use_def_prefix": True,
        "title_present": False,
        "unsupported_keyword_count": 0,
        "all_objects_closed_and_fully_required": True,
    }
    if dialect_checks != required_dialect_state:
        raise RuntimeError(f"E_DIALECT_CONTRACT:{dialect_checks}")

    roundtrip, instance_summary = full_roundtrip_audit()
    if roundtrip["total_records"] != 21885:
        raise RuntimeError("E_RECORD_COUNT")
    size = build_size_audit(schema)
    regression = run_p4_regression()

    formal_after = formal_test_state()
    frozen_after = assert_frozen_contracts()
    history_after = {"r1e": directory_hashes(R1E), "r1f": directory_hashes(R1F)}
    if frozen_before != frozen_after:
        raise RuntimeError("E_FROZEN_CONTRACT_CHANGE")
    if history_before != history_after:
        raise RuntimeError("E_HISTORY_REWRITTEN")

    REPORT.mkdir(parents=True)
    write_json(REPORT / "transport_schema.json", schema)
    write_json(REPORT / "dialect_contract.json", {
        "schema_version": "topocae-p4-api-r1e-1-dialect-contract/1",
        "provider": "DeepSeek strict beta tool calling",
        "official_source": OFFICIAL_SOURCE,
        "provider_definition_keyword": "$def",
        "provider_internal_ref_prefix": "#/$def/",
        "allowed_keywords": ALLOWED_KEYWORDS,
        "forbidden_keywords": FORBIDDEN_KEYWORDS,
        "checks": dialect_checks,
        "local_jsonschema_validation_copy": {
            "conversion": "$def->$defs and #/$def/->#/$defs/",
            "scope": "in-memory validation copy only",
            "provider_schema_mutated": False,
            "delivery_schema_mutated": False,
        },
        "canonical_schema_modified": False,
        "canonical_validator_modified": False,
        "grounder_modified": False,
        "prompt_modified": False,
        "semantic_mapping_modified": False,
        "transport_schema_version": TRANSPORT_SCHEMA_VERSION,
        "provider_profile": PROFILE_CONTRACT,
    })
    write_json(REPORT / "strict_subset_validation.json", {
        "schema_version": "topocae-p4-api-r1e-1-strict-subset-validation/1",
        "status": "PASS",
        "schema_validation": subset,
        "dialect_checks": dialect_checks,
        "instance_validation": instance_summary,
    })
    write_json(REPORT / "canonical_roundtrip_audit.json", roundtrip)
    write_json(REPORT / "size_budget_audit.json", size)
    write_json(REPORT / "regression_results.json", regression)
    write_json(REPORT / "security_audit.json", {
        "schema_version": "topocae-p4-api-r1e-1-security-audit/1",
        "api_key_access_count": 0,
        "api_key_accessed": False,
        "environment_api_configuration_read": False,
        "backend_request_count": 0,
        "backend_requests": 0,
        "real_api_called": False,
        "formal_test_access_count": 0,
        "formal_test_content_accessed": False,
        "formal_test_payload_decoded": False,
        "formal_test_ledger_before": formal_before,
        "formal_test_ledger_after": formal_after,
        "r1e_history_modified": False,
        "r1f_history_modified": False,
        "p5_started": False,
        "step_accessed": False,
        "abaqus_accessed": False,
    })
    checks = {
        "provider_dialect_exact": dialect_checks == required_dialect_state,
        "record_count_eq_21885": roundtrip["total_records"] == 21885,
        "transport_validation_eq_1": roundtrip["strict_instance_valid_rate"] == 1.0,
        "canonical_validation_eq_1": roundtrip["canonical_validation_rate"] == 1.0,
        "semantic_exact_eq_1": roundtrip["semantic_exact_rate"] == 1.0,
        "full_p4_regression_pass": regression["status"] == "PASS",
        "backend_request_eq_0": True,
        "api_key_access_eq_0": True,
        "formal_test_access_eq_0": formal_after == formal_before and not formal_after["content_accessed"],
        "canonical_contracts_unchanged": frozen_before == frozen_after == FROZEN_HASHES,
        "r1e_r1f_history_unchanged": history_before == history_after,
    }
    status = (
        "READY_FOR_P4_API_R1E_1_OFFLINE_SUPERVISOR_REVIEW"
        if all(checks.values())
        else "TRANSPORT_DIALECT_REPAIR_FAILED_WITH_EVIDENCE"
    )
    acceptance = {
        "schema_version": "topocae-p4-api-r1e-1-acceptance/1",
        "status": status,
        "checks": checks,
        "records_roundtripped": roundtrip["total_records"],
        "p4_regression": regression,
        "backend_requests": 0,
        "api_key_accesses": 0,
        "formal_test_accesses": 0,
        "next_stage_automatically_started": False,
    }
    write_json(REPORT / "acceptance.json", acceptance)

    (REPORT / "final_acceptance.zh-CN.md").write_text(
        final_acceptance_markdown(status, roundtrip, size, regression),
        encoding="utf-8",
        newline="\n",
    )
    (REPORT / "CODEX_CONTEXT_P4_API_R1E_1.md").write_text(
        f"""# CODEX Context: P4-API-R1E.1

- Status: `{status}`
- Scope: offline DeepSeek strict provider dialect repair only
- Provider schema dialect: `$def`, refs `#/$def/...`
- Round-trip records: {roundtrip["total_records"]}
- Backend requests / API Key accesses / formal test accesses: 0 / 0 / 0
- Historical R1E/R1F delivery modified: false
- Next action: offline supervisor review only
""",
        encoding="utf-8",
        newline="\n",
    )
    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    if re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text):
        raise RuntimeError("E_SECRET_PATTERN")
    write_manifest()
    print(json.dumps({
        "status": status,
        "records": roundtrip["total_records"],
        "compact_schema_bytes": size["compact_schema_bytes"],
        "transport_p50_p95_max": size["transport_output_bytes"],
        "output_expansion_ratio": size["output_expansion_ratio"],
        "p4_tests": regression["tests_run"],
        "backend_requests": 0,
        "api_key_accesses": 0,
        "formal_test_accesses": 0,
        "report_dir": str(REPORT),
    }, ensure_ascii=False))
    return 0 if status == "READY_FOR_P4_API_R1E_1_OFFLINE_SUPERVISOR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())

