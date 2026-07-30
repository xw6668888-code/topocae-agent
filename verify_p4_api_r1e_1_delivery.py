"""Independently verify the P4-API-R1E.1 offline delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from run_p4_api_r1e_transport_schema import FROZEN_HASHES, formal_test_state
from utils.cae_language_grounding.deepseek_strict_transport import (
    STRICT_BASE_URL,
    build_mock_tool_payload,
    build_transport_schema,
    validate_deepseek_strict_subset_schema,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1e_1_dialect_repair_v1"
REQUIRED = {
    "transport_schema.json",
    "dialect_contract.json",
    "strict_subset_validation.json",
    "canonical_roundtrip_audit.json",
    "size_budget_audit.json",
    "regression_results.json",
    "security_audit.json",
    "acceptance.json",
    "final_acceptance.zh-CN.md",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R1E_1.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_bytes(value: Any) -> int:
    return len(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def json_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_keys(child)


def main() -> None:
    files = {path.name for path in REPORT.iterdir() if path.is_file()}
    assert REQUIRED <= files

    manifest = json.loads((REPORT / "delivery_manifest.json").read_text(encoding="utf-8"))
    assets = {row["path"]: row for row in manifest["assets"]}
    assert set(assets) == files - {"delivery_manifest.json"}
    for name, row in assets.items():
        path = REPORT / name
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == sha256(path)

    schema = json.loads((REPORT / "transport_schema.json").read_text(encoding="utf-8"))
    assert schema == build_transport_schema()
    payload = build_mock_tool_payload(
        base_url=STRICT_BASE_URL,
        model="offline-verifier",
        messages=[],
    )
    assert payload["tools"][0]["function"]["parameters"] == schema
    dictionaries = [node for node in walk(schema) if isinstance(node, Mapping)]
    assert "$def" in schema
    assert all("$defs" not in node for node in dictionaries)
    assert all("title" not in node for node in dictionaries)
    refs = [node["$ref"] for node in dictionaries if "$ref" in node]
    assert refs and all(ref.startswith("#/$def/") for ref in refs)
    subset = validate_deepseek_strict_subset_schema(schema)
    assert subset["valid"] and not subset["errors"]

    strict = json.loads((REPORT / "strict_subset_validation.json").read_text(encoding="utf-8"))
    roundtrip = json.loads((REPORT / "canonical_roundtrip_audit.json").read_text(encoding="utf-8"))
    size = json.loads((REPORT / "size_budget_audit.json").read_text(encoding="utf-8"))
    regression = json.loads((REPORT / "regression_results.json").read_text(encoding="utf-8"))
    security = json.loads((REPORT / "security_audit.json").read_text(encoding="utf-8"))
    acceptance = json.loads((REPORT / "acceptance.json").read_text(encoding="utf-8"))
    dialect = json.loads((REPORT / "dialect_contract.json").read_text(encoding="utf-8"))

    assert strict["status"] == "PASS"
    assert strict["dialect_checks"]["unsupported_keyword_count"] == 0
    assert dialect["local_jsonschema_validation_copy"]["provider_schema_mutated"] is False
    assert dialect["local_jsonschema_validation_copy"]["delivery_schema_mutated"] is False
    assert roundtrip["total_records"] == 21885
    assert roundtrip["strict_instance_valid_count"] == 21885
    assert roundtrip["canonical_validation_pass_count"] == 21885
    assert roundtrip["semantic_exact_count"] == 21885
    assert roundtrip["strict_instance_valid_rate"] == 1.0
    assert roundtrip["canonical_validation_rate"] == 1.0
    assert roundtrip["semantic_exact_rate"] == 1.0
    assert roundtrip["test_payload_decoded"] is False
    assert all(audit["decoded_test_lines"] == 0 for audit in roundtrip["decode_audits"].values())

    assert size["record_count"] == 21885
    assert size["compact_schema_bytes"] == compact_bytes(schema)
    assert size["transport_output_bytes"] == {
        "p50": 1124,
        "p95": 1608,
        "max": 1669,
        "percentile_method": "nearest-rank",
    }
    assert size["output_expansion_ratio"] > 1.0
    assert regression["status"] == "PASS"
    assert regression["tests_run"] == 71
    assert regression["return_code"] == 0

    assert security["backend_request_count"] == 0
    assert security["api_key_access_count"] == 0
    assert security["formal_test_access_count"] == 0
    assert security["real_api_called"] is False
    assert security["p5_started"] is False
    assert security["step_accessed"] is False
    assert security["abaqus_accessed"] is False
    assert formal_test_state() == {
        "run_count": 0,
        "content_accessed": False,
        "unlock_authorized": False,
    }
    assert acceptance["status"] == "READY_FOR_P4_API_R1E_1_OFFLINE_SUPERVISOR_REVIEW"
    assert all(acceptance["checks"].values())
    assert {
        name: sha256(ROOT / name) for name in FROZEN_HASHES
    } == FROZEN_HASHES

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    assert re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text) is None
    for path in REPORT.glob("*.json"):
        keys = set(json_keys(json.loads(path.read_text(encoding="utf-8"))))
        assert "query" not in keys
        assert "ground_truth" not in keys

    print(json.dumps({
        "status": "PASS",
        "required_files": len(REQUIRED),
        "manifest_assets": len(assets),
        "records": roundtrip["total_records"],
        "p4_tests": regression["tests_run"],
        "compact_schema_bytes": size["compact_schema_bytes"],
        "backend_requests": 0,
        "api_key_accesses": 0,
        "formal_test_accesses": 0,
    }))


if __name__ == "__main__":
    main()

