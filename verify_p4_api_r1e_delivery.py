"""Independent verifier for the P4-API-R1E offline delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.deepseek_strict_transport import (
    PROFILE_NAME,
    STRICT_BASE_URL,
    recompute_r1c_combined_schema_metrics,
    validate_deepseek_strict_subset_schema,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1e_transport_schema_v1"
R1C = ROOT / "reports" / "p4_api_r1c_structured_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
EXPECTED_FILES = {
    "transport_schema.json",
    "transport_contract.json",
    "canonical_roundtrip_audit.json",
    "strict_subset_validation.json",
    "mock_transport_audit.json",
    "r1c_metric_consistency_audit.json",
    "security_audit.json",
    "acceptance.json",
    "final_acceptance.zh-CN.md",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R1E.md",
}
FROZEN_HASHES = {
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    "utils/cae_language_grounding/llm_adapter.py": "5c6137cefb44175e3f8818a59cedc87c585ab1adb761cabec521afe3e9dc6fe1",
    "utils/cae_language_grounding/llm_adapter_r0.py": "56e37a521316961f4e44682446ed0ef03a7255784f37e4afd3938a615f121710",
}
FORBIDDEN = {
    "oneOf", "allOf", "if", "then", "else", "not", "const",
    "minLength", "maxLength", "minItems", "maxItems", "uniqueItems",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_keys(child)


def walk_schema(node: Any, *, definitions: dict[str, Any], path: tuple[Any, ...] = ()) -> None:
    assert isinstance(node, dict), path
    assert not (set(node) & FORBIDDEN), (path, set(node) & FORBIDDEN)
    if node.get("type") == "object":
        properties = node.get("properties")
        assert isinstance(properties, dict), path
        assert set(node.get("required", [])) == set(properties), path
        assert node.get("additionalProperties") is False, path
        for name, child in properties.items():
            walk_schema(child, definitions=definitions, path=path + ("properties", name))
    if "items" in node:
        walk_schema(node["items"], definitions=definitions, path=path + ("items",))
    for index, child in enumerate(node.get("anyOf", [])):
        walk_schema(child, definitions=definitions, path=path + ("anyOf", index))
    if "$ref" in node:
        assert node["$ref"].startswith("#/$defs/"), (path, node["$ref"])
        assert node["$ref"].split("/", 2)[2] in definitions, (path, node["$ref"])
    if "$defs" in node:
        assert not path
        for name, child in node["$defs"].items():
            walk_schema(child, definitions=definitions, path=("$defs", name))


def main() -> None:
    files = {path.name for path in REPORT.iterdir() if path.is_file()}
    assert files == EXPECTED_FILES, (files, EXPECTED_FILES)

    manifest = json.loads((REPORT / "delivery_manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    assert len(assets) == len(EXPECTED_FILES) - 1
    assert {asset["path"] for asset in assets} == EXPECTED_FILES - {"delivery_manifest.json"}
    for asset in assets:
        path = REPORT / asset["path"]
        assert path.stat().st_size == asset["bytes"]
        assert sha256(path) == asset["sha256"]

    schema = json.loads((REPORT / "transport_schema.json").read_text(encoding="utf-8"))
    contract = json.loads((REPORT / "transport_contract.json").read_text(encoding="utf-8"))
    roundtrip = json.loads((REPORT / "canonical_roundtrip_audit.json").read_text(encoding="utf-8"))
    subset = json.loads((REPORT / "strict_subset_validation.json").read_text(encoding="utf-8"))
    mock = json.loads((REPORT / "mock_transport_audit.json").read_text(encoding="utf-8"))
    r1c_audit = json.loads((REPORT / "r1c_metric_consistency_audit.json").read_text(encoding="utf-8"))
    security = json.loads((REPORT / "security_audit.json").read_text(encoding="utf-8"))
    acceptance = json.loads((REPORT / "acceptance.json").read_text(encoding="utf-8"))

    independent_subset = validate_deepseek_strict_subset_schema(schema)
    assert independent_subset["valid"], independent_subset["errors"][:3]
    assert schema["type"] == "object"
    assert len(schema["properties"]["result"]["anyOf"]) == 3
    walk_schema(schema, definitions=schema["$defs"])

    profile = contract["provider_profile"]
    assert profile["name"] == PROFILE_NAME
    assert profile["base_url"] == STRICT_BASE_URL
    assert profile["request_mode"] == "tools/function/strict"
    assert profile["function_strict"] is True
    assert profile["tool_choice"] == "forced_named_function"
    assert profile["endpoint_fallback"] is False
    assert profile["provider_fallback"] is False
    assert profile["model_fallback"] is False
    assert contract["mapping"]["engineering_sentinel_values_used"] is False

    assert roundtrip["total_records"] == 21885
    assert roundtrip["semantic_exact_count"] == 21885
    assert roundtrip["strict_instance_valid_count"] == 21885
    assert roundtrip["canonical_validation_pass_count"] == 21885
    assert roundtrip["decode_audits"]["train"]["decoded_test_lines"] == 0
    assert roundtrip["decode_audits"]["val"]["decoded_test_lines"] == 0
    assert subset["status"] == "PASS"
    assert mock["status"] == "PASS" and mock["passed"] == mock["total"] == 11

    train = json.loads((R1C / "train_metrics.json").read_text(encoding="utf-8"))
    val = json.loads((R1C / "val_metrics.json").read_text(encoding="utf-8"))
    combined = json.loads((R1C / "combined_metrics.json").read_text(encoding="utf-8"))
    corrected = recompute_r1c_combined_schema_metrics(train, val)
    assert corrected == r1c_audit["corrected"]
    assert corrected["schema_valid_rate_after_retry"] == 0.89
    assert combined["language_cae_ir"]["schema_valid_rate_after_retry"] == 1.0
    assert r1c_audit["historical_report_rewritten"] is False

    assert {name: sha256(ROOT / name) for name in FROZEN_HASHES} == FROZEN_HASHES
    assert verify_full_freeze()["status"] == "PASS"
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in ledger["runs"].values()) == 0
    assert ledger["test_content_accessed"] is False
    assert ledger["test_unlock_authorized"] is False

    assert security["api_key_accessed"] is False
    assert security["environment_api_configuration_read"] is False
    assert security["real_client_created"] is False
    assert security["backend_requests"] == 0
    assert security["secret_leak_count"] == 0
    assert acceptance["status"] == "READY_FOR_P4_API_R1E_OFFLINE_SUPERVISOR_REVIEW"
    assert all(acceptance["checks"].values())
    assert acceptance["p4_regression"]["tests_run"] == 62

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
        "files": len(files),
        "manifest_assets": len(assets),
        "records": roundtrip["total_records"],
        "p4_tests": acceptance["p4_regression"]["tests_run"],
        "backend_requests": 0,
    }))


if __name__ == "__main__":
    main()
