"""Independent verifier for the P4-API-R1F strict handshake delivery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from p4_api_r0_1_audit import verify_full_freeze


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_strict_handshake_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
EXPECTED_FILES = {
    "preflight.json", "runtime_identity.json", "schema_handshake.json",
    "canonical_validation.json", "cache_audit.json", "cost_latency_audit.json",
    "security_audit.json", "acceptance.json", "final_acceptance.zh-CN.md",
    "delivery_manifest.json", "CODEX_CONTEXT_P4_API_R1F.md",
    "request_counter.json",
}
FROZEN_HASHES = {
    "reports/p4_api_r1e_transport_schema_v1/transport_schema.json": "55b26f9a55b6276628dd648a77c3eb09eba06f3bd268bb99b5e176892c6bf162",
    "utils/cae_language_grounding/deepseek_strict_transport.py": "3453b190fad7e58c661b328465da5e5edfd9f4ad0ed7e60ce2cc12849c21c65a",
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    "prompts/p4_a1_structured_llm_v1.md": "e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da",
    "utils/cae_intent.py": "163ae1189a81f607187e195c02b8444b716651393813aa3b2ec9c27c2cfb9c8e",
    "utils/cae_language_grounding/grounder.py": "b2b7a80240452eb0cadb04c1f189e2f2e8587ab76874907b61d53ebd6836ded9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


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

    preflight = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
    identity = json.loads((REPORT / "runtime_identity.json").read_text(encoding="utf-8"))
    handshake = json.loads((REPORT / "schema_handshake.json").read_text(encoding="utf-8"))
    canonical = json.loads((REPORT / "canonical_validation.json").read_text(encoding="utf-8"))
    cache = json.loads((REPORT / "cache_audit.json").read_text(encoding="utf-8"))
    cost = json.loads((REPORT / "cost_latency_audit.json").read_text(encoding="utf-8"))
    security = json.loads((REPORT / "security_audit.json").read_text(encoding="utf-8"))
    acceptance = json.loads((REPORT / "acceptance.json").read_text(encoding="utf-8"))
    counter = json.loads((REPORT / "request_counter.json").read_text(encoding="utf-8"))

    assert preflight["status"] == "PASS_BEFORE_API"
    assert preflight["provider_profile"] == "deepseek_strict_tool_beta"
    assert preflight["base_url"] == "https://api.deepseek.com/beta"
    assert preflight["endpoint"] == "https://api.deepseek.com/beta/chat/completions"
    assert preflight["model"] == "deepseek-v4-pro"
    assert preflight["budget"]["worst_case_cost_cny"] < 1.0
    assert preflight["schema_retry_count"] == 0
    assert preflight["http_retry_count"] == 0
    assert preflight["sample"]["split"] == "val"
    assert preflight["sample"]["query_written"] is False
    assert preflight["sample"]["ground_truth_written"] is False

    assert identity["provider_profile"] == "deepseek_strict_tool_beta"
    assert identity["actual_model"] is None
    assert handshake["status"] == "REJECTED"
    assert handshake["server_schema_accepted"] is False
    assert handshake["http_status"] == 400
    assert handshake["error_category"] == "STRICT_SCHEMA_REJECTED"
    assert handshake["backend_request_count"] == 1
    assert handshake["schema_parameter_path"] is None or isinstance(
        handshake["schema_parameter_path"], str
    )
    assert handshake["raw_error_persisted"] is False
    assert handshake["raw_response_persisted"] is False
    assert handshake["schema_mutation_retry_attempted"] is False
    assert handshake["defs_to_def_retry_attempted"] is False
    assert canonical["status"] == "NOT_RUN"
    assert cache["status"] == "NOT_RUN_AFTER_REJECTION"
    assert cache["total_backend_request_count"] == 1
    assert cost["backend_requests"] == 1
    assert cost["backend_request_cap"] == 1
    assert cost["usage_available"] is False
    assert cost["accounted_cost_cny_conservative"] == 0.0
    assert cost["schema_retry_count"] == 0
    assert cost["http_retry_count"] == 0
    assert counter["backend_request_count"] == counter["backend_request_cap"] == 1

    assert acceptance["status"] == "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE"
    assert acceptance["backend_request_count"] == 1
    assert acceptance["frozen_hash_changes"] == 0
    for name in (
        "provider_fallback", "endpoint_fallback", "model_fallback",
        "protocol_fallback", "content_fallback", "later_stage_started",
    ):
        assert acceptance[name] is False
    assert security["api_key_written"] is False
    assert security["api_key_hashed"] is False
    assert security["authorization_header_written"] is False
    assert security["raw_error_response_written"] is False
    assert security["query_written"] is False
    assert security["ground_truth_written"] is False
    assert security["secret_leak_count"] == 0

    assert {name: sha256(ROOT / name) for name in FROZEN_HASHES} == FROZEN_HASHES
    assert verify_full_freeze()["status"] == "PASS"
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in ledger["runs"].values()) == 0
    assert ledger["test_content_accessed"] is False
    assert ledger["test_unlock_authorized"] is False

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    assert not api_key or api_key not in delivery_text
    assert re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text) is None
    for path in REPORT.glob("*.json"):
        keys = set(walk_keys(json.loads(path.read_text(encoding="utf-8"))))
        assert "query" not in keys
        assert "ground_truth" not in keys

    print(json.dumps({
        "status": "PASS",
        "acceptance": acceptance["status"],
        "files": len(files),
        "manifest_assets": len(assets),
        "backend_requests": counter["backend_request_count"],
        "http_status": handshake["http_status"],
        "error_category": handshake["error_category"],
    }))


if __name__ == "__main__":
    main()
