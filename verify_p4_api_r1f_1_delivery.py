"""Independently verify the P4-API-R1F.1 single-handshake delivery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_1_strict_handshake_v1"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
ALLOWED_STATUSES = {
    "READY_FOR_P4_API_R1F_1_FIVE_PAIR_SMOKE_REVIEW",
    "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE",
    "BLOCKED_WITH_EVIDENCE",
}
REQUIRED_FILES = {
    "final_acceptance.zh-CN.md",
    "acceptance.json",
    "schema_handshake.json",
    "runtime_identity.json",
    "security_audit.json",
    "freeze_audit.json",
    "formal_test_ledger_audit.json",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R1F_1.md",
    "preflight.json",
    "canonical_validation.json",
    "cache_audit.json",
    "cost_latency_audit.json",
    "request_counter.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)


def load(name: str) -> dict[str, Any]:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def main() -> None:
    assert REPORT.is_dir(), "R1F.1 report directory is absent"
    files = {path.name for path in REPORT.iterdir() if path.is_file()}
    assert REQUIRED_FILES <= files

    manifest = load("delivery_manifest.json")
    assets = manifest["assets"]
    assert {row["path"] for row in assets} == files - {"delivery_manifest.json"}
    for row in assets:
        path = REPORT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]

    acceptance = load("acceptance.json")
    preflight = load("preflight.json")
    handshake = load("schema_handshake.json")
    identity = load("runtime_identity.json")
    canonical = load("canonical_validation.json")
    cache = load("cache_audit.json")
    cost = load("cost_latency_audit.json")
    counter = load("request_counter.json")
    security = load("security_audit.json")
    freeze = load("freeze_audit.json")
    formal = load("formal_test_ledger_audit.json")

    status = acceptance["status"]
    assert status in ALLOWED_STATUSES
    assert preflight["base_url"] == "https://api.deepseek.com/beta"
    assert preflight["endpoint"] == "https://api.deepseek.com/beta/chat/completions"
    assert preflight["protocol"] == "tools/function/strict with forced named function"
    assert preflight["sample"]["split"] == "val"
    assert preflight["sample"]["formal_test"] is False
    assert preflight["sample"]["query_persisted"] is False
    assert preflight["sample"]["ground_truth_persisted"] is False
    assert preflight["http_retry_count"] == 0
    assert preflight["schema_retry_count"] == 0
    assert preflight["prompt_repair_retry_count"] == 0
    assert preflight["budget"]["backend_request_cap"] == 1
    assert preflight["budget"]["cost_cap_cny"] == 1.0
    assert preflight["budget"]["worst_case_cost_cny"] < 1.0

    backend_requests = counter["backend_request_count"]
    assert backend_requests == acceptance["backend_request_count"]
    assert counter["backend_request_cap"] == 1
    assert backend_requests in {0, 1}
    assert cost["backend_requests"] == backend_requests
    assert cost["schema_retry_count"] == 0
    assert cost["http_retry_count"] == 0
    assert cost["prompt_repair_retry_count"] == 0
    assert float(cost["accounted_cost_cny_conservative"]) <= 1.0

    assert identity["configured_model"] == preflight["configured_model"]
    assert identity["base_url"] == preflight["base_url"]
    assert identity["endpoint"] == preflight["endpoint"]
    assert handshake["raw_response_persisted"] is False
    assert handshake["raw_tool_arguments_persisted"] is False
    assert handshake["raw_provider_error_body_persisted"] is False

    if status == "READY_FOR_P4_API_R1F_1_FIVE_PAIR_SMOKE_REVIEW":
        assert backend_requests == 1
        assert handshake["http_success"] is True
        assert handshake["server_schema_accepted"] is True
        assert handshake["finish_reason_reasonable"] is True
        assert handshake["tool_call_count"] == 1
        assert handshake["tool_name_matches"] is True
        assert handshake["tool_arguments_json_parsed"] is True
        assert handshake["transport_validation_pass"] is True
        assert canonical["transport_to_canonical_pass"] is True
        assert canonical["canonical_validation_pass"] is True
        assert identity["model_identity_match"] is True
        assert cache["replay_cache_hit"] is True
        assert cache["replay_backend_request_count"] == 0
        assert cache["total_backend_request_count"] == 1
    elif status == "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE":
        assert backend_requests == 1
        assert handshake["server_schema_accepted"] is False
        assert handshake["error_category"] == "STRICT_SCHEMA_REJECTED"
        assert handshake["http_status"] in {400, 422}
    else:
        assert handshake["status"] in {"NOT_CALLED", "FAILED"}

    assert freeze["frozen_hash_changes"] == 0
    assert freeze["history_directory_changes"] == 0
    assert freeze["transport_schema_matches_r1e_1"] is True
    assert freeze["transport_schema_root_has_def"] is True
    assert freeze["transport_schema_root_has_defs"] is False
    assert formal["before"] == formal["after"]
    assert formal["after"] == {
        "run_count": 0,
        "content_accessed": False,
        "unlock_authorized": False,
    }

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in ledger["runs"].values()) == 0
    assert ledger["test_content_accessed"] is False
    assert ledger["test_unlock_authorized"] is False

    assert security["api_key_persisted"] is False
    assert security["authorization_header_persisted"] is False
    assert security["raw_query_persisted"] is False
    assert security["ground_truth_persisted"] is False
    assert security["raw_response_persisted"] is False
    assert security["raw_tool_arguments_persisted"] is False
    assert security["raw_provider_error_body_persisted"] is False
    assert security["secret_leak_count"] == 0
    assert security["formal_test_content_accessed"] is False
    assert security["later_stage_started"] is False

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    assert not api_key or api_key not in delivery_text
    assert re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text) is None
    for name in ("query", "ground_truth", "raw_model_response", "raw_tool_arguments"):
        assert f'"{name}"' not in delivery_text

    print(json.dumps({
        "status": "PASS",
        "acceptance": status,
        "files": len(files),
        "backend_requests": backend_requests,
        "http_status": handshake.get("http_status"),
        "configured_model": identity["configured_model"],
        "actual_model": identity.get("actual_model"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
