"""Independently verify the P4-API-R1F.2 minimal strict control delivery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_2_minimal_strict_control_v1"
R1F1_ACCEPTANCE = ROOT / "reports" / "p4_api_r1f_1_strict_handshake_v1" / "acceptance.json"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
ALLOWED_STATUSES = {
    "MINIMAL_STRICT_ACCEPTED_FULL_SCHEMA_REJECTED",
    "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE",
    "MINIMAL_STRICT_REJECTED_UNRESOLVED",
    "BLOCKED_WITH_EVIDENCE",
}
REQUIRED_FILES = {
    "final_acceptance.zh-CN.md",
    "acceptance.json",
    "minimal_schema.json",
    "control_handshake.json",
    "runtime_identity.json",
    "security_audit.json",
    "freeze_audit.json",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R1F_2.md",
    "preflight.json",
    "cost_audit.json",
    "request_counter.json",
    "formal_test_ledger_audit.json",
}
EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean", "description": "Return true."},
    },
    "required": ["ok"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, Any]:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def main() -> None:
    assert REPORT.is_dir(), "R1F.2 report directory is absent"
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
    schema = load("minimal_schema.json")
    handshake = load("control_handshake.json")
    identity = load("runtime_identity.json")
    security = load("security_audit.json")
    freeze = load("freeze_audit.json")
    formal = load("formal_test_ledger_audit.json")
    counter = load("request_counter.json")
    cost = load("cost_audit.json")

    status = acceptance["status"]
    assert status in ALLOWED_STATUSES
    assert schema == EXPECTED_SCHEMA
    schema_text = json.dumps(schema, sort_keys=True)
    assert all(token not in schema_text for token in ("$def", "$ref", "anyOf", "array"))

    assert preflight["endpoint"] == "https://api.deepseek.com/beta/chat/completions"
    assert preflight["configured_model"] == "deepseek-v4-pro"
    assert preflight["function_strict"] is True
    assert preflight["forced_named_function"] is True
    assert preflight["data_access"]["train"] is False
    assert preflight["data_access"]["val"] is False
    assert preflight["data_access"]["test"] is False
    assert preflight["http_retry_count"] == 0
    assert preflight["schema_retry_count"] == 0
    assert preflight["prompt_retry_count"] == 0
    assert preflight["budget"]["backend_request_cap"] == 1
    assert preflight["budget"]["cost_cap_cny"] == 0.20
    assert preflight["budget"]["worst_case_cost_cny"] < 0.20

    requests = counter["backend_request_count"]
    assert requests == acceptance["backend_request_count"]
    assert counter["backend_request_cap"] == 1
    assert requests in {0, 1}
    assert cost["backend_requests"] == requests
    assert float(cost["accounted_cost_cny_conservative"]) <= 0.20
    assert cost["http_retry_count"] == 0
    assert cost["schema_retry_count"] == 0
    assert cost["prompt_retry_count"] == 0

    assert identity["configured_model"] == preflight["configured_model"]
    assert identity["endpoint"] == preflight["endpoint"]
    assert handshake["raw_response_persisted"] is False
    assert handshake["raw_arguments_persisted"] is False
    assert handshake["raw_provider_error_body_persisted"] is False
    assert handshake["provider_error_message_persisted"] is False

    if status == "MINIMAL_STRICT_ACCEPTED_FULL_SCHEMA_REJECTED":
        assert requests == 1
        assert handshake["http_success"] is True
        assert identity["actual_model"]
        assert identity["model_identity_match"] is True
        assert handshake["tool_call_count"] == 1
        assert handshake["tool_name_matches"] is True
        assert handshake["arguments_exact_ok_true"] is True
        assert handshake["local_schema_validation_pass"] is True
    elif status == "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE":
        assert requests == 1
        assert handshake["http_success"] is False
        assert handshake["capability_unavailable_language"] is True
        assert any(handshake["mentions"][name] for name in ("strict", "tool", "model"))
        assert handshake["stable_error_code"] == "E_DEEPSEEK_STRICT_PROFILE_UNAVAILABLE"
    elif status == "MINIMAL_STRICT_REJECTED_UNRESOLVED":
        assert requests == 1
        assert handshake["stable_error_code"] in {
            "E_MINIMAL_STRICT_REJECTED_UNRESOLVED",
            "E_MINIMAL_CONTROL_PROTOCOL",
        }
    else:
        assert handshake["status"] in {"NOT_CALLED", "FAILED"}

    assert freeze["r1f_1_acceptance_status"] == "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE"
    assert freeze["frozen_manifest_changes"] == 0
    assert freeze["history_directory_changes"] == 0
    assert freeze["dataset_payload_read"] is False
    assert formal["before"] == formal["after"] == {
        "run_count": 0,
        "content_accessed": False,
        "unlock_authorized": False,
    }

    r1f1 = json.loads(R1F1_ACCEPTANCE.read_text(encoding="utf-8"))
    assert r1f1["status"] == "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE"
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in ledger["runs"].values()) == 0
    assert ledger["test_content_accessed"] is False
    assert ledger["test_unlock_authorized"] is False

    assert security["api_key_persisted"] is False
    assert security["authorization_header_persisted"] is False
    assert security["raw_user_message_persisted"] is False
    assert security["raw_response_persisted"] is False
    assert security["raw_arguments_persisted"] is False
    assert security["raw_provider_error_body_persisted"] is False
    assert security["provider_error_message_persisted"] is False
    assert security["train_val_test_data_accessed"] is False
    assert security["formal_test_content_accessed"] is False
    assert security["later_stage_started"] is False
    assert security["secret_leak_count"] == 0

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    assert not api_key or api_key not in delivery_text
    assert re.search(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text) is None
    assert '"provider_error_message"' not in delivery_text
    assert '"raw_arguments"' not in delivery_text
    assert '"raw_response"' not in delivery_text

    print(json.dumps({
        "status": "PASS",
        "acceptance": status,
        "files": len(files),
        "backend_requests": requests,
        "http_status": handshake.get("http_status"),
        "configured_model": identity["configured_model"],
        "actual_model": identity.get("actual_model"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
