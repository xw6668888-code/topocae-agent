"""Run the one authorized P4-API-R1F.1 DeepSeek strict handshake."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.deepseek_strict_real_adapter import (
    DeepSeekStrictToolBetaAdapter,
    MAX_OUTPUT_TOKENS,
    PROFILE_NAME,
    STRICT_BASE_URL,
    STRICT_CHAT_ENDPOINT,
    SingleEntryHandshakeCache,
    StrictHandshakeError,
    _default_transport,
)
from utils.cae_language_grounding.deepseek_strict_transport import (
    TOOL_NAME,
    build_transport_schema,
    validate_deepseek_strict_subset_schema,
)
from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_1_strict_handshake_v1"
COUNTER = REPORT / "request_counter.json"
R1E1_SCHEMA = ROOT / "reports" / "p4_api_r1e_1_dialect_repair_v1" / "transport_schema.json"
PROMPT = ROOT / "prompts" / "p4_a1_structured_llm_v1.md"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
HISTORY_DIRS = (
    "reports/p4_api_r1e_transport_schema_v1",
    "reports/p4_api_r1f_strict_handshake_v1",
    "reports/p4_api_r1e_1_dialect_repair_v1",
)
FROZEN_HASHES = {
    "reports/p4_api_r1e_1_dialect_repair_v1/transport_schema.json": "9a0f08555fa16e027e8a993d533b4fddabd4b14cb1b075d8f3e10130b53a242c",
    "utils/cae_language_grounding/deepseek_strict_transport.py": "c55e27b6125bf6ab59ace83cca4aa82b1735cae1aa2587f7a10c719adb375164",
    "utils/cae_language_grounding/deepseek_strict_real_adapter.py": "f91816deef48d4100290e166dbaf224381fa64f2d1912f2ec9123e4ba9e7f9d3",
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    "utils/cae_intent.py": "163ae1189a81f607187e195c02b8444b716651393813aa3b2ec9c27c2cfb9c8e",
    "utils/cae_language_grounding/grounder.py": "b2b7a80240452eb0cadb04c1f189e2f2e8587ab76874907b61d53ebd6836ded9",
    "utils/cae_language_grounding/evaluation.py": "93c53ca9d9ad660d660f92d46fa5b3637a97ef97fe8a022dcdb2a89635fa1e08",
    "prompts/p4_a1_structured_llm_v1.md": "e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da",
    "prompts/p4_a1_direct_llm_v1.md": "ed21697b97c114ec12f2eab9e13aeb8c1a69afdc68efd03f7c458d693861a581",
}

BACKEND_REQUEST_CAP = 1
COST_CAP_CNY = 1.0
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
USD_CNY_CONSERVATIVE = 8.0
ALLOWED_FINISH_REASONS = {"tool_calls", "stop"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def request_counter() -> int:
    if not COUNTER.exists():
        return 0
    return int(json.loads(COUNTER.read_text(encoding="utf-8"))["backend_request_count"])


def current_frozen_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in FROZEN_HASHES}


def history_snapshot() -> dict[str, Any]:
    directories: dict[str, Any] = {}
    for relative in HISTORY_DIRS:
        directory = ROOT / relative
        files = {
            str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        directories[relative] = {"file_count": len(files), "aggregate_sha256": digest}
    return directories


def formal_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }


def conservative_cost_cny(prompt_units: int, completion_tokens: int) -> float:
    return (
        prompt_units * INPUT_USD_PER_MILLION
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


class PersistentOneRequestTransport:
    """Persist the attempt count before I/O and retain only sanitized metadata."""

    def __init__(self) -> None:
        self.response_metadata: dict[str, Any] = {}

    def __call__(self, endpoint, headers, payload, timeout):
        count = request_counter()
        if count >= BACKEND_REQUEST_CAP:
            raise StrictHandshakeError("E_BACKEND_REQUEST_CAP")
        if endpoint != STRICT_CHAT_ENDPOINT:
            raise StrictHandshakeError("E_ENDPOINT_IDENTITY")
        write_json(COUNTER, {
            "schema_version": "topocae-p4-api-r1f-1-request-counter/1",
            "backend_request_count": count + 1,
            "backend_request_cap": BACKEND_REQUEST_CAP,
        })
        status, response = _default_transport(endpoint, headers, payload, timeout)
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        tool_name = None
        if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], Mapping):
            function = calls[0].get("function", {})
            if isinstance(function, Mapping):
                tool_name = function.get("name")
        self.response_metadata = {
            "http_status": status,
            "actual_model": str(response.get("model") or ""),
            "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
            "tool_call_count": len(calls) if isinstance(calls, list) else None,
            "tool_name": tool_name,
        }
        return status, response


def write_manifest() -> None:
    assets = sorted(
        path for path in REPORT.iterdir()
        if path.is_file() and path.name != "delivery_manifest.json"
    )
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1f-1-delivery-manifest/1",
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
    })


def finalize(
    *,
    status: str,
    preflight: Mapping[str, Any],
    handshake: Mapping[str, Any],
    canonical: Mapping[str, Any],
    cache: Mapping[str, Any],
    cost: Mapping[str, Any],
    frozen_before: Mapping[str, str],
    history_before: Mapping[str, Any],
    formal_before: Mapping[str, Any],
    api_key: str,
) -> str:
    frozen_after = current_frozen_hashes()
    history_after = history_snapshot()
    formal_after = formal_state()
    frozen_changes = sum(
        frozen_before.get(name) != frozen_after.get(name) for name in FROZEN_HASHES
    )
    history_changes = sum(
        history_before.get(name) != history_after.get(name) for name in HISTORY_DIRS
    )
    if frozen_changes or history_changes or formal_before != formal_after:
        status = "BLOCKED_WITH_EVIDENCE"

    write_json(REPORT / "runtime_identity.json", handshake["runtime_identity"])
    write_json(REPORT / "schema_handshake.json", handshake)
    write_json(REPORT / "canonical_validation.json", canonical)
    write_json(REPORT / "cache_audit.json", cache)
    write_json(REPORT / "cost_latency_audit.json", cost)
    write_json(REPORT / "freeze_audit.json", {
        "schema_version": "topocae-p4-api-r1f-1-freeze-audit/1",
        "expected_hashes": FROZEN_HASHES,
        "before": frozen_before,
        "after": frozen_after,
        "frozen_hash_changes": frozen_changes,
        "history_before": history_before,
        "history_after": history_after,
        "history_directory_changes": history_changes,
        "transport_schema_matches_r1e_1": preflight["transport_schema_matches_r1e_1"],
        "transport_schema_root_has_def": preflight["transport_schema_root_has_def"],
        "transport_schema_root_has_defs": preflight["transport_schema_root_has_defs"],
    })
    write_json(REPORT / "formal_test_ledger_audit.json", {
        "schema_version": "topocae-p4-api-r1f-1-formal-ledger-audit/1",
        "ledger_path": "output_p4_a1_language_grounding_v1/test_run_ledger.json",
        "before": formal_before,
        "after": formal_after,
        "formal_test_payload_accessed": False,
    })
    security = {
        "schema_version": "topocae-p4-api-r1f-1-security-audit/1",
        "api_key_source": "TOPOCAE_LLM_API_KEY environment variable",
        "api_key_persisted": False,
        "api_key_hashed": False,
        "authorization_header_persisted": False,
        "raw_query_persisted": False,
        "ground_truth_persisted": False,
        "raw_response_persisted": False,
        "raw_tool_arguments_persisted": False,
        "raw_provider_error_body_persisted": False,
        "canonical_payload_persisted": False,
        "formal_test_content_accessed": False,
        "later_stage_started": False,
        "p5_started": False,
        "step_accessed": False,
        "abaqus_accessed": False,
        "secret_leak_count": 0,
    }
    write_json(REPORT / "security_audit.json", security)
    acceptance = {
        "schema_version": "topocae-p4-api-r1f-1-acceptance/1",
        "status": status,
        "backend_request_count": request_counter(),
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cost_cap_cny": COST_CAP_CNY,
        "schema_retry_count": 0,
        "http_retry_count": 0,
        "prompt_repair_retry_count": 0,
        "frozen_hash_changes": frozen_changes,
        "history_directory_changes": history_changes,
        "provider_fallback": False,
        "endpoint_fallback": False,
        "model_fallback": False,
        "protocol_fallback": False,
        "content_fallback": False,
        "later_stage_started": False,
    }
    write_json(REPORT / "acceptance.json", acceptance)

    descriptions = {
        "READY_FOR_P4_API_R1F_1_FIVE_PAIR_SMOKE_REVIEW": "DeepSeek beta 接受了 R1E.1 冻结的 `$def` transport schema；唯一 tool call 已无损映射并通过 canonical validator，缓存重放未产生第二次请求。",
        "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE": "DeepSeek beta 拒绝了未修改的 R1E.1 `$def` transport schema；未进行 schema repair、fallback 或第二次请求。",
        "BLOCKED_WITH_EVIDENCE": "前置条件、运行身份或响应验证未全部通过；未扩大请求预算，也未进入后续阶段。",
    }
    final_text = f"""# P4-API-R1F.1 DeepSeek Strict Schema Single Handshake

状态：`{status}`

{descriptions[status]}

- Base URL：`{STRICT_BASE_URL}`
- Endpoint：`{STRICT_CHAT_ENDPOINT}`
- Configured model：`{preflight['configured_model']}`
- Actual model：`{handshake['runtime_identity'].get('actual_model')}`
- Backend requests：{request_counter()}/1
- HTTP / schema / prompt-repair retry：0 / 0 / 0
- Formal test content accessed：false
- P5、STEP、Abaqus：未启动
"""
    (REPORT / "final_acceptance.zh-CN.md").write_text(
        final_text, encoding="utf-8", newline="\n"
    )
    context = f"""# CODEX Context: P4-API-R1F.1

- Status: `{status}`
- Endpoint: `{STRICT_CHAT_ENDPOINT}`
- Configured/actual model: `{preflight['configured_model']}` / `{handshake['runtime_identity'].get('actual_model')}`
- Backend requests: {request_counter()}/1
- Retries: HTTP=0, schema=0, prompt-repair=0
- Formal test accessed: false
- Next action: supervisor review only; no five-pair smoke was executed.
"""
    (REPORT / "CODEX_CONTEXT_P4_API_R1F_1.md").write_text(
        context, encoding="utf-8", newline="\n"
    )

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    if (api_key and api_key in delivery_text) or re.search(
        r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text
    ):
        raise RuntimeError("E_SECRET_DELIVERY")
    write_manifest()
    return status


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1F_1_REPORT_EXISTS")

    model = os.environ.get("TOPOCAE_LLM_MODEL", "").strip()
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    generated_schema = build_transport_schema()
    frozen_schema = json.loads(R1E1_SCHEMA.read_text(encoding="utf-8"))
    subset = validate_deepseek_strict_subset_schema(generated_schema)
    frozen_before = current_frozen_hashes()
    history_before = history_snapshot()
    formal_before = formal_state()

    gate_errors: list[str] = []
    if verify_full_freeze()["status"] != "PASS":
        gate_errors.append("E_FULL_FREEZE")
    if frozen_before != FROZEN_HASHES:
        gate_errors.append("E_FROZEN_HASH_CHANGE")
    if formal_before != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        gate_errors.append("E_FORMAL_TEST_GUARD")
    if generated_schema != frozen_schema:
        gate_errors.append("E_R1E_1_SCHEMA_IDENTITY")
    if not subset["valid"]:
        gate_errors.append("E_STRICT_SUBSET")
    if "$def" not in generated_schema or "$defs" in generated_schema:
        gate_errors.append("E_DEF_DIALECT")
    if model != "deepseek-v4-pro":
        gate_errors.append("E_MODEL_IDENTITY")
    if not api_key:
        gate_errors.append("E_API_KEY_MISSING")

    example = next(iter(iter_language_examples("val")))
    messages = [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(
            example.inference_input(), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )},
    ]
    payload_bytes = 0
    worst_cost = 0.0
    if not gate_errors:
        planning_adapter = DeepSeekStrictToolBetaAdapter(api_key=api_key, model=model)
        payload = planning_adapter.build_payload(messages)
        payload_bytes = len(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"))
        worst_cost = conservative_cost_cny(payload_bytes, MAX_OUTPUT_TOKENS)
        if worst_cost >= COST_CAP_CNY:
            gate_errors.append("E_COST_PLAN_CAP")

    REPORT.mkdir(parents=True)
    write_json(COUNTER, {
        "schema_version": "topocae-p4-api-r1f-1-request-counter/1",
        "backend_request_count": 0,
        "backend_request_cap": BACKEND_REQUEST_CAP,
    })
    preflight = {
        "schema_version": "topocae-p4-api-r1f-1-preflight/1",
        "status": "PASS_BEFORE_API" if not gate_errors else "BLOCKED_BEFORE_API",
        "authorized_scope": "ONE_DEEPSEEK_STRICT_SCHEMA_HANDSHAKE",
        "provider_profile": PROFILE_NAME,
        "base_url": STRICT_BASE_URL,
        "endpoint": STRICT_CHAT_ENDPOINT,
        "protocol": "tools/function/strict with forced named function",
        "configured_model": model or "UNCONFIGURED",
        "api_key_configured": bool(api_key),
        "ambient_base_url_ignored": True,
        "transport_schema_matches_r1e_1": generated_schema == frozen_schema,
        "transport_schema_sha256": sha256(R1E1_SCHEMA),
        "transport_schema_root_has_def": "$def" in generated_schema,
        "transport_schema_root_has_defs": "$defs" in generated_schema,
        "strict_subset_validation_pass": subset["valid"],
        "sample": {
            "split": "val",
            "formal_test": False,
            "selection_policy": "first record in frozen val iterator",
            "sample_id": example.example_id,
            "query_persisted": False,
            "ground_truth_persisted": False,
        },
        "budget": {
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "cost_cap_cny": COST_CAP_CNY,
            "payload_utf8_bytes": payload_bytes,
            "input_token_upper_bound_equals_payload_bytes": payload_bytes,
            "output_token_upper_bound": MAX_OUTPUT_TOKENS,
            "worst_case_cost_cny": worst_cost,
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "usd_cny_conservative": USD_CNY_CONSERVATIVE,
        },
        "schema_retry_count": 0,
        "http_retry_count": 0,
        "prompt_repair_retry_count": 0,
        "gate_errors": gate_errors,
    }
    write_json(REPORT / "preflight.json", preflight)

    base_identity = {
        "provider_profile": PROFILE_NAME,
        "base_url": STRICT_BASE_URL,
        "endpoint": STRICT_CHAT_ENDPOINT,
        "configured_model": model or "UNCONFIGURED",
        "actual_model": None,
        "model_identity_match": None,
    }
    common_raw_guards = {
        "raw_response_persisted": False,
        "raw_tool_arguments_persisted": False,
        "raw_provider_error_body_persisted": False,
    }
    if gate_errors:
        status = finalize(
            status="BLOCKED_WITH_EVIDENCE",
            preflight=preflight,
            handshake={
                "schema_version": "topocae-p4-api-r1f-1-schema-handshake/1",
                "status": "NOT_CALLED",
                "server_schema_accepted": False,
                "http_status": None,
                "gate_errors": gate_errors,
                "runtime_identity": base_identity,
                **common_raw_guards,
            },
            canonical={"status": "NOT_RUN", "canonical_payload_persisted": False},
            cache={"status": "NOT_RUN", "replay_cache_hit": False, "total_backend_request_count": 0},
            cost={
                "backend_requests": 0,
                "backend_request_cap": 1,
                "accounted_cost_cny_conservative": 0.0,
                "cost_cap_cny": COST_CAP_CNY,
                "schema_retry_count": 0,
                "http_retry_count": 0,
                "prompt_repair_retry_count": 0,
            },
            frozen_before=frozen_before,
            history_before=history_before,
            formal_before=formal_before,
            api_key=api_key,
        )
        print(status)
        return 2

    transport = PersistentOneRequestTransport()
    adapter = DeepSeekStrictToolBetaAdapter(
        api_key=api_key, model=model, transport=transport
    )
    cache = SingleEntryHandshakeCache()
    cache_key = hashlib.sha256(json.dumps({
        "sample_id": example.example_id,
        "model": model,
        "profile": PROFILE_NAME,
        "transport_schema_sha256": sha256(R1E1_SCHEMA),
        "prompt_sha256": sha256(PROMPT),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    try:
        outcome = cache.execute(cache_key, adapter, messages)
        replay = cache.execute(cache_key, adapter, messages)
        identity_match = outcome.actual_model == model
        finish_reason_ok = outcome.finish_reason in ALLOWED_FINISH_REASONS
        checks = {
            "http_success": 200 <= outcome.http_status < 300,
            "model_identity_match": identity_match,
            "finish_reason_reasonable": finish_reason_ok,
            "tool_call_count_one": outcome.tool_call_count == 1,
            "tool_name_matches": outcome.tool_name == TOOL_NAME,
            "tool_arguments_json_parsed": True,
            "transport_validation_pass": outcome.transport_valid,
            "transport_to_canonical_pass": True,
            "canonical_validation_pass": outcome.canonical_valid,
            "cache_replay_pass": replay.cache_hit and replay.backend_request_count == 0,
            "backend_request_count_one": request_counter() == 1 and adapter.backend_request_count == 1,
        }
        actual_cost = conservative_cost_cny(outcome.prompt_tokens, outcome.completion_tokens)
        checks["cost_within_cap"] = actual_cost <= COST_CAP_CNY
        status = (
            "READY_FOR_P4_API_R1F_1_FIVE_PAIR_SMOKE_REVIEW"
            if all(checks.values()) else "BLOCKED_WITH_EVIDENCE"
        )
        identity = {
            **base_identity,
            "actual_model": outcome.actual_model,
            "model_identity_match": identity_match,
        }
        status = finalize(
            status=status,
            preflight=preflight,
            handshake={
                "schema_version": "topocae-p4-api-r1f-1-schema-handshake/1",
                "status": "ACCEPTED" if status.startswith("READY") else "FAILED",
                "server_schema_accepted": True,
                "http_success": checks["http_success"],
                "http_status": outcome.http_status,
                "finish_reason": outcome.finish_reason,
                "finish_reason_reasonable": finish_reason_ok,
                "tool_name": outcome.tool_name,
                "tool_name_matches": checks["tool_name_matches"],
                "tool_call_count": outcome.tool_call_count,
                "tool_arguments_json_parsed": True,
                "transport_validation_pass": outcome.transport_valid,
                "runtime_identity": identity,
                "checks": checks,
                **common_raw_guards,
            },
            canonical={
                "schema_version": "topocae-p4-api-r1f-1-canonical-validation/1",
                "transport_to_canonical_pass": True,
                "canonical_validation_pass": outcome.canonical_valid,
                "canonical_payload_persisted": False,
            },
            cache={
                "schema_version": "topocae-p4-api-r1f-1-cache-audit/1",
                "first_call_cache_hit": outcome.cache_hit,
                "replay_cache_hit": replay.cache_hit,
                "cache_misses": cache.misses,
                "cache_hits": cache.hits,
                "first_backend_request_count": outcome.backend_request_count,
                "replay_backend_request_count": replay.backend_request_count,
                "total_backend_request_count": request_counter(),
                "cache_type": "in_memory_single_entry",
                "cached_payload_persisted": False,
            },
            cost={
                "schema_version": "topocae-p4-api-r1f-1-cost-latency/1",
                "backend_requests": request_counter(),
                "backend_request_cap": 1,
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "prompt_cache_hit_tokens": outcome.cache_read_input_tokens,
                "latency_ms": outcome.latency_ms,
                "accounted_cost_cny_conservative": actual_cost,
                "cost_cap_cny": COST_CAP_CNY,
                "schema_retry_count": 0,
                "http_retry_count": 0,
                "prompt_repair_retry_count": 0,
            },
            frozen_before=frozen_before,
            history_before=history_before,
            formal_before=formal_before,
            api_key=api_key,
        )
        print(status)
        return 0 if status.startswith("READY") else 2
    except StrictHandshakeError as exc:
        rejection = exc.public_metadata()
        is_schema_rejection = exc.category == "STRICT_SCHEMA_REJECTED"
        status = (
            "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE"
            if is_schema_rejection else "BLOCKED_WITH_EVIDENCE"
        )
        metadata = transport.response_metadata
        actual_model = metadata.get("actual_model") or None
        identity = {
            **base_identity,
            "actual_model": actual_model,
            "model_identity_match": actual_model == model if actual_model else None,
        }
        status = finalize(
            status=status,
            preflight=preflight,
            handshake={
                "schema_version": "topocae-p4-api-r1f-1-schema-handshake/1",
                "status": "REJECTED" if is_schema_rejection else "FAILED",
                "server_schema_accepted": False,
                "http_success": bool(metadata.get("http_status") and 200 <= metadata["http_status"] < 300),
                **rejection,
                "finish_reason": metadata.get("finish_reason"),
                "tool_name": metadata.get("tool_name"),
                "tool_call_count": metadata.get("tool_call_count"),
                "runtime_identity": identity,
                "schema_mutation_retry_attempted": False,
                "def_dialect_retry_attempted": False,
                **common_raw_guards,
            },
            canonical={
                "status": "NOT_COMPLETED",
                "transport_to_canonical_pass": False,
                "canonical_validation_pass": False,
                "canonical_payload_persisted": False,
            },
            cache={
                "status": "NOT_REPLAYABLE",
                "replay_cache_hit": False,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "replay_backend_request_count": 0,
                "total_backend_request_count": request_counter(),
            },
            cost={
                "backend_requests": request_counter(),
                "backend_request_cap": 1,
                "usage_available": False,
                "accounted_cost_cny_conservative": worst_cost if request_counter() else 0.0,
                "cost_cap_cny": COST_CAP_CNY,
                "latency_ms": exc.latency_ms,
                "schema_retry_count": 0,
                "http_retry_count": 0,
                "prompt_repair_retry_count": 0,
            },
            frozen_before=frozen_before,
            history_before=history_before,
            formal_before=formal_before,
            api_key=api_key,
        )
        print(status)
        return 3 if is_schema_rejection else 2


if __name__ == "__main__":
    raise SystemExit(main())
