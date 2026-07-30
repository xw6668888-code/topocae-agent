"""Execute the single authorized P4-API-R1F strict schema handshake."""

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
from utils.cae_language_grounding.deepseek_strict_transport import TOOL_NAME
from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_strict_handshake_v1"
COUNTER = REPORT / "request_counter.json"
R1E = ROOT / "reports" / "p4_api_r1e_transport_schema_v1"
TRANSPORT_SCHEMA = R1E / "transport_schema.json"
PROMPT = ROOT / "prompts" / "p4_a1_structured_llm_v1.md"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"

EXPECTED_HASHES = {
    "reports/p4_api_r1e_transport_schema_v1/transport_schema.json": "55b26f9a55b6276628dd648a77c3eb09eba06f3bd268bb99b5e176892c6bf162",
    "utils/cae_language_grounding/deepseek_strict_transport.py": "3453b190fad7e58c661b328465da5e5edfd9f4ad0ed7e60ce2cc12849c21c65a",
    "schemas/cae_intent_result.schema.json": "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
    "schemas/cae_intent.schema.json": "8e94cd9a6f3dad42ccad58d47eb82e801e2cb66870ac03dcd940a56d076aeac4",
    "prompts/p4_a1_structured_llm_v1.md": "e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da",
    "utils/cae_intent.py": "163ae1189a81f607187e195c02b8444b716651393813aa3b2ec9c27c2cfb9c8e",
    "utils/cae_language_grounding/grounder.py": "b2b7a80240452eb0cadb04c1f189e2f2e8587ab76874907b61d53ebd6836ded9",
}

BACKEND_REQUEST_CAP = 1
COST_CAP_CNY = 1.0
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
USD_CNY_CONSERVATIVE = 8.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def formal_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    state = {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }
    if state != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    return state


def frozen_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in EXPECTED_HASHES}


def assert_frozen() -> dict[str, str]:
    actual = frozen_hashes()
    if actual != EXPECTED_HASHES:
        raise RuntimeError("E_FROZEN_HASH_CHANGE")
    return actual


def request_counter() -> int:
    if not COUNTER.exists():
        return 0
    return int(json.loads(COUNTER.read_text(encoding="utf-8"))["backend_request_count"])


class PersistentOneRequestTransport:
    def __call__(self, endpoint, headers, payload, timeout):
        count = request_counter()
        if count >= BACKEND_REQUEST_CAP:
            raise StrictHandshakeError("E_BACKEND_REQUEST_CAP")
        write_json(COUNTER, {
            "schema_version": "topocae-p4-api-r1f-request-counter/1",
            "backend_request_count": count + 1,
            "backend_request_cap": BACKEND_REQUEST_CAP,
        })
        return _default_transport(endpoint, headers, payload, timeout)


def conservative_cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * INPUT_USD_PER_MILLION
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def write_manifest() -> None:
    assets = sorted(
        path for path in REPORT.iterdir()
        if path.is_file() and path.name != "delivery_manifest.json"
    )
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1f-delivery-manifest/1",
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
    cost_latency: Mapping[str, Any],
    api_key: str,
) -> None:
    formal_after = formal_state()
    hashes_after = assert_frozen()
    if hashes_after != preflight["frozen_hashes_before"]:
        raise RuntimeError("E_FROZEN_HASH_CHANGE_DURING_HANDSHAKE")
    backend_count = request_counter()
    security = {
        "schema_version": "topocae-p4-api-r1f-security-audit/1",
        "api_key_source": "TOPOCAE_LLM_API_KEY environment variable",
        "api_key_written": False,
        "api_key_hashed": False,
        "authorization_header_written": False,
        "raw_response_written": False,
        "raw_error_response_written": False,
        "raw_tool_arguments_written": False,
        "query_written": False,
        "ground_truth_written": False,
        "canonical_payload_written": False,
        "formal_test_accessed": False,
        "backend_requests": backend_count,
        "secret_leak_count": 0,
    }
    write_json(REPORT / "runtime_identity.json", handshake.get("runtime_identity", {}))
    write_json(REPORT / "schema_handshake.json", handshake)
    write_json(REPORT / "canonical_validation.json", canonical)
    write_json(REPORT / "cache_audit.json", cache)
    write_json(REPORT / "cost_latency_audit.json", cost_latency)
    write_json(REPORT / "security_audit.json", security)
    acceptance = {
        "schema_version": "topocae-p4-api-r1f-acceptance/1",
        "status": status,
        "backend_request_count": backend_count,
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cost_cap_cny": COST_CAP_CNY,
        "formal_test_after": formal_after,
        "frozen_hash_changes": 0,
        "schema_retry_count": 0,
        "http_retry_count": 0,
        "provider_fallback": False,
        "endpoint_fallback": False,
        "model_fallback": False,
        "protocol_fallback": False,
        "content_fallback": False,
        "later_stage_started": False,
    }
    write_json(REPORT / "acceptance.json", acceptance)
    status_text = {
        "READY_FOR_P4_API_R1F_FIVE_PAIR_SMOKE_REVIEW": "完整 R1E transport schema 被服务端接受，强制 tool call 已恢复并通过冻结 canonical validator。",
        "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE": "服务端拒绝了未修改的 R1E transport schema；未尝试 `$defs` 到 `$def` 的改写或第二次请求。",
        "BLOCKED_WITH_EVIDENCE": "环境、预算或非 schema 运行门禁未通过；没有扩大调用。",
    }[status]
    final_text = f'''# P4-API-R1F DeepSeek Strict Schema Handshake

## 状态

`{status}`

{status_text}

## 执行边界

- provider profile：`{PROFILE_NAME}`
- base URL：`{STRICT_BASE_URL}`
- endpoint：`{STRICT_CHAT_ENDPOINT}`
- backend requests：{backend_count}/{BACKEND_REQUEST_CAP}
- schema retry：0；HTTP retry：0
- formal test：run=0、content_accessed=false
- 未运行其余 val、Direct、P5、STEP 或 Abaqus
'''
    (REPORT / "final_acceptance.zh-CN.md").write_text(final_text, encoding="utf-8", newline="\n")
    context = f'''# CODEX Context: P4-API-R1F

- Status: `{status}`
- Provider: `{PROFILE_NAME}`
- Endpoint: `{STRICT_CHAT_ENDPOINT}`
- Backend requests: {backend_count}
- Schema/HTTP retries: 0/0
- Formal test accessed: false
- Next action: supervisor review only; no later stage started.
'''
    (REPORT / "CODEX_CONTEXT_P4_API_R1F.md").write_text(context, encoding="utf-8", newline="\n")

    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    if (api_key and api_key in delivery_text) or re.search(
        r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text
    ):
        raise RuntimeError("E_SECRET_DELIVERY")
    write_manifest()


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1F_REPORT_EXISTS")
    full_freeze = verify_full_freeze()
    if full_freeze["status"] != "PASS":
        raise RuntimeError("E_FULL_FREEZE")
    frozen_before = assert_frozen()
    formal_before = formal_state()
    model = os.environ.get("TOPOCAE_LLM_MODEL", "").strip()
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    gate_errors = []
    if model != "deepseek-v4-pro":
        gate_errors.append("E_MODEL_IDENTITY")
    if not api_key:
        gate_errors.append("E_API_KEY_MISSING")

    example = next(iter(iter_language_examples("val")))
    inference_input = example.inference_input()
    prompt = PROMPT.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                inference_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    ]

    if not gate_errors:
        planning_adapter = DeepSeekStrictToolBetaAdapter(api_key=api_key, model=model)
        payload = planning_adapter.build_payload(messages)
        payload_bytes = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        worst_cost = conservative_cost_cny(payload_bytes, MAX_OUTPUT_TOKENS)
        if worst_cost > COST_CAP_CNY:
            gate_errors.append("E_COST_PLAN_CAP")
    else:
        payload_bytes = 0
        worst_cost = 0.0

    REPORT.mkdir(parents=True)
    write_json(COUNTER, {
        "schema_version": "topocae-p4-api-r1f-request-counter/1",
        "backend_request_count": 0,
        "backend_request_cap": BACKEND_REQUEST_CAP,
    })
    preflight = {
        "schema_version": "topocae-p4-api-r1f-preflight/1",
        "status": "PASS_BEFORE_API" if not gate_errors else "BLOCKED_BEFORE_API",
        "authorized_scope": "ONE_DEEPSEEK_STRICT_SCHEMA_HANDSHAKE",
        "provider_profile": PROFILE_NAME,
        "base_url": STRICT_BASE_URL,
        "endpoint": STRICT_CHAT_ENDPOINT,
        "protocol": "tools/function/strict with forced named tool_choice",
        "model": model or "UNCONFIGURED",
        "api_key_configured": bool(api_key),
        "api_key_value_read_for_request_only": bool(api_key),
        "ambient_provider_and_base_not_consulted": True,
        "frozen_hashes_before": frozen_before,
        "full_freeze_status": full_freeze["status"],
        "formal_test_before": formal_before,
        "sample": {
            "split": "val",
            "selection_policy": "first_record_in_frozen_val_language_iteration",
            "sample_id": example.example_id,
            "query_written": False,
            "ground_truth_written": False,
        },
        "budget": {
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "cost_cap_cny": COST_CAP_CNY,
            "payload_utf8_bytes": payload_bytes,
            "input_token_upper_bound_equals_payload_bytes": payload_bytes,
            "output_token_upper_bound": MAX_OUTPUT_TOKENS,
            "worst_case_cost_cny": worst_cost,
            "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing",
            "input_cache_miss_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "usd_cny_conservative": USD_CNY_CONSERVATIVE,
        },
        "schema_retry_count": 0,
        "http_retry_count": 0,
        "gate_errors": gate_errors,
    }
    write_json(REPORT / "preflight.json", preflight)

    if gate_errors:
        empty_handshake = {
            "schema_version": "topocae-p4-api-r1f-schema-handshake/1",
            "status": "NOT_CALLED",
            "gate_errors": gate_errors,
            "runtime_identity": {
                "provider_profile": PROFILE_NAME,
                "base_url": STRICT_BASE_URL,
                "endpoint": STRICT_CHAT_ENDPOINT,
                "configured_model": model or "UNCONFIGURED",
                "actual_model": None,
            },
        }
        finalize(
            status="BLOCKED_WITH_EVIDENCE",
            preflight=preflight,
            handshake=empty_handshake,
            canonical={"status": "NOT_RUN", "canonical_payload_persisted": False},
            cache={"status": "NOT_RUN", "hits": 0, "misses": 0},
            cost_latency={"backend_requests": 0, "accounted_cost_cny": 0.0},
            api_key=api_key,
        )
        print("BLOCKED_WITH_EVIDENCE")
        return 2

    adapter = DeepSeekStrictToolBetaAdapter(
        api_key=api_key,
        model=model,
        transport=PersistentOneRequestTransport(),
    )
    cache = SingleEntryHandshakeCache()
    cache_key = hashlib.sha256(
        json.dumps({
            "sample_id": example.example_id,
            "model": model,
            "profile": PROFILE_NAME,
            "schema_sha256": frozen_before[
                "reports/p4_api_r1e_transport_schema_v1/transport_schema.json"
            ],
            "prompt_sha256": frozen_before["prompts/p4_a1_structured_llm_v1.md"],
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    try:
        outcome = cache.execute(cache_key, adapter, messages)
        replay = cache.execute(cache_key, adapter, messages)
        if adapter.backend_request_count != 1 or request_counter() != 1:
            raise RuntimeError("E_BACKEND_REQUEST_ACCOUNTING")
        if not replay.cache_hit or replay.backend_request_count != 0:
            raise RuntimeError("E_CACHE_REPLAY")
        actual_cost = conservative_cost_cny(
            outcome.prompt_tokens, outcome.completion_tokens
        )
        if actual_cost > COST_CAP_CNY:
            raise RuntimeError("E_ACTUAL_COST_CAP")
        identity_ok = outcome.actual_model == model
        if not identity_ok:
            raise StrictHandshakeError(
                "E_ACTUAL_MODEL_IDENTITY", http_status=outcome.http_status,
                category="MODEL_IDENTITY_ERROR", latency_ms=outcome.latency_ms,
                backend_request_count=1,
            )
        handshake = {
            "schema_version": "topocae-p4-api-r1f-schema-handshake/1",
            "status": "ACCEPTED",
            "server_schema_accepted": True,
            "http_status": outcome.http_status,
            "finish_reason": outcome.finish_reason,
            "tool_name": outcome.tool_name,
            "tool_call_count": outcome.tool_call_count,
            "transport_valid": outcome.transport_valid,
            "schema_parameter_path": None,
            "raw_response_persisted": False,
            "tool_arguments_persisted": False,
            "runtime_identity": {
                "provider_profile": PROFILE_NAME,
                "base_url": STRICT_BASE_URL,
                "endpoint": STRICT_CHAT_ENDPOINT,
                "configured_model": model,
                "actual_model": outcome.actual_model,
                "model_identity_match": identity_ok,
            },
        }
        canonical = {
            "schema_version": "topocae-p4-api-r1f-canonical-validation/1",
            "transport_validation_pass": outcome.transport_valid,
            "transport_to_canonical_pass": True,
            "frozen_canonical_validation_pass": outcome.canonical_valid,
            "canonical_payload_persisted": False,
        }
        cache_audit = {
            "schema_version": "topocae-p4-api-r1f-cache-audit/1",
            "first_call_cache_hit": outcome.cache_hit,
            "replay_cache_hit": replay.cache_hit,
            "cache_misses": cache.misses,
            "cache_hits": cache.hits,
            "first_backend_request_count": outcome.backend_request_count,
            "replay_backend_request_count": replay.backend_request_count,
            "total_backend_request_count": request_counter(),
            "cache_type": "in_memory_single_entry",
            "cached_payload_persisted": False,
        }
        cost_latency = {
            "schema_version": "topocae-p4-api-r1f-cost-latency/1",
            "backend_requests": request_counter(),
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "prompt_cache_hit_tokens": outcome.cache_read_input_tokens,
            "latency_ms": outcome.latency_ms,
            "accounted_cost_cny_conservative": actual_cost,
            "cost_cap_cny": COST_CAP_CNY,
            "pricing_assumption": "all prompt tokens charged at cache-miss rate",
            "schema_retry_count": 0,
            "http_retry_count": 0,
        }
        finalize(
            status="READY_FOR_P4_API_R1F_FIVE_PAIR_SMOKE_REVIEW",
            preflight=preflight,
            handshake=handshake,
            canonical=canonical,
            cache=cache_audit,
            cost_latency=cost_latency,
            api_key=api_key,
        )
        print("READY_FOR_P4_API_R1F_FIVE_PAIR_SMOKE_REVIEW")
        return 0
    except StrictHandshakeError as exc:
        rejection = exc.public_metadata()
        is_schema_rejection = exc.category == "STRICT_SCHEMA_REJECTED"
        status = (
            "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE"
            if is_schema_rejection else "BLOCKED_WITH_EVIDENCE"
        )
        handshake = {
            "schema_version": "topocae-p4-api-r1f-schema-handshake/1",
            "status": "REJECTED" if is_schema_rejection else "FAILED",
            "server_schema_accepted": False,
            **rejection,
            "tool_name": None,
            "tool_call_count": None,
            "finish_reason": None,
            "raw_response_persisted": False,
            "runtime_identity": {
                "provider_profile": PROFILE_NAME,
                "base_url": STRICT_BASE_URL,
                "endpoint": STRICT_CHAT_ENDPOINT,
                "configured_model": model,
                "actual_model": None,
                "model_identity_match": None,
            },
            "schema_mutation_retry_attempted": False,
            "defs_to_def_retry_attempted": False,
        }
        finalize(
            status=status,
            preflight=preflight,
            handshake=handshake,
            canonical={
                "schema_version": "topocae-p4-api-r1f-canonical-validation/1",
                "status": "NOT_RUN",
                "transport_validation_pass": False,
                "transport_to_canonical_pass": False,
                "frozen_canonical_validation_pass": False,
                "canonical_payload_persisted": False,
            },
            cache={
                "schema_version": "topocae-p4-api-r1f-cache-audit/1",
                "status": "NOT_RUN_AFTER_REJECTION",
                "cache_hits": 0,
                "cache_misses": cache.misses,
                "total_backend_request_count": request_counter(),
            },
            cost_latency={
                "schema_version": "topocae-p4-api-r1f-cost-latency/1",
                "backend_requests": request_counter(),
                "backend_request_cap": BACKEND_REQUEST_CAP,
                "usage_available": False,
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_ms": exc.latency_ms,
                "accounted_cost_cny_conservative": 0.0,
                "cost_cap_cny": COST_CAP_CNY,
                "schema_retry_count": 0,
                "http_retry_count": 0,
            },
            api_key=api_key,
        )
        print(status)
        return 3 if is_schema_rejection else 2


if __name__ == "__main__":
    raise SystemExit(main())
