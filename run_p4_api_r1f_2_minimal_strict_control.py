"""Run the one authorized P4-API-R1F.2 minimal strict capability control."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1f_2_minimal_strict_control_v1"
COUNTER = REPORT / "request_counter.json"
ENDPOINT = "https://api.deepseek.com/beta/chat/completions"
BASE_URL = "https://api.deepseek.com/beta"
TOOL_NAME = "minimal_strict_capability_control"
CONTROL_MESSAGE = "Call the function and return ok=true."
MAX_OUTPUT_TOKENS = 64
BACKEND_REQUEST_CAP = 1
COST_CAP_CNY = 0.20
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
USD_CNY_CONSERVATIVE = 8.0

MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {
            "type": "boolean",
            "description": "Return true.",
        }
    },
    "required": ["ok"],
    "additionalProperties": False,
}

R1F1_ACCEPTANCE = (
    ROOT / "reports" / "p4_api_r1f_1_strict_handshake_v1" / "acceptance.json"
)
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
FROZEN_MANIFEST_HASHES = {
    "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json": "41067a5b8916bb3ad222c9e873d8576a4ec7aaa3634c79d89d0eebdcba63468a",
    "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json": "7a9e8f184013eeeff0db794a95defc73d19438c89e7cbbf6609abdea1a6a2845",
    "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json": "44b4be432a432bce02158ab3fddc1bcdc569cb3d86587286c5ab866b193349b9",
    "reports/p4_api_r1f_1_strict_handshake_v1/delivery_manifest.json": "2b6df79088c12b9af4e8f8ef74dff087345ccae1ba0342f9128f6c02023b07bd",
    "reports/p4_api_r1e_1_dialect_repair_v1/delivery_manifest.json": "b67b44275e4f07422e45d6faf0d327b9c496a8c0e49c9d837677184964e10009",
}
HISTORY_DIRS = (
    "reports/p4_api_r1f_1_strict_handshake_v1",
    "reports/p4_api_r1e_1_dialect_repair_v1",
)
EXPECTED_HISTORY = {
    "reports/p4_api_r1f_1_strict_handshake_v1": {
        "file_count": 14,
        "aggregate_sha256": "4912a7e3971c7bb96cca9672a7d043e54f59bc22b93f8469182b9480bd5916e9",
    },
    "reports/p4_api_r1e_1_dialect_repair_v1": {
        "file_count": 11,
        "aggregate_sha256": "1ac02e566c8b055c53e2d7b02fdea0f1a71c957cf7eb29eea065ed899604a766",
    },
}


class ControlProtocolError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ControlHttpError(RuntimeError):
    def __init__(self, evidence: Mapping[str, Any], latency_ms: float) -> None:
        self.evidence = dict(evidence)
        self.latency_ms = latency_ms
        super().__init__(str(evidence["stable_error_code"]))


class ControlRuntimeError(RuntimeError):
    def __init__(self, code: str, latency_ms: float = 0.0) -> None:
        self.code = code
        self.latency_ms = latency_ms
        super().__init__(code)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so one configured endpoint means one HTTP request."""

    def redirect_request(
        self, req, fp, code, msg, headers, newurl
    ):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_payload(model: str) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("E_MODEL_REQUIRED")
    return {
        "model": model,
        "messages": [{"role": "user", "content": CONTROL_MESSAGE}],
        "tools": [{
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Return the minimal strict capability control result.",
                "strict": True,
                "parameters": MINIMAL_SCHEMA,
            },
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }


def _safe_provider_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return "REDACTED_UNSAFE_VALUE"
    if len(value) <= 200 and re.fullmatch(r"[A-Za-z0-9_.$\[\]/:-]*", value):
        return value
    return "REDACTED_UNSAFE_VALUE"


def sanitize_provider_error(http_status: int, raw: bytes) -> dict[str, Any]:
    decoded = raw.decode("utf-8", errors="replace")
    lowered = decoded.lower()
    error: Mapping[str, Any] = {}
    try:
        body = json.loads(decoded)
        candidate = body.get("error", {}) if isinstance(body, Mapping) else {}
        if isinstance(candidate, Mapping):
            error = candidate
    except json.JSONDecodeError:
        pass
    unavailable_phrases = (
        "does not support", "not support", "unsupported", "not available",
        "unavailable", "not enabled", "not allowed", "not exist",
        "unknown model", "invalid model", "not found",
    )
    return {
        "http_status": int(http_status),
        "provider_error_code": _safe_provider_scalar(error.get("code")),
        "provider_error_type": _safe_provider_scalar(error.get("type")),
        "provider_error_param": _safe_provider_scalar(error.get("param")),
        "mentions": {
            "schema": "schema" in lowered,
            "strict": "strict" in lowered,
            "tool": "tool" in lowered or "function" in lowered,
            "model": "model" in lowered,
        },
        "capability_unavailable_language": any(
            phrase in lowered for phrase in unavailable_phrases
        ),
        "raw_provider_error_body_persisted": False,
        "provider_error_message_persisted": False,
    }


def classify_http_failure(
    http_status: int, evidence: Mapping[str, Any]
) -> tuple[str, str]:
    mentions = evidence.get("mentions", {})
    capability_subject = any(
        bool(mentions.get(name)) for name in ("strict", "tool", "model")
    ) if isinstance(mentions, Mapping) else False
    if (
        http_status in {400, 404, 422}
        and evidence.get("capability_unavailable_language") is True
        and capability_subject
    ):
        return (
            "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE",
            "E_DEEPSEEK_STRICT_PROFILE_UNAVAILABLE",
        )
    if http_status in {400, 404, 422}:
        return (
            "MINIMAL_STRICT_REJECTED_UNRESOLVED",
            "E_MINIMAL_STRICT_REJECTED_UNRESOLVED",
        )
    if http_status in {401, 403}:
        return "BLOCKED_WITH_EVIDENCE", "E_AUTHENTICATION_REJECTED"
    if http_status == 429:
        return "BLOCKED_WITH_EVIDENCE", "E_RATE_LIMITED"
    if 500 <= http_status <= 599:
        return "BLOCKED_WITH_EVIDENCE", "E_PROVIDER_SERVER_ERROR"
    return "BLOCKED_WITH_EVIDENCE", f"E_HTTP_{http_status}"


def _strict_json_object(text: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(str(key))
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs_hook)
    except json.JSONDecodeError as exc:
        raise ControlProtocolError("E_ARGUMENTS_JSON") from exc
    if duplicates:
        raise ControlProtocolError("E_ARGUMENTS_DUPLICATE_KEY")
    if not isinstance(value, dict):
        raise ControlProtocolError("E_ARGUMENTS_OBJECT")
    return value


def parse_success_response(http_status: int, response: Mapping[str, Any]) -> dict[str, Any]:
    if not 200 <= http_status < 300:
        raise ControlProtocolError("E_HTTP_NOT_SUCCESS")
    try:
        choices = response["choices"]
        choice = choices[0]
        message = choice["message"]
        calls = message["tool_calls"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ControlProtocolError("E_RESPONSE_SHAPE") from exc
    if not isinstance(calls, list) or len(calls) != 1:
        raise ControlProtocolError("E_TOOL_CALL_COUNT")
    call = calls[0]
    try:
        if call["type"] != "function":
            raise ControlProtocolError("E_TOOL_CALL_TYPE")
        function = call["function"]
        if function["name"] != TOOL_NAME:
            raise ControlProtocolError("E_TOOL_NAME")
        arguments_text = function["arguments"]
    except (KeyError, TypeError) as exc:
        raise ControlProtocolError("E_RESPONSE_SHAPE") from exc
    if not isinstance(arguments_text, str):
        raise ControlProtocolError("E_ARGUMENTS_STRING")
    arguments = _strict_json_object(arguments_text)
    errors = list(Draft202012Validator(MINIMAL_SCHEMA).iter_errors(arguments))
    if errors:
        raise ControlProtocolError("E_LOCAL_SCHEMA_VALIDATION")
    if arguments != {"ok": True}:
        raise ControlProtocolError("E_ARGUMENTS_NOT_EXACT_TRUE")
    actual_model = response.get("model")
    if not isinstance(actual_model, str) or not actual_model:
        raise ControlProtocolError("E_ACTUAL_MODEL_MISSING")
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    return {
        "http_status": http_status,
        "actual_model": actual_model,
        "finish_reason": choice.get("finish_reason"),
        "tool_call_count": 1,
        "tool_name": TOOL_NAME,
        "arguments_exact": True,
        "local_schema_valid": True,
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
    }


def request_counter() -> int:
    if not COUNTER.exists():
        return 0
    return int(json.loads(COUNTER.read_text(encoding="utf-8"))["backend_request_count"])


def manifest_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in FROZEN_MANIFEST_HASHES}


def history_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in HISTORY_DIRS:
        directory = ROOT / relative
        files = {
            str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
        result[relative] = {
            "file_count": len(files),
            "aggregate_sha256": hashlib.sha256(json.dumps(
                files, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        }
    return result


def formal_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }


def conservative_cost_cny(input_units: int, output_tokens: int) -> float:
    return (
        input_units * INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def post_once(api_key: str, payload: Mapping[str, Any], timeout: float = 180.0):
    if request_counter() >= BACKEND_REQUEST_CAP:
        raise ControlRuntimeError("E_BACKEND_REQUEST_CAP")
    write_json(COUNTER, {
        "schema_version": "topocae-p4-api-r1f-2-request-counter/1",
        "backend_request_count": request_counter() + 1,
        "backend_request_cap": BACKEND_REQUEST_CAP,
    })
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(NoRedirectHandler())
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_error = exc.read()
        evidence = sanitize_provider_error(int(exc.code), raw_error)
        terminal_status, stable_code = classify_http_failure(int(exc.code), evidence)
        evidence.update({
            "terminal_status": terminal_status,
            "stable_error_code": stable_code,
        })
        raise ControlHttpError(evidence, latency_ms) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ControlRuntimeError(
            "E_HTTP_TIMEOUT", (time.perf_counter() - started) * 1000.0
        ) from exc
    except urllib.error.URLError as exc:
        raise ControlRuntimeError(
            "E_HTTP_NETWORK", (time.perf_counter() - started) * 1000.0
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlProtocolError("E_RESPONSE_JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ControlProtocolError("E_RESPONSE_OBJECT")
    return status, parsed, latency_ms


def write_manifest() -> None:
    assets = sorted(
        path for path in REPORT.iterdir()
        if path.is_file() and path.name != "delivery_manifest.json"
    )
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1f-2-delivery-manifest/1",
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
    identity: Mapping[str, Any],
    cost: Mapping[str, Any],
    manifests_before: Mapping[str, str],
    history_before: Mapping[str, Any],
    formal_before: Mapping[str, Any],
    api_key: str,
) -> str:
    manifests_after = manifest_hashes()
    history_after = history_snapshot()
    formal_after = formal_state()
    manifest_changes = sum(
        manifests_before.get(name) != manifests_after.get(name)
        for name in FROZEN_MANIFEST_HASHES
    )
    history_changes = sum(
        history_before.get(name) != history_after.get(name) for name in HISTORY_DIRS
    )
    if manifest_changes or history_changes or formal_before != formal_after:
        status = "BLOCKED_WITH_EVIDENCE"

    write_json(REPORT / "control_handshake.json", handshake)
    write_json(REPORT / "runtime_identity.json", identity)
    write_json(REPORT / "cost_audit.json", cost)
    write_json(REPORT / "freeze_audit.json", {
        "schema_version": "topocae-p4-api-r1f-2-freeze-audit/1",
        "expected_manifest_hashes": FROZEN_MANIFEST_HASHES,
        "manifest_hashes_before": manifests_before,
        "manifest_hashes_after": manifests_after,
        "frozen_manifest_changes": manifest_changes,
        "expected_history": EXPECTED_HISTORY,
        "history_before": history_before,
        "history_after": history_after,
        "history_directory_changes": history_changes,
        "r1f_1_acceptance_status": "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE",
        "dataset_payload_read": False,
        "freeze_verification_scope": "manifest identity and R1E.1/R1F.1 delivery snapshots only",
    })
    write_json(REPORT / "formal_test_ledger_audit.json", {
        "schema_version": "topocae-p4-api-r1f-2-formal-ledger-audit/1",
        "before": formal_before,
        "after": formal_after,
        "formal_test_payload_read": False,
    })
    security = {
        "schema_version": "topocae-p4-api-r1f-2-security-audit/1",
        "api_key_persisted": False,
        "api_key_hashed": False,
        "authorization_header_persisted": False,
        "raw_user_message_persisted": False,
        "raw_response_persisted": False,
        "raw_arguments_persisted": False,
        "raw_provider_error_body_persisted": False,
        "provider_error_message_persisted": False,
        "train_val_test_data_accessed": False,
        "formal_test_content_accessed": False,
        "later_stage_started": False,
        "five_pair_smoke_started": False,
        "p5_started": False,
        "step_accessed": False,
        "abaqus_accessed": False,
        "secret_leak_count": 0,
    }
    write_json(REPORT / "security_audit.json", security)
    write_json(REPORT / "acceptance.json", {
        "schema_version": "topocae-p4-api-r1f-2-acceptance/1",
        "status": status,
        "backend_request_count": request_counter(),
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cost_cap_cny": COST_CAP_CNY,
        "http_retry_count": 0,
        "schema_retry_count": 0,
        "prompt_retry_count": 0,
        "provider_fallback": False,
        "endpoint_fallback": False,
        "model_fallback": False,
        "protocol_fallback": False,
        "second_request_attempted": False,
        "later_stage_started": False,
    })
    descriptions = {
        "MINIMAL_STRICT_ACCEPTED_FULL_SCHEMA_REJECTED": "最小 strict tool schema 被当前 DeepSeek beta profile 接受；R1F.1 完整 schema 的拒绝因此可归因于完整 schema 兼容性，而不是基础 strict 能力缺失。",
        "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE": "Provider 明确返回当前模型、tool 或 strict capability 不可用的安全分类证据。",
        "MINIMAL_STRICT_REJECTED_UNRESOLVED": "最小 strict control 被拒绝或未形成合规 tool call，但证据不足以断言 strict profile 整体不可用。",
        "BLOCKED_WITH_EVIDENCE": "环境、身份、预算、网络或冻结门禁阻止了能力判定。",
    }
    (REPORT / "final_acceptance.zh-CN.md").write_text(
        f"""# P4-API-R1F.2 DeepSeek Minimal Strict Capability Control

状态：`{status}`

{descriptions[status]}

- Endpoint：`{ENDPOINT}`
- Configured model：`{preflight['configured_model']}`
- Actual model：`{identity.get('actual_model')}`
- Backend requests：{request_counter()}/1
- HTTP / schema / prompt retry：0 / 0 / 0
- Train / val / test data accessed：false / false / false
- 五条 smoke、P5、STEP、Abaqus：未启动
""",
        encoding="utf-8",
        newline="\n",
    )
    (REPORT / "CODEX_CONTEXT_P4_API_R1F_2.md").write_text(
        f"""# CODEX Context: P4-API-R1F.2

- Status: `{status}`
- Endpoint: `{ENDPOINT}`
- Configured/actual model: `{preflight['configured_model']}` / `{identity.get('actual_model')}`
- Backend requests: {request_counter()}/1
- Retries: HTTP=0, schema=0, prompt=0
- Dataset access: none
- Next action: supervisor review only; no subsequent stage was started.
""",
        encoding="utf-8",
        newline="\n",
    )
    delivery_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REPORT.iterdir() if path.is_file()
    )
    if (api_key and api_key in delivery_text) or re.search(
        r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", delivery_text
    ):
        raise RuntimeError("E_SECRET_DELIVERY")
    if CONTROL_MESSAGE in delivery_text:
        raise RuntimeError("E_RAW_CONTROL_MESSAGE_DELIVERY")
    write_manifest()
    return status


def _base_handshake() -> dict[str, Any]:
    return {
        "schema_version": "topocae-p4-api-r1f-2-control-handshake/1",
        "http_status": None,
        "http_success": False,
        "provider_error_code": None,
        "provider_error_type": None,
        "provider_error_param": None,
        "mentions": {"schema": False, "strict": False, "tool": False, "model": False},
        "capability_unavailable_language": False,
        "raw_response_persisted": False,
        "raw_arguments_persisted": False,
        "raw_provider_error_body_persisted": False,
        "provider_error_message_persisted": False,
    }


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("E_R1F_2_REPORT_EXISTS")

    model = os.environ.get("TOPOCAE_LLM_MODEL", "").strip()
    api_key = os.environ.get("TOPOCAE_LLM_API_KEY", "")
    manifests_before = manifest_hashes()
    history_before = history_snapshot()
    formal_before = formal_state()
    r1f1 = json.loads(R1F1_ACCEPTANCE.read_text(encoding="utf-8"))
    gate_errors: list[str] = []
    if model != "deepseek-v4-pro":
        gate_errors.append("E_MODEL_IDENTITY")
    if not api_key:
        gate_errors.append("E_API_KEY_MISSING")
    if manifests_before != FROZEN_MANIFEST_HASHES:
        gate_errors.append("E_FROZEN_MANIFEST_IDENTITY")
    if history_before != EXPECTED_HISTORY:
        gate_errors.append("E_HISTORY_IDENTITY")
    if r1f1.get("status") != "STRICT_SCHEMA_REJECTED_WITH_EVIDENCE":
        gate_errors.append("E_R1F_1_STATUS")
    if formal_before != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        gate_errors.append("E_FORMAL_LEDGER")
    if MINIMAL_SCHEMA != {
        "type": "object",
        "properties": {"ok": {"type": "boolean", "description": "Return true."}},
        "required": ["ok"],
        "additionalProperties": False,
    }:
        gate_errors.append("E_MINIMAL_SCHEMA_IDENTITY")

    payload = build_payload(model) if model else {}
    payload_bytes = len(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")) if payload else 0
    worst_cost = conservative_cost_cny(payload_bytes, MAX_OUTPUT_TOKENS)
    if worst_cost >= COST_CAP_CNY:
        gate_errors.append("E_COST_PLAN_CAP")

    REPORT.mkdir(parents=True)
    write_json(REPORT / "minimal_schema.json", MINIMAL_SCHEMA)
    write_json(COUNTER, {
        "schema_version": "topocae-p4-api-r1f-2-request-counter/1",
        "backend_request_count": 0,
        "backend_request_cap": BACKEND_REQUEST_CAP,
    })
    preflight = {
        "schema_version": "topocae-p4-api-r1f-2-preflight/1",
        "status": "PASS_BEFORE_API" if not gate_errors else "BLOCKED_BEFORE_API",
        "authorized_scope": "ONE_MINIMAL_DEEPSEEK_STRICT_CONTROL",
        "base_url": BASE_URL,
        "endpoint": ENDPOINT,
        "configured_model": model or "UNCONFIGURED",
        "api_key_configured": bool(api_key),
        "function_strict": True,
        "forced_named_function": True,
        "user_message_policy": "single minimal control instruction; not persisted",
        "data_access": {"train": False, "val": False, "test": False},
        "budget": {
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "cost_cap_cny": COST_CAP_CNY,
            "payload_utf8_bytes": payload_bytes,
            "input_token_upper_bound_equals_payload_bytes": payload_bytes,
            "output_token_upper_bound": MAX_OUTPUT_TOKENS,
            "worst_case_cost_cny": worst_cost,
        },
        "http_retry_count": 0,
        "schema_retry_count": 0,
        "prompt_retry_count": 0,
        "gate_errors": gate_errors,
    }
    write_json(REPORT / "preflight.json", preflight)

    identity = {
        "provider_profile": "deepseek_beta_minimal_strict_control",
        "base_url": BASE_URL,
        "endpoint": ENDPOINT,
        "configured_model": model or "UNCONFIGURED",
        "actual_model": None,
        "model_identity_match": None,
    }
    if gate_errors:
        handshake = {
            **_base_handshake(),
            "status": "NOT_CALLED",
            "stable_error_code": "E_PREFLIGHT_BLOCKED",
            "gate_errors": gate_errors,
        }
        status = finalize(
            status="BLOCKED_WITH_EVIDENCE",
            preflight=preflight,
            handshake=handshake,
            identity=identity,
            cost={
                "backend_requests": 0,
                "backend_request_cap": 1,
                "accounted_cost_cny_conservative": 0.0,
                "cost_cap_cny": COST_CAP_CNY,
                "http_retry_count": 0,
                "schema_retry_count": 0,
                "prompt_retry_count": 0,
            },
            manifests_before=manifests_before,
            history_before=history_before,
            formal_before=formal_before,
            api_key=api_key,
        )
        print(status)
        return 2

    try:
        http_status, response, latency_ms = post_once(api_key, payload)
        parsed = parse_success_response(http_status, response)
        identity = {
            **identity,
            "actual_model": parsed["actual_model"],
            "model_identity_match": parsed["actual_model"] == model,
        }
        accepted = bool(identity["model_identity_match"])
        status = (
            "MINIMAL_STRICT_ACCEPTED_FULL_SCHEMA_REJECTED"
            if accepted else "BLOCKED_WITH_EVIDENCE"
        )
        handshake = {
            **_base_handshake(),
            "status": "ACCEPTED" if accepted else "FAILED",
            "stable_error_code": None if accepted else "E_ACTUAL_MODEL_IDENTITY",
            "http_status": parsed["http_status"],
            "http_success": True,
            "finish_reason": parsed["finish_reason"],
            "tool_call_count": parsed["tool_call_count"],
            "tool_name": parsed["tool_name"],
            "tool_name_matches": parsed["tool_name"] == TOOL_NAME,
            "arguments_exact_ok_true": parsed["arguments_exact"],
            "local_schema_validation_pass": parsed["local_schema_valid"],
            "second_request_attempted": False,
            "latency_ms": latency_ms,
        }
        if parsed["prompt_tokens"] is not None and parsed["completion_tokens"] is not None:
            accounted_cost = conservative_cost_cny(
                parsed["prompt_tokens"], parsed["completion_tokens"]
            )
        else:
            accounted_cost = worst_cost
        cost = {
            "backend_requests": request_counter(),
            "backend_request_cap": 1,
            "prompt_tokens": parsed["prompt_tokens"],
            "completion_tokens": parsed["completion_tokens"],
            "usage_available": parsed["prompt_tokens"] is not None,
            "accounted_cost_cny_conservative": accounted_cost,
            "cost_cap_cny": COST_CAP_CNY,
            "http_retry_count": 0,
            "schema_retry_count": 0,
            "prompt_retry_count": 0,
        }
    except ControlHttpError as exc:
        status = str(exc.evidence["terminal_status"])
        handshake = {
            **_base_handshake(),
            **exc.evidence,
            "status": "REJECTED",
            "http_success": False,
            "second_request_attempted": False,
            "latency_ms": exc.latency_ms,
        }
        cost = {
            "backend_requests": request_counter(),
            "backend_request_cap": 1,
            "usage_available": False,
            "accounted_cost_cny_conservative": worst_cost,
            "cost_cap_cny": COST_CAP_CNY,
            "http_retry_count": 0,
            "schema_retry_count": 0,
            "prompt_retry_count": 0,
        }
    except ControlProtocolError as exc:
        status = "MINIMAL_STRICT_REJECTED_UNRESOLVED"
        handshake = {
            **_base_handshake(),
            "status": "FAILED",
            "http_status": 200,
            "http_success": True,
            "stable_error_code": "E_MINIMAL_CONTROL_PROTOCOL",
            "protocol_error_code": exc.code,
            "second_request_attempted": False,
        }
        cost = {
            "backend_requests": request_counter(),
            "backend_request_cap": 1,
            "usage_available": False,
            "accounted_cost_cny_conservative": worst_cost,
            "cost_cap_cny": COST_CAP_CNY,
            "http_retry_count": 0,
            "schema_retry_count": 0,
            "prompt_retry_count": 0,
        }
    except ControlRuntimeError as exc:
        status = "BLOCKED_WITH_EVIDENCE"
        handshake = {
            **_base_handshake(),
            "status": "FAILED",
            "stable_error_code": exc.code,
            "second_request_attempted": False,
            "latency_ms": exc.latency_ms,
        }
        cost = {
            "backend_requests": request_counter(),
            "backend_request_cap": 1,
            "usage_available": False,
            "accounted_cost_cny_conservative": worst_cost if request_counter() else 0.0,
            "cost_cap_cny": COST_CAP_CNY,
            "http_retry_count": 0,
            "schema_retry_count": 0,
            "prompt_retry_count": 0,
        }

    status = finalize(
        status=status,
        preflight=preflight,
        handshake=handshake,
        identity=identity,
        cost=cost,
        manifests_before=manifests_before,
        history_before=history_before,
        formal_before=formal_before,
        api_key=api_key,
    )
    print(status)
    return 0 if status == "MINIMAL_STRICT_ACCEPTED_FULL_SCHEMA_REJECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
