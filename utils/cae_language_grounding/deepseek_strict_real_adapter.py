"""Single-request real adapter for the DeepSeek strict-tool beta handshake."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import socket
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

from utils.cae_intent import validate_intent_result
from .deepseek_strict_transport import (
    PROFILE_NAME,
    STRICT_BASE_URL,
    TOOL_NAME,
    build_mock_tool_payload,
    parse_mock_tool_call_response,
)


STRICT_CHAT_ENDPOINT = "https://api.deepseek.com/beta/chat/completions"
MAX_OUTPUT_TOKENS = 4096


class StrictHandshakeError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        http_status: int | None = None,
        category: str = "RUNTIME_ERROR",
        schema_parameter_path: str | None = None,
        latency_ms: float = 0.0,
        backend_request_count: int = 0,
    ) -> None:
        self.code = code
        self.http_status = http_status
        self.category = category
        self.schema_parameter_path = schema_parameter_path
        self.latency_ms = latency_ms
        self.backend_request_count = backend_request_count
        super().__init__(code)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "http_status": self.http_status,
            "error_category": self.category,
            "schema_parameter_path": self.schema_parameter_path,
            "latency_ms": self.latency_ms,
            "backend_request_count": self.backend_request_count,
            "raw_error_persisted": False,
        }


@dataclass(frozen=True)
class HandshakeOutcome:
    canonical: Mapping[str, Any]
    http_status: int
    actual_model: str
    finish_reason: str | None
    tool_name: str
    tool_call_count: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    latency_ms: float
    backend_request_count: int
    transport_valid: bool
    canonical_valid: bool
    cache_hit: bool = False

    def public_metadata(self) -> dict[str, Any]:
        return {
            "http_status": self.http_status,
            "actual_model": self.actual_model,
            "finish_reason": self.finish_reason,
            "tool_name": self.tool_name,
            "tool_call_count": self.tool_call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "latency_ms": self.latency_ms,
            "backend_request_count": self.backend_request_count,
            "transport_valid": self.transport_valid,
            "canonical_valid": self.canonical_valid,
            "cache_hit": self.cache_hit,
            "raw_response_persisted": False,
            "tool_arguments_persisted": False,
            "canonical_payload_persisted": False,
        }


RawTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    tuple[int, Mapping[str, Any]],
]


def _extract_schema_path(message: str) -> str | None:
    patterns = (
        r"(?i)(?:schema|parameters?)(?:\s+path|\s+at|\s*=|:)\s*['\"]?([/$A-Za-z0-9_.\[\]-]{1,200})",
        r"(\$defs(?:[/.$][A-Za-z0-9_$-]+)*)",
        r"(?i)(properties(?:[/.$][A-Za-z0-9_$-]+)+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return None


def _error_category(status: int, message: str) -> str:
    lowered = message.lower()
    if status in {400, 422} and any(
        token in lowered
        for token in ("schema", "$defs", "$def", "additionalproperties", "strict")
    ):
        return "STRICT_SCHEMA_REJECTED"
    if status in {401, 403}:
        return "AUTHENTICATION_REJECTED"
    if status == 429:
        return "RATE_LIMITED"
    if 500 <= status <= 599:
        return "PROVIDER_SERVER_ERROR"
    return "HTTP_REJECTED"


def _sanitized_http_error(exc: urllib.error.HTTPError, latency_ms: float) -> StrictHandshakeError:
    raw = exc.read()
    message = ""
    try:
        body = json.loads(raw.decode("utf-8"))
        error = body.get("error", {}) if isinstance(body, Mapping) else {}
        if isinstance(error, Mapping):
            message = str(error.get("message") or "")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        message = ""
    category = _error_category(int(exc.code), message)
    return StrictHandshakeError(
        "E_STRICT_SCHEMA_REJECTED" if category == "STRICT_SCHEMA_REJECTED" else f"E_HTTP_{int(exc.code)}",
        http_status=int(exc.code),
        category=category,
        schema_parameter_path=_extract_schema_path(message),
        latency_ms=latency_ms,
        backend_request_count=1,
    )


def _default_transport(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[int, Mapping[str, Any]]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise StrictHandshakeError(
            "E_HTTP_TIMEOUT", category="NETWORK_ERROR", backend_request_count=1
        ) from exc
    except urllib.error.URLError as exc:
        raise StrictHandshakeError(
            "E_HTTP_NETWORK", category="NETWORK_ERROR", backend_request_count=1
        ) from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictHandshakeError(
            "E_RESPONSE_JSON", http_status=status,
            category="PROTOCOL_ERROR", backend_request_count=1,
        ) from exc
    if not isinstance(value, Mapping):
        raise StrictHandshakeError(
            "E_RESPONSE_TYPE", http_status=status,
            category="PROTOCOL_ERROR", backend_request_count=1,
        )
    return status, value


class DeepSeekStrictToolBetaAdapter:
    """Exact-profile adapter with no retry, repair, fallback, or cache."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        profile: str = PROFILE_NAME,
        base_url: str = STRICT_BASE_URL,
        timeout_seconds: float = 180.0,
        transport: RawTransport | None = None,
    ) -> None:
        if profile != PROFILE_NAME:
            raise ValueError("E_STRICT_PROFILE_IDENTITY")
        if base_url != STRICT_BASE_URL:
            raise ValueError("E_STRICT_BASE_URL_IDENTITY")
        if model != "deepseek-v4-pro":
            raise ValueError("E_STRICT_MODEL_IDENTITY")
        if not api_key:
            raise ValueError("E_API_KEY_REQUIRED")
        self.api_key = api_key
        self.model = model
        self.profile = profile
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _default_transport
        self.backend_request_count = 0

    def build_payload(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        payload = build_mock_tool_payload(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
        )
        payload.update({
            "temperature": 0.0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "disabled"},
            "stream": False,
        })
        return payload

    def handshake(self, messages: Sequence[Mapping[str, str]]) -> HandshakeOutcome:
        if self.backend_request_count >= 1:
            raise StrictHandshakeError("E_BACKEND_REQUEST_CAP")
        payload = self.build_payload(messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.backend_request_count += 1
        started = time.perf_counter()
        try:
            status, response = self.transport(
                STRICT_CHAT_ENDPOINT, headers, payload, self.timeout_seconds
            )
        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            raise _sanitized_http_error(exc, latency_ms) from exc
        except StrictHandshakeError as exc:
            if exc.latency_ms == 0.0:
                exc.latency_ms = (time.perf_counter() - started) * 1000.0
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            choice = response["choices"][0]
            message = choice["message"]
            tool_calls = message["tool_calls"]
            finish_reason_raw = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise StrictHandshakeError(
                "E_TOOL_RESPONSE_SHAPE", http_status=status,
                category="PROTOCOL_ERROR", latency_ms=latency_ms,
                backend_request_count=1,
            ) from exc
        if not isinstance(tool_calls, list):
            raise StrictHandshakeError(
                "E_TOOL_CALL_COUNT", http_status=status,
                category="PROTOCOL_ERROR", latency_ms=latency_ms,
                backend_request_count=1,
            )
        tool_call_count = len(tool_calls)
        tool_name = ""
        if tool_call_count == 1 and isinstance(tool_calls[0], Mapping):
            function = tool_calls[0].get("function", {})
            if isinstance(function, Mapping):
                tool_name = str(function.get("name") or "")
        try:
            canonical = parse_mock_tool_call_response(response)
        except Exception as exc:
            code = getattr(exc, "code", "E_TOOL_ARGUMENTS_INVALID")
            raise StrictHandshakeError(
                str(code), http_status=status,
                category="TOOL_ARGUMENT_VALIDATION_ERROR",
                latency_ms=latency_ms, backend_request_count=1,
            ) from exc
        canonical_validation = validate_intent_result(canonical)
        if not canonical_validation["valid"]:
            raise StrictHandshakeError(
                "E_CANONICAL_VALIDATION", http_status=status,
                category="CANONICAL_VALIDATION_ERROR",
                latency_ms=latency_ms, backend_request_count=1,
            )
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise StrictHandshakeError(
                "E_USAGE_MISSING", http_status=status,
                category="PROTOCOL_ERROR", latency_ms=latency_ms,
                backend_request_count=1,
            )
        try:
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            cache_read = int(usage.get("prompt_cache_hit_tokens", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise StrictHandshakeError(
                "E_USAGE_INVALID", http_status=status,
                category="PROTOCOL_ERROR", latency_ms=latency_ms,
                backend_request_count=1,
            ) from exc
        return HandshakeOutcome(
            canonical=canonical,
            http_status=status,
            actual_model=str(response.get("model") or ""),
            finish_reason=(str(finish_reason_raw) if finish_reason_raw is not None else None),
            tool_name=tool_name,
            tool_call_count=tool_call_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cache_read,
            latency_ms=latency_ms,
            backend_request_count=1,
            transport_valid=True,
            canonical_valid=True,
        )


class SingleEntryHandshakeCache:
    """In-memory cache used only to prove a zero-backend local replay."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._value: HandshakeOutcome | None = None
        self.hits = 0
        self.misses = 0

    def execute(
        self,
        key: str,
        adapter: DeepSeekStrictToolBetaAdapter,
        messages: Sequence[Mapping[str, str]],
    ) -> HandshakeOutcome:
        if self._key == key and self._value is not None:
            self.hits += 1
            return replace(
                self._value,
                backend_request_count=0,
                latency_ms=0.0,
                cache_hit=True,
            )
        if self._value is not None:
            raise StrictHandshakeError("E_CACHE_KEY_MISMATCH")
        self.misses += 1
        outcome = adapter.handshake(messages)
        self._key = key
        self._value = outcome
        return outcome


__all__ = [
    "DeepSeekStrictToolBetaAdapter", "HandshakeOutcome", "MAX_OUTPUT_TOKENS",
    "PROFILE_NAME", "STRICT_BASE_URL", "STRICT_CHAT_ENDPOINT",
    "SingleEntryHandshakeCache", "StrictHandshakeError",
]
