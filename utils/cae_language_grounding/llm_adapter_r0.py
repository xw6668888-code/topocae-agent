"""Provider-aware, offline-testable JSON adapter for P4 language baselines.

This package is the R0 implementation.  The sibling ``llm_adapter.py`` remains
as the frozen legacy provider used for regression comparison.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from utils.cae_intent import validate_intent_result
from .records import assert_inference_input_safe
from .reliable_runner import AtomicJSONCache, build_request_hash, redact_secrets


ROOT = Path(__file__).resolve().parents[2]
INTENT_RESULT_SCHEMA_PATH = ROOT / "schemas" / "cae_intent_result.schema.json"
STRUCTURED_PROMPT_PATH = ROOT / "prompts" / "p4_a1_structured_llm_v1.md"
DEFAULT_SEED = 20260401


class LLMRuntimeUnavailable(RuntimeError):
    pass


class LLMProtocolError(RuntimeError):
    def __init__(self, code: str, detail: str = "", *, request_count: int = 0,
                 latency_ms: float = 0.0) -> None:
        self.code = code
        self.request_count = request_count
        self.latency_ms = latency_ms
        super().__init__(f"{code}:{detail}" if detail else code)


class LLMTransportError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None,
                 retryable: bool, request_count: int = 0) -> None:
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.request_count = request_count
        super().__init__(code)


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    response_format: str
    send_seed: bool
    send_temperature: bool
    send_thinking: bool
    send_max_tokens: bool


PROVIDER_PROFILES = {
    "openai_json_schema": ProviderProfile(
        "openai_json_schema", "json_schema", True, True, False, True
    ),
    "deepseek_json_object": ProviderProfile(
        "deepseek_json_object", "json_object", False, True, True, True
    ),
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    return _env_bool(name, False) if name in os.environ else None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    base_url: str
    model: str
    api_key: str | None
    seed: int
    timeout_seconds: float
    revision: str
    dtype: str
    quantization: str
    backend: str
    gpu: str
    context_length: int | None
    constrained_decoding: bool
    provider_profile: str = "openai_json_schema"
    temperature: float = 0.0
    max_tokens: int = 4096
    send_seed: bool | None = None
    send_temperature: bool | None = None
    send_thinking: bool | None = None
    send_max_tokens: bool | None = None
    max_http_retries: int = 2
    retry_backoff_seconds: float = 1.0
    cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "LLMRuntimeConfig":
        context_raw = os.environ.get("TOPOCAE_LLM_CONTEXT_LENGTH", "").strip()
        cache_default = str(
            ROOT / "output_p4_a1_language_grounding_v1" / "runtime_cache"
        )
        return cls(
            base_url=os.environ.get("TOPOCAE_LLM_BASE_URL", "").strip(),
            model=os.environ.get("TOPOCAE_LLM_MODEL", "").strip(),
            api_key=os.environ.get("TOPOCAE_LLM_API_KEY") or None,
            seed=int(os.environ.get("TOPOCAE_LLM_SEED", str(DEFAULT_SEED))),
            timeout_seconds=float(os.environ.get("TOPOCAE_LLM_TIMEOUT_SECONDS", "120")),
            revision=os.environ.get("TOPOCAE_LLM_REVISION", "UNSPECIFIED"),
            dtype=os.environ.get("TOPOCAE_LLM_DTYPE", "UNSPECIFIED"),
            quantization=os.environ.get("TOPOCAE_LLM_QUANTIZATION", "UNSPECIFIED"),
            backend=os.environ.get("TOPOCAE_LLM_BACKEND", "openai-compatible"),
            gpu=os.environ.get("TOPOCAE_LLM_GPU", "UNSPECIFIED"),
            context_length=int(context_raw) if context_raw else None,
            constrained_decoding=_env_bool("TOPOCAE_LLM_CONSTRAINED_DECODING", True),
            provider_profile=os.environ.get(
                "TOPOCAE_LLM_PROVIDER_PROFILE", "openai_json_schema"
            ).strip(),
            temperature=float(os.environ.get("TOPOCAE_LLM_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("TOPOCAE_LLM_MAX_TOKENS", "4096")),
            send_seed=_env_optional_bool("TOPOCAE_LLM_SEND_SEED"),
            send_temperature=_env_optional_bool("TOPOCAE_LLM_SEND_TEMPERATURE"),
            send_thinking=_env_optional_bool("TOPOCAE_LLM_SEND_THINKING"),
            send_max_tokens=_env_optional_bool("TOPOCAE_LLM_SEND_MAX_TOKENS"),
            max_http_retries=int(os.environ.get("TOPOCAE_LLM_HTTP_RETRIES", "2")),
            retry_backoff_seconds=float(
                os.environ.get("TOPOCAE_LLM_RETRY_BACKOFF_SECONDS", "1")
            ),
            cache_dir=os.environ.get(
                "TOPOCAE_LLM_CACHE_DIR", cache_default
            ).strip() or None,
        )

    @property
    def profile(self) -> ProviderProfile:
        try:
            profile = PROVIDER_PROFILES[self.provider_profile]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported provider profile: {self.provider_profile!r}"
            ) from exc
        overrides = {
            name: getattr(self, name)
            for name in ("send_seed", "send_temperature", "send_thinking", "send_max_tokens")
            if getattr(self, name) is not None
        }
        return replace(profile, **overrides)

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def missing_required(self) -> list[str]:
        missing = []
        if not self.base_url:
            missing.append("TOPOCAE_LLM_BASE_URL")
        if not self.model:
            missing.append("TOPOCAE_LLM_MODEL")
        return missing

    def public_metadata(self) -> dict[str, Any]:
        profile = self.profile
        return redact_secrets({
            "runtime_available": self.available,
            "base_url_configured": bool(self.base_url),
            "api_key_configured": bool(self.api_key),
            "provider_profile": profile.name,
            "response_format": profile.response_format,
            "model": self.model or "UNCONFIGURED",
            "revision": self.revision,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "backend": self.backend,
            "gpu": self.gpu,
            "context_length": self.context_length,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "payload_switches": {
                "seed": profile.send_seed,
                "temperature": profile.send_temperature,
                "thinking": profile.send_thinking,
                "max_tokens": profile.send_max_tokens,
            },
            "constrained_decoding_requested": self.constrained_decoding,
            "max_schema_repair_retries": 1,
            "max_http_retries": self.max_http_retries,
            "cache_enabled": bool(self.cache_dir),
            "silent_json_repair": False,
            "silent_protocol_fallback": False,
        })


@dataclass(frozen=True)
class JSONInferenceOutcome:
    value: Mapping[str, Any] | None
    schema_valid_first_round: bool
    schema_valid_after_retry: bool
    retry_count: int
    failure_code: str | None
    validation_errors: tuple[Mapping[str, Any], ...]
    input_tokens: int
    output_tokens: int
    backend_request_count: int
    actual_model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    http_retry_count: int = 0
    cache_hit: bool = False
    request_hash: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        return redact_secrets({
            "actual_model": self.actual_model,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "http_retry_count": self.http_retry_count,
            "backend_request_count": self.backend_request_count,
            "cache_hit": self.cache_hit,
            "request_hash": self.request_hash,
        })

    def cache_value(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))

    @classmethod
    def from_cache(cls, value: Mapping[str, Any], request_hash: str):
        try:
            return cls(
                value=value.get("value"),
                schema_valid_first_round=bool(value["schema_valid_first_round"]),
                schema_valid_after_retry=bool(value["schema_valid_after_retry"]),
                retry_count=int(value["retry_count"]),
                failure_code=value.get("failure_code"),
                validation_errors=tuple(value.get("validation_errors", ())),
                input_tokens=int(value.get("input_tokens", 0)),
                output_tokens=int(value.get("output_tokens", 0)),
                backend_request_count=0,
                actual_model=value.get("actual_model"),
                system_fingerprint=value.get("system_fingerprint"),
                finish_reason=value.get("finish_reason"),
                latency_ms=float(value.get("latency_ms", 0.0)),
                cache_hit=True,
                request_hash=request_hash,
            )
        except (KeyError, TypeError, ValueError):
            return None


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]
Validator = Callable[[Any], Mapping[str, Any]]


def _strict_object_json(text: str) -> Mapping[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                duplicates.append(str(key))
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=pairs_hook)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError("E_JSON_PARSE", exc.msg) from exc
    if duplicates:
        raise LLMProtocolError(
            "E_JSON_DUPLICATE_KEY", ",".join(sorted(set(duplicates)))
        )
    if not isinstance(value, Mapping):
        raise LLMProtocolError("E_JSON_TYPE", "top-level JSON must be an object")
    return value


def _default_transport(url: str, headers: Mapping[str, str],
                       payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raise LLMTransportError(
            f"E_HTTP_{status}", status_code=status,
            retryable=status == 429 or 500 <= status <= 599,
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMTransportError("E_HTTP_TIMEOUT", retryable=True) from exc
    except urllib.error.URLError as exc:
        code = "E_HTTP_TIMEOUT" if isinstance(
            exc.reason, (TimeoutError, socket.timeout)
        ) else "E_HTTP_NETWORK"
        raise LLMTransportError(code, retryable=True) from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMProtocolError("E_RUNTIME_RESPONSE_JSON") from exc
    if not isinstance(value, Mapping):
        raise LLMProtocolError("E_RUNTIME_RESPONSE_TYPE")
    return value


@dataclass(frozen=True)
class _Response:
    value: Mapping[str, Any]
    input_tokens: int
    output_tokens: int
    actual_model: str
    system_fingerprint: str | None
    finish_reason: str | None
    latency_ms: float
    request_count: int
    http_retry_count: int


class OpenAICompatibleJSONClient:
    """Explicit-provider client with HTTP retries, schema repair, and cache."""

    def __init__(self, config: LLMRuntimeConfig, *, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.transport = transport or _default_transport
        self.sleep = sleep
        self.cache = AtomicJSONCache(config.cache_dir) if config.cache_dir else None

    def _endpoint(self) -> str:
        parsed = urlsplit(self.config.base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("TOPOCAE_LLM_BASE_URL must be an absolute HTTP(S) URL")
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            endpoint_path = path
        elif path.endswith("/v1"):
            endpoint_path = path + "/chat/completions"
        else:
            endpoint_path = path + "/v1/chat/completions"
        return urlunsplit(
            (parsed.scheme, parsed.netloc, endpoint_path, parsed.query, parsed.fragment)
        )

    def _payload(self, messages, schema_name, schema) -> dict[str, Any]:
        profile = self.config.profile
        payload: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if profile.send_temperature:
            payload["temperature"] = self.config.temperature
        if profile.send_seed:
            payload["seed"] = self.config.seed
        if profile.send_max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        if profile.send_thinking:
            payload["thinking"] = {"type": "disabled"}
        if profile.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.config.constrained_decoding:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        return payload

    def _send(self, headers, payload):
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.transport(
                    self._endpoint(), headers, payload, self.config.timeout_seconds
                )
                if not isinstance(response, Mapping):
                    raise LLMProtocolError("E_RUNTIME_RESPONSE_TYPE")
                return response, attempts, attempts - 1
            except LLMTransportError as exc:
                if not exc.retryable or attempts > self.config.max_http_retries:
                    exc.request_count = attempts
                    raise
                self.sleep(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))

    def _request_once(self, messages, schema_name, schema) -> _Response:
        if not self.config.available:
            raise LLMRuntimeUnavailable(
                "Missing required runtime variables: " + ", ".join(self.config.missing_required)
            )
        payload = self._payload(messages, schema_name, schema)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        started = time.perf_counter()
        response, request_count, http_retries = self._send(headers, payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
            finish_raw = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError(
                "E_RUNTIME_RESPONSE_SHAPE", request_count=request_count,
                latency_ms=latency_ms,
            ) from exc
        finish_reason = str(finish_raw) if finish_raw is not None else None
        if finish_reason == "length":
            raise LLMProtocolError(
                "E_RUNTIME_TRUNCATED", request_count=request_count,
                latency_ms=latency_ms,
            )
        if isinstance(content, str) and not content.strip():
            raise LLMProtocolError(
                "E_RUNTIME_EMPTY_CONTENT", request_count=request_count,
                latency_ms=latency_ms,
            )
        try:
            if isinstance(content, Mapping):
                value = dict(content)
            elif isinstance(content, str):
                value = _strict_object_json(content)
            else:
                raise LLMProtocolError("E_RUNTIME_CONTENT_TYPE")
        except LLMProtocolError as exc:
            exc.request_count = request_count
            exc.latency_ms = latency_ms
            raise
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise LLMProtocolError(
                "E_RUNTIME_USAGE_MISSING", request_count=request_count,
                latency_ms=latency_ms,
            )
        try:
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMProtocolError(
                "E_RUNTIME_USAGE_MISSING", request_count=request_count,
                latency_ms=latency_ms,
            ) from exc
        return _Response(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            actual_model=str(response.get("model") or self.config.model),
            system_fingerprint=(
                str(response["system_fingerprint"])
                if response.get("system_fingerprint") is not None else None
            ),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            request_count=request_count,
            http_retry_count=http_retries,
        )

    def infer_json(self, *, inference_input, system_prompt, schema_name,
                   schema, validator) -> JSONInferenceOutcome:
        assert_inference_input_safe(inference_input)
        request_hash = build_request_hash(
            provider=self.config.provider_profile,
            model=self.config.model,
            revision=self.config.revision,
            prompt=system_prompt,
            schema=schema,
            baseline=schema_name,
            inference_input=inference_input,
            generation={
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "max_tokens": self.config.max_tokens,
                "profile": asdict(self.config.profile),
            },
        )
        if self.cache is not None:
            cached = self.cache.get(schema_name, request_hash)
            if cached is not None:
                hit = JSONInferenceOutcome.from_cache(cached, request_hash)
                if hit is not None and hit.schema_valid_after_retry:
                    return hit

        user_text = json.dumps(
            dict(inference_input), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        input_tokens = output_tokens = request_count = http_retries = 0
        latency_ms = 0.0
        errors: list[Mapping[str, Any]] = []
        failure_code = "E_SCHEMA_VALIDATION"
        actual_model = fingerprint = finish_reason = None
        for attempt in range(2):
            try:
                response = self._request_once(messages, schema_name, schema)
                input_tokens += response.input_tokens
                output_tokens += response.output_tokens
                request_count += response.request_count
                http_retries += response.http_retry_count
                latency_ms += response.latency_ms
                actual_model = response.actual_model
                fingerprint = response.system_fingerprint
                finish_reason = response.finish_reason
                value = response.value
                validation = validator(value)
                errors = list(validation.get("errors", []))
                if validation.get("valid"):
                    outcome = JSONInferenceOutcome(
                        value, attempt == 0, True, attempt, None, (),
                        input_tokens, output_tokens, request_count,
                        actual_model, fingerprint, finish_reason, latency_ms,
                        http_retries, False, request_hash,
                    )
                    if self.cache is not None:
                        self.cache.put(schema_name, request_hash, outcome.cache_value())
                    return outcome
                failure_code = "E_SCHEMA_VALIDATION"
            except LLMProtocolError as exc:
                value = None
                failure_code = exc.code
                request_count += exc.request_count
                latency_ms += exc.latency_ms
                errors = [{"code": failure_code, "message": str(exc)}]
            if attempt == 0:
                messages += [
                    {
                        "role": "assistant",
                        "content": json.dumps(value, ensure_ascii=False)
                        if value is not None else "<invalid-json>",
                    },
                    {
                        "role": "user",
                        "content": "The previous answer was invalid. Return a complete replacement JSON object matching the schema. Errors: "
                        + json.dumps(errors, ensure_ascii=False),
                    },
                ]
        return JSONInferenceOutcome(
            None, False, False, 1, failure_code, tuple(errors),
            input_tokens, output_tokens, request_count, actual_model,
            fingerprint, finish_reason, latency_ms, http_retries, False,
            request_hash,
        )

    def infer_structured(self, inference_input: Mapping[str, str]) -> JSONInferenceOutcome:
        schema = json.loads(INTENT_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
        prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
        return self.infer_json(
            inference_input=inference_input,
            system_prompt=prompt,
            schema_name="topocae_cae_intent_result_v1",
            schema=schema,
            validator=validate_intent_result,
        )


__all__ = [
    "DEFAULT_SEED", "JSONInferenceOutcome", "LLMProtocolError",
    "LLMRuntimeConfig", "LLMRuntimeUnavailable", "LLMTransportError",
    "OpenAICompatibleJSONClient", "PROVIDER_PROFILES", "ProviderProfile",
]
