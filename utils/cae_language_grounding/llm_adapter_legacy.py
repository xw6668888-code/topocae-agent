"""Strict OpenAI-compatible JSON adapter for P4-A1 language baselines.

The adapter is inference-only.  It never downloads models, trains parameters,
reads geometry, or accepts answer-bearing fields in its language input.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from utils.cae_intent import validate_intent_result

from .records import assert_inference_input_safe


ROOT = Path(__file__).resolve().parents[2]
INTENT_RESULT_SCHEMA_PATH = ROOT / "schemas" / "cae_intent_result.schema.json"
STRUCTURED_PROMPT_PATH = ROOT / "prompts" / "p4_a1_structured_llm_v1.md"
DEFAULT_SEED = 20260401


class LLMRuntimeUnavailable(RuntimeError):
    """Raised when an endpoint/model was not supplied through the environment."""


class LLMProtocolError(RuntimeError):
    """Raised when the HTTP backend violates the declared response protocol."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    @classmethod
    def from_env(cls) -> "LLMRuntimeConfig":
        context_raw = os.environ.get("TOPOCAE_LLM_CONTEXT_LENGTH", "").strip()
        return cls(
            base_url=os.environ.get("TOPOCAE_LLM_BASE_URL", "").strip(),
            model=os.environ.get("TOPOCAE_LLM_MODEL", "").strip(),
            api_key=os.environ.get("TOPOCAE_LLM_API_KEY") or None,
            seed=int(os.environ.get("TOPOCAE_LLM_SEED", str(DEFAULT_SEED))),
            timeout_seconds=float(
                os.environ.get("TOPOCAE_LLM_TIMEOUT_SECONDS", "120")
            ),
            revision=os.environ.get("TOPOCAE_LLM_REVISION", "UNSPECIFIED"),
            dtype=os.environ.get("TOPOCAE_LLM_DTYPE", "UNSPECIFIED"),
            quantization=os.environ.get(
                "TOPOCAE_LLM_QUANTIZATION", "UNSPECIFIED"
            ),
            backend=os.environ.get("TOPOCAE_LLM_BACKEND", "openai-compatible"),
            gpu=os.environ.get("TOPOCAE_LLM_GPU", "UNSPECIFIED"),
            context_length=int(context_raw) if context_raw else None,
            constrained_decoding=_env_bool(
                "TOPOCAE_LLM_CONSTRAINED_DECODING", True
            ),
        )

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
        """Return reproducibility metadata without endpoint or secret values."""

        return {
            "runtime_available": self.available,
            "base_url_configured": bool(self.base_url),
            "api_key_configured": bool(self.api_key),
            "model": self.model or "UNCONFIGURED",
            "revision": self.revision,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "backend": self.backend,
            "gpu": self.gpu,
            "context_length": self.context_length,
            "temperature": 0,
            "seed": self.seed,
            "constrained_decoding_requested": self.constrained_decoding,
            "max_repair_retries": 1,
            "silent_json_repair": False,
        }


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


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]
Validator = Callable[[Any], Mapping[str, Any]]


def _strict_object_json(text: str) -> Mapping[str, Any]:
    duplicate_keys: list[str] = []

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                duplicate_keys.append(str(key))
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError(f"E_JSON_PARSE:{exc.msg}") from exc
    if duplicate_keys:
        raise LLMProtocolError(
            "E_JSON_DUPLICATE_KEY:" + ",".join(sorted(set(duplicate_keys)))
        )
    if not isinstance(value, Mapping):
        raise LLMProtocolError("E_JSON_TYPE:top-level JSON must be an object")
    return value


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMProtocolError(f"E_RUNTIME_HTTP:{type(exc).__name__}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMProtocolError("E_RUNTIME_RESPONSE_JSON") from exc
    if not isinstance(value, Mapping):
        raise LLMProtocolError("E_RUNTIME_RESPONSE_TYPE")
    return value


class OpenAICompatibleJSONClient:
    """OpenAI-compatible chat-completions client with one explicit retry."""

    def __init__(
        self,
        config: LLMRuntimeConfig,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or _default_transport

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _request_once(
        self,
        *,
        messages: list[Mapping[str, str]],
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], int, int]:
        if not self.config.available:
            raise LLMRuntimeUnavailable(
                "Missing required runtime variables: "
                + ", ".join(self.config.missing_required)
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "seed": self.config.seed,
        }
        if self.config.constrained_decoding:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = self.transport(
            self._endpoint(),
            headers,
            payload,
            self.config.timeout_seconds,
        )
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError("E_RUNTIME_RESPONSE_SHAPE") from exc
        if isinstance(content, Mapping):
            value = dict(content)
        elif isinstance(content, str):
            value = _strict_object_json(content)
        else:
            raise LLMProtocolError("E_RUNTIME_CONTENT_TYPE")
        usage = response.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        return value, input_tokens, output_tokens

    def infer_json(
        self,
        *,
        inference_input: Mapping[str, Any],
        system_prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        validator: Validator,
    ) -> JSONInferenceOutcome:
        assert_inference_input_safe(inference_input)
        user_text = json.dumps(
            dict(inference_input),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        messages: list[Mapping[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        total_input_tokens = 0
        total_output_tokens = 0
        errors: list[Mapping[str, Any]] = []
        for attempt in range(2):
            try:
                value, input_tokens, output_tokens = self._request_once(
                    messages=messages,
                    schema_name=schema_name,
                    schema=schema,
                )
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                result = validator(value)
                errors = list(result.get("errors", []))
                if result.get("valid"):
                    return JSONInferenceOutcome(
                        value=value,
                        schema_valid_first_round=attempt == 0,
                        schema_valid_after_retry=True,
                        retry_count=attempt,
                        failure_code=None,
                        validation_errors=(),
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        backend_request_count=attempt + 1,
                    )
                failure_code = "E_SCHEMA_VALIDATION"
            except LLMProtocolError as exc:
                value = None
                failure_code = str(exc).split(":", 1)[0]
                errors = [{"code": failure_code, "message": str(exc)}]
            if attempt == 0:
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": (
                            json.dumps(value, ensure_ascii=False)
                            if value is not None
                            else "<invalid-json>"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous answer was invalid. Return a complete "
                            "replacement JSON object matching the schema. Errors: "
                            + json.dumps(errors, ensure_ascii=False)
                        ),
                    },
                ]
        return JSONInferenceOutcome(
            value=None,
            schema_valid_first_round=False,
            schema_valid_after_retry=False,
            retry_count=1,
            failure_code=failure_code,
            validation_errors=tuple(errors),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            backend_request_count=2,
        )

    def infer_structured(
        self, inference_input: Mapping[str, str]
    ) -> JSONInferenceOutcome:
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
    "DEFAULT_SEED",
    "JSONInferenceOutcome",
    "LLMProtocolError",
    "LLMRuntimeConfig",
    "LLMRuntimeUnavailable",
    "OpenAICompatibleJSONClient",
]
