"""Canonical P4 R0.1 provider adapter.

The pre-supervisor implementation is retained explicitly as
``llm_adapter_r0``; the historical single-provider implementation is retained
as ``llm_adapter_legacy``.  This module is the only canonical import path.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping

from . import llm_adapter_r0 as _r0
from .reliable_runner import build_request_hash
from .run_identity import (
    canonical_chat_endpoint,
    endpoint_public_identity,
    normalize_backend_identity,
)


DEFAULT_SEED = _r0.DEFAULT_SEED
JSONInferenceOutcome = _r0.JSONInferenceOutcome
LLMProtocolError = _r0.LLMProtocolError
LLMRuntimeUnavailable = _r0.LLMRuntimeUnavailable
LLMTransportError = _r0.LLMTransportError
PROVIDER_PROFILES = _r0.PROVIDER_PROFILES
ProviderProfile = _r0.ProviderProfile


class LLMRuntimeConfig(_r0.LLMRuntimeConfig):
    """R0 config with public, credential-free endpoint provenance."""

    def public_metadata(self) -> dict[str, Any]:
        value = super().public_metadata()
        if self.base_url:
            value.update(endpoint_public_identity(self.base_url))
        else:
            value.update({
                "endpoint_sha256": None,
                "endpoint_origin_redacted": None,
            })
        value["backend"] = normalize_backend_identity(self.backend)
        return value


class OpenAICompatibleJSONClient(_r0.OpenAICompatibleJSONClient):
    """Provider client whose cache key binds endpoint and backend identity."""

    def _endpoint(self) -> str:
        return canonical_chat_endpoint(self.config.base_url)

    def infer_json(
        self,
        *,
        inference_input: Mapping[str, Any],
        system_prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        validator,
    ) -> JSONInferenceOutcome:
        _r0.assert_inference_input_safe(inference_input)
        request_hash = build_request_hash(
            provider=self.config.provider_profile,
            backend=self.config.backend,
            endpoint=self._endpoint(),
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
                "constrained_decoding": self.config.constrained_decoding,
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
                        self.cache.put(
                            schema_name, request_hash, outcome.cache_value()
                        )
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
                        "content": (
                            "The previous answer was invalid. Return a complete "
                            "replacement JSON object matching the schema. Errors: "
                            + json.dumps(errors, ensure_ascii=False)
                        ),
                    },
                ]
        return JSONInferenceOutcome(
            None, False, False, 1, failure_code, tuple(errors),
            input_tokens, output_tokens, request_count, actual_model,
            fingerprint, finish_reason, latency_ms, http_retries, False,
            request_hash,
        )


__all__ = [
    "DEFAULT_SEED", "JSONInferenceOutcome", "LLMProtocolError",
    "LLMRuntimeConfig", "LLMRuntimeUnavailable", "LLMTransportError",
    "OpenAICompatibleJSONClient", "PROVIDER_PROFILES", "ProviderProfile",
]
