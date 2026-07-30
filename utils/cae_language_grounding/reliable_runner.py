"""R0.1 reliability primitives with endpoint/backend cache provenance."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .reliable_runner_r0 import (
    AtomicJSONCache,
    FormalTestLockedError,
    assert_split_authorized,
    canonical_json_bytes,
    deterministic_window,
    redact_secrets,
    sha256_json,
)
from .run_identity import canonical_chat_endpoint, normalize_backend_identity


def build_request_hash(
    *, provider: str, model: str, revision: str, prompt: str,
    schema: Mapping[str, Any], baseline: str,
    inference_input: Mapping[str, Any], generation: Mapping[str, Any],
    backend: str = "UNSPECIFIED_LEGACY",
    endpoint: str = "http://unspecified.invalid/v1/chat/completions",
) -> str:
    """Hash response-affecting inputs, including private endpoint identity."""

    endpoint_value = canonical_chat_endpoint(endpoint)
    return sha256_json({
        "provider": str(provider),
        "backend": normalize_backend_identity(backend),
        "endpoint_sha256": hashlib.sha256(
            endpoint_value.encode("utf-8")
        ).hexdigest(),
        "model": str(model),
        "revision": str(revision),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": sha256_json(schema),
        "baseline": str(baseline),
        "input": inference_input,
        "generation": generation,
    })


__all__ = [
    "AtomicJSONCache", "FormalTestLockedError", "assert_split_authorized",
    "build_request_hash", "canonical_json_bytes", "deterministic_window",
    "redact_secrets", "sha256_json",
]
