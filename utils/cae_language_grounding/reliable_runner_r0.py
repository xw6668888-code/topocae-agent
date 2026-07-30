"""Offline-safe reliability primitives for P4 API baseline runners."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar
from urllib.parse import urlsplit, urlunsplit


T = TypeVar("T")
_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "bearer", "credential",
    "credentials", "password", "secret", "token", "access_token",
    "refresh_token",
}
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


class FormalTestLockedError(PermissionError):
    """Raised when this R0 runner is asked to access the formal test split."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_request_hash(
    *, provider: str, model: str, revision: str, prompt: str,
    schema: Mapping[str, Any], baseline: str,
    inference_input: Mapping[str, Any], generation: Mapping[str, Any],
) -> str:
    """Hash every frozen input that can alter a Structured/Direct response."""

    return sha256_json({
        "provider": provider,
        "model": model,
        "revision": revision,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": sha256_json(schema),
        "baseline": baseline,
        "input": inference_input,
        "generation": generation,
    })


class AtomicJSONCache:
    """Namespaced content-addressed cache using same-directory atomic replace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _namespace(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        if not cleaned:
            raise ValueError("Cache namespace must not be empty")
        return cleaned

    def path_for(self, namespace: str, request_hash: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise ValueError("request_hash must be lowercase SHA-256")
        return self.root / self._namespace(namespace) / f"{request_hash}.json"

    def get(self, namespace: str, request_hash: str) -> Mapping[str, Any] | None:
        path = self.path_for(namespace, request_hash)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, Mapping) or value.get("request_hash") != request_hash:
            return None
        payload = value.get("outcome")
        return payload if isinstance(payload, Mapping) else None

    def put(
        self, namespace: str, request_hash: str, outcome: Mapping[str, Any]
    ) -> Path:
        path = self.path_for(namespace, request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": "topocae-p4-api-cache/1",
            "request_hash": request_hash,
            "namespace": self._namespace(namespace),
            "outcome": redact_secrets(dict(outcome)),
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{request_hash}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return path


def deterministic_window(
    values: Iterable[T], *, offset: int = 0, limit: int | None = None,
    shard_index: int = 0, shard_count: int = 1,
) -> list[T]:
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("offset and limit must be non-negative")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    sharded = [
        value for index, value in enumerate(values)
        if index % shard_count == shard_index
    ]
    end = None if limit is None else offset + limit
    return sharded[offset:end]


def assert_split_authorized(split: str) -> None:
    """R0 is development-only; formal test cannot be unlocked in this module."""

    if split not in {"train", "val"}:
        raise FormalTestLockedError(
            "E_FORMAL_TEST_LOCKED:P4_API_R0 permits train/val only"
        )


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or parsed.hostname is None or parsed.username is None:
        return value
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((
        parsed.scheme,
        f"REDACTED@{parsed.hostname}{port}",
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))


def redact_secrets(value: Any) -> Any:
    """Recursively redact credentials while preserving numeric token metadata."""

    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith("_api_key"):
                result[str(key)] = "REDACTED"
            else:
                result[str(key)] = redact_secrets(child)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _BEARER_RE.sub("Bearer REDACTED", _redact_url(value))
    return value


__all__ = [
    "AtomicJSONCache", "FormalTestLockedError", "assert_split_authorized",
    "build_request_hash", "canonical_json_bytes", "deterministic_window",
    "redact_secrets", "sha256_json",
]
