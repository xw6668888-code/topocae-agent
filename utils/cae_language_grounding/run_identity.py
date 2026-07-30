"""Canonical endpoint provenance and immutable P4 API run identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class RunIdentityError(RuntimeError):
    """Stable fail-closed run identity error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_backend_identity(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().split())
    if not normalized:
        raise RunIdentityError("E_BACKEND_IDENTITY_EMPTY")
    return normalized


def canonical_chat_endpoint(value: str) -> str:
    """Normalize a base or complete endpoint while retaining private identity."""

    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise RunIdentityError("E_ENDPOINT_INVALID")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    default_port = 80 if scheme == "http" else 443
    port = f":{parsed.port}" if parsed.port and parsed.port != default_port else ""
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += ":" + parsed.password
        userinfo += "@"
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = path + "/chat/completions"
    else:
        endpoint_path = path + "/v1/chat/completions"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, f"{userinfo}{host}{port}", endpoint_path, query, ""))


def endpoint_public_identity(value: str) -> dict[str, str]:
    canonical = canonical_chat_endpoint(value)
    parsed = urlsplit(canonical)
    port = f":{parsed.port}" if parsed.port is not None else ""
    return {
        "endpoint_sha256": sha256_bytes(canonical.encode("utf-8")),
        "endpoint_origin_redacted": f"{parsed.scheme}://{parsed.hostname}{port}",
    }


def build_run_manifest(
    *, endpoint: str, backend: str, provider_profile: str, model: str,
    revision: str, prompt: str, schema: Mapping[str, Any],
    runtime_config: Mapping[str, Any], baseline: str, split: str,
    offset: int, limit: int | None, shard_index: int, shard_count: int,
) -> dict[str, Any]:
    endpoint_identity = endpoint_public_identity(endpoint)
    identity = {
        "endpoint_sha256": endpoint_identity["endpoint_sha256"],
        "backend": normalize_backend_identity(backend),
        "provider_profile": str(provider_profile),
        "model": str(model),
        "revision": str(revision),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
        "runtime_config_sha256": sha256_bytes(canonical_json_bytes(runtime_config)),
        "baseline": str(baseline),
        "split": str(split),
        "offset": int(offset),
        "limit": None if limit is None else int(limit),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
    }
    full_hash = sha256_bytes(canonical_json_bytes(identity))
    return {
        "schema_version": "topocae-p4-api-run-identity/1",
        "run_id": f"p4api-{full_hash[:32]}",
        "identity_sha256": full_hash,
        "identity": identity,
        "public_endpoint": endpoint_identity,
        "immutable": True,
    }


def prepare_immutable_run_directory(
    root: str | Path, manifest: Mapping[str, Any]
) -> Path:
    run_id = str(manifest.get("run_id", ""))
    if not run_id.startswith("p4api-"):
        raise RunIdentityError("E_RUN_ID_INVALID")
    run_dir = Path(root) / run_id
    manifest_path = run_dir / "run_manifest.json"
    expected = canonical_json_bytes(manifest)
    if manifest_path.exists():
        try:
            actual_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunIdentityError("E_RUN_MANIFEST_CORRUPT") from exc
        if canonical_json_bytes(actual_value) != expected:
            raise RunIdentityError("E_RUN_ID_MANIFEST_MISMATCH")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunIdentityError("E_RUN_DIRECTORY_UNCLAIMED")
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".run_manifest.", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(expected + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return run_dir


__all__ = [
    "RunIdentityError", "build_run_manifest", "canonical_chat_endpoint",
    "canonical_json_bytes", "endpoint_public_identity",
    "normalize_backend_identity", "prepare_immutable_run_directory",
    "sha256_bytes",
]
