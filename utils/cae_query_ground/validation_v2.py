"""Performance-safe validation facade; validation rules remain unchanged."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import validation as base


_uncached_sha256_file = base.sha256_file


@lru_cache(maxsize=1024)
def cached_sha256_file(path: Path) -> str:
    return _uncached_sha256_file(Path(path))


# All records sharing a source graph see the same immutable, read-only hash result.
base.sha256_file = cached_sha256_file

audit_pilot = base.audit_pilot
semantic_identity = base.semantic_identity
split_leakage_audit = base.split_leakage_audit
validate_manifest = base.validate_manifest
validate_record = base.validate_record

__all__ = [
    "audit_pilot",
    "semantic_identity",
    "split_leakage_audit",
    "validate_manifest",
    "validate_record",
]
