"""Strict cache identity checks for G09 prediction reuse."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


REQUIRED_FIELDS = (
    "schema",
    "dataset",
    "split",
    "configuration",
    "seed",
    "sample_count",
    "training_data",
    "architecture",
    "checkpoint",
    "normalization",
    "prediction_entry",
    "sample_manifest",
)


class CacheIdentityMismatch(ValueError):
    def __init__(self, mismatches: list[dict[str, Any]]) -> None:
        self.mismatches = mismatches
        fields = ", ".join(row["field"] for row in mismatches)
        super().__init__(f"Cache identity mismatch: {fields}")


def identity_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_seed_count_match(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> bool:
    """Document the insufficient G08 compatibility check used by the fixture."""
    return all(expected.get(field) == actual.get(field) for field in ("seed", "sample_count"))


def compare_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[dict[str, Any]]:
    mismatches = []
    for field in REQUIRED_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value != actual_value:
            mismatches.append(
                {"field": field, "expected": expected_value, "actual": actual_value}
            )
    return mismatches


def require_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> Mapping[str, Any]:
    mismatches = compare_identity(expected, actual)
    if mismatches:
        raise CacheIdentityMismatch(mismatches)
    return actual
