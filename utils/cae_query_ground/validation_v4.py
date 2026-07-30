"""Cached unique-payload P2 validation for the grouped pilot audit."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

from . import validation as base
from .validation_v2 import cached_sha256_file  # noqa: F401


_p2_module = base.p2_api()
_uncached_validate_intent_result = _p2_module.validate_intent_result


@lru_cache(maxsize=4096)
def _cached_p2(payload: str) -> dict[str, Any]:
    return _uncached_validate_intent_result(json.loads(payload))


def _validate_intent_result_cached(value: Any) -> dict[str, Any]:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return copy.deepcopy(_cached_p2(payload))


_p2_module.validate_intent_result = _validate_intent_result_cached

from .validation_v3 import audit_pilot, validate_manifest  # noqa: E402

__all__ = ["audit_pilot", "validate_manifest"]
