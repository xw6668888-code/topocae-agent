from __future__ import annotations

from collections import Counter

from p1_c import formal_plan as core
from p1_c import formal_plan_v2 as revision2


PLAN_REVISION = "p1_c_clean_context_latin_plan_r3"
_REVISION2_SECONDARY_STREAM = revision2._secondary_stream


def _compatible_pair(left: str, right: str) -> bool:
    return left != right and {left, right} != {"rib", "web"}


def _pair_compatible(values: list[str]) -> list[str]:
    original = Counter(values)
    counts = Counter(values)
    result: list[str] = []
    while counts:
        ordered = sorted(counts, key=lambda value: (-counts[value], value))
        selected: tuple[str, str] | None = None
        for left in ordered:
            for right in ordered:
                if _compatible_pair(left, right):
                    selected = (left, right)
                    break
            if selected is not None:
                break
        if selected is None:
            raise AssertionError(f"cannot form compatible triple pairs from {dict(counts)}")
        result.extend(selected)
        for value in selected:
            counts[value] -= 1
            if counts[value] == 0:
                del counts[value]
    if Counter(result) != original:
        raise AssertionError("compatible triple pairing changed secondary quotas")
    if any(not _compatible_pair(result[index], result[index + 1]) for index in range(0, len(result), 2)):
        raise AssertionError("compatible triple pairing retained a forbidden pair")
    return result


def _secondary_stream(primary_index: int) -> list[str]:
    stream = _REVISION2_SECONDARY_STREAM(primary_index)
    return stream[:120] + _pair_compatible(stream[120:])


core._secondary_stream = _secondary_stream
core.PLAN_REVISION = PLAN_REVISION

CONTEXT_QUOTAS = core.CONTEXT_QUOTAS
PlannedSample = core.PlannedSample
audit_plan = core.audit_plan
build_plan = core.build_plan
compatibility_matrix = core.compatibility_matrix
load_plan = core.load_plan
sha256_file = core.sha256_file
write_plan = core.write_plan


__all__ = [
    "CONTEXT_QUOTAS",
    "PLAN_REVISION",
    "PlannedSample",
    "audit_plan",
    "build_plan",
    "compatibility_matrix",
    "load_plan",
    "sha256_file",
    "write_plan",
]
