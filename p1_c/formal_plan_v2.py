from __future__ import annotations

from collections import Counter
import heapq

from p1_c import formal_plan as core


PLAN_REVISION = "p1_c_clean_context_latin_plan_r2"
_ORIGINAL_SECONDARY_STREAM = core._secondary_stream


def _pair_distinct(values: list[str]) -> list[str]:
    """Reorder a multiset so each consecutive pair has distinct classes."""
    original = Counter(values)
    heap = [(-count, value) for value, count in original.items()]
    heapq.heapify(heap)
    result: list[str] = []
    while heap:
        first_count, first_value = heapq.heappop(heap)
        if not heap:
            raise AssertionError(f"cannot form distinct secondary pairs from {dict(original)}")
        second_count, second_value = heapq.heappop(heap)
        result.extend((first_value, second_value))
        first_count += 1
        second_count += 1
        if first_count < 0:
            heapq.heappush(heap, (first_count, first_value))
        if second_count < 0:
            heapq.heappush(heap, (second_count, second_value))
    if Counter(result) != original:
        raise AssertionError("triple pairing changed the secondary quota")
    if any(result[index] == result[index + 1] for index in range(0, len(result), 2)):
        raise AssertionError("triple pairing retained a duplicate secondary")
    return result


def _secondary_stream(primary_index: int) -> list[str]:
    stream = _ORIGINAL_SECONDARY_STREAM(primary_index)
    return stream[:120] + _pair_distinct(stream[120:])


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
