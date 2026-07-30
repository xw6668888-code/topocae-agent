from __future__ import annotations

from collections import Counter
from functools import lru_cache

from p1_c import formal_plan as core
from p1_c.all_features import MACRO_CLASSES


PLAN_REVISION = "p1_c_clean_context_constrained_latin_plan_r4"
FORBIDDEN_CLASS_PAIRS = (
    frozenset(("rib", "web")),
    frozenset(("fillet", "chamfer")),
)


def _compatible(left: str, right: str) -> bool:
    return left != right and frozenset((left, right)) not in FORBIDDEN_CLASS_PAIRS


def compatibility_matrix() -> dict[str, list[str]]:
    return {
        primary: [secondary for secondary in MACRO_CLASSES if _compatible(primary, secondary)]
        for primary in MACRO_CLASSES
    }


@lru_cache(maxsize=1)
def _all_secondary_streams() -> tuple[tuple[str, ...], ...]:
    class_count = len(MACRO_CLASSES)
    directed_counts: Counter[tuple[int, int]] = Counter()
    streams: list[list[str]] = [[] for _ in range(class_count)]
    for occurrence in range(200):
        preferred_shift = occurrence % (class_count - 1) + 1
        preferences: dict[int, list[int]] = {}
        for row in range(class_count):
            preferred = (row + preferred_shift) % class_count
            candidates = [
                column
                for column in range(class_count)
                if _compatible(MACRO_CLASSES[row], MACRO_CLASSES[column])
                and directed_counts[(row, column)] < 13
            ]
            candidates.sort(
                key=lambda column: (
                    directed_counts[(row, column)],
                    0 if column == preferred else 1,
                    (column - preferred) % class_count,
                    column,
                )
            )
            preferences[row] = candidates
        column_owner: dict[int, int] = {}

        def assign(row: int, seen: set[int]) -> bool:
            for column in preferences[row]:
                if column in seen:
                    continue
                seen.add(column)
                owner = column_owner.get(column)
                if owner is None or assign(owner, seen):
                    column_owner[column] = row
                    return True
            return False

        row_order = sorted(
            range(class_count),
            key=lambda row: (len(preferences[row]), -max((directed_counts[(row, column)] for column in preferences[row]), default=0), row),
        )
        for row in row_order:
            if not assign(row, set()):
                raise AssertionError(f"could not construct constrained Latin permutation at occurrence {occurrence}")
        if len(column_owner) != class_count:
            raise AssertionError(f"Latin permutation is incomplete at occurrence {occurrence}")
        row_to_column = {row: column for column, row in column_owner.items()}
        for row in range(class_count):
            column = row_to_column[row]
            directed_counts[(row, column)] += 1
            streams[row].append(MACRO_CLASSES[column])
    for row, stream in enumerate(streams):
        streams[row] = stream[:120] + _pair_compatible(stream[120:])
    if max(directed_counts.values(), default=0) > 13:
        raise AssertionError("constrained Latin directed-pair cap was exceeded")
    aggregate = Counter(value for stream in streams for value in stream)
    if any(aggregate[macro] != 200 for macro in MACRO_CLASSES):
        raise AssertionError(f"constrained Latin column quota mismatch: {aggregate}")
    return tuple(tuple(stream) for stream in streams)


def _pair_compatible(values: list[str]) -> list[str]:
    original = Counter(values)
    counts = Counter(values)
    result: list[str] = []
    while counts:
        ordered = sorted(counts, key=lambda value: (-counts[value], value))
        selected: tuple[str, str] | None = None
        for left in ordered:
            for right in ordered:
                if _compatible(left, right):
                    selected = (left, right)
                    break
            if selected is not None:
                break
        if selected is None:
            raise AssertionError(f"cannot form compatible triple pair from {dict(counts)}")
        result.extend(selected)
        for value in selected:
            counts[value] -= 1
            if counts[value] == 0:
                del counts[value]
    if Counter(result) != original:
        raise AssertionError("triple pairing changed constrained Latin quotas")
    if any(not _compatible(result[index], result[index + 1]) for index in range(0, len(result), 2)):
        raise AssertionError("triple pairing retained a forbidden class pair")
    return result


def _secondary_stream(primary_index: int) -> list[str]:
    return list(_all_secondary_streams()[primary_index])


core._secondary_stream = _secondary_stream
core.compatibility_matrix = compatibility_matrix
core.PLAN_REVISION = PLAN_REVISION

CONTEXT_QUOTAS = core.CONTEXT_QUOTAS
PlannedSample = core.PlannedSample
audit_plan = core.audit_plan
build_plan = core.build_plan
load_plan = core.load_plan
sha256_file = core.sha256_file
write_plan = core.write_plan


__all__ = [
    "CONTEXT_QUOTAS",
    "FORBIDDEN_CLASS_PAIRS",
    "PLAN_REVISION",
    "PlannedSample",
    "audit_plan",
    "build_plan",
    "compatibility_matrix",
    "load_plan",
    "sha256_file",
    "write_plan",
]
