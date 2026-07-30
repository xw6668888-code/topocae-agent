"""Split-safe read access for the frozen CAE-QueryGround 1.1 dataset.

The upstream v1 files are physically separated by split.  The P3-A1R supplement
is mixed, but its frozen generator has a deterministic line schedule.  Before
H4 this module decodes only train/val lines.  Test lines are crossed as opaque
bytes and are never UTF-8 decoded or JSON parsed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = ROOT / "output_cae_query_ground_v1"
SUPPLEMENT_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"

ALLOWED_DEVELOPMENT_SPLITS = frozenset({"train", "val"})
FEATURE_COUNT = 17
MODELS_PER_FEATURE = 36
TRAIN_MODELS_PER_FEATURE = 28
VAL_MODELS_PER_FEATURE = 4
CANONICAL_PER_MODEL = 5
RENDERINGS_PER_CANONICAL = 3

EXPECTED_SUPPLEMENT_LINES = {
    "canonical": FEATURE_COUNT * MODELS_PER_FEATURE * CANONICAL_PER_MODEL,
    "renderings": (
        FEATURE_COUNT
        * MODELS_PER_FEATURE
        * CANONICAL_PER_MODEL
        * RENDERINGS_PER_CANONICAL
    ),
}


class DatasetIsolationError(RuntimeError):
    """Raised when a caller attempts pre-H4 access outside train/val."""


@dataclass
class DecodeAudit:
    """Machine-checkable evidence of which split payloads were decoded."""

    source_kind: str
    requested_split: str
    total_lines_seen: int = 0
    decoded_train_lines: int = 0
    decoded_val_lines: int = 0
    decoded_test_lines: int = 0
    opaque_test_lines_crossed: int = 0

    @property
    def decoded_lines(self) -> int:
        return (
            self.decoded_train_lines
            + self.decoded_val_lines
            + self.decoded_test_lines
        )

    def record_decoded(self, split: str) -> None:
        if split == "train":
            self.decoded_train_lines += 1
        elif split == "val":
            self.decoded_val_lines += 1
        elif split == "test":
            self.decoded_test_lines += 1
        else:
            raise DatasetIsolationError(f"Unknown dataset split: {split!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "requested_split": self.requested_split,
            "total_lines_seen": self.total_lines_seen,
            "decoded_lines": self.decoded_lines,
            "decoded_train_lines": self.decoded_train_lines,
            "decoded_val_lines": self.decoded_val_lines,
            "decoded_test_lines": self.decoded_test_lines,
            "opaque_test_lines_crossed": self.opaque_test_lines_crossed,
        }


def _require_development_split(split: str) -> None:
    if split not in ALLOWED_DEVELOPMENT_SPLITS:
        raise DatasetIsolationError(
            "H4-before-freeze access is limited to train/val; test requires the "
            "formal test-run ledger and is deliberately unsupported here."
        )


def _supplement_split_for_line(kind: str, line_index: int) -> str:
    if kind == "canonical":
        records_per_model = CANONICAL_PER_MODEL
    elif kind == "renderings":
        records_per_model = CANONICAL_PER_MODEL * RENDERINGS_PER_CANONICAL
    else:
        raise ValueError(f"Unsupported supplement kind: {kind!r}")
    model_index = line_index // records_per_model
    variant_index = model_index % MODELS_PER_FEATURE
    if variant_index < TRAIN_MODELS_PER_FEATURE:
        return "train"
    if variant_index < TRAIN_MODELS_PER_FEATURE + VAL_MODELS_PER_FEATURE:
        return "val"
    return "test"


def _decode_json_line(
    raw_line: bytes,
    *,
    decoder: Callable[[str], Any],
) -> dict[str, Any]:
    value = decoder(raw_line.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset JSONL record must be an object")
    return value


def _iter_upstream(
    kind: str,
    split: str,
    *,
    audit: DecodeAudit,
    decoder: Callable[[str], Any],
) -> Iterator[dict[str, Any]]:
    path = UPSTREAM_ROOT / kind / f"{split}.jsonl"
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            audit.total_lines_seen += 1
            record = _decode_json_line(raw_line, decoder=decoder)
            actual_split = str(record.get("source_ref", {}).get("dataset_split", ""))
            if actual_split != split:
                raise DatasetIsolationError(
                    f"Physical {split} file contains a {actual_split!r} record"
                )
            audit.record_decoded(split)
            yield record


def _iter_supplement(
    kind: str,
    split: str,
    *,
    audit: DecodeAudit,
    decoder: Callable[[str], Any],
) -> Iterator[dict[str, Any]]:
    path = SUPPLEMENT_ROOT / "queries" / f"{kind}.jsonl"
    nonempty_index = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            line_split = _supplement_split_for_line(kind, nonempty_index)
            nonempty_index += 1
            audit.total_lines_seen += 1
            if line_split == "test":
                audit.opaque_test_lines_crossed += 1
                continue
            if line_split != split:
                continue
            record = _decode_json_line(raw_line, decoder=decoder)
            if kind == "canonical":
                actual_split = str(
                    record.get("source_ref", {}).get("dataset_split", "")
                )
                if actual_split != split:
                    raise DatasetIsolationError(
                        "Frozen supplement line plan disagrees with canonical "
                        f"record split: expected {split!r}, got {actual_split!r}"
                    )
            audit.record_decoded(split)
            yield record
    expected = EXPECTED_SUPPLEMENT_LINES[kind]
    if nonempty_index != expected:
        raise DatasetIsolationError(
            f"Frozen supplement {kind} line count changed: "
            f"expected {expected}, got {nonempty_index}"
        )


def _iter_records(
    kind: str,
    split: str,
    *,
    audit: DecodeAudit | None,
    decoder: Callable[[str], Any],
) -> Iterable[dict[str, Any]]:
    _require_development_split(split)
    actual_audit = audit if audit is not None else DecodeAudit(kind, split)
    if actual_audit.source_kind != kind or actual_audit.requested_split != split:
        raise ValueError("DecodeAudit does not match the requested source kind/split")
    yield from _iter_upstream(
        kind,
        split,
        audit=actual_audit,
        decoder=decoder,
    )
    yield from _iter_supplement(
        kind,
        split,
        audit=actual_audit,
        decoder=decoder,
    )
    if actual_audit.decoded_test_lines:
        raise DatasetIsolationError("Test payload was decoded before H4 authorization")


def iter_canonical_records(
    split: str,
    *,
    audit: DecodeAudit | None = None,
    decoder: Callable[[str], Any] = json.loads,
) -> Iterable[dict[str, Any]]:
    """Yield train/val canonical records without decoding test payloads."""

    return _iter_records("canonical", split, audit=audit, decoder=decoder)


def iter_rendering_records(
    split: str,
    *,
    audit: DecodeAudit | None = None,
    decoder: Callable[[str], Any] = json.loads,
) -> Iterable[dict[str, Any]]:
    """Yield train/val rendering records without decoding test payloads."""

    return _iter_records("renderings", split, audit=audit, decoder=decoder)


__all__ = [
    "DatasetIsolationError",
    "DecodeAudit",
    "iter_canonical_records",
    "iter_rendering_records",
]
