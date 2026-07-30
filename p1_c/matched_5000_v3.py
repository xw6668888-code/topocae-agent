"""Deterministic class-balanced matched-5000 selector."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


SELECTION_REVISION = "rotating_largest_remainder_fail_closed_v3"


def write_matched_5000(
    output: Path,
    records: list[dict[str, Any]],
    *,
    sha256_file: Callable[[Path], str],
) -> dict[str, Any]:
    targets = {
        "train": {"target_only": 2400, "pair": 1200, "triple": 400},
        "val": {"target_only": 300, "pair": 150, "triple": 50},
        "development_test": {"target_only": 300, "pair": 150, "triple": 50},
    }
    classes = sorted({record["primary_target"] for record in records})
    if len(classes) != 18:
        raise RuntimeError(f"matched-5000 requires 18 primary classes, got {len(classes)}")
    selected: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    extra_cursor = 0
    for split, context_targets in targets.items():
        for context, target_count in context_targets.items():
            base, remainder = divmod(target_count, len(classes))
            extra_indices = {
                (extra_cursor + offset) % len(classes) for offset in range(remainder)
            }
            extra_cursor = (extra_cursor + remainder) % len(classes)
            for class_position, primary in enumerate(classes):
                quota = base + (1 if class_position in extra_indices else 0)
                candidates = [
                    record
                    for record in records
                    if record["split"] == split
                    and record["context"] == context
                    and record["primary_target"] == primary
                ]
                candidates.sort(
                    key=lambda item: hashlib.sha256(
                        (
                            "P1-C-matched-5000-strict-v3::"
                            f"{item['sample_id']}"
                        ).encode("utf-8")
                    ).hexdigest()
                )
                if len(candidates) < quota:
                    raise RuntimeError(
                        f"matched-5000 stratum underflow: {split}/{context}/{primary}"
                    )
                chosen = candidates[:quota]
                selected.extend(chosen)
                stratum_counts[f"{split}/{context}/{primary}"] = len(chosen)

    matched_root = output / "splits_matched_5000"
    matched_root.mkdir(parents=True, exist_ok=True)
    selected.sort(key=lambda item: item["sample_id"])
    selected_ids = [record["sample_id"] for record in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("matched-5000 selector produced duplicate sample IDs")
    manifest_path = matched_root / "sample_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in selected:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    split_counts = Counter(record["split"] for record in selected)
    context_counts = Counter(record["context"] for record in selected)
    class_counts = Counter(record["primary_target"] for record in selected)
    for split in ("train", "val", "development_test"):
        with (matched_root / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for record in selected:
                if record["split"] == split:
                    stream.write(
                        json.dumps(
                            record, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
    checks = {
        "sample_count_5000": len(selected) == 5000,
        "sample_ids_unique": len(selected_ids) == len(set(selected_ids)),
        "split_counts_exact": dict(split_counts)
        == {"train": 4000, "val": 500, "development_test": 500},
        "context_counts_exact": dict(context_counts)
        == {"target_only": 3000, "pair": 1500, "triple": 500},
        "class_count_range_at_most_1": (
            max(class_counts.values()) - min(class_counts.values()) <= 1
        ),
    }
    payload = {
        "schema_version": "topocae_p1_c_matched_5000_3",
        "selection_revision": SELECTION_REVISION,
        "frozen_before_training": True,
        "sample_count": len(selected),
        "split_counts": dict(split_counts),
        "context_counts": dict(context_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "class_count_range": [min(class_counts.values()), max(class_counts.values())],
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "passed": all(checks.values()),
    }
    (matched_root / "matched_5000_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise RuntimeError(f"matched-5000 fail-closed audit failed: {checks}")
    return payload


__all__ = ["SELECTION_REVISION", "write_matched_5000"]
