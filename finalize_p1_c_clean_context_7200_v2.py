from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import generate_p1_c_clean_context_7200 as protocol
import generate_p1_c_clean_context_7200_v2  # activates formal generator r5 in protocol


def _write_matched_5000_balanced(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    targets = {
        "train": {"target_only": 2400, "pair": 1200, "triple": 400},
        "val": {"target_only": 300, "pair": 150, "triple": 50},
        "development_test": {"target_only": 300, "pair": 150, "triple": 50},
    }
    classes = sorted({record["primary_target"] for record in records})
    selected: list[dict[str, Any]] = []
    extra_cursor = 0
    import hashlib

    for split, context_targets in targets.items():
        for context, target_count in context_targets.items():
            base, remainder = divmod(target_count, len(classes))
            extra_indices = {(extra_cursor + offset) % len(classes) for offset in range(remainder)}
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
                        f"P1-C-matched-5000-balanced-v2::{item['sample_id']}".encode("utf-8")
                    ).hexdigest()
                )
                if len(candidates) < quota:
                    raise RuntimeError(f"matched-5000 stratum underflow: {split}/{context}/{primary}")
                selected.extend(candidates[:quota])
    matched_root = output / "splits_matched_5000"
    selected.sort(key=lambda item: item["sample_id"])
    with (matched_root / "sample_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    split_counts = Counter(record["split"] for record in selected)
    context_counts = Counter(record["context"] for record in selected)
    class_counts = Counter(record["primary_target"] for record in selected)
    for split in ("train", "val", "development_test"):
        with (matched_root / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in selected:
                if record["split"] == split:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        "schema_version": "topocae_p1_c_matched_5000_2",
        "selection_revision": "rotating_largest_remainder_v2",
        "frozen_before_training": True,
        "sample_count": len(selected),
        "split_counts": dict(split_counts),
        "context_counts": dict(context_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "class_count_range": [min(class_counts.values()), max(class_counts.values())],
        "sample_manifest_sha256": protocol.sha256_file(matched_root / "sample_manifest.jsonl"),
        "passed": len(selected) == 5000
        and dict(split_counts) == {"train": 4000, "val": 500, "development_test": 500}
        and dict(context_counts) == {"target_only": 3000, "pair": 1500, "triple": 500}
        and max(class_counts.values()) - min(class_counts.values()) <= 1,
    }
    (matched_root / "matched_5000_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


protocol._write_matched_5000 = _write_matched_5000_balanced


def main() -> int:
    payload = protocol.finalize(protocol.OUTPUT.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
