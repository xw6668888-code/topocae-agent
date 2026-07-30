"""H-C2 protocol frozen to the approved strict H-C1 r2 review."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from p1_c import pilot_h_c2 as protocol
from p1_c import all_features_h_c2_v2 as generator


CANONICAL_SEED_BASE = 2026090000
H_C1_REVIEW_RELATIVE_PATH = Path(
    "output_generation_p1_c1_strict_concept_hosts_pilot_96_r2/human_review_canonical_24.csv"
)


def verify_h_c1(root: Path) -> dict[str, Any]:
    path = root / H_C1_REVIEW_RELATIVE_PATH
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("single_solid", "clean_host", "feature_concept_valid", "decision")
    invalid = []
    for row in rows:
        values = {field: row[field].strip().upper() for field in required}
        if any(values[field] != "PASS" for field in required):
            invalid.append({"review_id": row["review_id"], **values})
        step_path = root / row["step_path"]
        if not step_path.is_file() or protocol.sha256_file(step_path) != row["step_sha256"]:
            invalid.append({"review_id": row["review_id"], "step_integrity": "FAIL"})
    result = {
        "path": path.relative_to(root).as_posix(),
        "sha256": protocol.sha256_file(path),
        "row_count": len(rows),
        "pass_count": len(rows) - len({item["review_id"] for item in invalid}),
        "invalid_rows": invalid,
        "passed": len(rows) == 24 and not invalid,
    }
    if not result["passed"]:
        raise RuntimeError(f"strict H-C1 r2 is not 24/24 PASS: {result}")
    return result


protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.build_feature_sample = generator.build_feature_sample
protocol.verify_h_c1 = verify_h_c1
protocol.CANONICAL_SEED_BASE = CANONICAL_SEED_BASE

generate_h_c2 = protocol.generate_h_c2


__all__ = ["generate_h_c2", "verify_h_c1"]
