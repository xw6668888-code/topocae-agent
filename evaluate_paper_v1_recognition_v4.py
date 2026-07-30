from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import evaluate_paper_v1_recognition as base
from evaluate_paper_v1_recognition_v2 import fast_bootstrap_object_metrics
from evaluate_paper_v1_recognition_v3 import resumable_cached_predict_dataset


SERIALIZED_FLOAT_SIGNIFICANT_DIGITS = 12


def normalize_floats(value: Any) -> Any:
    """Make JSON output byte-stable across deterministic CPU BLAS reductions."""
    if isinstance(value, dict):
        return {key: normalize_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_floats(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float cannot be serialized: {value}")
        return float(format(value, f".{SERIALIZED_FLOAT_SIGNIFICANT_DIGITS}g"))
    return value


def deterministic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_floats(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_existing_run(run_root: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in run_root.rglob("*.json")
        if path.parent.name in {"predictions", "feature_objects"}
    )
    for path in files:
        deterministic_write_json(path, json.loads(path.read_text(encoding="utf-8")))
    return {
        "run_root": str(run_root.resolve()),
        "normalized_file_count": len(files),
        "serialized_float_significant_digits": SERIALIZED_FLOAT_SIGNIFICANT_DIGITS,
    }


def normalize_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize an already completed paper-v1 run to the frozen JSON precision."
    )
    parser.add_argument("--normalize-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_existing_run(args.normalize_run.resolve()), indent=2))


def main() -> None:
    if "--normalize-run" in sys.argv:
        normalize_cli()
        return
    base.bootstrap_object_metrics = fast_bootstrap_object_metrics
    base.predict_dataset = resumable_cached_predict_dataset
    base.write_json = deterministic_write_json
    base.main()


if __name__ == "__main__":
    main()
