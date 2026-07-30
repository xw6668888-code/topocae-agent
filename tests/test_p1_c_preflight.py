from __future__ import annotations

import json
from pathlib import Path

from p1_c.preflight import FROZEN_METRICS, PROTECTED_RELATIVE_PATHS, _markdown


def test_protected_paths_are_explicit_and_exclude_denied_directory() -> None:
    assert len(PROTECTED_RELATIVE_PATHS) == 7
    assert all("test_part_s3" not in path for path in PROTECTED_RELATIVE_PATHS)


def test_frozen_metrics_match_declared_p1_b_baseline() -> None:
    assert FROZEN_METRICS["p1_b_dataset_sample_count"] == 5000
    assert FROZEN_METRICS["p1_b_dataset_instance_count"] == 8050
    assert FROZEN_METRICS["synthetic_val_three_seed_mean"]["face_macro_f1"] == 0.9774265760696488
    assert FROZEN_METRICS["real_dev_area_macro"]["f1"] == 0.6


def test_generation_config_has_exact_18_classes_and_7200_quota() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "p1_c" / "generation.json").read_text(encoding="utf-8"))
    assert len(config["classes"]) == 18
    assert len(set(config["classes"])) == 18
    dataset = config["dataset"]
    assert dataset["samples_per_primary_class"] * 18 == 7200
    assert dataset["target_only_per_class"] + dataset["pair_per_class"] + dataset["triple_per_class"] == 400


def test_preflight_markdown_stays_below_250_lines() -> None:
    report = {
        "generated_at_utc": "2026-07-18T00:00:00+00:00",
        "single_entrypoint": "p1_c_pipeline.py",
        "new_paths": ["reports/p1_c_preflight"],
        "environment": {
            "python_version": "3.11.14",
            "python_executable": "python.exe",
            "platform": "Windows",
            "cpu_count": 1,
            "packages": {},
            "cuda": {"available": False, "runtime_version": None, "devices": []},
        },
        "frozen_metrics": FROZEN_METRICS,
        "protected_summary": {
            "aggregate_sha256": "0" * 64,
            "directories": [
                {"path": path, "exists": True, "file_count": 1, "byte_count": 1}
                for path in PROTECTED_RELATIVE_PATHS
            ],
        },
    }
    assert len(_markdown(report).splitlines()) <= 250
