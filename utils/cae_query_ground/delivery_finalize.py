"""Pilot-only delivery inventory; this is not a freeze manifest."""

from __future__ import annotations

from pathlib import Path

from .constants import PILOT_ROOT, REPORT_ROOT, ROOT
from .io import write_json
from .reporting import pilot_delivery_manifest


def finalize() -> Path:
    assets: list[tuple[Path, str]] = [
        (ROOT / "GOAL_P3_A1_CAE_QUERYGROUND_DATASET.zh-CN.md", "task_contract"),
        (ROOT / "build_cae_query_grounding_dataset_pilot.py", "functional_pilot_entrypoint"),
        (ROOT / "build_cae_query_grounding_dataset.py", "superseded_preliminary_entrypoint"),
        (ROOT / "utils" / "run_p3_a1_cae_query_ground_regression.py", "passing_regression_runner"),
        (ROOT / "tests" / "cae_query_ground_pilot_regression.py", "passing_regression_tests"),
        (ROOT / "utils" / "run_p3_a1_cae_query_ground_tests.py", "superseded_test_scaffold"),
        (ROOT / "tests" / "test_cae_query_ground.py", "superseded_test_scaffold"),
        (ROOT / "schemas" / "cae_query_ground_record.schema.json", "record_schema"),
        (ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json", "dataset_manifest_schema"),
        (ROOT / "docs" / "cae_query_ground_annotation_guide.zh-CN.md", "annotation_guide"),
        (ROOT / "configs" / "paper_v1" / "cae_query_role_crosswalk.yaml", "role_crosswalk"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.en-US.yaml", "english_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.zh-CN.yaml", "chinese_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.json", "template_split_manifest"),
    ]
    assets.extend(
        (path, "generator_module")
        for path in sorted((ROOT / "utils" / "cae_query_ground").glob("*.py"))
    )
    assets.extend(
        (path, "pilot_data_or_manifest")
        for path in sorted(PILOT_ROOT.rglob("*"))
        if path.is_file()
    )
    assets.extend(
        (path, "pilot_report_or_h3_review_asset")
        for path in sorted(REPORT_ROOT.rglob("*"))
        if path.is_file() and path.name != "pilot_delivery_manifest.json"
    )
    output = REPORT_ROOT / "pilot_delivery_manifest.json"
    write_json(output, pilot_delivery_manifest(assets), force=True)
    return output


if __name__ == "__main__":
    print(finalize())

