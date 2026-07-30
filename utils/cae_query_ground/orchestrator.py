from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import PILOT_ROOT, REPORT_ROOT, ROOT
from .io import write_json
from .pipeline import run_audit, run_pilot
from .reporting import pilot_delivery_manifest


def _finalize_delivery() -> Path:
    paths: list[tuple[Path, str]] = [
        (ROOT / "build_cae_query_grounding_dataset_pilot.py", "functional_pilot_entrypoint"),
        (ROOT / "build_cae_query_grounding_dataset.py", "preliminary_entrypoint_preserved"),
        (ROOT / "utils" / "run_p3_a1_cae_query_ground_tests.py", "test_runner"),
        (ROOT / "tests" / "test_cae_query_ground.py", "regression_tests"),
        (ROOT / "schemas" / "cae_query_ground_record.schema.json", "record_schema"),
        (ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json", "dataset_manifest_schema"),
        (ROOT / "docs" / "cae_query_ground_annotation_guide.zh-CN.md", "annotation_guide"),
        (ROOT / "configs" / "paper_v1" / "cae_query_role_crosswalk.yaml", "role_crosswalk"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.en-US.yaml", "english_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.zh-CN.yaml", "chinese_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.json", "template_split_manifest"),
    ]
    for path in sorted((ROOT / "utils" / "cae_query_ground").glob("*.py")):
        paths.append((path, "generator_module"))
    for path in sorted(PILOT_ROOT.rglob("*")):
        if path.is_file():
            paths.append((path, "pilot_data_or_manifest"))
    for path in sorted(REPORT_ROOT.rglob("*")):
        if path.is_file() and path.name != "pilot_delivery_manifest.json":
            paths.append((path, "pilot_report_or_h3_review_asset"))
    delivery_path = REPORT_ROOT / "pilot_delivery_manifest.json"
    write_json(delivery_path, pilot_delivery_manifest(paths), force=True)
    return delivery_path


def execute(*, stage: str, force: bool) -> dict[str, Any]:
    audit, crosswalk = run_audit(force=force)
    if stage == "audit":
        return {
            "stage": "audit",
            "automatic_pass": True,
            "source": audit["source"],
        }
    result = run_pilot(force=force, source_audit=audit, crosswalk=crosswalk)
    result["delivery_manifest"] = str(_finalize_delivery().resolve())
    return result

