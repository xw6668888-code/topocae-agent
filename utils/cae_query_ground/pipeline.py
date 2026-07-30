from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .audit_v2 import audit_sources, render_audit_markdown
from .constants import (
    CROSSWALK_PATH,
    PILOT_ROOT,
    REPORT_ROOT,
    ROOT,
    SEED,
    SOURCE_AUDIT_JSON,
    SOURCE_AUDIT_MD,
)
from .io import canonical_sha256, read_yaml, write_json, write_jsonl, write_text, write_yaml
from .planner import make_record
from .planner_v2 import generate_pilot_plans
from .renderer import render_all
from .reporting import (
    canonical_rendering_index,
    dataset_card,
    dataset_manifest,
    human_fields_are_blank,
    pilot_delivery_manifest,
)
from .review import (
    review_distribution,
    review_readme,
    review_rows,
    select_review_records,
    write_expression_csv,
    write_review_csv,
)
from .validation_v4 import audit_pilot, validate_manifest


def run_audit(*, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    audit, crosswalk = audit_sources()
    if not audit["automatic_pass"]:
        raise RuntimeError("Source/role audit failed; pilot generation is prohibited")
    write_json(SOURCE_AUDIT_JSON, audit, force=force)
    write_text(SOURCE_AUDIT_MD, render_audit_markdown(audit), force=force)
    write_yaml(CROSSWALK_PATH, crosswalk, force=force)
    return audit, crosswalk


def _build_records(plans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_records: list[dict[str, Any]] = []
    rendering_records: list[dict[str, Any]] = []
    for plan in plans:
        canonical_query, queries = render_all(plan)
        canonical_records.append(make_record(plan, record_kind="canonical", query=canonical_query))
        rendering_records.extend(
            make_record(plan, record_kind="rendering", query=query) for query in queries
        )
    return canonical_records, rendering_records


def _split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result[record["source_ref"]["dataset_split"]].append(record)
    return {key: result[key] for key in ("train", "val", "test")}


def _assert_review_files_safe(review_csv: Path, expression_csv: Path, *, force: bool) -> None:
    if not force:
        return
    if review_csv.exists() and not human_fields_are_blank(review_csv):
        raise RuntimeError("Refusing to overwrite non-blank H3-01 human review")
    if expression_csv.exists():
        import csv

        with expression_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            if any(row.get("natural_expression", "").strip() for row in csv.DictReader(handle)):
                raise RuntimeError("Refusing to overwrite H3-01 human expression additions")


def run_pilot(*, force: bool, source_audit: dict[str, Any], crosswalk: dict[str, Any]) -> dict[str, Any]:
    plans = generate_pilot_plans(crosswalk, seed=SEED)
    canonical_records, rendering_records = _build_records(plans)

    # Same source + same seed must produce byte-identical semantic arrays in memory.
    repeat_plans = generate_pilot_plans(crosswalk, seed=SEED)
    repeat_canonical, repeat_renderings = _build_records(repeat_plans)
    deterministic = {
        "canonical_sha256": canonical_sha256(canonical_records),
        "repeat_canonical_sha256": canonical_sha256(repeat_canonical),
        "renderings_sha256": canonical_sha256(rendering_records),
        "repeat_renderings_sha256": canonical_sha256(repeat_renderings),
    }
    deterministic["automatic_pass"] = (
        deterministic["canonical_sha256"] == deterministic["repeat_canonical_sha256"]
        and deterministic["renderings_sha256"] == deterministic["repeat_renderings_sha256"]
    )
    if not deterministic["automatic_pass"]:
        raise RuntimeError("Deterministic regeneration check failed")

    quality = audit_pilot(plans, canonical_records, rendering_records)
    if not quality["automatic_pass"]:
        raise RuntimeError("Pilot automatic validation failed; no pilot will be published")

    selection = select_review_records(plans, rendering_records, count=120)
    rows = review_rows(selection)
    distribution = review_distribution(rows)
    if distribution["review_decisions_blank"] != 120:
        raise RuntimeError("H3-01 review decisions must remain blank")

    review_root = REPORT_ROOT / "human_review_package"
    review_csv = review_root / "cae_query_review_120.csv"
    expression_csv = review_root / "human_expression_additions_20.csv"
    review_readme_path = review_root / "README_H3_01.zh-CN.md"
    _assert_review_files_safe(review_csv, expression_csv, force=force)

    canonical_by_split = _split(canonical_records)
    rendering_by_split = _split(rendering_records)
    data_files: list[tuple[Path, str, int]] = []
    for split in ("train", "val", "test"):
        path = PILOT_ROOT / "canonical" / f"{split}.jsonl"
        count = write_jsonl(path, canonical_by_split[split], force=force)
        data_files.append((path, f"canonical_{split}", count))
    for split in ("train", "val", "test"):
        path = PILOT_ROOT / "renderings" / f"{split}.jsonl"
        count = write_jsonl(path, rendering_by_split[split], force=force)
        data_files.append((path, f"rendering_{split}", count))
    index_path = PILOT_ROOT / "manifests" / "canonical_to_rendering.jsonl"
    index_count = write_jsonl(
        index_path,
        canonical_rendering_index(canonical_records, rendering_records),
        force=force,
    )
    data_files.append((index_path, "canonical_to_rendering_index", index_count))

    write_review_csv(review_csv, rows, force=force)
    write_expression_csv(expression_csv, force=force)
    write_text(
        review_readme_path,
        review_readme(distribution, review_csv, expression_csv),
        force=force,
    )
    if not human_fields_are_blank(review_csv):
        raise RuntimeError("H3-01 package was not emitted with blank human fields")

    quality_path = REPORT_ROOT / "quality_audit.json"
    leakage_path = REPORT_ROOT / "split_leakage_audit.json"
    card_path = REPORT_ROOT / "dataset_card.zh-CN.md"
    write_json(quality_path, quality, force=force)
    write_json(leakage_path, quality["split_leakage"], force=force)
    write_text(card_path, dataset_card(quality, distribution), force=force)

    manifest = dataset_manifest(
        source_audit=source_audit,
        quality=quality,
        data_files=data_files,
        review_csv=review_csv,
        expression_csv=expression_csv,
    )
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise RuntimeError(f"Dataset manifest schema failed: {manifest_errors}")
    manifest_path = PILOT_ROOT / "manifests" / "dataset_manifest.json"
    write_json(manifest_path, manifest, force=force)

    summary = {
        "schema_version": "cae-query-ground-pilot-test-summary/1",
        "status": "BOUNDED-PASS_WAITING_H3_01",
        "automatic_pass": True,
        "source_and_role_audit": source_audit["automatic_pass"],
        "pilot_quality_audit": quality["automatic_pass"],
        "dataset_manifest_schema": "PASS",
        "deterministic_regeneration": deterministic,
        "h3_01": {
            "status": "WAITING_FOR_REAL_USER_REVIEW",
            "reviewed_rows": 0,
            "required_rows": 120,
            "distribution": distribution,
        },
        "p1_p2_frozen_hash_changes": source_audit["p2_freeze"]["mismatch_count"],
        "scope_guards": quality["scope_guards"],
    }
    test_summary_path = REPORT_ROOT / "test_summary.json"
    write_json(test_summary_path, summary, force=force)

    delivery_paths = [
        (ROOT / "build_cae_query_grounding_dataset.py", "entrypoint"),
        (ROOT / "schemas" / "cae_query_ground_record.schema.json", "record_schema"),
        (ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json", "dataset_manifest_schema"),
        (ROOT / "docs" / "cae_query_ground_annotation_guide.zh-CN.md", "annotation_guide"),
        (CROSSWALK_PATH, "role_crosswalk"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.en-US.yaml", "english_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.zh-CN.yaml", "chinese_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.json", "template_split_manifest"),
        (SOURCE_AUDIT_JSON, "source_audit"),
        (SOURCE_AUDIT_MD, "source_audit_report"),
        (quality_path, "quality_audit"),
        (leakage_path, "split_leakage_audit"),
        (card_path, "pilot_dataset_card"),
        (test_summary_path, "test_summary"),
        (review_csv, "h3_review_csv"),
        (expression_csv, "h3_expression_csv"),
        (review_readme_path, "h3_review_instructions"),
        (manifest_path, "pilot_dataset_manifest"),
    ]
    delivery_paths.extend((path, role) for path, role, _ in data_files)
    delivery = pilot_delivery_manifest(delivery_paths)
    delivery_path = REPORT_ROOT / "pilot_delivery_manifest.json"
    write_json(delivery_path, delivery, force=force)
    return {
        "status": "BOUNDED-PASS_WAITING_H3_01",
        "canonical_records": len(canonical_records),
        "renderings": len(rendering_records),
        "review_rows": len(rows),
        "reviewed_rows": 0,
        "dataset_manifest": str(manifest_path.resolve()),
        "review_csv": str(review_csv.resolve()),
        "expression_csv": str(expression_csv.resolve()),
        "delivery_manifest": str(delivery_path.resolve()),
    }

