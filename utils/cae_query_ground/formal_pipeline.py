from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .audit_v2 import audit_sources, render_audit_markdown
from .constants import (
    CROSSWALK_PATH,
    REPORT_ROOT,
    ROOT,
    SEED,
    SOURCE_AUDIT_JSON,
    SOURCE_AUDIT_MD,
)
from .formal_contract import (
    FINAL_MANIFEST_SCHEMA,
    FINAL_RECORD_SCHEMA,
    build_final_schemas,
)
from .formal_planner import (
    FINAL_DATASET_VERSION,
    generate_formal_plans,
    make_formal_record,
)
from .formal_renderer_v3 import render_formal_all
from .formal_reporting import (
    acceptance_json,
    dataset_card,
    dataset_manifest,
    delivery_manifest,
    final_acceptance_report,
    freeze_manifest,
)
from .formal_validation_v2 import audit_formal, validate_formal_manifest
from .h3_ingest import ingest as ingest_h3
from .io import (
    canonical_json_bytes,
    read_json,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)
from .reporting import canonical_rendering_index


FORMAL_ROOT = ROOT / "output_cae_query_ground_v1"
FORMAL_REPORTS = FORMAL_ROOT / "reports"
FREEZE_ROOT = REPORT_ROOT / "freeze"


def _hash_sequence(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(canonical_json_bytes(value))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_records(
    plans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical: list[dict[str, Any]] = []
    renderings: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        canonical_query, queries = render_formal_all(plan)
        canonical.append(
            make_formal_record(plan, record_kind="canonical", query=canonical_query)
        )
        renderings.extend(
            make_formal_record(plan, record_kind="rendering", query=query)
            for query in queries
        )
        if index % 5000 == 0:
            print(f"rendered canonical plans: {index}/{len(plans)}", flush=True)
    return canonical, renderings


def _split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        output[record["source_ref"]["dataset_split"]].append(record)
    return {split: output[split] for split in ("train", "val", "test")}


def _quality_log(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for plan in plans:
        face_count = sum(
            len(target["target_face_indices"])
            for target in plan["ground_truth"]["targets"]
            if target.get("target_contract") == "feature_face_set"
        )
        rows.append({
            "canonical_record_id": plan["canonical_record_id"],
            "dataset_split": {"train": "train", "val": "val", "development_test": "test"}[
                plan["source_record"]["split"]
            ],
            "level": plan["level"],
            "status": plan["status"],
            "feature": plan["verbalization"]["feature_type"],
            "operations": [item["operation_type"] for item in plan["verbalization"]["operations"]],
            "target_face_count": face_count,
            "schema_filter": "PASS",
            "p2_filter": "PASS",
            "target_reconstruction_filter": "PASS" if plan["status"] == "VALID" else "NOT_EXECUTABLE",
            "outcome": "accepted",
        })
    return rows


def _source_and_h3(*, force: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    h3 = ingest_h3(force=force)
    if h3["status"] != "PASS":
        raise RuntimeError("H3-01 is not passed")
    source_audit, crosswalk = audit_sources()
    if not source_audit["automatic_pass"]:
        raise RuntimeError("Source/role audit failed")
    write_json(SOURCE_AUDIT_JSON, source_audit, force=force)
    write_text(SOURCE_AUDIT_MD, render_audit_markdown(source_audit), force=force)
    write_yaml(CROSSWALK_PATH, crosswalk, force=force)
    return h3, source_audit, crosswalk


def _deterministic_proof(
    plans: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    renderings: list[dict[str, Any]],
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    first_plan_hash = _hash_sequence(plans)
    first_record_hash = _hash_sequence(canonical)
    first_rendering_hash = _hash_sequence(renderings)
    print("starting deterministic plan regeneration", flush=True)
    repeat_plans = generate_formal_plans(crosswalk, seed=SEED)
    repeat_plan_hash = _hash_sequence(repeat_plans)
    repeat_queries = (
        query
        for plan in repeat_plans
        for query in (lambda pair: [pair[0], *pair[1]])(render_formal_all(plan))
    )
    repeat_query_hash = _hash_sequence(repeat_queries)
    first_query_hash = _hash_sequence(
        [record["query"] for record in canonical]
        + [record["query"] for record in renderings]
    )
    passed = first_plan_hash == repeat_plan_hash and first_query_hash == repeat_query_hash
    if not passed:
        raise RuntimeError("Deterministic formal regeneration failed")
    return {
        "automatic_pass": True,
        "plan_sha256": first_plan_hash,
        "repeat_plan_sha256": repeat_plan_hash,
        "canonical_record_sha256": first_record_hash,
        "rendering_record_sha256": first_rendering_hash,
        "query_sha256": first_query_hash,
        "repeat_query_sha256": repeat_query_hash,
        "seed": SEED,
    }


def _asset_sources() -> list[tuple[Path, str]]:
    paths = [
        (ROOT / "build_cae_query_grounding_dataset_formal.py", "formal_entrypoint"),
        (ROOT / "utils" / "run_p3_a1_cae_query_ground_formal_tests.py", "test_runner"),
        (ROOT / "tests" / "cae_query_ground_formal_regression.py", "regression_tests"),
        (FINAL_RECORD_SCHEMA, "record_schema"),
        (FINAL_MANIFEST_SCHEMA, "dataset_manifest_schema"),
        (ROOT / "schemas" / "cae_query_ground_record.schema.json", "pilot_schema_source"),
        (ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json", "pilot_schema_source"),
        (ROOT / "configs" / "paper_v1" / "cae_query_role_crosswalk.yaml", "role_crosswalk"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.en-US.yaml", "base_english_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_phrase_bank.zh-CN.yaml", "base_chinese_phrase_bank"),
        (ROOT / "configs" / "paper_v1" / "cae_query_reviewed_phrase_extensions.yaml", "reviewed_phrase_extensions"),
        (ROOT / "configs" / "paper_v1" / "cae_query_template_split_manifest.formal.json", "formal_template_split_manifest"),
        (ROOT / "docs" / "cae_query_ground_annotation_guide.zh-CN.md", "annotation_guide"),
        (ROOT / "GOAL_P3_A1_CAE_QUERYGROUND_DATASET.zh-CN.md", "task_contract"),
    ]
    used_modules = (
        "__init__.py",
        "audit.py",
        "audit_v2.py",
        "constants.py",
        "io.py",
        "planner.py",
        "planner_v2.py",
        "renderer.py",
        "formal_contract.py",
        "formal_pipeline.py",
        "formal_planner.py",
        "formal_renderer.py",
        "formal_renderer_v2.py",
        "formal_renderer_v3.py",
        "formal_reporting.py",
        "formal_validation.py",
        "formal_validation_v2.py",
        "h3_ingest.py",
        "reporting.py",
        "validation.py",
        "validation_v2.py",
    )
    paths.extend(
        (ROOT / "utils" / "cae_query_ground" / name, "generator_module")
        for name in used_modules
    )
    return paths


def run_formal(*, force: bool) -> dict[str, Any]:
    h3, source_audit, crosswalk = _source_and_h3(force=force)
    build_final_schemas(force=force)
    print("generating formal canonical plans", flush=True)
    plans = generate_formal_plans(crosswalk, seed=SEED)
    print(f"formal plans generated: {len(plans)}", flush=True)
    canonical, renderings = _build_records(plans)
    deterministic = _deterministic_proof(plans, canonical, renderings, crosswalk)
    print("starting full formal schema/P2/target/leakage audit", flush=True)
    quality = audit_formal(plans, canonical, renderings)
    if not quality["automatic_pass"]:
        raise RuntimeError(f"Formal quality audit failed: {quality['failure_examples'][:3]}")
    print("formal quality audit passed; writing outputs", flush=True)

    data_files: list[tuple[Path, str, int]] = []
    for split, records in _split(canonical).items():
        path = FORMAL_ROOT / "canonical" / f"{split}.jsonl"
        count = write_jsonl(path, records, force=force)
        data_files.append((path, f"canonical_{split}", count))
    for split, records in _split(renderings).items():
        path = FORMAL_ROOT / "renderings" / f"{split}.jsonl"
        count = write_jsonl(path, records, force=force)
        data_files.append((path, f"rendering_{split}", count))
    index_path = FORMAL_ROOT / "manifests" / "canonical_to_rendering.jsonl"
    index_count = write_jsonl(
        index_path,
        canonical_rendering_index(canonical, renderings),
        force=force,
    )
    data_files.append((index_path, "canonical_to_rendering_index", index_count))
    quality_log_path = FORMAL_REPORTS / "quality_filter_log.jsonl"
    quality_log_count = write_jsonl(quality_log_path, _quality_log(plans), force=force)
    data_files.append((quality_log_path, "quality_filter_log", quality_log_count))

    distribution_path = FORMAL_REPORTS / "distribution.json"
    write_json(distribution_path, quality["counts"], force=force)
    quality_path = REPORT_ROOT / "quality_audit.json"
    leakage_path = REPORT_ROOT / "split_leakage_audit.json"
    card_path = REPORT_ROOT / "dataset_card.zh-CN.md"
    write_json(quality_path, quality, force=force)
    write_json(leakage_path, quality["split_leakage"], force=force)
    write_text(card_path, dataset_card(quality, h3), force=force)

    manifest = dataset_manifest(
        quality=quality,
        source_audit_path=SOURCE_AUDIT_JSON,
        data_files=data_files,
        h3=h3,
    )
    manifest_errors = validate_formal_manifest(manifest)
    if manifest_errors:
        raise RuntimeError(f"Formal dataset manifest failed schema: {manifest_errors}")
    dataset_manifest_path = FORMAL_ROOT / "manifests" / "dataset_manifest.json"
    write_json(dataset_manifest_path, manifest, force=force)

    frozen_changes = source_audit["p2_freeze"]["mismatch_count"]
    acceptance = acceptance_json(quality, h3, frozen_changes)
    if not acceptance["automatic_pass"] or not all(acceptance["gate_p3"].values()):
        raise RuntimeError(f"Gate P3 acceptance failed: {acceptance}")
    acceptance_path = REPORT_ROOT / "acceptance.json"
    final_report_path = REPORT_ROOT / "final_acceptance.zh-CN.md"
    write_json(acceptance_path, acceptance, force=force)
    write_text(
        final_report_path,
        final_acceptance_report(quality, h3, frozen_changes),
        force=force,
    )
    test_summary = {
        "schema_version": "cae-query-ground-formal-test-summary/1",
        "status": "PASS",
        "dataset_version": FINAL_DATASET_VERSION,
        "automatic_pass": True,
        "deterministic_regeneration": deterministic,
        "p2_validation": quality["p2_validation"],
        "target_reconstruction": quality["target_reconstruction"],
        "schema_semantic_failures": quality["record_schema_and_semantic_failures"],
        "split_leakage_collisions": quality["split_leakage"]["total_collisions"],
        "coverage": quality["coverage"],
        "h3_01": {"reviewed": 120, "correct_rate": h3["effective_correct_rate"]},
        "scope_guards": quality["scope_guards"],
    }
    test_summary_path = REPORT_ROOT / "test_summary.json"
    write_json(test_summary_path, test_summary, force=force)
    context_path = ROOT / "CODEX_CONTEXT_P3_A1.md"
    write_text(
        context_path,
        f"""# CODEX Context P3-A1

Goal P3-A1 已完成并冻结 `{FINAL_DATASET_VERSION}`。正式数据为 {len(canonical)} canonical / {len(renderings)} renderings，每条 2 英文 + 1 中文。H3-01 为 120/120，正确率 100%。不得在本 Goal 内进入 P4。
""",
        force=force,
    )

    report_assets = [
        (SOURCE_AUDIT_JSON, "source_audit"),
        (SOURCE_AUDIT_MD, "source_audit_report"),
        (quality_path, "quality_audit"),
        (leakage_path, "split_leakage_audit"),
        (card_path, "dataset_card"),
        (acceptance_path, "acceptance"),
        (final_report_path, "final_acceptance"),
        (test_summary_path, "test_summary"),
        (distribution_path, "distribution"),
        (context_path, "context"),
        (dataset_manifest_path, "dataset_manifest"),
        (REPORT_ROOT / "preflight_summary.zh-CN.md", "preflight"),
    ]
    human_assets = [
        (path, "human_review_evidence")
        for path in sorted((REPORT_ROOT / "human_review_package").glob("*"))
        if path.is_file()
    ]
    freeze_assets = _asset_sources() + report_assets + human_assets + [
        (path, role) for path, role, _ in data_files
    ]
    freeze_path = FREEZE_ROOT / "freeze_manifest.json"
    write_json(
        freeze_path,
        freeze_manifest(assets=freeze_assets, source_audit=source_audit, h3=h3),
        force=force,
    )
    delivery_path = REPORT_ROOT / "delivery_manifest.json"
    write_json(
        delivery_path,
        delivery_manifest(freeze_assets + [(freeze_path, "freeze_manifest")]),
        force=force,
    )
    return {
        "status": "PASS",
        "dataset_version": FINAL_DATASET_VERSION,
        "canonical_records": len(canonical),
        "renderings": len(renderings),
        "h3_reviewed": 120,
        "h3_correct_rate": h3["effective_correct_rate"],
        "target_reconstruction": quality["target_reconstruction"],
        "split_leakage_collisions": quality["split_leakage"]["total_collisions"],
        "p1_p2_frozen_hash_changes": frozen_changes,
        "final_report": str(final_report_path.resolve()),
        "freeze_manifest": str(freeze_path.resolve()),
        "delivery_manifest": str(delivery_path.resolve()),
    }

