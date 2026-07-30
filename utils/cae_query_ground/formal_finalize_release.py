"""Finalize already-generated P3-A1 formal outputs without regenerating data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .audit_v2 import audit_sources, render_audit_markdown
from .constants import (
    CROSSWALK_PATH,
    P2_POLICY,
    REPORT_ROOT,
    ROOT,
    SEED,
    SOURCE_AUDIT_JSON,
    SOURCE_AUDIT_MD,
)
from .formal_contract import FINAL_MANIFEST_SCHEMA, FINAL_RECORD_SCHEMA
from .formal_contract_release import build_final_schemas
from .formal_planner import FINAL_DATASET_VERSION
from .formal_reporting import (
    acceptance_json,
    dataset_card,
    dataset_manifest,
    delivery_manifest,
    final_acceptance_report,
    freeze_manifest,
)
from .formal_validation import validate_formal_manifest
from .h3_ingest import ingest as ingest_h3
from .io import read_json, read_yaml, sha256_file, write_json, write_text


OUTPUT = ROOT / "output_cae_query_ground_v1"
OUTPUT_REPORTS = OUTPUT / "reports"
FREEZE_ROOT = REPORT_ROOT / "freeze"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _jsonl_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _deduplicate(assets: Iterable[tuple[Path, str]]) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, role in assets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        _require(path.is_file(), f"Freeze asset is missing: {path}")
        seen.add(resolved)
        result.append((path, role))
    return result


def _data_files(
    quality: dict[str, Any], deterministic: dict[str, Any]
) -> list[tuple[Path, str, int]]:
    counts = quality["counts"]["splits"]
    expected_hashes = deterministic["existing_file_sha256"]
    result: list[tuple[Path, str, int]] = []
    for split in ("train", "val", "test"):
        path = OUTPUT / "canonical" / f"{split}.jsonl"
        count = _jsonl_count(path)
        _require(count == counts[split], f"Canonical count mismatch for {split}: {count}")
        role = f"canonical_{split}"
        _require(sha256_file(path) == expected_hashes[role], f"Hash mismatch: {path}")
        result.append((path, role, count))
    for split in ("train", "val", "test"):
        path = OUTPUT / "renderings" / f"{split}.jsonl"
        count = _jsonl_count(path)
        _require(count == counts[split] * 3, f"Rendering count mismatch for {split}: {count}")
        role = f"rendering_{split}"
        _require(sha256_file(path) == expected_hashes[role], f"Hash mismatch: {path}")
        result.append((path, role, count))
    extras = (
        (
            OUTPUT / "manifests" / "canonical_to_rendering.jsonl",
            "canonical_to_rendering_index",
            quality["counts"]["canonical_records"],
        ),
        (
            OUTPUT_REPORTS / "quality_filter_log.jsonl",
            "quality_filter_log",
            quality["counts"]["canonical_records"],
        ),
    )
    for path, role, expected_count in extras:
        count = _jsonl_count(path)
        _require(count == expected_count, f"Count mismatch for {role}: {count}")
        _require(sha256_file(path) == expected_hashes[role], f"Hash mismatch: {path}")
        result.append((path, role, count))
    return result


def _assert_gate(quality: dict[str, Any], h3: dict[str, Any], deterministic: dict[str, Any]) -> None:
    counts = quality["counts"]
    _require(quality["automatic_pass"], "Formal quality audit is not PASS")
    _require(deterministic["automatic_pass"] and not deterministic["mismatches"], "Deterministic regeneration failed")
    _require(h3["status"] == "PASS" and h3["review_rows"] == 120, "H3-01 is not PASS 120/120")
    _require(h3["effective_correct_rate"] >= 0.95, "H3-01 effective correctness is below 95%")
    _require(h3["expression_rows_populated"] >= 10, "Fewer than 10 human expressions were supplied")
    _require(counts["canonical_records"] >= 20_000, "Canonical minimum not met")
    _require(counts["renderings"] >= 60_000, "Rendering minimum not met")
    _require(counts["languages"]["en-US"] == 2 * counts["canonical_records"], "English rendering cardinality failed")
    _require(counts["languages"]["zh-CN"] == counts["canonical_records"], "Chinese rendering cardinality failed")
    _require(quality["record_schema_and_semantic_failures"] == 0, "Schema/H2 semantic failures are non-zero")
    _require(quality["rendering_identity_failures"] == 0, "Rendering identity failures are non-zero")
    _require(quality["rendering_cardinality_failures"] == 0, "Rendering cardinality failures are non-zero")
    _require(quality["duplicate_canonical_payloads"] == 0, "Duplicate canonical payloads are non-zero")
    _require(quality["target_reconstruction"]["passed"] == quality["target_reconstruction"]["total"], "Target reconstruction is not 100%")
    _require(quality["p2_validation"]["passed"] == quality["p2_validation"]["unique_intent_result_payloads"], "Frozen P2 validation is not 100%")
    _require(quality["split_leakage"]["total_collisions"] == 0, "Cross-split leakage is non-zero")
    _require(all(quality["coverage"].values()), "Required coverage is incomplete")
    _require(not any(quality["scope_guards"].values()), "A prohibited scope guard is true")


def _collect_freeze_assets(
    data_files: list[tuple[Path, str, int]],
    report_paths: list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    assets: list[tuple[Path, str]] = [
        (ROOT / "GOAL_P3_A1_CAE_QUERYGROUND_DATASET.zh-CN.md", "task_contract"),
        (ROOT / "CODEX_CONTEXT_P3_A1.md", "context"),
        (FINAL_RECORD_SCHEMA, "record_schema"),
        (FINAL_MANIFEST_SCHEMA, "dataset_manifest_schema"),
        (ROOT / "docs" / "cae_query_ground_annotation_guide.zh-CN.md", "annotation_guide"),
    ]
    assets.extend((path, "entrypoint_or_runner") for path in sorted(ROOT.glob("build_cae_query_grounding_dataset*.py")))
    assets.extend((path, "entrypoint_or_runner") for path in sorted(ROOT.glob("run_cae_query_ground*.py")))
    assets.append((ROOT / "finalize_cae_query_grounding_dataset_release.py", "finalizer_runner"))
    assets.extend((path, "generator_renderer_or_validator_module") for path in sorted((ROOT / "utils" / "cae_query_ground").glob("*.py")))
    assets.extend((path, "test_runner") for path in sorted((ROOT / "utils").glob("run_p3_a1_cae_query_ground*.py")))
    assets.extend((path, "regression_test") for path in sorted((ROOT / "tests").glob("*cae_query*.py")))
    assets.extend((path, "schema") for path in sorted((ROOT / "schemas").glob("cae_query_ground*.json")))
    assets.extend((path, "template_crosswalk_or_phrase_bank") for path in sorted((ROOT / "configs" / "paper_v1").glob("cae_query*")) if path.is_file())
    assets.extend((path, "human_review_evidence") for path in sorted((REPORT_ROOT / "human_review_package").glob("*")) if path.is_file())
    assets.extend((path, "pilot_provenance") for path in sorted((OUTPUT / "pilot").rglob("*")) if path.is_file())
    assets.extend(report_paths)
    assets.extend((path, role) for path, role, _ in data_files)
    return _deduplicate(assets)


def _final_report(quality: dict[str, Any], h3: dict[str, Any], frozen_changes: int) -> str:
    base = final_acceptance_report(quality, h3, frozen_changes).rstrip()
    deterministic = read_json(REPORT_ROOT / "deterministic_regeneration.json")
    return base + f"""

## 冻结补充证据

- 输出级确定性复现：PASS，mismatches = 0
- deterministic plan SHA-256：`{deterministic['plan_sha256']}`
- 每条 canonical rendering 组成：2 英文 + 1 中文
- 正式规模已生成：是（仅表示 P3-A1 正式数据已生成，不是禁止行为）
- 下一阶段建议：`GO_P4_BASELINES`（仅供监督者决策；本任务未进入 P4）

本报告与交付清单提交监督者复核；本执行在 P3-A1 冻结后停止。
"""


def finalize(*, force: bool = True) -> dict[str, Any]:
    h3 = ingest_h3(force=force)
    quality = read_json(REPORT_ROOT / "quality_audit.json")
    deterministic = read_json(REPORT_ROOT / "deterministic_regeneration.json")
    _assert_gate(quality, h3, deterministic)

    source_audit, computed_crosswalk = audit_sources()
    _require(source_audit["automatic_pass"], "Source/role audit failed during finalization")
    _require(source_audit["p2_freeze"]["mismatch_count"] == 0, "P2 frozen assets changed")
    _require(read_yaml(CROSSWALK_PATH) == computed_crosswalk, "Frozen formal crosswalk no longer matches the source audit")
    write_json(SOURCE_AUDIT_JSON, source_audit, force=force)
    write_text(SOURCE_AUDIT_MD, render_audit_markdown(source_audit), force=force)

    build_final_schemas(force=True)
    corrected_schema = read_json(FINAL_MANIFEST_SCHEMA)
    _require(
        corrected_schema["properties"]["scope_guards"]["properties"]["formal_scale_generated"]["const"] is True,
        "Formal manifest schema was not corrected to formal_scale_generated=true",
    )
    data_files = _data_files(quality, deterministic)
    manifest = dataset_manifest(
        quality=quality,
        source_audit_path=SOURCE_AUDIT_JSON,
        data_files=data_files,
        h3=h3,
    )
    manifest["scope_guards"]["formal_scale_generated"] = True
    errors = validate_formal_manifest(manifest)
    _require(not errors, f"Formal dataset manifest failed schema: {errors}")
    manifest_path = OUTPUT / "manifests" / "dataset_manifest.json"
    write_json(manifest_path, manifest, force=force)

    frozen_changes = source_audit["p2_freeze"]["mismatch_count"]
    acceptance = acceptance_json(quality, h3, frozen_changes)
    acceptance["gate_p3"].update({
        "deterministic_regeneration": deterministic["automatic_pass"] and not deterministic["mismatches"],
        "p2_validation_100_percent": quality["p2_validation"]["passed"] == quality["p2_validation"]["unique_intent_result_payloads"],
        "rendering_identity_zero": quality["rendering_identity_failures"] == 0,
        "rendering_cardinality_zero": quality["rendering_cardinality_failures"] == 0,
        "duplicate_payloads_zero": quality["duplicate_canonical_payloads"] == 0,
        "bilingual_2en_1zh": quality["counts"]["languages"]["en-US"] == 2 * quality["counts"]["canonical_records"] and quality["counts"]["languages"]["zh-CN"] == quality["counts"]["canonical_records"],
        "test_part_s3_not_accessed": not quality["scope_guards"]["test_part_s3_accessed"],
    })
    acceptance["automatic_pass"] = acceptance["automatic_pass"] and all(acceptance["gate_p3"].values())
    _require(acceptance["automatic_pass"], f"Gate P3 acceptance failed: {acceptance}")

    acceptance_path = REPORT_ROOT / "acceptance.json"
    final_path = REPORT_ROOT / "final_acceptance.zh-CN.md"
    card_path = REPORT_ROOT / "dataset_card.zh-CN.md"
    leakage_path = REPORT_ROOT / "split_leakage_audit.json"
    test_summary_path = REPORT_ROOT / "test_summary.json"
    context_path = ROOT / "CODEX_CONTEXT_P3_A1.md"
    distribution_path = OUTPUT_REPORTS / "distribution.json"
    write_json(acceptance_path, acceptance, force=force)
    write_text(final_path, _final_report(quality, h3, frozen_changes), force=force)
    write_text(card_path, dataset_card(quality, h3), force=force)
    write_json(leakage_path, quality["split_leakage"], force=force)
    write_json(distribution_path, quality["counts"], force=force)
    test_summary = {
        "schema_version": "cae-query-ground-formal-test-summary/1",
        "status": "PASS",
        "dataset_version": FINAL_DATASET_VERSION,
        "automatic_pass": True,
        "gate_p3": acceptance["gate_p3"],
        "deterministic_regeneration": deterministic,
        "p2_validation": quality["p2_validation"],
        "target_reconstruction": quality["target_reconstruction"],
        "schema_semantic_failures": quality["record_schema_and_semantic_failures"],
        "rendering_identity_failures": quality["rendering_identity_failures"],
        "rendering_cardinality_failures": quality["rendering_cardinality_failures"],
        "duplicate_canonical_payloads": quality["duplicate_canonical_payloads"],
        "split_leakage_collisions": quality["split_leakage"]["total_collisions"],
        "coverage": quality["coverage"],
        "h3_01": {"reviewed": h3["review_rows"], "correct_rate": h3["effective_correct_rate"]},
        "p1_p2_frozen_hash_changes": frozen_changes,
        "scope_guards": quality["scope_guards"],
    }
    write_json(test_summary_path, test_summary, force=force)
    write_text(
        context_path,
        f"""# CODEX Context P3-A1

Goal P3-A1 已完成正式数据构造并冻结 `{FINAL_DATASET_VERSION}`。正式数据为 {quality['counts']['canonical_records']} canonical / {quality['counts']['renderings']} renderings，每条 canonical 固定 2 英文 + 1 中文。H3-01 为 {h3['review_rows']}/120，有效正确率 {h3['effective_correct_rate']:.1%}；20 条人工表达均已记录采用或拒绝理由。输出级确定性复现、P2 校验、目标重建和跨 split 泄漏审计均通过。P1/P2 冻结资产变化为 0。

本 Goal 未调用 LLM、未训练模型、未执行 Abaqus/CalculiX、未运行 Holdout、未访问真实件或 `test_part_s3`，未进入 P4。下一步仅可将验收报告交监督者审核；不得在本 Goal 内进入 P4。
""",
        force=force,
    )

    report_assets = [
        (SOURCE_AUDIT_JSON, "source_audit"),
        (SOURCE_AUDIT_MD, "source_audit_report"),
        (REPORT_ROOT / "quality_audit.json", "quality_audit"),
        (leakage_path, "split_leakage_audit"),
        (REPORT_ROOT / "deterministic_regeneration.json", "deterministic_regeneration"),
        (card_path, "dataset_card"),
        (acceptance_path, "acceptance"),
        (final_path, "final_acceptance"),
        (test_summary_path, "test_summary"),
        (distribution_path, "distribution"),
        (manifest_path, "dataset_manifest"),
        (REPORT_ROOT / "preflight_summary.zh-CN.md", "preflight"),
    ]
    freeze_assets = _collect_freeze_assets(data_files, report_assets)
    freeze = freeze_manifest(assets=freeze_assets, source_audit=source_audit, h3=h3)
    freeze["input_references"].update({
        "source_audit_sha256": sha256_file(SOURCE_AUDIT_JSON),
        "source_dataset_meta_sha256": source_audit["source"]["dataset_meta_sha256"],
        "source_artifact_manifest_sha256": source_audit["source"]["artifact_manifest_sha256"],
        "source_split_file_sha256": source_audit["source"]["split_file_sha256"],
        "crosswalk_sha256": sha256_file(CROSSWALK_PATH),
        "p2_policy_sha256": sha256_file(P2_POLICY),
        "deterministic_regeneration_sha256": sha256_file(REPORT_ROOT / "deterministic_regeneration.json"),
    })
    freeze["scope_guards"]["formal_scale_generated"] = True
    freeze_path = FREEZE_ROOT / "freeze_manifest.json"
    write_json(freeze_path, freeze, force=force)
    delivery_path = REPORT_ROOT / "delivery_manifest.json"
    write_json(
        delivery_path,
        delivery_manifest(freeze_assets + [(freeze_path, "freeze_manifest")]),
        force=force,
    )
    return {
        "status": "PASS",
        "dataset_version": FINAL_DATASET_VERSION,
        "canonical_records": quality["counts"]["canonical_records"],
        "renderings": quality["counts"]["renderings"],
        "deterministic_mismatches": len(deterministic["mismatches"]),
        "h3_reviewed": h3["review_rows"],
        "h3_correct_rate": h3["effective_correct_rate"],
        "p1_p2_frozen_hash_changes": frozen_changes,
        "dataset_manifest": str(manifest_path.resolve()),
        "final_report": str(final_path.resolve()),
        "freeze_manifest": str(freeze_path.resolve()),
        "delivery_manifest": str(delivery_path.resolve()),
    }

