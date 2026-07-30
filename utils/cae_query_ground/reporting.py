from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CROSSWALK_PATH,
    DATASET_VERSION,
    P2_POLICY,
    PILOT_ROOT,
    REPORT_ROOT,
    ROOT,
    SEED,
    SOURCE_AUDIT_JSON,
    SOURCE_MANIFEST,
    SOURCE_ROOT,
)
from .io import sha256_file


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for _ in handle)


def file_entry(path: Path, role: str, *, record_count: int | None = None) -> dict[str, Any]:
    if record_count is None:
        record_count = line_count(path) if path.suffix == ".jsonl" else 0
    return {
        "role": role,
        "path": relative_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": record_count,
    }


def dataset_manifest(
    *,
    source_audit: dict[str, Any],
    quality: dict[str, Any],
    data_files: Iterable[tuple[Path, str, int]],
    review_csv: Path,
    expression_csv: Path,
) -> dict[str, Any]:
    files = [file_entry(path, role, record_count=count) for path, role, count in data_files]
    files.extend(
        [
            file_entry(review_csv, "h3_01_review_csv", record_count=120),
            file_entry(expression_csv, "h3_01_expression_csv", record_count=20),
        ]
    )
    return {
        "schema_version": "cae-query-ground-dataset-manifest/1.0.0-h3.01-pilot",
        "dataset_version": DATASET_VERSION,
        "status": "pilot_waiting_h3_01",
        "seed": SEED,
        "source": {
            "dataset_root": str(SOURCE_ROOT.resolve()),
            "sample_manifest_sha256": source_audit["source"]["sample_manifest_sha256"],
            "source_audit_sha256": sha256_file(SOURCE_AUDIT_JSON),
            "crosswalk_sha256": sha256_file(CROSSWALK_PATH),
            "p2_policy_sha256": sha256_file(P2_POLICY),
        },
        "counts": quality["counts"],
        "files": files,
        "template_split_manifest": "configs/paper_v1/cae_query_template_split_manifest.json",
        "review_gate": {
            "gate": "H3-01",
            "status": "WAITING_FOR_REAL_USER_REVIEW",
            "required_rows": 120,
            "reviewed_rows": 0,
            "review_csv": relative_path(review_csv),
            "expression_csv": relative_path(expression_csv),
        },
        "scope_guards": quality["scope_guards"],
    }


def canonical_rendering_index(
    canonical_records: list[dict[str, Any]], rendering_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_canonical: dict[str, list[str]] = {}
    for record in rendering_records:
        canonical_id = record["source_ref"]["canonical_record_id"]
        by_canonical.setdefault(canonical_id, []).append(record["query"]["rendering_id"])
    return [
        {
            "canonical_record_id": record["source_ref"]["canonical_record_id"],
            "canonical_semantics_sha256": record["source_ref"]["canonical_semantics_sha256"],
            "dataset_split": record["source_ref"]["dataset_split"],
            "rendering_ids": sorted(by_canonical[record["source_ref"]["canonical_record_id"]]),
        }
        for record in canonical_records
    ]


def dataset_card(quality: dict[str, Any], review_distribution: dict[str, Any]) -> str:
    counts = quality["counts"]
    leakage = quality["split_leakage"]
    return f"""# CAE-QueryGround v1 Pilot Dataset Card

状态：`BOUNDED-PASS / WAITING_H3_01`  
版本：`{DATASET_VERSION}`  
固定随机种子：`{SEED}`

本交付仅为 H3-01 前 pilot，不是正式 20,000 条数据，也不是 P3 冻结或最终验收资产。

## 数据规模与覆盖

- canonical records：{counts['canonical_records']}
- controlled renderings：{counts['renderings']}（每条 canonical：2 条 en-US + 1 条 zh-CN）
- source split：{counts['splits']}
- language：{counts['languages']}
- level：{counts['levels']}
- status：{counts['statuses']}
- feature：{counts['features']}
- operation：{counts['operations']}

## 数据契约

记录严格分离 `source_ref`、`query`、冻结 P2 `intent_result` 与仅监督使用的 `ground_truth`。可执行 feature target 的 face set 可从只读源 graph、role crosswalk 与 target instance 重新计算。自然语言和 CAE-IR 不包含 sample/part/file、face ID、instance ID、真值面积或 geometry source。

双语文本完全由受控 phrase bank 和 split 专属 template family 生成；没有调用 LLM。P2 CAE-IR 版本保持 `1.0.0-h2.01`。

## 自动质量结果

- record/schema/semantic failures：{quality['record_schema_and_semantic_failures']}
- target reconstruction：{quality['target_reconstruction']['passed']}/{quality['target_reconstruction']['total']}
- rendering identity failures：{quality['rendering_identity_failures']}
- rendering cardinality failures：{quality['rendering_cardinality_failures']}
- cross-split leakage collisions：{leakage['total_collisions']}
- automatic pilot checks：`{'PASS' if quality['automatic_pass'] else 'FAIL'}`

## H3-01 人工 Gate

已抽取 {review_distribution['row_count']} 条：语言 {review_distribution['languages']}，Level {review_distribution['levels']}，Status {review_distribution['statuses']}。当前人工决定空白 {review_distribution['review_decisions_blank']}/120。

自动校验不能替代工程师对表达自然度、CAE 工况合理性和 role 可理解性的判断。必须由用户真实回填 120/120 行，并补充至少 10 条自然表达；在此之前状态保持 `WAITING_H3_01`。

## 范围边界

未调用 LLM，未训练模型，未执行 Abaqus/CalculiX，未读取/运行 Holdout 或真实件，未生成正式规模，未进入 P4。P1/P2 冻结资产只读。
"""


def pilot_delivery_manifest(paths: Iterable[tuple[Path, str]]) -> dict[str, Any]:
    entries = []
    for path, role in paths:
        entries.append(file_entry(path, role))
    return {
        "schema_version": "cae-query-ground-pilot-delivery/1",
        "status": "BOUNDED-PASS_WAITING_H3_01",
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "asset_count": len(entries),
        "assets": entries,
        "excluded_by_design": [
            "formal_20000_canonical_dataset",
            "p3_freeze_manifest",
            "p3_acceptance",
            "p3_final_acceptance",
            "p4_assets",
        ],
    }


def human_fields_are_blank(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ("review_decision", "error_category", "corrected_query", "reviewer_note")
    return len(rows) == 120 and all(not row[field].strip() for row in rows for field in fields)

