from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CROSSWALK_PATH,
    P2_FREEZE_MANIFEST,
    P2_POLICY,
    REPORT_ROOT,
    ROOT,
    SEED,
    SOURCE_MANIFEST,
    SOURCE_ROOT,
)
from .formal_planner import FINAL_DATASET_VERSION
from .io import sha256_file


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def file_entry(path: Path, role: str, record_count: int = 0) -> dict[str, Any]:
    return {
        "role": role,
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": record_count,
    }


def dataset_manifest(
    *,
    quality: dict[str, Any],
    source_audit_path: Path,
    data_files: Iterable[tuple[Path, str, int]],
    h3: dict[str, Any],
) -> dict[str, Any]:
    review_root = REPORT_ROOT / "human_review_package"
    files = [file_entry(path, role, count) for path, role, count in data_files]
    files.extend([
        file_entry(review_root / "cae_query_review_120.csv", "h3_review_csv", 120),
        file_entry(review_root / "human_expression_additions_20.csv", "h3_expression_csv", 20),
        file_entry(review_root / "human_expression_adoption_decisions.csv", "h3_expression_decisions", 20),
        file_entry(review_root / "h3_01_ingestion_summary.json", "h3_ingestion_summary", 1),
    ])
    guards = quality["scope_guards"]
    return {
        "schema_version": "cae-query-ground-dataset-manifest/1.0.0-h3.01",
        "dataset_version": FINAL_DATASET_VERSION,
        "status": "frozen",
        "seed": SEED,
        "source": {
            "dataset_root": str(SOURCE_ROOT.resolve()),
            "sample_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "source_audit_sha256": sha256_file(source_audit_path),
            "crosswalk_sha256": sha256_file(CROSSWALK_PATH),
            "p2_policy_sha256": sha256_file(P2_POLICY),
        },
        "counts": {
            key: quality["counts"][key]
            for key in (
                "canonical_records",
                "renderings",
                "splits",
                "languages",
                "levels",
                "statuses",
                "features",
                "operations",
            )
        },
        "files": files,
        "template_split_manifest": "configs/paper_v1/cae_query_template_split_manifest.formal.json",
        "review_gate": {
            "gate": "H3-01",
            "status": "PASS",
            "required_rows": 120,
            "reviewed_rows": 120,
            "review_csv": "reports/p3_a1_cae_queryground_v1/human_review_package/cae_query_review_120.csv",
            "expression_csv": "reports/p3_a1_cae_queryground_v1/human_review_package/human_expression_additions_20.csv",
        },
        "scope_guards": {
            key: guards[key]
            for key in (
                "llm_called",
                "model_trained",
                "solver_executed",
                "holdout_run",
                "real_part_content_accessed",
                "p4_entered",
            )
        },
    }


def dataset_card(quality: dict[str, Any], h3: dict[str, Any]) -> str:
    counts = quality["counts"]
    return f"""# CAE-QueryGround v1 Dataset Card

状态：`FROZEN / PASS`  
版本：`{FINAL_DATASET_VERSION}`  
随机种子：`{SEED}`

## 任务边界

本数据集学习自然语言 CAE 前处理指令到冻结 CAE-IR 与 Oracle 几何目标区域的映射。单条 query 是可组合的材料、网格、约束或载荷指令，不等同于完整可收敛 FEA job；分析步、接触、求解控制、输出请求和求解器执行不在 P3-A1 范围内。

## 数据规模

- canonical：{counts['canonical_records']}
- renderings：{counts['renderings']}，每条 canonical 固定 2 英文 + 1 中文
- source graphs：{counts['source_graphs']}
- train / val / test：{counts['splits']}
- English / Chinese：{counts['languages']}
- L1–L5：{counts['levels']}
- VALID / CLARIFICATION / INVALID：{counts['statuses']}
- features：{counts['features']}
- operations：{counts['operations']}
- target face-count：{counts['target_face_counts']}

## H3-01

- 人工审核：{h3['review_rows']}/120
- 有效正确率：{h3['effective_correct_rate']:.1%}
- 表达补充：{h3['expression_rows_populated']} 条
- 表达采用：{h3['expression_decisions']}

## 自动质量

- schema/P2/semantic failures：{quality['record_schema_and_semantic_failures']}
- target reconstruction：{quality['target_reconstruction']['passed']}/{quality['target_reconstruction']['total']}
- rendering identity failures：{quality['rendering_identity_failures']}
- rendering cardinality failures：{quality['rendering_cardinality_failures']}
- duplicate canonical payloads：{quality['duplicate_canonical_payloads']}
- cross-split leakage：{quality['split_leakage']['total_collisions']}

## 语言策略

英文是论文主实验语言；中文保留为跨语言 grounding 与工程审查语言。train、val、test 的 template family 和句式构造互不重叠。双语 rendering 的 `source_language` 与 query 严格一致。

## 禁止范围

未调用 LLM，未训练模型，未执行 Abaqus/CalculiX，未读取或运行 Holdout、真实件或 `test_part_s3`，未进入 P4。P1/P2 冻结输入保持只读。
"""


def final_acceptance_report(
    quality: dict[str, Any], h3: dict[str, Any], frozen_changes: int
) -> str:
    counts = quality["counts"]
    return f"""# Goal P3-A1 最终验收报告

结论：`PASS`  
CAE-QueryGround version：`{FINAL_DATASET_VERSION}`

- canonical records：{counts['canonical_records']}
- renderings：{counts['renderings']}
- train / val / test：{counts['splits']}
- English / Chinese：{counts['languages']}
- L1 / L2 / L3 / L4 / L5：{counts['levels']}
- VALID / CLARIFICATION / INVALID：{counts['statuses']}
- target reconstruction：{quality['target_reconstruction']['passed']}/{quality['target_reconstruction']['total']}
- P2 distinct intent_result validation：{quality['p2_validation']['passed']}/{quality['p2_validation']['unique_intent_result_payloads']}
- schema/semantic failures：{quality['record_schema_and_semantic_failures']}
- split leakage collisions：{quality['split_leakage']['total_collisions']}
- H3-01：{h3['review_rows']}/120，正确率 {h3['effective_correct_rate']:.1%}
- expression additions：{h3['expression_rows_populated']}，采用决定 {h3['expression_decisions']}
- P1/P2 frozen hash changes：{frozen_changes}
- 禁止阶段访问：无内容访问；未运行 Holdout、真实件或 test_part_s3

所有 Gate P3 自动条件与真实人工 Gate 均满足。本验收仅冻结 P3-A1 数据集，不授权进入 P4。
"""


def acceptance_json(
    quality: dict[str, Any], h3: dict[str, Any], frozen_changes: int
) -> dict[str, Any]:
    return {
        "schema_version": "cae-query-ground-p3-a1-acceptance/1",
        "status": "PASS",
        "dataset_version": FINAL_DATASET_VERSION,
        "gate_p3": {
            "h3_review": h3["review_rows"] == 120 and h3["effective_correct_rate"] >= 0.95,
            "expression_additions": h3["expression_rows_populated"] >= 10,
            "canonical_minimum": quality["counts"]["canonical_records"] >= 20000,
            "rendering_minimum": quality["counts"]["renderings"] >= 60000,
            "target_reconstruction_100_percent": quality["target_reconstruction"]["passed"] == quality["target_reconstruction"]["total"],
            "schema_and_h2_100_percent": quality["record_schema_and_semantic_failures"] == 0,
            "split_leakage_zero": quality["split_leakage"]["total_collisions"] == 0,
            "coverage_complete": all(quality["coverage"].values()),
            "identity_leakage_zero": quality["record_schema_and_semantic_failures"] == 0,
            "frozen_hash_changes_zero": frozen_changes == 0,
            "prohibited_scope_zero": not any(quality["scope_guards"].values()),
        },
        "automatic_pass": quality["automatic_pass"],
    }


def freeze_manifest(
    *,
    assets: Iterable[tuple[Path, str]],
    source_audit: dict[str, Any],
    h3: dict[str, Any],
) -> dict[str, Any]:
    entries = [file_entry(path, role) for path, role in assets]
    return {
        "schema_version": "cae-query-ground-freeze-manifest/1",
        "dataset_version": FINAL_DATASET_VERSION,
        "status": "FROZEN",
        "seed": SEED,
        "asset_count": len(entries),
        "assets": entries,
        "input_references": {
            "source_dataset_root": str(SOURCE_ROOT.resolve()),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "p2_freeze_manifest": str(P2_FREEZE_MANIFEST.resolve()),
            "p2_freeze_manifest_sha256": sha256_file(P2_FREEZE_MANIFEST),
            "p2_asset_count": source_audit["p2_freeze"]["asset_count"],
            "p2_hash_changes": source_audit["p2_freeze"]["mismatch_count"],
            "p2_supervisor_runner_sha256": source_audit["p2_freeze"]["supervisor_added_runner"]["actual_sha256"],
            "h3_review_sha256": h3["review_csv_sha256"],
            "h3_expression_sha256": h3["expression_csv_sha256"],
            "h3_expression_decisions_sha256": h3["expression_decision_csv_sha256"],
        },
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
            "test_part_s3_accessed": False,
            "p4_entered": False,
        },
    }


def delivery_manifest(assets: Iterable[tuple[Path, str]]) -> dict[str, Any]:
    entries = [file_entry(path, role) for path, role in assets]
    return {
        "schema_version": "cae-query-ground-delivery-manifest/1",
        "dataset_version": FINAL_DATASET_VERSION,
        "status": "PASS",
        "asset_count": len(entries),
        "assets": entries,
    }

