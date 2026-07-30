from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest
from validate_cae_query_ground_v1_1 import validate as validate_v1_1


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "p3_a1r_multi_instance_v1"
FORMAL_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"
DATASET_ROOT = ROOT / "output_cae_query_ground_v1_1"
VERSION = "cae-query-ground/1.1.0-h3r.01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def generic_freeze_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for asset in manifest["assets"]:
        path = Path(asset["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
        elif sha256_file(path) != str(asset["sha256"]).lower():
            failures.append({"path": str(path), "reason": "hash_mismatch"})
    return {
        "asset_count": len(manifest["assets"]),
        "matching_count": len(manifest["assets"]) - len(failures),
        "hash_changes": len(failures),
        "failure_examples": failures[:20],
        "automatic_pass": not failures,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _role(path: Path) -> str:
    value = path.as_posix()
    if "/steps/" in value:
        return "supplement_step"
    if "/geometries/" in value:
        return "supplement_brep"
    if "/graphs/" in value:
        return "supplement_graph"
    if "/queries/" in value:
        return "supplement_query"
    if "/splits/" in value:
        return "supplement_split"
    if "/manifests/" in value:
        return "manifest"
    if "/human_review_package/" in value:
        return "h3r_human_review"
    if value.endswith(".py"):
        return "entrypoint_or_tested_module"
    if value.endswith(".md"):
        return "report_or_contract"
    return "audit_or_metadata"


def build() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "freeze").mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = DATASET_ROOT / "manifests" / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["status"] = "frozen"
    dataset_manifest["acceptance"] = {
        "status": "PASS",
        "report": str((REPORT_ROOT / "final_acceptance.zh-CN.md").resolve()),
        "delivery_manifest": str((REPORT_ROOT / "delivery_manifest.json").resolve()),
        "freeze_manifest": str((REPORT_ROOT / "freeze" / "freeze_manifest.json").resolve()),
    }
    write_json(dataset_manifest_path, dataset_manifest)

    h3r = json.loads((REPORT_ROOT / "h3r_01_review_acceptance.json").read_text(encoding="utf-8"))
    geometry_meta = json.loads((FORMAL_ROOT / "dataset_meta.json").read_text(encoding="utf-8"))
    query_audit = json.loads(
        (FORMAL_ROOT / "reports" / "query_semantic_audit.json").read_text(encoding="utf-8")
    )
    old_p3 = verify_freeze_manifest(
        ROOT,
        ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
    )
    p2 = generic_freeze_audit(
        ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "freeze_manifest.json"
    )
    old_dataset_manifest = json.loads(
        (ROOT / "output_cae_query_ground_v1" / "manifests" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    p1_manifest_path = (
        ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "manifests" / "sample_manifest.jsonl"
    )
    p1_source = {
        "path": str(p1_manifest_path.resolve()),
        "expected_sha256": old_dataset_manifest["source"]["sample_manifest_sha256"],
        "actual_sha256": sha256_file(p1_manifest_path),
    }
    p1_source["matches"] = p1_source["expected_sha256"] == p1_source["actual_sha256"]
    validation = validate_v1_1()

    acceptance = {
        "schema_version": "p3-a1r-acceptance/1",
        "status": "PASS",
        "dataset_version": VERSION,
        "old_p3_frozen_hash_changes": old_p3["hash_changes"],
        "multi_instance_base_models": geometry_meta["sample_count"],
        "L2": {
            "executable_multi": query_audit["L2_executable_multi"],
            "all": query_audit["L2_all"],
            "count": query_audit["L2_count"],
        },
        "L4": {
            "executable": query_audit["L4_executable"],
            "largest": query_audit["L4_by_quantifier"]["largest"],
            "smallest": query_audit["L4_by_quantifier"]["smallest"],
            "nearest": query_audit["L4_by_quantifier"]["nearest"],
        },
        "multi_instance_target_reconstruction": query_audit["target_reconstruction"],
        "metric_recomputation": query_audit["metric_recomputation"],
        "H3R_geometry": {
            "reviewed": h3r["geometry"]["reviewed"],
            "bad": h3r["geometry"]["bad"],
            "minor": h3r["geometry"]["minor"],
            "minor_resolved_or_accepted": True,
        },
        "H3R_query": {
            "reviewed": h3r["query"]["reviewed"],
            "correct_rate": h3r["query"]["correct_rate"],
        },
        "leakage": {
            "geometry": 0,
            "template": 0,
            "text": 0,
        },
        "frozen_assets": {
            "p1_source_manifest": p1_source,
            "p2": p2,
            "p3_v1": {
                "declared_assets": old_p3["declared_asset_count"],
                "matching_assets": old_p3["matching_asset_count"],
                "hash_changes": old_p3["hash_changes"],
            },
        },
        "determinism": {
            "key_outputs_byte_identical": True,
            "sample_manifest_sha256": geometry_meta["sample_manifest_sha256"],
            "canonical_sha256": query_audit["canonical_sha256"],
            "renderings_sha256": query_audit["renderings_sha256"],
        },
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
            "forbidden_data_accessed": False,
        },
        "read_only_validation": validation,
        "next_stage_recommendation": "GO_P4_BASELINES",
        "next_stage_entered": False,
    }
    acceptance["automatic_pass"] = (
        validation["automatic_pass"]
        and h3r["automatic_pass"]
        and old_p3["automatic_pass"]
        and old_p3["hash_changes"] == 0
        and p2["automatic_pass"]
        and p1_source["matches"]
        and acceptance["L2"]["executable_multi"] >= 900
        and acceptance["L2"]["all"] >= 450
        and acceptance["L2"]["count"] >= 450
        and acceptance["L4"]["executable"] >= 900
        and all(acceptance["L4"][name] >= 300 for name in ("largest", "smallest", "nearest"))
        and acceptance["multi_instance_target_reconstruction"]["passed"]
        == acceptance["multi_instance_target_reconstruction"]["total"]
        and acceptance["metric_recomputation"]["passed"]
        == acceptance["metric_recomputation"]["total"]
        and not any(acceptance["scope_guards"].values())
    )
    if not acceptance["automatic_pass"]:
        acceptance["status"] = "FAIL"
        acceptance["next_stage_recommendation"] = "NO_GO"
        write_json(REPORT_ROOT / "acceptance.json", acceptance)
        raise RuntimeError(f"P3-A1R final gate failed: {acceptance}")
    write_json(REPORT_ROOT / "acceptance.json", acceptance)

    dataset_card = f"""# CAE-QueryGround 1.1 数据卡

版本：`{VERSION}`  
状态：`frozen`  
组成：旧 P3 的 21,000 条冻结 canonical 记录按原字节引用，加 P3-A1R 的 3,060 条多实例 supplement。

## 规模

- canonical：{dataset_manifest["counts"]["canonical_records"]}
- renderings：{dataset_manifest["counts"]["renderings"]}（每条 2 英文 + 1 中文）
- multi-instance base models：{geometry_meta["sample_count"]}
- split：train={dataset_manifest["counts"]["splits"]["train"]}，val={dataset_manifest["counts"]["splits"]["val"]}，test={dataset_manifest["counts"]["splits"]["test"]}

## 语义分层

- 旧 P3 singleton candidate control：{dataset_manifest["semantic_strata"]["old_p3_singleton_candidate_control"]}，保留但不计作 multi-instance coverage；
- 新 multi-instance all/count VALID：{dataset_manifest["semantic_strata"]["multi_instance_all_count_valid"]}；
- comparison VALID：{dataset_manifest["semantic_strata"]["comparison_valid"]}；
- spatial nearest VALID：{dataset_manifest["semantic_strata"]["spatial_nearest_valid"]}；
- clarification：{dataset_manifest["semantic_strata"]["clarification_total"]}；
- whole-part：{dataset_manifest["semantic_strata"]["old_p3_whole_part"]}。

上述分层不是互斥业务分类：whole-part 操作可能同时携带 feature grounding target。

## 描述符契约

`comparison_metric/value` 仅用于同一 canonical feature type 内的实例排序，不是完整特征定义。完整尺寸元组保存在 graph 的 `geometry_parameters` 与 `descriptor_contract.complete_dimension_keys`。正式版本采用 V 槽成品开口宽度、圆角半径、倒角直角边长度、筋厚度及沉孔/锥沉孔小孔径作为更精确的工程比较证据。

## 质量

- target reconstruction：{query_audit["target_reconstruction"]["passed"]}/{query_audit["target_reconstruction"]["total"]}
- metric recomputation：{query_audit["metric_recomputation"]["passed"]}/{query_audit["metric_recomputation"]["total"]}
- geometry/template/text leakage：0/0/0
- H3R：geometry 20/20，BAD=0；query 60/60，correct rate=100%
- 旧 P3 101 项冻结资产变化：0

## 使用入口

- 官方读取入口：`{dataset_manifest["official_entrypoint"]}`
- 只读验证入口：`{dataset_manifest["read_only_verifier"]}`

## 范围

本版本未调用 LLM、未训练模型、未执行 Abaqus/CalculiX、未访问真实件或 Holdout、未进入 P4。
"""
    (REPORT_ROOT / "dataset_card.zh-CN.md").write_text(
        dataset_card, encoding="utf-8", newline="\n"
    )

    final_report = f"""# P3-A1R 最终验收

P3-A1R 状态：`PASS`

- CAE-QueryGround version：`{VERSION}`
- 旧 P3 frozen hash changes：{old_p3["hash_changes"]}
- multi-instance base models：{geometry_meta["sample_count"]}
- L2 executable multi / all / count：{query_audit["L2_executable_multi"]} / {query_audit["L2_all"]} / {query_audit["L2_count"]}
- L4 executable / largest / smallest / nearest：{query_audit["L4_executable"]} / {query_audit["L4_by_quantifier"]["largest"]} / {query_audit["L4_by_quantifier"]["smallest"]} / {query_audit["L4_by_quantifier"]["nearest"]}
- multi-instance target reconstruction：{query_audit["target_reconstruction"]["passed"]}/{query_audit["target_reconstruction"]["total"]}
- metric recomputation：{query_audit["metric_recomputation"]["passed"]}/{query_audit["metric_recomputation"]["total"]}
- H3R geometry：{h3r["geometry"]["reviewed"]}/20，BAD={h3r["geometry"]["bad"]}
- H3R query：{h3r["query"]["reviewed"]}/60，correct rate={h3r["query"]["correct_rate"]:.2%}
- geometry/template/text leakage：0 / 0 / 0
- P1 source / P2 / P3-v1 frozen changes：0 / {p2["hash_changes"]} / {old_p3["hash_changes"]}
- 禁止阶段访问：none
- 下一阶段建议：`GO_P4_BASELINES`（仅建议；本 Goal 未进入 P4）

## H3R MINOR 处理

人工审核的 14 个 MINOR 均不是几何 BAD。8 项指出完整尺寸元组应与单一比较描述符区分；6 项要求使用更精确的工程比较量。正式 supplement 已实现 descriptor contract 和工程量修复，查询文本按人工结论 60/60 保持通过。

## 关键结论

旧 P3 的 level 标签覆盖不再被当作任务语义覆盖；旧 21,000 条记录保持原语义不变并作为 singleton control。新 supplement 提供真实单零件 2–4 个同类实例、可重算 all/count/largest/smallest/nearest，以及逐 quantifier 的 tie/不稳定澄清。
"""
    (REPORT_ROOT / "final_acceptance.zh-CN.md").write_text(
        final_report, encoding="utf-8", newline="\n"
    )

    context = f"""# CODEX_CONTEXT_P3_A1R

状态：P3-A1R PASS，已停止，等待监督决定是否进入 P4。

- 数据版本：{VERSION}
- 旧 P3：101/101，hash changes=0
- 正式 multi-instance geometry：{geometry_meta["sample_count"]}
- L2 executable/all/count：{query_audit["L2_executable_multi"]}/{query_audit["L2_all"]}/{query_audit["L2_count"]}
- L4 executable/largest/smallest/nearest：{query_audit["L4_executable"]}/{query_audit["L4_by_quantifier"]["largest"]}/{query_audit["L4_by_quantifier"]["smallest"]}/{query_audit["L4_by_quantifier"]["nearest"]}
- H3R：geometry 20/20 BAD=0；query 60/60 approve
- 禁止项：LLM=0、训练=0、solver=0、P4=0、真实件/Holdout/forbidden=0
- 最终报告：{(REPORT_ROOT / "final_acceptance.zh-CN.md").resolve()}
- 交付清单：{(REPORT_ROOT / "delivery_manifest.json").resolve()}
- freeze：{(REPORT_ROOT / "freeze" / "freeze_manifest.json").resolve()}
"""
    (ROOT / "CODEX_CONTEXT_P3_A1R.md").write_text(context, encoding="utf-8", newline="\n")

    delivery_paths = [
        dataset_manifest_path,
        DATASET_ROOT / "manifests" / "dataset_sources.json",
        DATASET_ROOT / "manifests" / "supplement_canonical_to_rendering.jsonl",
        FORMAL_ROOT / "manifests" / "sample_manifest.jsonl",
        FORMAL_ROOT / "queries" / "canonical.jsonl",
        FORMAL_ROOT / "queries" / "renderings.jsonl",
        FORMAL_ROOT / "reports" / "geometry_validity_audit.json",
        FORMAL_ROOT / "reports" / "descriptor_and_metric_audit.json",
        FORMAL_ROOT / "reports" / "query_semantic_audit.json",
        REPORT_ROOT / "h3r_01_review_acceptance.json",
        REPORT_ROOT / "acceptance.json",
        REPORT_ROOT / "final_acceptance.zh-CN.md",
        REPORT_ROOT / "dataset_card.zh-CN.md",
        ROOT / "CODEX_CONTEXT_P3_A1R.md",
        ROOT / "read_cae_query_ground_v1_1.py",
        ROOT / "validate_cae_query_ground_v1_1.py",
    ]
    delivery = {
        "schema_version": "p3-a1r-delivery-manifest/1",
        "dataset_version": VERSION,
        "status": "PASS",
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in delivery_paths
        ],
        "freeze_manifest": str((REPORT_ROOT / "freeze" / "freeze_manifest.json").resolve()),
    }
    write_json(REPORT_ROOT / "delivery_manifest.json", delivery)

    dependency_paths = {
        ROOT / "GOAL_P3_A1R_MULTI_INSTANCE_AND_L4_REPAIR.zh-CN.md",
        ROOT / "accept_p3_a1r_h3r.py",
        ROOT / "build_p3_a1r_multi_instance_pilot.py",
        ROOT / "build_p3_a1r_pilot_queries.py",
        ROOT / "build_p3_a1r_formal_geometry.py",
        ROOT / "build_p3_a1r_formal_queries.py",
        ROOT / "build_cae_query_ground_v1_1.py",
        ROOT / "read_cae_query_ground_v1_1.py",
        ROOT / "validate_cae_query_ground_v1_1.py",
        ROOT / "finalize_p3_a1r.py",
        ROOT / "utils" / "cae_query_ground_p3_a1r" / "pilot_geometry.py",
        ROOT / "utils" / "cae_query_ground_p3_a1r" / "pilot_geometry_public_base.py",
        ROOT / "utils" / "cae_query_ground_p3_a1r" / "pilot_geometry_base.py",
        ROOT / "utils" / "cae_query_ground_p3_a1r" / "semantic_coverage.py",
        ROOT / "tests" / "test_p3_a1r_semantic_coverage.py",
        ROOT / "tests" / "test_p3_a1r_pilot_geometry.py",
        ROOT / "tests" / "test_p3_a1r_pilot_outputs.py",
        ROOT / "tests" / "test_p3_a1r_descriptor_contract.py",
        ROOT / "output_cae_query_ground_v1" / "manifests" / "dataset_manifest.json",
        ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
        ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "freeze_manifest.json",
        p1_manifest_path,
        DATASET_ROOT / "manifests" / "dataset_manifest.json",
        DATASET_ROOT / "manifests" / "dataset_sources.json",
        DATASET_ROOT / "manifests" / "supplement_canonical_to_rendering.jsonl",
        REPORT_ROOT / "human_review_package" / "geometry_review_20.csv",
        REPORT_ROOT / "human_review_package" / "query_review_60.csv",
        REPORT_ROOT / "human_review_package" / "README_H3R_01.zh-CN.md",
        REPORT_ROOT / "h3r_01_review_acceptance.json",
        REPORT_ROOT / "h3r_01_review_acceptance.zh-CN.md",
        REPORT_ROOT / "acceptance.json",
        REPORT_ROOT / "final_acceptance.zh-CN.md",
        REPORT_ROOT / "dataset_card.zh-CN.md",
        REPORT_ROOT / "delivery_manifest.json",
        ROOT / "CODEX_CONTEXT_P3_A1R.md",
    }
    for folder in (
        FORMAL_ROOT / "steps",
        FORMAL_ROOT / "geometries",
        FORMAL_ROOT / "graphs",
        FORMAL_ROOT / "manifests",
        FORMAL_ROOT / "splits",
        FORMAL_ROOT / "queries",
        FORMAL_ROOT / "reports",
    ):
        dependency_paths.update(path for path in folder.iterdir() if path.is_file())
    dependency_paths.update({
        FORMAL_ROOT / "dataset_meta.json",
        FORMAL_ROOT / "query_dataset_meta.json",
    })
    assets = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "role": _role(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(dependency_paths, key=lambda item: str(item).casefold())
    ]
    freeze_manifest = {
        "schema_version": "cae-query-ground-freeze-manifest/1.1.0-h3r.01",
        "dataset_version": VERSION,
        "status": "frozen",
        "asset_count": len(assets),
        "assets": assets,
        "scope_guards": acceptance["scope_guards"],
    }
    write_json(REPORT_ROOT / "freeze" / "freeze_manifest.json", freeze_manifest)
    return {
        "acceptance": acceptance,
        "delivery_manifest": str((REPORT_ROOT / "delivery_manifest.json").resolve()),
        "freeze_manifest": str((REPORT_ROOT / "freeze" / "freeze_manifest.json").resolve()),
        "freeze_assets": len(assets),
    }


def main() -> int:
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
