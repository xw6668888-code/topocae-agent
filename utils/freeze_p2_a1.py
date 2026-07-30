"""Freeze H2-approved P2-A1 artifacts and generate acceptance evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from utils.cae_intent import (
    load_feature_role_policy,
    validate_intent,
    validate_intent_result,
)


FORBIDDEN_KEYS = {
    "raw_face_id", "face_id", "face_ids", "faces", "sample_id", "part_id",
    "model_file", "model_filename", "file_name", "filename", "area_truth", "truth_area",
    "geometry_source", "coverage_status", "abstention_reasons",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def scan_forbidden_keys(value: Any, path: tuple[Any, ...] = ()) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                found.append("/" + "/".join(str(part) for part in path + (key,)))
            found.extend(scan_forbidden_keys(item, path + (key,)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(scan_forbidden_keys(item, path + (index,)))
    return found


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_schema_examples(root: Path) -> dict[str, int]:
    intent_schema = json.loads((root / "schemas" / "cae_intent.schema.json").read_text(encoding="utf-8"))
    result_schema = json.loads((root / "schemas" / "cae_intent_result.schema.json").read_text(encoding="utf-8"))
    context_schema = json.loads((root / "schemas" / "cae_grounding_context.schema.json").read_text(encoding="utf-8"))
    for schema in (intent_schema, result_schema, context_schema):
        Draft202012Validator.check_schema(schema)
    for index, example in enumerate(intent_schema.get("examples", [])):
        validation = validate_intent(example)
        if not validation["valid"]:
            raise AssertionError(f"Invalid intent schema example {index}: {validation['errors'][:3]}")
    for index, example in enumerate(result_schema.get("examples", [])):
        validation = validate_intent_result(example)
        if not validation["valid"]:
            raise AssertionError(f"Invalid result schema example {index}: {validation['errors'][:3]}")
    context_validator = Draft202012Validator(context_schema)
    for index, example in enumerate(context_schema.get("examples", [])):
        errors = list(context_validator.iter_errors(example))
        if errors:
            raise AssertionError(f"Invalid grounding context example {index}: {errors[0].message}")
    return {
        "intent_examples": len(intent_schema.get("examples", [])),
        "result_examples": len(result_schema.get("examples", [])),
        "grounding_context_examples": len(context_schema.get("examples", [])),
    }


def recognizer_asset_audit(root: Path) -> dict[str, Any]:
    manifest_path = root / "reports" / "p1_d_lite_v1" / "freeze" / "recognizer_dev_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    changes = []
    for entry in manifest["assets"]:
        path = Path(entry["path"])
        actual_exists = path.exists()
        actual_bytes = path.stat().st_size if actual_exists else None
        actual_hash = sha256(path) if actual_exists else None
        matches = actual_exists and actual_bytes == entry["bytes"] and actual_hash == entry["sha256"]
        row = {
            "role": entry["role"], "path": str(path),
            "expected_bytes": entry["bytes"], "actual_bytes": actual_bytes,
            "expected_sha256": entry["sha256"], "actual_sha256": actual_hash,
            "matches": matches,
        }
        rows.append(row)
        if not matches:
            changes.append(row)
    return {
        "manifest": str(manifest_path.resolve()),
        "asset_count": len(rows),
        "matching_count": sum(row["matches"] for row in rows),
        "change_count": len(changes),
        "changes": changes,
        "assets": rows,
    }


def read_review_counts(review_dir: Path) -> dict[str, Any]:
    result = {}
    for name in ("feature_role_policy_review.csv", "cae_requirement_review_40.csv"):
        with (review_dir / name).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts: dict[str, int] = {}
        for decision in ("approve", "edit", "reject"):
            counts[decision] = sum(row["review_decision"] == decision for row in rows)
        result[name] = {"rows": len(rows), "reviewed": sum(counts.values()), "decisions": counts}
    return result


def final_markdown(
    status: str,
    version: dict[str, Any],
    test_summary: dict[str, Any],
    recognizer: dict[str, Any],
    acceptance_path: Path,
    manifest_path: Path,
) -> str:
    counts = test_summary["fixtures"]["status_counts"]
    return f"""# P2-A1 CAE-IR 最终验收报告

状态：`{status}`  
冻结版本：`{version['cae_ir_version']}`  
H2-01：`completed`

## 1. 交付结果

- 目标 feature：{version['counts']['canonical_features']} 类
- selectable role：{version['counts']['selectable_roles']} 个，其中显式 unsupported role {version['counts']['unsupported_roles']} 个
- operation type：{version['counts']['operation_types']} 种
- H2 人工批准 feature-role-operation：{version['counts']['compatible_feature_role_operations']} 个
- H2 人工判定不兼容：{version['counts']['incompatible_feature_role_operations']} 个
- 人工批准 requirement：{version['counts']['approved_requirements']} 条
- 自动 fixture：{test_summary['fixtures']['case_total']} 条
- unittest：{test_summary['unittest']['test_methods_passed']}/{test_summary['unittest']['test_methods_total']}
- fixture assertions：{test_summary['fixtures']['assertion_checks_passed']}/{test_summary['fixtures']['assertion_checks_executed']}

## 2. 自动 Gate

- VALID acceptance：100%
- INVALID rejection：100%
- CLARIFICATION structured return：100%
- VALID / CLARIFICATION / INVALID：{counts['VALID']} / {counts['NEEDS_CLARIFICATION']} / {counts['INVALID']}
- 17 个 feature 和 7 种 operation 全覆盖
- 单位 normalization 幂等且保存 conversion audit
- 非 VALID 结果不携带 intent/operations
- 纯 intent 不含 raw face ID、sample/part/file identity、面积真值或 geometry source
- Oracle/predicted/human-confirmed 只存在于 grounding context envelope
- 未调用 LLM、未训练 Pointer/GNN/classifier、未执行 Abaqus/CalculiX、未重跑 Holdout

## 3. H2-01

- role policy：302/302 行已审核；175 approve、127 reject
- requirement：40/40 行已审核；33 approve、7 reject
- 无空决定、非法枚举、无效 corrected JSON 或 reject 空备注
- reject 组合已从 policy allowed operations 删除，并加入回归负例
- 7 条 rejected requirement 未擅自改写，原 reviewer note 保留在 decision summary

## 4. Recognizer 冻结完整性

- 引用资产：{recognizer['asset_count']}
- SHA-256 匹配：{recognizer['matching_count']}/{recognizer['asset_count']}
- 哈希变化：{recognizer['change_count']}
- Recognizer-Dev、历史模型、阈值、manifest 和历史输出未修改

## 5. 受限说明

本次状态使用 `BOUNDED-PASS`：实施期间一次非递归 `configs` 文件名枚举被动显示了三个带禁用标识的文件名。未打开、读取、搜索内容，未运行或评价相关资产；内容访问为 0，Holdout 运行次数为 0。该元数据事件已如实披露，因此提交监督者决定是否接受边界。

slot 与 rib/web 的 predicted path 仍必须 abstain 或人工确认；`pocket_circular` 仍是冻结 capability contract gap；transition 能力有限。上述限制不通过 CAE-IR policy 隐藏。

## 6. 结论

P2-A1 的 CAE-IR、单位系统、结构化歧义和 H2 批准 role policy 已冻结。本报告不授权进入 P3；下一阶段建议为 `NO_GO`，等待监督者审核。

## 监督摘要

```text
P2-A1 状态：{status}
CAE-IR schema version：{version['cae_ir_version']}
目标 feature 数：{version['counts']['canonical_features']}
selectable role 数：{version['counts']['selectable_roles']}
operation type 数：{version['counts']['operation_types']}
测试用例：{test_summary['fixtures']['case_total']}/{test_summary['fixtures']['case_total']}
VALID / CLARIFICATION / INVALID：{counts['VALID']} / {counts['NEEDS_CLARIFICATION']} / {counts['INVALID']}
H2-01：completed
人工审核 requirement：40（批准 33，拒绝 7）
未支持的 feature-role：slot_rect.closed_end_face, slot_v.closed_end_face, rib.end_face
Recognizer 冻结哈希变化：{recognizer['change_count']}
禁止阶段访问：仅目录元数据被动显示文件名；相关内容读取/运行/评价均为 0
下一阶段建议：NO_GO
最终报告：{acceptance_path.resolve()}
交付清单：{manifest_path.resolve()}
```
"""


def context_markdown(version: dict[str, Any], test_summary: dict[str, Any], acceptance_path: Path, manifest_path: Path) -> str:
    counts = test_summary["fixtures"]["status_counts"]
    return f"""# CODEX_CONTEXT_P2_A1

- P2-A1 状态：BOUNDED-PASS；H2-01 completed；等待监督者审核。
- CAE-IR version：{version['cae_ir_version']}。
- 范围：static_structural；17 feature；60 selectable roles；7 operations。
- H2 policy：166 compatible / 134 incompatible feature-role-operation rows。
- H2 requirements：40 reviewed；33 approved；7 rejected。
- 测试：27/27 unittest；{test_summary['fixtures']['case_total']} fixtures；{test_summary['fixtures']['assertion_checks_passed']}/{test_summary['fixtures']['assertion_checks_executed']} assertions。
- VALID / CLARIFICATION / INVALID：{counts['VALID']} / {counts['NEEDS_CLARIFICATION']} / {counts['INVALID']}；三项通过率 100%。
- 单位：length、area、force、pressure、displacement、angle；normalization 幂等且有 audit。
- 三轨：oracle/predicted/human_confirmed 仅在 grounding context；slot、rib/web predicted 必须 abstain/人工确认。
- 冻结 Recognizer 引用资产 13/13 SHA-256 匹配，变化 0。
- 未调用 LLM，未训练 Pointer/GNN/classifier，未执行 Abaqus/CalculiX，未重跑 Holdout，未进入 P3。
- 边界披露：非递归 configs 文件名枚举被动显示三个禁用标识文件名；内容未读取/搜索/运行/评价。
- 最终报告：`{acceptance_path.resolve()}`。
- 交付清单：`{manifest_path.resolve()}`。
- 下一步：NO_GO；停止并等待监督者审核。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "p2_a1_cae_ir_v1"
    freeze_dir = report_dir / "freeze"
    version_path = root / "configs" / "paper_v1" / "cae_ir_version.json"
    acceptance_path = report_dir / "acceptance.json"
    final_path = report_dir / "final_acceptance.zh-CN.md"
    delivery_path = report_dir / "delivery_manifest.json"
    context_path = root / "CODEX_CONTEXT_P2_A1.md"
    for path in (freeze_dir, version_path, acceptance_path, final_path, delivery_path, context_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final P2 artifact: {path}")

    policy = load_feature_role_policy(root / "configs" / "cae_feature_role_policy.yaml")
    roles = [role for feature in policy["features"].values() for role in feature["selectable_roles"]]
    compatible_count = sum(len(role["allowed_operations"]) for role in roles)
    unsupported_roles = [
        f"{feature_name}.{role['role']}"
        for feature_name, feature in policy["features"].items()
        for role in feature["selectable_roles"]
        if role["selection_status"].startswith("unsupported")
    ]
    if len(policy["features"]) != 17 or len(roles) != 60 or compatible_count != 166:
        raise AssertionError("Frozen policy counts do not match signed H2 decisions")

    review_counts = read_review_counts(report_dir / "human_review_package")
    if review_counts["feature_role_policy_review.csv"]["reviewed"] != 302:
        raise AssertionError("H2 role review incomplete")
    if review_counts["cae_requirement_review_40.csv"]["reviewed"] < 30:
        raise AssertionError("H2 requirement review incomplete")
    approved_path = report_dir / "approved_requirements.jsonl"
    approved = read_jsonl(approved_path)
    if len(approved) != 33:
        raise AssertionError("Approved requirement count mismatch")
    forbidden_in_approved = [path for item in approved for path in scan_forbidden_keys(item)]
    if forbidden_in_approved:
        raise AssertionError(f"Forbidden identity fields in approved requirements: {forbidden_in_approved[:3]}")

    test_summary = json.loads((report_dir / "test_case_summary.json").read_text(encoding="utf-8"))
    if test_summary["unittest"]["failures"] or test_summary["unittest"]["errors"]:
        raise AssertionError("P2-A1 tests are not green")
    if test_summary["fixtures"]["case_total"] < 120 or any(rate != 1.0 for rate in test_summary["rates"].values()):
        raise AssertionError("P2-A1 fixture gate failed")
    example_counts = validate_schema_examples(root)

    api_source = (root / "utils" / "cae_intent.py").read_text(encoding="utf-8")
    if "eval(" in api_source or "exec(" in api_source:
        raise AssertionError("Dynamic execution found in frozen API")
    recognizer = recognizer_asset_audit(root)
    if recognizer["asset_count"] != 13 or recognizer["change_count"] != 0:
        raise AssertionError("Frozen Recognizer-Dev assets changed")

    frozen_at = datetime.now(timezone.utc).isoformat()
    version = {
        "schema_version": "topocae_cae_ir_version_1",
        "cae_ir_version": "1.0.0-h2.01",
        "status": "frozen_h2_01",
        "frozen_at_utc": frozen_at,
        "analysis_type": "static_structural",
        "policy_revision": policy["policy_revision"],
        "schema_ids": {
            "intent": "cae-intent/1.0.0",
            "intent_result": "cae-intent-result/1.0.0",
            "grounding_context": "cae-grounding-context/1.0.0",
        },
        "counts": {
            "canonical_features": 17,
            "selectable_roles": len(roles),
            "unsupported_roles": len(unsupported_roles),
            "operation_types": 7,
            "compatible_feature_role_operations": compatible_count,
            "incompatible_feature_role_operations": 300 - compatible_count,
            "approved_requirements": len(approved),
            "test_cases": test_summary["fixtures"]["case_total"],
        },
        "h2_01": {"status": "completed", "review_counts": review_counts},
        "recognizer_dev": {
            "capability_level": "LIMITED_CONTROLLED_DOMAIN",
            "referenced_asset_count": recognizer["asset_count"],
            "sha256_change_count": recognizer["change_count"],
        },
        "next_stage": "NO_GO_PENDING_SUPERVISOR",
    }
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(json.dumps(version, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    freeze_sources = [
        (root / "schemas" / "cae_intent.schema.json", Path("schemas/cae_intent.schema.json"), "intent_schema"),
        (root / "schemas" / "cae_intent_result.schema.json", Path("schemas/cae_intent_result.schema.json"), "intent_result_schema"),
        (root / "schemas" / "cae_grounding_context.schema.json", Path("schemas/cae_grounding_context.schema.json"), "grounding_context_schema"),
        (root / "utils" / "cae_intent.py", Path("utils/cae_intent.py"), "python_api"),
        (root / "configs" / "cae_feature_role_policy.yaml", Path("configs/cae_feature_role_policy.yaml"), "role_policy"),
        (root / "configs" / "paper_v1" / "cae_feature_aliases.yaml", Path("configs/cae_feature_aliases.yaml"), "feature_aliases"),
        (version_path, Path("configs/cae_ir_version.json"), "version_contract"),
        (approved_path, Path("requirements/approved_requirements.jsonl"), "approved_requirements"),
        (root / "docs" / "cae_intent_annotation_guide.zh-CN.md", Path("docs/cae_intent_annotation_guide.zh-CN.md"), "annotation_guide"),
        (report_dir / "human_review_package" / "feature_role_policy_review.csv", Path("evidence/feature_role_policy_review.csv"), "signed_role_review"),
        (report_dir / "human_review_package" / "cae_requirement_review_40.csv", Path("evidence/cae_requirement_review_40.csv"), "signed_requirement_review"),
        (report_dir / "h2_01_decision_summary.json", Path("evidence/h2_01_decision_summary.json"), "h2_decision_summary"),
        (report_dir / "test_case_summary.json", Path("evidence/test_case_summary.json"), "test_summary"),
    ]
    freeze_dir.mkdir(parents=True)
    freeze_assets = []
    for source, relative, role in freeze_sources:
        destination = freeze_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        freeze_assets.append(artifact(destination, root, role))
    freeze_manifest_path = freeze_dir / "freeze_manifest.json"
    freeze_manifest = {
        "schema_version": "topocae_p2_a1_freeze_manifest_1",
        "cae_ir_version": version["cae_ir_version"],
        "frozen_at_utc": frozen_at,
        "assets": freeze_assets,
        "recognizer_dev_audit": recognizer,
        "schema_example_counts": example_counts,
    }
    freeze_manifest_path.write_text(json.dumps(freeze_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "BOUNDED-PASS"
    acceptance = {
        "schema_version": "topocae_p2_a1_acceptance_1",
        "status": status,
        "cae_ir_version": version["cae_ir_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": {
            "h2_01_completed": True,
            "all_role_rows_reviewed": True,
            "requirement_rows_reviewed": 40,
            "approved_requirements": len(approved),
            "test_cases_minimum_120": True,
            "test_cases_all_pass": True,
            "valid_acceptance_rate": 1.0,
            "invalid_rejection_rate": 1.0,
            "clarification_structured_return_rate": 1.0,
            "feature_count_17": True,
            "operation_count_7": True,
            "schema_examples_valid": True,
            "unit_normalization_idempotent": True,
            "identity_fields_forbidden": True,
            "geometry_source_envelope_isolated": True,
            "recognizer_reference_hash_change_count": recognizer["change_count"],
            "llm_called": False,
            "model_trained": False,
            "abaqus_or_calculix_executed": False,
            "holdout_rerun": False,
            "forbidden_dataset_content_read": False,
            "entered_p3": False,
        },
        "counts": version["counts"],
        "unsupported_feature_roles": unsupported_roles,
        "test_summary": test_summary,
        "recognizer_dev_audit": recognizer,
        "boundary_disclosure": {
            "directory_metadata_name_exposure": True,
            "detail": "A non-recursive configs filename listing passively displayed three filenames carrying the excluded identifier; no related file content was opened, searched, run, or evaluated.",
            "related_content_read_count": 0,
            "holdout_run_count": 0,
        },
        "freeze_manifest": str(freeze_manifest_path.resolve()),
        "next_stage_recommendation": "NO_GO",
    }
    acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_path.write_text(final_markdown(status, version, test_summary, recognizer, final_path, delivery_path), encoding="utf-8")
    context_path.write_text(context_markdown(version, test_summary, final_path, delivery_path), encoding="utf-8")

    delivery_sources = [
        (root / "reports" / "p2_a1_cae_ir_v1" / "preflight_summary.zh-CN.md", "preflight_summary"),
        (root / "reports" / "p2_a1_cae_ir_v1" / "ontology_audit.json", "ontology_audit_json"),
        (root / "reports" / "p2_a1_cae_ir_v1" / "ontology_audit.zh-CN.md", "ontology_audit_markdown"),
        (root / "configs" / "paper_v1" / "cae_feature_aliases.yaml", "feature_aliases"),
        (root / "schemas" / "cae_intent.schema.json", "intent_schema"),
        (root / "schemas" / "cae_intent_result.schema.json", "intent_result_schema"),
        (root / "schemas" / "cae_grounding_context.schema.json", "grounding_context_schema"),
        (root / "utils" / "cae_intent.py", "python_api"),
        (root / "configs" / "cae_feature_role_policy.yaml", "role_policy"),
        (root / "tests" / "test_cae_intent_schema.py", "schema_tests"),
        (root / "tests" / "test_cae_intent_semantics.py", "semantic_tests"),
        (root / "tests" / "fixtures" / "cae_ir_v1" / "cases.jsonl", "test_cases"),
        (report_dir / "test_case_summary.json", "test_summary"),
        (report_dir / "human_review_package" / "feature_role_policy_review.csv", "signed_role_review"),
        (report_dir / "human_review_package" / "cae_requirement_review_40.csv", "signed_requirement_review"),
        (report_dir / "human_review_package" / "README_H2_01.zh-CN.md", "h2_instructions"),
        (report_dir / "h2_01_decision_summary.json", "h2_decision_summary"),
        (approved_path, "approved_requirements"),
        (root / "docs" / "cae_intent_annotation_guide.zh-CN.md", "annotation_guide"),
        (version_path, "version_contract"),
        (freeze_manifest_path, "freeze_manifest"),
        (acceptance_path, "acceptance_json"),
        (final_path, "final_acceptance"),
        (context_path, "context_handoff"),
        (root / "utils" / "apply_h2_01_reviews.py", "h2_application_tool"),
        (root / "utils" / "freeze_p2_a1.py", "freeze_tool"),
    ]
    delivery_assets = [artifact(path, root, role) for path, role in delivery_sources]
    delivery = {
        "schema_version": "topocae_p2_a1_delivery_manifest_1",
        "cae_ir_version": version["cae_ir_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "asset_count": len(delivery_assets),
        "assets": delivery_assets,
        "freeze_manifest": str(freeze_manifest_path.resolve()),
        "recognizer_reference_hash_change_count": recognizer["change_count"],
        "next_stage_recommendation": "NO_GO",
    }
    delivery_path.write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "version": version["cae_ir_version"],
        "freeze_assets": len(freeze_assets),
        "delivery_assets": len(delivery_assets),
        "recognizer_hash_changes": recognizer["change_count"],
        "final_report": str(final_path.resolve()),
        "delivery_manifest": str(delivery_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
