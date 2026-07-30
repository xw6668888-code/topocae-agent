from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_COLUMNS = [
    "review_id",
    "canonical_record_id",
    "rendering_id",
    "query",
    "language",
    "difficulty_level",
    "intent_status",
    "ir_summary",
    "canonical_feature",
    "source_macro_provenance",
    "face_role",
    "quantifier",
    "operation_type",
    "target_instance_count",
    "target_face_count",
    "uniqueness_or_clarification_rationale",
    "source_geometry_readonly_path",
    "source_graph_readonly_path",
    "review_decision",
    "error_category",
    "corrected_query",
    "reviewer_note",
]


def _stable(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _operations(plan: dict[str, Any]) -> list[str]:
    return sorted({item["operation_type"] for item in plan["verbalization"]["operations"]})


def _candidate(plan: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": plan,
        "record": record,
        "language": record["query"]["language"],
        "level": plan["level"],
        "status": plan["status"],
        "feature": plan["verbalization"]["feature_type"],
        "operations": _operations(plan),
    }


def select_review_records(
    plans: list[dict[str, Any]],
    rendering_records: list[dict[str, Any]],
    *,
    count: int = 120,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if count != 120:
        raise ValueError("H3-01 review size is fixed at 120")
    plan_by_id = {plan["canonical_record_id"]: plan for plan in plans}
    candidates = [
        _candidate(plan_by_id[record["source_ref"]["canonical_record_id"]], record)
        for record in rendering_records
    ]
    cells = [(level, language) for level in ("L1", "L2", "L3", "L4", "L5") for language in ("en-US", "zh-CN")]
    cell_quota = count // len(cells)
    selected: list[dict[str, Any]] = []
    used_renderings: set[str] = set()
    used_canonical: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for level, language in cells:
        pool = [item for item in candidates if item["level"] == level and item["language"] == language]
        for _ in range(cell_quota):
            available = [item for item in pool if item["record"]["query"]["rendering_id"] not in used_renderings]
            if not available:
                raise ValueError(f"Insufficient H3 candidates for {level}/{language}")

            def score(item: dict[str, Any]) -> tuple[Any, ...]:
                feature_gain = 20 if feature_counts[item["feature"]] == 0 else 5 / (1 + feature_counts[item["feature"]])
                operation_gain = sum(15 if operation_counts[op] == 0 else 3 / (1 + operation_counts[op]) for op in item["operations"])
                status_gain = 12 if status_counts[item["status"]] == 0 else 2 / (1 + status_counts[item["status"]])
                duplicate_penalty = used_canonical[item["plan"]["canonical_record_id"]] * 25
                stable = _stable(item["record"]["query"]["rendering_id"])
                return (feature_gain + operation_gain + status_gain - duplicate_penalty, stable)

            choice = max(available, key=score)
            selected.append(choice)
            rendering_id = choice["record"]["query"]["rendering_id"]
            used_renderings.add(rendering_id)
            used_canonical[choice["plan"]["canonical_record_id"]] += 1
            feature_counts[choice["feature"]] += 1
            status_counts[choice["status"]] += 1
            operation_counts.update(choice["operations"])

    if len(selected) != count:
        raise AssertionError("Review selection size mismatch")
    required_features = {item["feature"] for item in candidates}
    required_operations = {operation for item in candidates for operation in item["operations"]}
    if set(feature_counts) != required_features:
        raise ValueError(f"Review package misses features: {sorted(required_features - set(feature_counts))}")
    if set(operation_counts) != required_operations:
        raise ValueError(f"Review package misses operations: {sorted(required_operations - set(operation_counts))}")
    if set(status_counts) != {"VALID", "NEEDS_CLARIFICATION", "INVALID"}:
        raise ValueError("Review package misses intent status")
    return [(item["plan"], item["record"]) for item in selected]


def _ir_summary(plan: dict[str, Any]) -> str:
    result = plan["intent_result"]
    if result["status"] != "VALID":
        return f"{result['status']} | {result['error_code']} @ {result['field_path']}"
    parts = []
    for operation in result["intent"]["operations"]:
        target = operation["target"]
        if target.get("scope") == "whole_part":
            target_text = "whole_part"
        else:
            target_text = f"{target['feature_type']}.{target['face_role']}[{target['quantifier']}]"
        parts.append(f"{operation['operation_type']} -> {target_text}")
    return "VALID | " + "; ".join(parts)


def _source_macros(plan: dict[str, Any]) -> list[str]:
    macros = {
        macro
        for target in plan["ground_truth"]["targets"]
        if target.get("target_contract") == "feature_face_set"
        for macro in target["source_macro_features"]
    }
    if macros:
        return sorted(macros)
    focus = plan["focus_feature"]
    source_macros = set(plan["source_record"].get("macro_instance_counts", {}))
    if focus == "rib" and "web" in source_macros:
        return ["web"]
    return [focus]


def _target_counts(plan: dict[str, Any]) -> tuple[int, int]:
    instance_ids = {
        instance_id
        for target in plan["ground_truth"]["targets"]
        if target.get("target_contract") == "feature_face_set"
        for instance_id in target["target_instance_ids"]
    }
    face_indices = {
        face_index
        for target in plan["ground_truth"]["targets"]
        if target.get("target_contract") == "feature_face_set"
        for face_index in target["target_face_indices"]
    }
    return len(instance_ids), len(face_indices)


def _rationale(plan: dict[str, Any]) -> str:
    if plan["ground_truth"]["contract_status"] == "non_executable":
        return plan["ground_truth"]["non_executable_reason"]
    rationales = [
        target["uniqueness_contract"]["rationale"]
        for target in plan["ground_truth"]["targets"]
        if target.get("target_contract") == "feature_face_set"
    ]
    if not rationales:
        return "Whole-part operation; face grounding is not required."
    return " | ".join(rationales)


def review_rows(selection: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for index, (plan, record) in enumerate(selection, start=1):
        instance_count, face_count = _target_counts(plan)
        verbal = plan["verbalization"]
        rows.append(
            {
                "review_id": f"H3-{index:03d}",
                "canonical_record_id": plan["canonical_record_id"],
                "rendering_id": record["query"]["rendering_id"],
                "query": record["query"]["text"],
                "language": record["query"]["language"],
                "difficulty_level": plan["level"],
                "intent_status": plan["status"],
                "ir_summary": _ir_summary(plan),
                "canonical_feature": verbal["feature_type"],
                "source_macro_provenance": "+".join(_source_macros(plan)),
                "face_role": verbal["face_role"],
                "quantifier": verbal.get("quantifier") or "",
                "operation_type": "+".join(_operations(plan)),
                "target_instance_count": instance_count,
                "target_face_count": face_count,
                "uniqueness_or_clarification_rationale": _rationale(plan),
                "source_geometry_readonly_path": record["source_ref"]["geometry_path"],
                "source_graph_readonly_path": record["source_ref"]["graph_path"],
                "review_decision": "",
                "error_category": "",
                "corrected_query": "",
                "reviewer_note": "",
            }
        )
    return rows


def write_review_csv(path: Path, rows: list[dict[str, Any]], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite review file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_expression_csv(path: Path, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite expression file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_id",
        "language",
        "natural_expression",
        "intended_operation",
        "intended_feature_or_scope",
        "reviewer_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(1, 21):
            writer.writerow({"row_id": f"EX-{index:02d}", **{column: "" for column in columns[1:]}})


def review_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    operation_counts: Counter[str] = Counter()
    for row in rows:
        operation_counts.update(item for item in row["operation_type"].split("+") if item)
    return {
        "row_count": len(rows),
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "levels": dict(sorted(Counter(row["difficulty_level"] for row in rows).items())),
        "features": dict(sorted(Counter(row["canonical_feature"] for row in rows).items())),
        "operations": dict(sorted(operation_counts.items())),
        "statuses": dict(sorted(Counter(row["intent_status"] for row in rows).items())),
        "review_decisions_blank": sum(not row["review_decision"] for row in rows),
    }


def review_readme(distribution: dict[str, Any], review_path: Path, expression_path: Path) -> str:
    return f"""# H3-01 CAE-QueryGround Pilot 人工审查

状态：`WAITING_FOR_REAL_USER_REVIEW`  
审查行数：{distribution['row_count']}  
语言：{distribution['languages']}  
Level：{distribution['levels']}  
Status：{distribution['statuses']}

## 为什么需要你

自动检查可以证明 schema、role policy、target face 重建和 split 隔离，但不能替代工程师对自然表达、工况合理性和面角色可理解性的判断。

## 你要检查的对象

- 审查表：`{review_path}`
- 常用表达补充表：`{expression_path}`

## 具体操作

1. 打开 `cae_query_review_120.csv`，逐行阅读 query、IR 摘要、feature/role/quantifier、operation 和唯一性/澄清说明。
2. 若 query 自然、工程语义合理、IR 摘要等价且目标 feature/face role 明确，填写 `review_decision=approve` 和 `error_category=none`。
3. 若仅需修改措辞且语义不变，填写 `review_decision=edit`、适当的 `error_category`，并在 `corrected_query` 填入完整改写。
4. 若目标不唯一、工程上不合理、role/operation 映射错误或数值/单位错误，填写 `review_decision=reject`，选择非 `none` error category，并在 `reviewer_note` 写一句原因。
5. 不需要核对或填写 face ID。只有怀疑几何语义时，才按 `source_geometry_readonly_path` 打开对应 STEP/BREP 辅助判断；不要修改源文件。
6. 在 `human_expression_additions_20.csv` 中补充至少 10 条常用自然表达。无需写 face ID；请尽量注明 intended operation 与 feature/scope。
7. 保存两个 CSV 后，在同一 Goal 通知继续。

允许的 `review_decision`：`approve | edit | reject`。  
允许的 `error_category`：`none | unnatural_language | wrong_cae_semantics | non_unique_target | wrong_target_role | wrong_value_or_unit | other`。

## 通过标准

- 120/120 行有合法人工决定；
- `edit` 必须有 corrected query；`reject` 必须有非 none error category 和原因；
- 审核后有效正确率至少 95%；
- expression additions 至少 10 条；
- 系统性问题修复后具备 regression test。

未经真实回填，不得自动批准 H3-01、生成正式 20,000 条、冻结 P3 或进入 P4。
"""
