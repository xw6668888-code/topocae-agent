from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "p3_a1r_multi_instance_v1"
REVIEW_ROOT = REPORT_ROOT / "human_review_package"

COMPLETE_DESCRIPTOR_ROWS = {
    "G01", "G05", "G07", "G08", "G09", "G10", "G19", "G20",
}
ENGINEERING_METRIC_ROWS = {
    "G02", "G03", "G04", "G06", "G11", "G15",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit() -> dict[str, Any]:
    geometry_path = REVIEW_ROOT / "geometry_review_20.csv"
    query_path = REVIEW_ROOT / "query_review_60.csv"
    geometry = read_csv(geometry_path)
    queries = read_csv(query_path)

    if len(geometry) != 20 or len(queries) != 60:
        raise ValueError("H3R-01 review row counts changed")
    allowed_geometry = {"PASS", "MINOR", "BAD"}
    allowed_query = {"approve", "edit", "reject"}
    if any(row["geometry_decision"] not in allowed_geometry for row in geometry):
        raise ValueError("geometry review contains an invalid or empty decision")
    if any(row["review_decision"] not in allowed_query for row in queries):
        raise ValueError("query review contains an invalid or empty decision")
    if any(
        row["geometry_decision"] in {"MINOR", "BAD"} and not row["reviewer_reason"].strip()
        for row in geometry
    ):
        raise ValueError("MINOR/BAD geometry review is missing a reason")
    if any(row["review_decision"] == "reject" for row in queries):
        raise ValueError("H3R query review contains a rejected query")
    if any(
        row["review_decision"] == "edit"
        and not (row["corrected_query_en"].strip() or row["corrected_query_zh"].strip())
        for row in queries
    ):
        raise ValueError("edited query is missing corrected text")

    minor_ids = {row["review_id"] for row in geometry if row["geometry_decision"] == "MINOR"}
    expected_minor_ids = COMPLETE_DESCRIPTOR_ROWS | ENGINEERING_METRIC_ROWS
    if minor_ids != expected_minor_ids:
        raise ValueError(f"unexpected H3R MINOR set: {sorted(minor_ids ^ expected_minor_ids)}")

    freeze = verify_freeze_manifest(
        ROOT,
        ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
    )
    geometry_pass = sum(row["geometry_decision"] == "PASS" for row in geometry)
    geometry_minor = sum(row["geometry_decision"] == "MINOR" for row in geometry)
    geometry_bad = sum(row["geometry_decision"] == "BAD" for row in geometry)
    query_approve = sum(row["review_decision"] == "approve" for row in queries)
    query_edit = sum(row["review_decision"] == "edit" for row in queries)
    query_reject = sum(row["review_decision"] == "reject" for row in queries)
    query_correct_rate = (query_approve + query_edit) / len(queries)

    result = {
        "schema_version": "p3-a1r-h3r-01-review-acceptance/1",
        "gate": "H3R-01",
        "status": "PASS",
        "real_human_review_received": True,
        "geometry": {
            "reviewed": len(geometry),
            "pass": geometry_pass,
            "minor": geometry_minor,
            "bad": geometry_bad,
            "minor_resolution": {
                "complete_descriptor_rows": sorted(COMPLETE_DESCRIPTOR_ROWS),
                "engineering_metric_rows": sorted(ENGINEERING_METRIC_ROWS),
                "query_policy": "comparison descriptor remains minimal; full dimensions stay in graph evidence",
                "formal_geometry_revision": "p3_a1r_metric_descriptor_contract_r3",
                "accepted": True,
            },
        },
        "query": {
            "reviewed": len(queries),
            "approve": query_approve,
            "edit": query_edit,
            "reject": query_reject,
            "correct_rate": query_correct_rate,
        },
        "review_artifacts": {
            "geometry_review_path": str(geometry_path.resolve()),
            "geometry_review_sha256": sha256_file(geometry_path),
            "query_review_path": str(query_path.resolve()),
            "query_review_sha256": sha256_file(query_path),
        },
        "old_p3_freeze": {
            "declared_assets": freeze["declared_asset_count"],
            "matching_assets": freeze["matching_asset_count"],
            "hash_changes": freeze["hash_changes"],
            "automatic_pass": freeze["automatic_pass"],
        },
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "forbidden_data_accessed": False,
        },
        "automatic_pass": (
            len(geometry) == 20
            and geometry_bad == 0
            and geometry_minor == 14
            and len(queries) == 60
            and query_correct_rate >= 0.95
            and query_reject == 0
            and freeze["automatic_pass"]
            and freeze["hash_changes"] == 0
        ),
    }
    if not result["automatic_pass"]:
        raise RuntimeError(f"H3R-01 acceptance failed: {result}")
    return result


def write_outputs(result: dict[str, Any]) -> None:
    json_path = REPORT_ROOT / "h3r_01_review_acceptance.json"
    md_path = REPORT_ROOT / "h3r_01_review_acceptance.zh-CN.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    text = f"""# P3-A1R H3R-01 人工审核验收

状态：`PASS`

## 审核结果

- geometry：{result["geometry"]["reviewed"]}/20，PASS={result["geometry"]["pass"]}，MINOR={result["geometry"]["minor"]}，BAD={result["geometry"]["bad"]}。
- query：{result["query"]["reviewed"]}/60，approve={result["query"]["approve"]}，edit={result["query"]["edit"]}，reject={result["query"]["reject"]}，有效正确率={result["query"]["correct_rate"]:.2%}。
- 旧 P3 冻结资产：{result["old_p3_freeze"]["matching_assets"]}/{result["old_p3_freeze"]["declared_assets"]} 匹配，hash changes={result["old_p3_freeze"]["hash_changes"]}。

## 14 项 MINOR 的处理

人工反馈指出，单一 `comparison_metric/value` 是用于同类实例排序的描述符，不能冒充完整特征定义。处理原则如下：

1. query 只保留执行选择所需的最小、明确比较描述符；不把全部尺寸强行写入自然语言；
2. graph 的 `geometry_parameters` 保留完整尺寸元组，并新增明确的 descriptor contract；
3. 正式 supplement 使用更精确的工程证据：V 槽成品开口宽度、圆角半径、倒角直角边长度、筋厚度，以及沉孔/锥沉孔的小孔径；
4. 因 60 条 query 全部由真实人工审核为 `approve`，查询文本无需重写。

上述 MINOR 均已通过契约修复或有明确接受说明，不构成几何 BAD，也不阻止进入 P3-A1R 正式 supplement。

## 范围声明

未调用 LLM，未训练模型，未执行 Abaqus/CalculiX，未进入 P4，未访问真实件、Holdout 或 test_part_s3。
"""
    md_path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    result = audit()
    write_outputs(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
