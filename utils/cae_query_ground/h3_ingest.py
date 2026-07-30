from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import REPORT_ROOT
from .io import sha256_file, write_json, write_text


REVIEW_PATH = REPORT_ROOT / "human_review_package" / "cae_query_review_120.csv"
EXPRESSION_PATH = REPORT_ROOT / "human_review_package" / "human_expression_additions_20.csv"
DECISION_PATH = REPORT_ROOT / "human_review_package" / "human_expression_adoption_decisions.csv"
SUMMARY_JSON = REPORT_ROOT / "human_review_package" / "h3_01_ingestion_summary.json"
SUMMARY_MD = REPORT_ROOT / "human_review_package" / "h3_01_decision_summary.zh-CN.md"


EXPRESSION_DECISIONS = {
    "EX-01": ("adopt_with_normalization", "补齐冻结 P2 要求的 E 和泊松比；Y 位移规范为 U2。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格；完全固定所有阶梯孔的完整特征区域，并在圆柱凸台顶面施加 U2=0.1 mm 位移。"),
    "EX-02": ("reject", "全局 ?Y 的符号已损坏且无法无歧义恢复；不猜测方向。", ""),
    "EX-03": ("adopt_with_normalization", "将元数据目标从完整通孔特征改为 query 实际表达的通孔壁。", "材料采用 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格；完全固定所有通孔壁，并对所有矩形口袋侧壁施加 2.3 MPa 表面法向压力。"),
    "EX-04": ("adopt_with_normalization", "补齐材料参数，并把圆柱明确为圆柱凸台。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格；完全固定所有沉孔，并在直径为 50 mm 的圆柱凸台顶面施加 U1=0.1 mm 位移。"),
    "EX-05": ("adopt_with_normalization", "将 10MP 修正为 10 MPa，并使用冻结 ontology 名称。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格，将所有加强筋局部网格设为 2 mm；完全固定所有阶梯孔和矩形口袋底面，并对圆柱凸台顶面施加 10 MPa 表面法向压力。"),
    "EX-06": ("adopt_with_normalization", "冻结 P2 不表达密度，去除密度但保留 E 和泊松比。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格，并对所有圆形口袋侧壁施加 1 MPa 表面法向压力。"),
    "EX-07": ("adopt_with_normalization", "补齐冻结 P2 材料参数并明确局部网格尺寸。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格，并将所有圆角的局部网格尺寸设为 2 mm。"),
    "EX-08": ("adopt_with_normalization", "去除重复词和不支持的密度，明确矩形槽完整特征区域。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格；完全固定所有锥沉孔，并对矩形槽完整特征区域施加 0.75 MPa、沿全局 +X 方向的表面牵引。"),
    "EX-09": ("adopt_with_normalization", "去除损坏且不受 P2 支持的密度字段。", "为整个零件赋予 CQG-LinearElastic（E=210 GPa，泊松比 0.3），设置 4 mm 全局网格；完全固定所有锥沉孔，并对矩形槽底面施加 0.75 MPa、沿全局 +X 方向的表面牵引。"),
    "EX-10": ("reject", "pressure 不能使用全局 ?Z 方向，且符号已损坏；改成法向会改变原语义。", ""),
    "EX-11": ("adopt_with_normalization", "补齐材料参数，并以 natural_expression 中的 1.8 MPa 为准修正元数据。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Fix all through-hole walls, and apply a 1.8 MPa surface traction in global +X to the cylindrical-boss side faces."),
    "EX-12": ("reject", "pressure 被写成全局 -Z 方向；冻结 P2 仅允许表面法向或反法向。", ""),
    "EX-13": ("adopt_with_normalization", "补齐材料参数并明确长圆孔完整特征区域。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Fix the complete feature regions of all obround holes, and apply a 0.5 MPa surface-normal pressure to all circular-pocket side walls."),
    "EX-14": ("adopt_with_normalization", "补齐材料参数并规范 feature/role 名称。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Fix the complete feature regions of all stepped holes, and apply a 1.6 MPa surface traction in global -Y to the cylindrical-boss side faces."),
    "EX-15": ("adopt_with_normalization", "补齐材料参数，并把遗漏的锥沉孔固定写入操作元数据。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Set the fillet local mesh size to 2 mm, fix all countersink holes, and apply a 1.3 MPa surface traction in global +X to the circular-pocket side walls."),
    "EX-16": ("reject", "pressure 使用全局 -Z 方向且密度指数损坏；不能无语义变化修复。", ""),
    "EX-17": ("adopt_with_normalization", "去除不支持且损坏的密度；将 densified by 2 mm 规范为局部网格尺寸。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Set all rib local mesh sizes to 2 mm, fix all countersink holes, and apply a 2.8 MPa surface traction in global -X to the rectangular-slot bottom faces."),
    "EX-18": ("adopt_with_normalization", "去除不支持且损坏的密度，并明确直径筛选作用于通孔。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Fix all through holes with diameter 50 mm, and apply a 1.1 MPa surface traction in global +Y to the rectangular-slot bottom faces."),
    "EX-19": ("adopt_with_normalization", "natural_expression 的 -Y 清楚可用；修正元数据损坏并补齐材料参数。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Set the local mesh size of through holes smaller than 5 mm to 2 mm, fix all obround holes, and apply a 0.95 MPa surface traction in global -Y to the rectangular-slot side walls."),
    "EX-20": ("adopt_with_normalization", "补齐材料参数并规范 rectangular boss 名称。", "Assign CQG-LinearElastic (E=210 GPa, Poisson ratio 0.3) to the whole part and use a 4 mm global mesh. Fix all blind holes, and apply a 3.2 MPa surface-normal pressure to the rectangular-boss top faces."),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ingest(*, force: bool = False) -> dict[str, Any]:
    reviews = _read(REVIEW_PATH)
    expressions = _read(EXPRESSION_PATH)
    errors: list[str] = []
    allowed = {"approve", "edit", "reject"}
    for row in reviews:
        decision = row["review_decision"].strip()
        category = row["error_category"].strip()
        if decision not in allowed:
            errors.append(f"{row['review_id']}: illegal/blank decision")
        elif decision == "approve" and category != "none":
            errors.append(f"{row['review_id']}: approve requires error_category=none")
        elif decision == "edit" and not row["corrected_query"].strip():
            errors.append(f"{row['review_id']}: edit requires corrected_query")
        elif decision == "reject" and (category in {"", "none"} or not row["reviewer_note"].strip()):
            errors.append(f"{row['review_id']}: reject requires category and note")
    if len(reviews) != 120:
        errors.append(f"expected 120 review rows, got {len(reviews)}")

    populated = [row for row in expressions if row["natural_expression"].strip()]
    if len(populated) < 10:
        errors.append(f"expression additions below 10: {len(populated)}")
    expression_ids = {row["row_id"] for row in populated}
    if expression_ids != set(EXPRESSION_DECISIONS):
        errors.append("expression decision IDs do not exactly cover populated expressions")
    if errors:
        raise ValueError("H3-01 ingestion failed: " + "; ".join(errors))

    decision_rows = []
    for row in expressions:
        decision, reason, normalized = EXPRESSION_DECISIONS[row["row_id"]]
        decision_rows.append({
            "row_id": row["row_id"],
            "language": row["language"],
            "original_expression": row["natural_expression"],
            "decision": decision,
            "decision_reason": reason,
            "normalized_expression": normalized,
        })
    if DECISION_PATH.exists() and not force:
        raise FileExistsError(DECISION_PATH)
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)

    decision_counts = Counter(row["review_decision"].strip() for row in reviews)
    expression_counts = Counter(row["decision"] for row in decision_rows)
    effective_correct = decision_counts["approve"] + decision_counts["edit"]
    summary = {
        "schema_version": "cae-query-ground-h3-01-ingestion/1",
        "gate": "H3-01",
        "status": "PASS",
        "review_csv": str(REVIEW_PATH.resolve()),
        "review_csv_sha256": sha256_file(REVIEW_PATH),
        "review_rows": len(reviews),
        "review_decisions": dict(sorted(decision_counts.items())),
        "effective_correct_rows": effective_correct,
        "effective_correct_rate": effective_correct / len(reviews),
        "language_counts": dict(sorted(Counter(row["language"] for row in reviews).items())),
        "level_counts": dict(sorted(Counter(row["difficulty_level"] for row in reviews).items())),
        "status_counts": dict(sorted(Counter(row["intent_status"] for row in reviews).items())),
        "expression_csv": str(EXPRESSION_PATH.resolve()),
        "expression_csv_sha256": sha256_file(EXPRESSION_PATH),
        "expression_rows_populated": len(populated),
        "expression_languages": dict(sorted(Counter(row["language"] for row in populated).items())),
        "expression_decisions": dict(sorted(expression_counts.items())),
        "expression_decision_csv": str(DECISION_PATH.resolve()),
        "expression_decision_csv_sha256": sha256_file(DECISION_PATH),
        "user_review_is_not_automatically_generated": True,
    }
    write_json(SUMMARY_JSON, summary, force=force)
    report = f"""# H3-01 真实人工审核与表达采用决定

状态：`PASS`

- 人工审核：{len(reviews)}/120
- approve / edit / reject：{dict(sorted(decision_counts.items()))}
- 审核后有效正确率：{summary['effective_correct_rate']:.1%}
- 英文 / 中文：{summary['language_counts']}
- 表达补充：{len(populated)} 条
- 表达采用决定：{dict(sorted(expression_counts.items()))}

原始 CSV 保持不变并按 SHA-256 引用。表达中的非法 pressure 方向、损坏方向符号、损坏密度指数或冻结 P2 无法无歧义表达的内容未被直接采用。其余表达只在不扩展 P2 schema 的前提下归一化采用；所有采用/拒绝理由见 `{DECISION_PATH}`。
"""
    write_text(SUMMARY_MD, report, force=force)
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(ingest(force=True), ensure_ascii=False, sort_keys=True))

