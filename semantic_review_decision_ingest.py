"""v5.20 User Semantic Review Ingestion.

Build human-review templates from v5.19 cross-macro semantic audit rows,
then ingest user decisions to decide the next code-changing step.

Diagnostic/workflow only — does not modify model, proposer, or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_USER_DECISIONS = {
    "accept_family_match",
    "keep_strict",
    "truth_label_should_change",
    "model_prediction_wrong",
    "needs_face_level_review",
}

OUTCOME_BY_DECISION = {
    "accept_family_match": "policy_recovered",
    "truth_label_should_change": "truth_correction_needed",
    "keep_strict": "strict_failure",
    "model_prediction_wrong": "model_or_proposer_error",
    "needs_face_level_review": "unresolved_manual_review",
    "": "pending_user_review",
}

REVIEW_COLUMNS = [
    "row_id",
    "sample",
    "truth_macro",
    "candidate_macro",
    "pair",
    "review_category",
    "truth_area",
    "candidate_area",
    "relative_error",
    "truth_face_count",
    "candidate_face_count",
    "candidate_source",
    "candidate_action",
    "candidate_confidence",
    "candidate_faces",
    "review_prompt",
    "user_decision",
    "user_note",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def validate_user_decision(value: str | None) -> str:
    token = str(value or "").strip()
    if token == "":
        return ""
    if token not in ALLOWED_USER_DECISIONS:
        raise ValueError(f"Invalid user_decision {token!r}; allowed values: {sorted(ALLOWED_USER_DECISIONS)}")
    return token


def classify_decision_outcome(user_decision: str | None) -> str:
    decision = validate_user_decision(user_decision)
    return OUTCOME_BY_DECISION[decision]


def review_prompt_for_category(category: str) -> str:
    prompts = {
        "transition_vs_nontransition_conflict": "判断该 fillet/chamfer 目标是否真正是独立的过渡面特征，还是 truth/候选的宏类别标注有误。若 paper 中 fillet/chamfer 为独立功能类别，建议保持严格。",
        "hole_subfeature_vs_composite_hole": "判断该孔特征是否应作为 hole-family 匹配（用于 CAE 区域选取评估），还是必须严格区分宏类别。若 paper 会显式评估 hole-family region selection，可接受为语义族匹配。",
        "rectangular_recess_slot_vs_pocket": "判断 slot_rect 和 pocket_rect 应保持严格独立类别，还是归为同一个 CAE recess-selection 语义族。如下游 CAE 操作将两者视为同类 recess 区域，可接受为语义族匹配。",
        "thinwall_rib_vs_boss_or_web": "判断该行是真正的 rib/web 目标，还是应标注为 boss/context/background。请基于几何语义决定，不建议自动接受为匹配。",
        "circular_recess_or_hole_conflict": "判断圆形 pocket/recess 与 hole 应保持严格区分，还是归为圆形 recess 语义族。",
        "unknown_cross_macro_conflict": "此行的宏类别对没有预定义的自动语义族。请手动审查。",
    }
    return prompts.get(category, "请手动审查此行。")


def build_review_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, source in enumerate(source_rows, start=1):
        row = {column: str(source.get(column, "")) for column in REVIEW_COLUMNS}
        row["row_id"] = f"R{index:03d}"
        row["review_prompt"] = review_prompt_for_category(row["review_category"])
        row["user_decision"] = str(source.get("user_decision", "") or "").strip()
        row["user_note"] = str(source.get("user_note", "") or "")
        rows.append(row)
    return rows


def write_review_template(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_decisions(rows: list[dict[str, str]]) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    outcome_category_counts: Counter[str] = Counter()
    for row in rows:
        decision = validate_user_decision(row.get("user_decision", ""))
        outcome = classify_decision_outcome(decision)
        category = str(row.get("review_category", ""))
        pair = str(row.get("pair", ""))
        decision_counts[decision or "blank"] += 1
        outcome_counts[outcome] += 1
        category_counts[category] += 1
        pair_counts[pair] += 1
        outcome_category_counts[f"{outcome}|{category}"] += 1
    return {
        "row_count": len(rows),
        "by_decision": dict(decision_counts.most_common()),
        "by_outcome": dict(outcome_counts.most_common()),
        "by_category": dict(category_counts.most_common()),
        "by_pair": dict(pair_counts.most_common()),
        "by_outcome_category": dict(outcome_category_counts.most_common()),
    }


def decide_next_action(summary: dict[str, Any]) -> tuple[str, str]:
    by_outcome = summary.get("by_outcome", {})
    pending = int(by_outcome.get("pending_user_review", 0))
    unresolved = int(by_outcome.get("unresolved_manual_review", 0))
    truth = int(by_outcome.get("truth_correction_needed", 0))
    policy = int(by_outcome.get("policy_recovered", 0))
    model = int(by_outcome.get("model_or_proposer_error", 0))
    strict = int(by_outcome.get("strict_failure", 0))
    if pending > 0:
        return "wait_for_user_semantic_review", "Some rows still have blank user_decision values."
    if unresolved > 0:
        return "prepare_face_level_review_package", "Some rows require face-level visual inspection."
    if truth >= max(policy, model, strict) and truth > 0:
        return "correct_area_truth_table_then_rerun_metrics", "Most actionable decisions indicate truth-label corrections."
    if policy >= max(truth, model, strict) and policy > 0:
        return "implement_evaluation_only_semantic_policy", "Most actionable decisions support semantic family matching for evaluation only."
    if model >= max(truth, policy, strict) and model > 0:
        return "targeted_proposal_or_classifier_repair", "Most actionable decisions indicate the model/proposer prediction is wrong."
    return "return_to_proposal_generation_repair_for_remaining_no_same_macro_misses", "Reviewed cross-macro rows mostly remain strict failures."


def write_template_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    category_counts = Counter(row["review_category"] for row in rows)
    lines = [
        "# v5.20 语义审查模板",
        "",
        f"- 待审查行数: `{len(rows)}`",
        "",
        "## 类别统计",
        "",
        "| 类别 | 数量 |",
        "|----------|-------|",
    ]
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")
    lines.extend([
        "",
        "## 用户操作指南",
        "",
        "打开 `semantic_review_template.csv`，填写 `user_decision` 列。",
        "",
        "允许的值：",
        "",
        "- `accept_family_match` — 接受为 CAE 语义族匹配，评估时可算作命中",
        "- `keep_strict` — 保持严格类别区分，应视为评估失败",
        "- `truth_label_should_change` — Truth 标签有误，应修正",
        "- `model_prediction_wrong` — 模型/候选预测错误",
        "- `needs_face_level_review` — 面积不足以判断，需要面级别视觉审查",
        "",
        "审查优先级：",
        "1. 先填所有 `transition_vs_nontransition_conflict` 行（fillet/chamfer 过渡面冲突）",
        "2. 再填 `thinwall_rib_vs_boss_or_web` 行（rib/web 语义边界）",
        "3. 最后填 hole/recess 行",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pending_review(path: Path, rows: list[dict[str, str]]) -> None:
    pending = [row for row in rows if validate_user_decision(row.get("user_decision", "")) == ""]
    lines = [
        "# v5.20 待用户审查",
        "",
        f"- 待审查行数: `{len(pending)}`",
        "",
        "| 行号 | 零件 | Truth | 候选 | 类别 | 面积 | 面ID |",
        "|-----|--------|-------|-----------|----------|------|-------|",
    ]
    for row in pending:
        lines.append(
            f"| {row['row_id']} | {row['sample']} | {row['truth_macro']} | {row['candidate_macro']} | "
            f"{row['review_category']} | {row['truth_area']}->{row['candidate_area']} | {row['candidate_faces']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_summary(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    recommendation, reason = decide_next_action(summary)
    lines = [
        "# v5.20 Semantic Review Decision Summary",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Outcome Counts",
        "",
        "| Outcome | Count |",
        "|---------|-------|",
    ]
    for outcome, count in summary["by_outcome"].items():
        lines.append(f"| {outcome} | {count} |")
    lines.extend(["", "## Reviewed Rows", "", "| Row | Sample | Truth | Candidate | Category | Decision | Outcome | Note |", "|-----|--------|-------|-----------|----------|----------|---------|------|"])
    for row in rows:
        decision = validate_user_decision(row.get("user_decision", ""))
        outcome = classify_decision_outcome(decision)
        lines.append(
            f"| {row['row_id']} | {row['sample']} | {row['truth_macro']} | {row['candidate_macro']} | "
            f"{row['review_category']} | {decision or 'blank'} | {outcome} | {row.get('user_note', '')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or ingest semantic review decisions.")
    parser.add_argument("--mode", choices=("build-template", "apply-decisions"), required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_csv_rows(args.input_csv)
    rows = build_review_rows(source_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "build-template":
        write_review_template(args.output_dir / "semantic_review_template.csv", rows)
        write_template_markdown(args.output_dir / "semantic_review_template.md", rows)
        write_pending_review(args.output_dir / "pending_user_review.md", rows)
        print(json.dumps({"mode": args.mode, "row_count": len(rows), "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))
        return 0

    summary = summarize_decisions(rows)
    write_json(args.output_dir / "semantic_review_decision_summary.json", {"summary": summary, "rows": rows})
    write_decision_summary(args.output_dir / "semantic_review_decision_summary.md", rows, summary)
    recommendation, reason = decide_next_action(summary)
    decision_lines = [
        "# v5.20 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
    ]
    (args.output_dir / "v5_20_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "summary": summary, "decision": [recommendation, reason]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
