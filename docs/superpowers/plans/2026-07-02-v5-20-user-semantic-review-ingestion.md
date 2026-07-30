# v5.20 User Semantic Review Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the v5.19 cross-macro semantic audit into a small human-review workflow, then ingest the user's decisions to decide whether to change evaluation policy, correct truth labels, or repair proposal generation.

**Architecture:** Do not modify inference, training, graph extraction, candidate proposal, topology postprocess, or existing evaluation behavior. Add a review-template builder and decision-ingestion script that creates an editable CSV with `user_decision` fields, validates completed decisions, summarizes decisions by semantic category and macro pair, and writes the next code-changing recommendation.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, CSV, JSON, Markdown, existing v5.19 semantic audit outputs.

---

## 0. Review of v5.19 Output

v5.19 satisfies the requested plan.

Verification:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_cross_macro_audit tests.test_proposal_miss_atlas -v
```

Observed:

```text
Ran 14 tests
OK
```

Generated files:

```text
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.json
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.csv
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.md
output_real_step_s3_v5_19_semantic_audit/reports/semantic_policy_simulation.md
output_real_step_s3_v5_19_semantic_audit/reports/user_semantic_review_instructions.md
output_real_step_s3_v5_19_semantic_audit/reports/v5_19_decision.md
```

Important v5.19 result:

```text
cross_macro_compatible_count: 24

transition_vs_nontransition_conflict: 11
thinwall_rib_vs_boss_or_web: 5
hole_subfeature_vs_composite_hole: 4
circular_recess_or_hole_conflict: 2
rectangular_recess_slot_vs_pocket: 1
unknown_cross_macro_conflict: 1
```

v5.19 recommendation:

```text
inspect_transition_truth_and_face_roles
```

This is reasonable because transition conflicts are the largest category. However, v5.19 cannot decide automatically whether those transition rows are:

- correct strict failures,
- truth-table mistakes,
- acceptable feature-family matches,
- or model/proposer errors.

Human review is now required, but the burden is small: only 24 cross-macro compatible rows, with priority on 11 transition rows.

## 1. Human Task for User

The user should review a generated CSV template, not the whole dataset.

The next agent will create:

```text
output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv
```

The user should open that CSV in Excel/WPS/LibreOffice and fill only this column:

```text
user_decision
```

Allowed values:

```text
accept_family_match
keep_strict
truth_label_should_change
model_prediction_wrong
needs_face_level_review
```

Decision meanings:

- `accept_family_match`: In CAE region-selection evaluation, this candidate should count as matching the truth even though the macro names differ.
- `keep_strict`: The macro distinction is meaningful and should remain a strict failure.
- `truth_label_should_change`: The truth row is mislabeled; the truth macro should be corrected.
- `model_prediction_wrong`: The candidate is genuinely wrong.
- `needs_face_level_review`: Area alone is insufficient; the STEP/face IDs need visual inspection.

Priority rows:

```text
review_category == transition_vs_nontransition_conflict
```

Default engineering guidance:

- For `fillet`/`chamfer`: keep them strict if they are independent transition targets in the first paper.
- If a truth row says `fillet` but the candidate is actually a hole/cavity face with identical area, mark `truth_label_should_change` if the truth table was likely mislabeled.
- For `hole_subfeature_vs_composite_hole`: mark `accept_family_match` only if the paper will explicitly evaluate "hole family" region selection in addition to strict macro classification.
- For `slot_rect` vs `pocket_rect`: mark `accept_family_match` only if the downstream CAE operation would treat both as the same recess-selection region.
- For `rib` vs `rectangular_boss`: do not auto-accept. Mark `truth_label_should_change`, `model_prediction_wrong`, or `needs_face_level_review` based on geometry.

## 2. Scope Control

### Create

- `semantic_review_decision_ingest.py`
- `tests/test_semantic_review_decision_ingest.py`
- `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv`
- `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.md`
- `output_real_step_s3_v5_20_user_semantic_review/reports/pending_user_review.md`

After the user fills the template, the same script will generate:

- `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_decision_summary.json`
- `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_decision_summary.md`
- `output_real_step_s3_v5_20_user_semantic_review/reports/v5_20_decision.md`

### Do Not Modify

- `utils/object_candidate_proposer.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- `utils/topology_postprocess.py`
- `object_proposal_recall_report.py`
- `proposal_miss_atlas.py`
- `semantic_cross_macro_audit.py`
- `train_gnn_baseline.py`
- `train_candidate_topology_classifier.py`
- any model checkpoint
- any existing v5.13/v5.17/v5.18/v5.19 report directory

This plan creates review and ingestion tooling only.

## 3. Fixed Inputs

Use:

```text
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.csv
```

Optional reference:

```text
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.md
output_real_step_s3_v5_19_semantic_audit/reports/v5_19_decision.md
```

## 4. Decision Rules

When user decisions are ingested:

```text
accept_family_match -> policy_recovered
truth_label_should_change -> truth_correction_needed
keep_strict -> strict_failure
model_prediction_wrong -> model_or_proposer_error
needs_face_level_review -> unresolved_manual_review
blank -> pending_user_review
```

Next recommendation:

- If any rows are `pending_user_review`, recommend `wait_for_user_semantic_review`.
- If any rows are `needs_face_level_review`, recommend `prepare_face_level_review_package`.
- If `truth_label_should_change` is the largest nonzero category, recommend `correct_area_truth_table_then_rerun_metrics`.
- If `accept_family_match` is the largest nonzero category, recommend `implement_evaluation_only_semantic_policy`.
- If `model_prediction_wrong` is the largest nonzero category, recommend `targeted_proposal_or_classifier_repair`.
- If `keep_strict` dominates and no other category is actionable, recommend `return_to_proposal_generation_repair_for_remaining_no_same_macro_misses`.

## 5. Task 1: Add Unit Tests

**Files:**

- Create: `tests/test_semantic_review_decision_ingest.py`
- Create: `semantic_review_decision_ingest.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_semantic_review_decision_ingest.py`:

```python
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from semantic_review_decision_ingest import (
    ALLOWED_USER_DECISIONS,
    build_review_rows,
    classify_decision_outcome,
    decide_next_action,
    read_csv_rows,
    summarize_decisions,
    validate_user_decision,
    write_review_template,
)


class SemanticReviewDecisionIngestTest(unittest.TestCase):
    def test_validate_user_decision_accepts_allowed_and_blank(self):
        self.assertEqual(validate_user_decision("accept_family_match"), "accept_family_match")
        self.assertEqual(validate_user_decision(" "), "")
        self.assertIn("keep_strict", ALLOWED_USER_DECISIONS)

    def test_validate_user_decision_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "Invalid user_decision"):
            validate_user_decision("maybe")

    def test_classify_decision_outcome_maps_values(self):
        self.assertEqual(classify_decision_outcome("accept_family_match"), "policy_recovered")
        self.assertEqual(classify_decision_outcome("truth_label_should_change"), "truth_correction_needed")
        self.assertEqual(classify_decision_outcome("keep_strict"), "strict_failure")
        self.assertEqual(classify_decision_outcome("model_prediction_wrong"), "model_or_proposer_error")
        self.assertEqual(classify_decision_outcome("needs_face_level_review"), "unresolved_manual_review")
        self.assertEqual(classify_decision_outcome(""), "pending_user_review")

    def test_build_review_rows_adds_row_id_decision_and_prompt(self):
        source_rows = [
            {
                "sample": "part3",
                "truth_macro": "fillet",
                "candidate_macro": "blind_hole_round",
                "pair": "blind_hole_round|fillet",
                "review_category": "transition_vs_nontransition_conflict",
                "truth_area": "235.62",
                "candidate_area": "235.62",
                "relative_error": "0.0",
                "truth_face_count": "1",
                "candidate_face_count": "1",
                "candidate_source": "same_instance",
                "candidate_action": "rejected_label",
                "candidate_confidence": "0.64",
                "candidate_faces": "22",
            }
        ]

        rows = build_review_rows(source_rows)

        self.assertEqual(rows[0]["row_id"], "R001")
        self.assertEqual(rows[0]["user_decision"], "")
        self.assertIn("fillet/chamfer", rows[0]["review_prompt"])

    def test_write_review_template_round_trips_user_decision_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            rows = build_review_rows([
                {
                    "sample": "part11",
                    "truth_macro": "slot_rect",
                    "candidate_macro": "pocket_rect",
                    "pair": "pocket_rect|slot_rect",
                    "review_category": "rectangular_recess_slot_vs_pocket",
                    "truth_area": "72.36",
                    "candidate_area": "72.36",
                    "relative_error": "0.0",
                    "truth_face_count": "3",
                    "candidate_face_count": "3",
                    "candidate_source": "hybrid_merged",
                    "candidate_action": "rejected_label",
                    "candidate_confidence": "0.78",
                    "candidate_faces": "28 29 30",
                }
            ])

            write_review_template(path, rows)
            loaded = read_csv_rows(path)

            self.assertEqual(loaded[0]["row_id"], "R001")
            self.assertIn("user_decision", loaded[0])
            self.assertIn("review_prompt", loaded[0])

    def test_summarize_decisions_counts_outcomes_and_categories(self):
        rows = [
            {"review_category": "transition_vs_nontransition_conflict", "pair": "a|b", "user_decision": "keep_strict"},
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "c|d", "user_decision": "accept_family_match"},
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "c|d", "user_decision": ""},
        ]

        summary = summarize_decisions(rows)

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["by_outcome"]["strict_failure"], 1)
        self.assertEqual(summary["by_outcome"]["policy_recovered"], 1)
        self.assertEqual(summary["by_outcome"]["pending_user_review"], 1)
        self.assertEqual(summary["by_category"]["hole_subfeature_vs_composite_hole"], 2)

    def test_decide_next_action_waits_for_pending_review_first(self):
        summary = {"by_outcome": {"pending_user_review": 1, "policy_recovered": 5}}
        recommendation, _ = decide_next_action(summary)
        self.assertEqual(recommendation, "wait_for_user_semantic_review")

    def test_decide_next_action_prefers_truth_correction_when_largest(self):
        summary = {"by_outcome": {"truth_correction_needed": 4, "policy_recovered": 2}}
        recommendation, _ = decide_next_action(summary)
        self.assertEqual(recommendation, "correct_area_truth_table_then_rerun_metrics")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_review_decision_ingest -v
```

Expected before implementation:

```text
ERROR
```

The error should mention missing `semantic_review_decision_ingest`.

## 6. Task 2: Implement Review Template and Decision Ingestion Script

**Files:**

- Modify: `semantic_review_decision_ingest.py`

- [ ] **Step 1: Create implementation**

Create `semantic_review_decision_ingest.py`:

```python
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
        "transition_vs_nontransition_conflict": "Decide whether this fillet/chamfer target is truly an independent transition face or whether the truth/candidate macro is wrong.",
        "hole_subfeature_vs_composite_hole": "Decide whether this can count as a hole-family match for CAE region selection, or whether strict macro labels must differ.",
        "rectangular_recess_slot_vs_pocket": "Decide whether slot_rect and pocket_rect should be strict separate classes or one CAE recess-selection family.",
        "thinwall_rib_vs_boss_or_web": "Decide whether this is truly rib/web, a boss/context face, or needs face-level review.",
        "circular_recess_or_hole_conflict": "Decide whether circular pocket/recess and hole should remain strict or count as a circular recess family.",
        "unknown_cross_macro_conflict": "Inspect this row manually; no safe automatic semantic family is defined.",
    }
    return prompts.get(category, "Inspect this row manually.")


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
        "# v5.20 Semantic Review Template",
        "",
        f"- row_count: `{len(rows)}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")
    lines.extend([
        "",
        "## User Instructions",
        "",
        "Open `semantic_review_template.csv` and fill the `user_decision` column.",
        "",
        "Allowed values:",
        "",
        "- `accept_family_match`",
        "- `keep_strict`",
        "- `truth_label_should_change`",
        "- `model_prediction_wrong`",
        "- `needs_face_level_review`",
        "",
        "Prioritize rows whose `review_category` is `transition_vs_nontransition_conflict`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pending_review(path: Path, rows: list[dict[str, str]]) -> None:
    pending = [row for row in rows if validate_user_decision(row.get("user_decision", "")) == ""]
    lines = [
        "# v5.20 Pending User Review",
        "",
        f"- pending_rows: `{len(pending)}`",
        "",
        "| Row | Sample | Truth | Candidate | Category | Area | Faces |",
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
```

- [ ] **Step 2: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_review_decision_ingest -v
```

Expected:

```text
OK
```

## 7. Task 3: Generate the User Review Template

**Files:**

- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv`
- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.md`
- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/pending_user_review.md`

- [ ] **Step 1: Build the review template**

Run:

```powershell
W:\python\envs\feagpt\python.exe semantic_review_decision_ingest.py `
  --mode build-template `
  --input-csv output_real_step_s3_v5_19_semantic_audit\reports\cross_macro_semantic_audit.csv `
  --output-dir output_real_step_s3_v5_20_user_semantic_review\reports
```

Expected console output:

```text
"mode": "build-template"
"row_count": 24
```

- [ ] **Step 2: Inspect template summary**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.md
```

Expected:

```text
The file lists category counts and allowed user_decision values.
```

- [ ] **Step 3: Inspect pending review**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_20_user_semantic_review\reports\pending_user_review.md -TotalCount 80
```

Expected:

```text
pending_rows: `24`
```

## 8. Task 4: Optional Dry-Run Apply Decisions Before User Edits

**Files:**

- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_decision_summary.json`
- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_decision_summary.md`
- Generate: `output_real_step_s3_v5_20_user_semantic_review/reports/v5_20_decision.md`

- [ ] **Step 1: Apply decisions to the blank template**

Run:

```powershell
W:\python\envs\feagpt\python.exe semantic_review_decision_ingest.py `
  --mode apply-decisions `
  --input-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --output-dir output_real_step_s3_v5_20_user_semantic_review\reports
```

Expected:

```text
"wait_for_user_semantic_review"
```

- [ ] **Step 2: Inspect decision**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_20_user_semantic_review\reports\v5_20_decision.md
```

Expected:

```text
Recommendation is wait_for_user_semantic_review.
```

## 9. Human Review Checkpoint

At this point the implementing agent must stop and report that user review is required.

The user must edit:

```text
output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv
```

The user should save it as:

```text
output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template_reviewed.csv
```

The user only needs to fill `user_decision`.

Recommended priority:

1. Fill all `transition_vs_nontransition_conflict` rows.
2. Fill all `thinwall_rib_vs_boss_or_web` rows.
3. Fill remaining hole/recess rows.

After the user saves the reviewed CSV, the next agent should run:

```powershell
W:\python\envs\feagpt\python.exe semantic_review_decision_ingest.py `
  --mode apply-decisions `
  --input-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template_reviewed.csv `
  --output-dir output_real_step_s3_v5_20_user_semantic_review\reports
```

## 10. Final Verification

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_review_decision_ingest tests.test_semantic_cross_macro_audit -v
```

Expected:

```text
OK
```

## 11. What the Implementing Agent Must Report

The final response must include:

```text
1. Test command and result.
2. Template CSV path.
3. Pending review count.
4. Category counts in the template.
5. Whether dry-run apply-decisions produced wait_for_user_semantic_review.
6. Exact instructions for the user to fill user_decision.
7. Confirmation that no inference/training/proposer files were modified.
```

## 12. What the User Should Do

Open:

```text
W:\python\FEA\dataset_generator\output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv
```

Fill `user_decision` for each row using exactly one of:

```text
accept_family_match
keep_strict
truth_label_should_change
model_prediction_wrong
needs_face_level_review
```

Save as:

```text
W:\python\FEA\dataset_generator\output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template_reviewed.csv
```

No new STEP modelling is required. No whole-part relabeling is required. This is only a 24-row semantic review.

## 13. How I Will Review This Work

I will check:

```text
semantic_review_decision_ingest.py
tests/test_semantic_review_decision_ingest.py
output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv
output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.md
output_real_step_s3_v5_20_user_semantic_review/reports/pending_user_review.md
output_real_step_s3_v5_20_user_semantic_review/reports/v5_20_decision.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_review_decision_ingest tests.test_semantic_cross_macro_audit -v
```

I will not recommend code-changing repairs until the reviewed CSV exists and the decision summary shows whether the dominant outcome is semantic policy, truth correction, model/proposer error, or face-level review.

## 14. English Handoff Summary

v5.19 showed transition-vs-nontransition conflicts dominate the cross-macro compatible misses. The next step is not another model or proposer change. Implement v5.20 to generate a 24-row user semantic review template, ingest the user's `user_decision` values, and produce a decision summary. Stop after generating the blank template and ask the user to fill it. Code-changing repairs come only after user decisions are ingested.
