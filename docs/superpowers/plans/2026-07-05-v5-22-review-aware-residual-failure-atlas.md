# v5.22 Review-Aware Residual Failure Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the v5.21 proposal-miss atlas into a review-aware residual failure atlas that respects the user's v5.20 semantic decisions, so the next engineering step targets true proposer/classifier failures instead of repeatedly asking for the same semantic review.

**Architecture:** Do not modify training, inference, graph extraction, STEP generation, candidate proposal, or topology postprocess in this version. Add one diagnostic script that joins v5.21 strict failures with the user's v5.20 review decisions, separates auxiliary semantic-policy accepted rows from confirmed strict failures, and writes a next-action decision memo. This fixes the v5.21 decision weakness: it treated all 24 cross-macro rows as unreviewed even though the user had already classified them.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, CSV, JSON, Markdown.

---

## 0. Review of v5.21 Output

v5.21 mostly satisfies the previous plan.

Verification command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

Observed:

```text
Ran 23 tests
OK
```

Correct v5.21 outputs:

```text
truth_export_diff.md:
  old_truth_count: 204
  new_truth_count: 204
  changed_sample_count: 1
  changed sample: part12 only
  part12 125.6637 and 157.0796 changed from through_hole to fillet

semantic_policy_recall_s3_v5_21.md:
  truth_count: 204
  strict_matched_truth_count: 106
  strict_recall: 0.5196
  policy_recovered_count: 2
  semantic_policy_matched_truth_count: 108
  semantic_policy_recall: 0.5294
```

Problem in v5.21 recommendation:

```text
v5_21_decision.md recommends inspect_remaining_cross_macro_semantics.
```

This recommendation is not the right next engineering action because the 24 `cross_macro_area_compatible` rows are not unreviewed. They include:

- 2 rows the user accepted only as auxiliary hole-family CAE region-selection matches.
- 18 rows the user explicitly marked `keep_strict`.
- 4 rows whose old truth labels were corrected in the Excel truth table; after truth refresh they become strict fillet failures, not semantic-review tasks.

Therefore v5.22 must produce a review-aware atlas before any proposer repair.

## 1. Human Task

No human intervention is required to implement v5.22.

The user should only inspect the generated v5.22 report after implementation:

```text
output_real_step_s3_v5_22_review_aware_residual_atlas\reports\review_aware_residual_atlas.md
output_real_step_s3_v5_22_review_aware_residual_atlas\reports\v5_22_decision.md
```

If the v5.22 report contains any `unreviewed_cross_macro` rows, the agent must stop and ask the user to review only those new rows. If there are no `unreviewed_cross_macro` rows, the next plan should be a targeted proposer/classifier repair plan.

## 2. Scope Control

### Create

- `review_aware_residual_failure_atlas.py`
- `tests/test_review_aware_residual_failure_atlas.py`
- `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/review_aware_residual_atlas.json`
- `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/review_aware_residual_atlas.md`
- `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/v5_22_decision.md`

### Do Not Modify

- `utils/object_candidate_proposer.py`
- `utils/topology_postprocess.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- `object_proposal_recall_report.py`
- `proposal_miss_atlas.py`
- `train_gnn_baseline.py`
- `train_candidate_topology_classifier.py`
- any graph files
- any STEP files
- any model checkpoints
- any existing v5.13-v5.21 report directory

v5.22 is diagnostic classification only.

## 3. Fixed Inputs

Use these exact files:

```text
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_proposal_miss_atlas.json
output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv
output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.json
```

Output directory:

```text
output_real_step_s3_v5_22_review_aware_residual_atlas\reports
```

## 4. Classification Rules

Each miss from v5.21 atlas should receive one `review_status`:

```text
semantic_policy_accepted_auxiliary
confirmed_strict_cross_macro
truth_corrected_strict_failure
unreviewed_cross_macro
non_cross_macro_engineering_failure
```

Rules:

- If a miss matches a v5.20 row with `accept_family_match` by `sample + normalized truth macro + area`, classify as `semantic_policy_accepted_auxiliary`.
- If a miss matches a v5.20 row with `keep_strict` by `sample + normalized truth macro + area`, classify as `confirmed_strict_cross_macro`.
- If a miss matches a v5.20 row with `truth_label_should_change` by `sample + area` after the truth JSON has been refreshed, classify as `truth_corrected_strict_failure`.
- If `miss_cause == cross_macro_area_compatible` and no review row matches, classify as `unreviewed_cross_macro`.
- All other misses are `non_cross_macro_engineering_failure`.

Macro aliases:

```text
through_hole -> through_hole_round
blind_hole -> blind_hole_round
stepped_hole -> stepped_hole_round
cylindrical_boss -> cylinder_boss
rectangle_boss -> rectangular_boss
```

Area matching:

```text
absolute difference <= 0.02 mm^2
```

Decision logic:

- If `unreviewed_cross_macro > 0`, recommendation is `user_review_new_cross_macro_rows`.
- Else if `confirmed_strict_cross_macro + truth_corrected_strict_failure + non_cross_macro_engineering_failure > 0`, recommendation is `write_targeted_proposer_repair_plan`.
- Else if no strict residual failures remain, recommendation is `stop_engineering_and_write_paper_results`.

## 5. Task 1: Add Unit Tests

**Files:**

- Create: `tests/test_review_aware_residual_failure_atlas.py`
- Create: `review_aware_residual_failure_atlas.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_review_aware_residual_failure_atlas.py`:

```python
from __future__ import annotations

import unittest

from review_aware_residual_failure_atlas import (
    classify_miss_with_review,
    choose_next_action,
    normalize_macro,
    summarize_review_aware_misses,
)


class ReviewAwareResidualFailureAtlasTest(unittest.TestCase):
    def test_normalize_macro_unifies_known_aliases(self):
        self.assertEqual(normalize_macro("through_hole"), "through_hole_round")
        self.assertEqual(normalize_macro("blind_hole"), "blind_hole_round")
        self.assertEqual(normalize_macro("stepped_hole"), "stepped_hole_round")
        self.assertEqual(normalize_macro("cylindrical_boss"), "cylinder_boss")
        self.assertEqual(normalize_macro("rectangle_boss"), "rectangular_boss")
        self.assertEqual(normalize_macro("fillet"), "fillet")

    def test_accept_family_match_becomes_auxiliary_not_new_review(self):
        miss = {"sample": "part10", "macro": "through_hole_round", "area": 1912.3562, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "semantic_policy_accepted_auxiliary")
        self.assertEqual(row["user_decision"], "accept_family_match")

    def test_keep_strict_becomes_confirmed_strict_failure(self):
        miss = {"sample": "part18", "macro": "slot_o", "area": 3553.0576, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part18", "truth_macro": "slot_o", "truth_area": "3553.0576", "user_decision": "keep_strict"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "confirmed_strict_cross_macro")

    def test_truth_label_change_matches_by_area_after_truth_refresh(self):
        miss = {"sample": "part12", "macro": "fillet", "area": 125.6637, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "truth_corrected_strict_failure")

    def test_cross_macro_without_review_is_unreviewed(self):
        miss = {"sample": "part99", "macro": "rib", "area": 100.0, "miss_cause": "cross_macro_area_compatible"}

        row = classify_miss_with_review(miss, [])

        self.assertEqual(row["review_status"], "unreviewed_cross_macro")

    def test_non_cross_macro_is_engineering_failure(self):
        miss = {"sample": "part4", "macro": "rib", "area": 458.4, "miss_cause": "no_related_candidate"}

        row = classify_miss_with_review(miss, [])

        self.assertEqual(row["review_status"], "non_cross_macro_engineering_failure")

    def test_summarize_review_aware_misses_counts_status_and_macro(self):
        rows = [
            {"review_status": "confirmed_strict_cross_macro", "macro": "rib", "miss_cause": "cross_macro_area_compatible"},
            {"review_status": "confirmed_strict_cross_macro", "macro": "rib", "miss_cause": "cross_macro_area_compatible"},
            {"review_status": "non_cross_macro_engineering_failure", "macro": "fillet", "miss_cause": "no_same_macro_candidates"},
        ]

        summary = summarize_review_aware_misses(rows)

        self.assertEqual(summary["miss_count"], 3)
        self.assertEqual(summary["by_review_status"]["confirmed_strict_cross_macro"], 2)
        self.assertEqual(summary["by_macro"]["rib"], 2)
        self.assertEqual(summary["by_miss_cause"]["cross_macro_area_compatible"], 2)

    def test_choose_next_action_stops_for_unreviewed_cross_macro(self):
        summary = {"by_review_status": {"unreviewed_cross_macro": 1, "confirmed_strict_cross_macro": 10}}

        recommendation, reason = choose_next_action(summary)

        self.assertEqual(recommendation, "user_review_new_cross_macro_rows")
        self.assertIn("unreviewed_cross_macro", reason)

    def test_choose_next_action_recommends_targeted_repair_after_review(self):
        summary = {
            "by_review_status": {
                "confirmed_strict_cross_macro": 18,
                "truth_corrected_strict_failure": 4,
                "non_cross_macro_engineering_failure": 39,
                "semantic_policy_accepted_auxiliary": 2,
            }
        }

        recommendation, reason = choose_next_action(summary)

        self.assertEqual(recommendation, "write_targeted_proposer_repair_plan")
        self.assertIn("strict residual failures", reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_review_aware_residual_failure_atlas -v
```

Expected before implementation:

```text
ERROR: ModuleNotFoundError: No module named 'review_aware_residual_failure_atlas'
```

## 6. Task 2: Implement Review-Aware Residual Atlas Script

**Files:**

- Modify: `review_aware_residual_failure_atlas.py`

- [ ] **Step 1: Create implementation**

Create `review_aware_residual_failure_atlas.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


MACRO_ALIASES = {
    "through_hole": "through_hole_round",
    "blind_hole": "blind_hole_round",
    "stepped_hole": "stepped_hole_round",
    "cylindrical_boss": "cylinder_boss",
    "rectangle_boss": "rectangular_boss",
}

AREA_TOLERANCE = 0.02


def normalize_macro(value: str) -> str:
    token = str(value or "").strip()
    return MACRO_ALIASES.get(token, token)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def area_close(left: Any, right: Any, tolerance: float = AREA_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def review_row_matches_macro_and_area(miss: dict[str, Any], review: dict[str, str]) -> bool:
    return (
        str(miss.get("sample", "")) == str(review.get("sample", ""))
        and normalize_macro(str(miss.get("macro", ""))) == normalize_macro(str(review.get("truth_macro", "")))
        and area_close(miss.get("area", 0.0), review.get("truth_area", 0.0))
    )


def review_row_matches_area_only(miss: dict[str, Any], review: dict[str, str]) -> bool:
    return (
        str(miss.get("sample", "")) == str(review.get("sample", ""))
        and area_close(miss.get("area", 0.0), review.get("truth_area", 0.0))
    )


def find_matching_review(miss: dict[str, Any], review_rows: list[dict[str, str]]) -> dict[str, str] | None:
    for review in review_rows:
        decision = str(review.get("user_decision", "")).strip()
        if decision in {"accept_family_match", "keep_strict"} and review_row_matches_macro_and_area(miss, review):
            return review
    for review in review_rows:
        decision = str(review.get("user_decision", "")).strip()
        if decision == "truth_label_should_change" and review_row_matches_area_only(miss, review):
            return review
    return None


def classify_miss_with_review(miss: dict[str, Any], review_rows: list[dict[str, str]]) -> dict[str, Any]:
    row = dict(miss)
    review = find_matching_review(miss, review_rows)
    if review is not None:
        decision = str(review.get("user_decision", "")).strip()
        row["user_decision"] = decision
        row["user_note"] = str(review.get("user_note", ""))
        row["review_row_id"] = str(review.get("row_id", ""))
        if decision == "accept_family_match":
            row["review_status"] = "semantic_policy_accepted_auxiliary"
        elif decision == "keep_strict":
            row["review_status"] = "confirmed_strict_cross_macro"
        elif decision == "truth_label_should_change":
            row["review_status"] = "truth_corrected_strict_failure"
        else:
            row["review_status"] = "unreviewed_cross_macro"
        return row
    row["user_decision"] = ""
    row["user_note"] = ""
    row["review_row_id"] = ""
    if str(miss.get("miss_cause", "")) == "cross_macro_area_compatible":
        row["review_status"] = "unreviewed_cross_macro"
    else:
        row["review_status"] = "non_cross_macro_engineering_failure"
    return row


def summarize_review_aware_misses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "miss_count": len(rows),
        "by_review_status": dict(Counter(str(row.get("review_status", "")) for row in rows).most_common()),
        "by_macro": dict(Counter(str(row.get("macro", "")) for row in rows).most_common()),
        "by_miss_cause": dict(Counter(str(row.get("miss_cause", "")) for row in rows).most_common()),
        "by_status_macro": dict(Counter(f"{row.get('review_status', '')}|{row.get('macro', '')}" for row in rows).most_common()),
        "by_part": dict(Counter(str(row.get("sample", "")) for row in rows).most_common()),
    }


def choose_next_action(summary: dict[str, Any]) -> tuple[str, str]:
    status = summary.get("by_review_status", {})
    unreviewed = int(status.get("unreviewed_cross_macro", 0))
    if unreviewed > 0:
        return "user_review_new_cross_macro_rows", "There are unreviewed_cross_macro rows after joining v5.20 decisions; ask the user to review only those rows."
    strict_residual = (
        int(status.get("confirmed_strict_cross_macro", 0))
        + int(status.get("truth_corrected_strict_failure", 0))
        + int(status.get("non_cross_macro_engineering_failure", 0))
    )
    if strict_residual > 0:
        return "write_targeted_proposer_repair_plan", "All cross-macro rows are reviewed; strict residual failures should drive targeted proposer/classifier repair."
    return "stop_engineering_and_write_paper_results", "No strict residual failures remain in the review-aware atlas."


def build_review_aware_atlas(atlas: dict[str, Any], review_rows: list[dict[str, str]], semantic_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    source_misses = atlas.get("misses", [])
    rows = [classify_miss_with_review(miss, review_rows) for miss in source_misses]
    summary = summarize_review_aware_misses(rows)
    recommendation, reason = choose_next_action(summary)
    return {
        "schema": "review_aware_residual_failure_atlas_v1",
        "source_atlas_schema": atlas.get("schema", "unknown"),
        "semantic_policy_recall": semantic_policy or {},
        "summary": summary,
        "recommendation": recommendation,
        "reason": reason,
        "misses": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# v5.22 Review-Aware Residual Failure Atlas",
        "",
        "## Recommendation",
        "",
        f"**{payload['recommendation']}**",
        "",
        "## Reasoning",
        "",
        payload["reason"],
        "",
        "## Summary",
        "",
        f"- miss_count: `{summary['miss_count']}`",
        "",
        "## Review Status Counts",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, count in summary["by_review_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Macro Counts", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Status by Macro", "", "| Status | Macro | Count |", "|--------|-------|-------|"])
    for key, count in summary["by_status_macro"].items():
        status, macro = key.split("|", 1)
        lines.append(f"| {status} | {macro} | {count} |")
    lines.extend([
        "",
        "## Top Residual Misses",
        "",
        "| Sample | Macro | Area | Faces | Miss Cause | Review Status | User Decision |",
        "|--------|-------|------|-------|------------|---------------|---------------|",
    ])
    for row in payload["misses"][:80]:
        lines.append(
            f"| {row.get('sample', '')} | {row.get('macro', '')} | {float(row.get('area', 0.0)):.4f} | "
            f"{row.get('face_count', '')} | {row.get('miss_cause', '')} | {row.get('review_status', '')} | "
            f"{row.get('user_decision', '')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# v5.22 Decision",
        "",
        "## Recommendation",
        "",
        f"**{payload['recommendation']}**",
        "",
        "## Reasoning",
        "",
        payload["reason"],
        "",
        "## Review-Aware Counts",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, count in summary["by_review_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Top Macros for Next Repair", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-aware residual failure atlas from v5.21 atlas and v5.20 user decisions.")
    parser.add_argument("--atlas-json", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--semantic-policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atlas = load_json(args.atlas_json)
    review_rows = read_review_rows(args.review_csv)
    semantic_policy = load_json(args.semantic_policy_json)
    payload = build_review_aware_atlas(atlas, review_rows, semantic_policy)
    write_json(args.output_dir / "review_aware_residual_atlas.json", payload)
    write_markdown(args.output_dir / "review_aware_residual_atlas.md", payload)
    write_decision(args.output_dir / "v5_22_decision.md", payload)
    print(json.dumps({"summary": payload["summary"], "recommendation": payload["recommendation"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_review_aware_residual_failure_atlas -v
```

Expected:

```text
OK
```

## 7. Task 3: Generate v5.22 Review-Aware Atlas

**Files:**

- Generate: `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/review_aware_residual_atlas.json`
- Generate: `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/review_aware_residual_atlas.md`
- Generate: `output_real_step_s3_v5_22_review_aware_residual_atlas/reports/v5_22_decision.md`

- [ ] **Step 1: Run script**

Run:

```powershell
W:\python\envs\feagpt\python.exe review_aware_residual_failure_atlas.py `
  --atlas-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\v5_21_proposal_miss_atlas.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --semantic-policy-json output_real_step_s3_v5_21_reviewed_truth_semantic_policy\reports\semantic_policy_recall_s3_v5_21.json `
  --output-dir output_real_step_s3_v5_22_review_aware_residual_atlas\reports
```

Expected:

```text
"recommendation": "write_targeted_proposer_repair_plan"
```

If the recommendation is `user_review_new_cross_macro_rows`, stop and report the generated `unreviewed_cross_macro` rows to the user instead of planning proposer repair.

- [ ] **Step 2: Inspect markdown**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_22_review_aware_residual_atlas\reports\review_aware_residual_atlas.md -TotalCount 160
```

Expected:

```text
The report includes Review Status Counts and Top Macros for Next Repair.
```

## 8. Task 4: Final Verification

**Files:**

- Verify all generated v5.22 files.

- [ ] **Step 1: Run all relevant tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_review_aware_residual_failure_atlas `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Verify outputs exist**

Run:

```powershell
Get-ChildItem -LiteralPath output_real_step_s3_v5_22_review_aware_residual_atlas\reports | Select-Object Name,Length,LastWriteTime
```

Expected files:

```text
review_aware_residual_atlas.json
review_aware_residual_atlas.md
v5_22_decision.md
```

## 9. Final Response Requirements

The implementing agent must report:

```text
1. Test command and result.
2. Review-aware status counts.
3. Whether any unreviewed_cross_macro rows remain.
4. Top macros for next repair.
5. v5.22 recommendation.
6. Confirmation that no training, inference, proposer, graph extractor, STEP generator, or model checkpoint files were modified.
```

## 10. How I Will Review This Work

I will check:

```text
review_aware_residual_failure_atlas.py
tests/test_review_aware_residual_failure_atlas.py
output_real_step_s3_v5_22_review_aware_residual_atlas\reports\review_aware_residual_atlas.json
output_real_step_s3_v5_22_review_aware_residual_atlas\reports\review_aware_residual_atlas.md
output_real_step_s3_v5_22_review_aware_residual_atlas\reports\v5_22_decision.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_review_aware_residual_failure_atlas `
  tests.test_export_area_truth_from_xlsx `
  tests.test_apply_semantic_review_policy_to_recall_report `
  tests.test_write_v5_21_decision `
  tests.test_semantic_review_decision_ingest `
  tests.test_semantic_cross_macro_audit -v
```

I will reject the implementation if:

- It asks the user to re-review all 24 v5.20 rows.
- It treats `semantic_policy_accepted_auxiliary` rows as strict successes.
- It modifies proposer or model code in v5.22.
- It ignores the four `truth_label_should_change` rows after truth refresh.
- It recommends model training before producing a review-aware residual atlas.

## 11. Next Plan After v5.22

If v5.22 recommends `write_targeted_proposer_repair_plan`, the next plan should be v5.23 and should modify proposer/classifier logic. It should prioritize the largest review-aware residual macro groups, expected to be:

```text
rib
fillet
transition-vs-hole/cavity role confusion
slot_rect / slot_o / slot_u residual proposal misses
cylinder_boss vs blind_hole role confusion
```

The v5.23 plan must be based on the actual `review_aware_residual_atlas.md` numbers, not on the old v5.21 `cross_macro_area_compatible` count.

## 12. English Handoff Summary

v5.21 refreshed the truth table correctly and separated strict recall from auxiliary hole-family recall, but its recommendation is incomplete because it treats previously reviewed cross-macro rows as unreviewed. v5.22 should join v5.21 failures with v5.20 user decisions and produce a review-aware residual atlas. Do not change model/proposer code in v5.22. If no new unreviewed cross-macro rows remain, the next version should be a targeted proposer repair plan.
