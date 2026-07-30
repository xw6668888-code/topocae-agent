# v5.19 Semantic Audit and Evaluation Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit cross-macro area-compatible S3 failures and simulate explicit CAE-semantic evaluation policies before changing proposal generation or model training.

**Architecture:** Keep all inference, model, proposer, topology postprocess, and dataset generation behavior unchanged. Add a diagnostic script that reads v5.18 proposal miss atlases, groups cross-macro misses by truth/predicted macro pair, assigns a review category, simulates safe/unsafe evaluation policy choices, and writes a human-review package plus a next-step decision.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, JSON, CSV, Markdown, existing v5.18 proposal miss atlas outputs.

---

## 0. Review of v5.18 Output

v5.18 satisfies the plan.

Verified command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_proposal_miss_atlas tests.test_object_proposal_recall_report tests.test_compare_candidate_classifier_recall_reports -v
```

Observed:

```text
Ran 21 tests
OK
```

Generated artifacts:

```text
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.json
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.json
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_repair_gate.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_next_step_recommendation.md
```

Important v5.13 miss cause counts:

```text
cross_macro_area_compatible: 24
no_same_macro_candidates: 17
cross_macro_area_near_miss: 9
no_related_candidate: 9
same_macro_area_near_miss: 3
same_macro_split_or_merge_suspected: 3
```

Top missed macros:

```text
rib: 23
fillet: 14
through_hole_round: 10
chamfer: 4
counterbore_hole: 3
slot_rect: 2
countersink_hole: 2
cylinder_boss: 2
```

Top missed parts:

```text
part20: 14
part4: 8
part21: 7
part3: 6
part12: 5
part1: 4
part18: 3
part7: 3
```

v5.18 recommendation:

```text
truth_or_macro_semantic_audit
```

This recommendation is reasonable. The largest failure category is not "no same-macro candidate exists"; it is "a different macro has an area-compatible candidate." Examples:

```text
through_hole_round -> counterbore_hole
through_hole_round -> countersink_hole
through_hole_round -> stepped_hole_round
slot_rect -> pocket_rect
slot_o -> fillet
rib -> rectangular_boss
fillet -> blind_hole_round / pocket_circular / through_hole_round
```

These are not all proposer failures. Some are likely evaluation-granularity or CAE-semantic mismatches. v5.19 must inspect and simulate policies before any proposer/code repair.

## 1. Scope Control

### Create

- `semantic_cross_macro_audit.py`
- `tests/test_semantic_cross_macro_audit.py`
- `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.json`
- `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.csv`
- `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.md`
- `output_real_step_s3_v5_19_semantic_audit/reports/semantic_policy_simulation.md`
- `output_real_step_s3_v5_19_semantic_audit/reports/v5_19_decision.md`

### Do Not Modify

- `utils/object_candidate_proposer.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- `utils/topology_postprocess.py`
- `train_gnn_baseline.py`
- `train_candidate_topology_classifier.py`
- `object_proposal_recall_report.py`
- `proposal_miss_atlas.py`
- any model checkpoint
- any generated v5.13/v5.17/v5.18 reports

This is diagnostic and policy simulation only.

## 2. Fixed Inputs

Primary input:

```text
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.json
```

Supporting input:

```text
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.json
```

Use v5.13 as primary because it remains the current best real-S3 baseline.

## 3. Semantic Review Categories

Every cross-macro area-compatible miss must receive exactly one `review_category`:

```text
hole_subfeature_vs_composite_hole
rectangular_recess_slot_vs_pocket
thinwall_rib_vs_boss_or_web
transition_vs_nontransition_conflict
circular_recess_or_hole_conflict
unknown_cross_macro_conflict
```

Definitions:

- `hole_subfeature_vs_composite_hole`: truth or candidate is one of `through_hole_round`, `blind_hole_round`, `stepped_hole_round`, `counterbore_hole`, `countersink_hole`.
- `rectangular_recess_slot_vs_pocket`: pair contains `slot_rect` and `pocket_rect`.
- `thinwall_rib_vs_boss_or_web`: pair contains `rib` or `web`, and the other macro is `rectangular_boss`, `cylinder_boss`, `slot_v`, or `slot_rect`.
- `transition_vs_nontransition_conflict`: exactly one macro is `fillet` or `chamfer`.
- `circular_recess_or_hole_conflict`: pair contains `pocket_circular` and a hole macro.
- `unknown_cross_macro_conflict`: none of the above.

## 4. Policy Simulation Modes

The script must simulate these policies:

```text
strict_current
safe_hole_composite
safe_hole_and_rectangular_recess
review_all_cross_macro_compatible
```

Meaning:

- `strict_current`: no cross-macro compatible misses are recovered.
- `safe_hole_composite`: recover only `hole_subfeature_vs_composite_hole`.
- `safe_hole_and_rectangular_recess`: recover `hole_subfeature_vs_composite_hole` and `rectangular_recess_slot_vs_pocket`.
- `review_all_cross_macro_compatible`: recover all `cross_macro_area_compatible` misses for upper-bound simulation only. This is not an automatic policy recommendation.

The simulation must output:

```text
baseline_miss_count
recovered_count
remaining_miss_count
recovered_by_category
recovered_by_pair
```

## 5. Task 1: Add Unit Tests

**Files:**

- Create: `tests/test_semantic_cross_macro_audit.py`
- Create: `semantic_cross_macro_audit.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_semantic_cross_macro_audit.py`:

```python
from __future__ import annotations

import unittest

from semantic_cross_macro_audit import (
    classify_review_category,
    macro_pair_key,
    simulate_policy,
    summarize_cross_macro_misses,
)


class SemanticCrossMacroAuditTest(unittest.TestCase):
    def test_macro_pair_key_is_ordered_and_stable(self):
        self.assertEqual(
            macro_pair_key("counterbore_hole", "through_hole_round"),
            "counterbore_hole|through_hole_round",
        )
        self.assertEqual(
            macro_pair_key("through_hole_round", "counterbore_hole"),
            "counterbore_hole|through_hole_round",
        )

    def test_classifies_hole_subfeature_vs_composite_hole(self):
        self.assertEqual(
            classify_review_category("through_hole_round", "counterbore_hole"),
            "hole_subfeature_vs_composite_hole",
        )
        self.assertEqual(
            classify_review_category("countersink_hole", "through_hole_round"),
            "hole_subfeature_vs_composite_hole",
        )

    def test_classifies_rectangular_recess_slot_vs_pocket(self):
        self.assertEqual(
            classify_review_category("slot_rect", "pocket_rect"),
            "rectangular_recess_slot_vs_pocket",
        )

    def test_classifies_rib_vs_boss_or_web(self):
        self.assertEqual(
            classify_review_category("rib", "rectangular_boss"),
            "thinwall_rib_vs_boss_or_web",
        )
        self.assertEqual(
            classify_review_category("web", "slot_v"),
            "thinwall_rib_vs_boss_or_web",
        )

    def test_classifies_transition_conflict(self):
        self.assertEqual(
            classify_review_category("fillet", "blind_hole_round"),
            "transition_vs_nontransition_conflict",
        )

    def test_summarize_cross_macro_misses_counts_pairs_and_categories(self):
        misses = [
            {
                "sample": "part1",
                "macro": "through_hole_round",
                "area": 100.0,
                "miss_cause": "cross_macro_area_compatible",
                "nearest_cross_macro": [{"proposal_macro": "counterbore_hole", "area": 100.1, "relative_error": 0.001}],
            },
            {
                "sample": "part2",
                "macro": "slot_rect",
                "area": 72.0,
                "miss_cause": "cross_macro_area_compatible",
                "nearest_cross_macro": [{"proposal_macro": "pocket_rect", "area": 72.0, "relative_error": 0.0}],
            },
        ]

        rows, summary = summarize_cross_macro_misses(misses)

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["cross_macro_compatible_count"], 2)
        self.assertEqual(summary["by_pair"]["counterbore_hole|through_hole_round"], 1)
        self.assertEqual(summary["by_category"]["hole_subfeature_vs_composite_hole"], 1)
        self.assertEqual(summary["by_category"]["rectangular_recess_slot_vs_pocket"], 1)

    def test_simulate_policy_recovers_only_allowed_categories(self):
        rows = [
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "counterbore_hole|through_hole_round"},
            {"review_category": "rectangular_recess_slot_vs_pocket", "pair": "pocket_rect|slot_rect"},
            {"review_category": "transition_vs_nontransition_conflict", "pair": "fillet|through_hole_round"},
        ]

        result = simulate_policy("safe_hole_and_rectangular_recess", rows, baseline_miss_count=65)

        self.assertEqual(result["policy"], "safe_hole_and_rectangular_recess")
        self.assertEqual(result["recovered_count"], 2)
        self.assertEqual(result["remaining_miss_count"], 63)
        self.assertEqual(result["recovered_by_category"]["hole_subfeature_vs_composite_hole"], 1)
        self.assertEqual(result["recovered_by_category"]["rectangular_recess_slot_vs_pocket"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_cross_macro_audit -v
```

Expected before implementation:

```text
ERROR
```

The error should mention missing `semantic_cross_macro_audit`.

## 6. Task 2: Implement Semantic Cross-Macro Audit

**Files:**

- Modify: `semantic_cross_macro_audit.py`

- [ ] **Step 1: Create the script**

Create `semantic_cross_macro_audit.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HOLE_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
}
TRANSITION_MACROS = {"fillet", "chamfer"}
RECTANGULAR_RECESS_MACROS = {"slot_rect", "pocket_rect"}
RIB_CONTEXT_MACROS = {"rib", "web", "rectangular_boss", "cylinder_boss", "slot_v", "slot_rect"}
POLICY_CATEGORIES = {
    "strict_current": set(),
    "safe_hole_composite": {"hole_subfeature_vs_composite_hole"},
    "safe_hole_and_rectangular_recess": {
        "hole_subfeature_vs_composite_hole",
        "rectangular_recess_slot_vs_pocket",
    },
    "review_all_cross_macro_compatible": {
        "hole_subfeature_vs_composite_hole",
        "rectangular_recess_slot_vs_pocket",
        "thinwall_rib_vs_boss_or_web",
        "transition_vs_nontransition_conflict",
        "circular_recess_or_hole_conflict",
        "unknown_cross_macro_conflict",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def macro_pair_key(left: str, right: str) -> str:
    return "|".join(sorted([str(left), str(right)]))


def classify_review_category(truth_macro: str, candidate_macro: str) -> str:
    pair = {truth_macro, candidate_macro}
    if truth_macro in HOLE_MACROS and candidate_macro in HOLE_MACROS and truth_macro != candidate_macro:
        return "hole_subfeature_vs_composite_hole"
    if pair == RECTANGULAR_RECESS_MACROS:
        return "rectangular_recess_slot_vs_pocket"
    if "pocket_circular" in pair and bool(pair & HOLE_MACROS):
        return "circular_recess_or_hole_conflict"
    if bool(pair & {"rib", "web"}) and bool(pair & RIB_CONTEXT_MACROS - {"rib", "web"}):
        return "thinwall_rib_vs_boss_or_web"
    if (truth_macro in TRANSITION_MACROS) != (candidate_macro in TRANSITION_MACROS):
        return "transition_vs_nontransition_conflict"
    return "unknown_cross_macro_conflict"


def best_cross_candidate(miss: dict[str, Any]) -> dict[str, Any] | None:
    candidates = miss.get("nearest_cross_macro", [])
    if not candidates:
        return None
    return candidates[0]


def summarize_cross_macro_misses(misses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for miss in misses:
        if miss.get("miss_cause") != "cross_macro_area_compatible":
            continue
        candidate = best_cross_candidate(miss)
        if not candidate:
            continue
        truth_macro = str(miss.get("macro", ""))
        candidate_macro = str(candidate.get("proposal_macro", ""))
        category = classify_review_category(truth_macro, candidate_macro)
        pair = macro_pair_key(truth_macro, candidate_macro)
        rows.append(
            {
                "sample": str(miss.get("sample", "")),
                "truth_macro": truth_macro,
                "candidate_macro": candidate_macro,
                "pair": pair,
                "review_category": category,
                "truth_area": float(miss.get("area", 0.0) or 0.0),
                "candidate_area": float(candidate.get("area", 0.0) or 0.0),
                "relative_error": float(candidate.get("relative_error", 0.0) or 0.0),
                "truth_face_count": int(miss.get("face_count", 0) or 0),
                "candidate_face_count": int(candidate.get("face_count", 0) or 0),
                "candidate_source": str(candidate.get("source", "")),
                "candidate_action": str(candidate.get("action", "")),
                "candidate_confidence": candidate.get("confidence"),
                "candidate_faces": list(candidate.get("face_indices", [])),
            }
        )
    summary = {
        "cross_macro_compatible_count": len(rows),
        "by_pair": dict(Counter(row["pair"] for row in rows).most_common()),
        "by_category": dict(Counter(row["review_category"] for row in rows).most_common()),
        "by_truth_macro": dict(Counter(row["truth_macro"] for row in rows).most_common()),
        "by_candidate_macro": dict(Counter(row["candidate_macro"] for row in rows).most_common()),
        "by_part": dict(Counter(row["sample"] for row in rows).most_common()),
    }
    return rows, summary


def simulate_policy(policy: str, rows: list[dict[str, Any]], baseline_miss_count: int) -> dict[str, Any]:
    if policy not in POLICY_CATEGORIES:
        raise ValueError(f"Unknown policy {policy!r}")
    allowed = POLICY_CATEGORIES[policy]
    recovered = [row for row in rows if row["review_category"] in allowed]
    return {
        "policy": policy,
        "baseline_miss_count": int(baseline_miss_count),
        "recovered_count": len(recovered),
        "remaining_miss_count": int(baseline_miss_count) - len(recovered),
        "recovered_by_category": dict(Counter(row["review_category"] for row in recovered).most_common()),
        "recovered_by_pair": dict(Counter(row["pair"] for row in recovered).most_common()),
    }


def build_audit(atlas: dict[str, Any]) -> dict[str, Any]:
    misses = list(atlas.get("misses", []))
    rows, summary = summarize_cross_macro_misses(misses)
    baseline_miss_count = int(atlas.get("summary", {}).get("miss_count", len(misses)))
    simulations = {
        policy: simulate_policy(policy, rows, baseline_miss_count)
        for policy in POLICY_CATEGORIES
    }
    return {
        "schema": "semantic_cross_macro_audit_v1",
        "baseline_miss_count": baseline_miss_count,
        "summary": summary,
        "rows": rows,
        "policy_simulations": simulations,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            written = dict(row)
            written["candidate_faces"] = " ".join(str(item) for item in row.get("candidate_faces", []))
            writer.writerow(written)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    lines = [
        "# v5.19 Cross-Macro Semantic Audit",
        "",
        "## Summary",
        "",
        f"- baseline_miss_count: `{audit['baseline_miss_count']}`",
        f"- cross_macro_compatible_count: `{summary['cross_macro_compatible_count']}`",
        "",
        "## Review Categories",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for category, count in summary["by_category"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Macro Pairs", "", "| Pair | Count |", "|------|-------|"])
    for pair, count in summary["by_pair"].items():
        lines.append(f"| {pair} | {count} |")
    lines.extend(["", "## Review Rows", "", "| Sample | Truth | Candidate | Category | Truth Area | Candidate Area | Rel Error | Faces |", "|--------|-------|-----------|----------|------------|----------------|-----------|-------|"])
    for row in audit["rows"]:
        lines.append(
            f"| {row['sample']} | {row['truth_macro']} | {row['candidate_macro']} | "
            f"{row['review_category']} | {row['truth_area']:.2f} | {row['candidate_area']:.2f} | "
            f"{row['relative_error']:.4f} | {' '.join(str(item) for item in row['candidate_faces'])} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_policy_simulation(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# v5.19 Semantic Policy Simulation",
        "",
        "| Policy | Baseline Misses | Recovered | Remaining Misses |",
        "|--------|-----------------|-----------|------------------|",
    ]
    for result in audit["policy_simulations"].values():
        lines.append(
            f"| {result['policy']} | {result['baseline_miss_count']} | "
            f"{result['recovered_count']} | {result['remaining_miss_count']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def decide_next_step(audit: dict[str, Any]) -> tuple[str, str]:
    summary = audit["summary"]
    category_counts = summary.get("by_category", {})
    cross_count = int(summary.get("cross_macro_compatible_count", 0))
    if cross_count <= 0:
        return "return_to_proposal_generation_repair", "No cross-macro compatible misses remain to audit."
    hole_count = int(category_counts.get("hole_subfeature_vs_composite_hole", 0))
    recess_count = int(category_counts.get("rectangular_recess_slot_vs_pocket", 0))
    transition_count = int(category_counts.get("transition_vs_nontransition_conflict", 0))
    rib_count = int(category_counts.get("thinwall_rib_vs_boss_or_web", 0))
    if hole_count + recess_count >= max(transition_count, rib_count) and (hole_count + recess_count) / cross_count >= 0.40:
        return (
            "define_safe_semantic_evaluation_policy",
            "Most cross-macro compatible misses are hole-composite or rectangular recess granularity conflicts. Define an explicit evaluation policy before changing proposer code.",
        )
    if transition_count >= max(hole_count + recess_count, rib_count) and transition_count / cross_count >= 0.30:
        return (
            "inspect_transition_truth_and_face_roles",
            "Transition conflicts dominate. Review fillet/chamfer truth rows and face roles before changing transition proposers.",
        )
    if rib_count >= max(hole_count + recess_count, transition_count) and rib_count / cross_count >= 0.30:
        return (
            "inspect_rib_boss_semantic_boundary",
            "Rib/boss conflicts dominate. Review whether the truth should mark these as rib, rectangular_boss, or context/background.",
        )
    return (
        "manual_semantic_review_before_code_changes",
        "Cross-macro compatible misses are heterogeneous. Manual semantic review should pick which policy is valid for the paper.",
    )


def write_decision(path: Path, audit: dict[str, Any]) -> None:
    recommendation, reason = decide_next_step(audit)
    lines = [
        "# v5.19 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Next Action",
        "",
        "If the recommendation is `define_safe_semantic_evaluation_policy`, the next implementation should add a separate evaluation-only semantic matching policy and compare strict vs policy-aware metrics. It must not change the inference output labels.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-macro compatible proposal misses.")
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(load_json(args.atlas))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "cross_macro_semantic_audit.json", audit)
    write_csv(args.output_dir / "cross_macro_semantic_audit.csv", audit["rows"])
    write_markdown(args.output_dir / "cross_macro_semantic_audit.md", audit)
    write_policy_simulation(args.output_dir / "semantic_policy_simulation.md", audit)
    write_decision(args.output_dir / "v5_19_decision.md", audit)
    print(json.dumps({"summary": audit["summary"], "decision": decide_next_step(audit)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_cross_macro_audit -v
```

Expected:

```text
OK
```

## 7. Task 3: Generate v5.19 Semantic Audit Reports

**Files:**

- Generate: `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.json`
- Generate: `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.csv`
- Generate: `output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.md`
- Generate: `output_real_step_s3_v5_19_semantic_audit/reports/semantic_policy_simulation.md`
- Generate: `output_real_step_s3_v5_19_semantic_audit/reports/v5_19_decision.md`

- [ ] **Step 1: Run audit script**

Run:

```powershell
W:\python\envs\feagpt\python.exe semantic_cross_macro_audit.py `
  --atlas output_real_step_s3_v5_18_proposal_miss_atlas\reports\v5_13_proposal_miss_atlas.json `
  --output-dir output_real_step_s3_v5_19_semantic_audit\reports
```

Expected console output:

```text
"summary"
"decision"
```

- [ ] **Step 2: Inspect semantic audit**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_19_semantic_audit\reports\cross_macro_semantic_audit.md
```

Expected:

```text
The report contains Summary, Review Categories, Macro Pairs, and Review Rows.
```

- [ ] **Step 3: Inspect policy simulation**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_19_semantic_audit\reports\semantic_policy_simulation.md
```

Expected:

```text
Rows for strict_current, safe_hole_composite, safe_hole_and_rectangular_recess, review_all_cross_macro_compatible.
```

- [ ] **Step 4: Inspect decision**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_19_semantic_audit\reports\v5_19_decision.md
```

Expected:

```text
One of:
define_safe_semantic_evaluation_policy
inspect_transition_truth_and_face_roles
inspect_rib_boss_semantic_boundary
manual_semantic_review_before_code_changes
return_to_proposal_generation_repair
```

## 8. Task 4: Add User Review Instructions

**Files:**

- Create: `output_real_step_s3_v5_19_semantic_audit/reports/user_semantic_review_instructions.md`

- [ ] **Step 1: Create user-facing review document**

Create `output_real_step_s3_v5_19_semantic_audit/reports/user_semantic_review_instructions.md` with this content after the reports are generated:

```markdown
# v5.19 User Semantic Review Instructions

Please review `cross_macro_semantic_audit.csv`.

You do not need to inspect every STEP face manually. Focus on the rows grouped by `review_category`:

1. `hole_subfeature_vs_composite_hole`

   Decide whether a composite hole prediction such as `counterbore_hole`, `countersink_hole`, or `stepped_hole_round` should count as satisfying a truth row labelled `through_hole_round` when their areas match.

2. `rectangular_recess_slot_vs_pocket`

   Decide whether `slot_rect` and `pocket_rect` should be separate strict classes in evaluation, or whether they may be counted as the same CAE recess-selection family when area and face count match.

3. `thinwall_rib_vs_boss_or_web`

   Decide whether these rows are truly rib/web targets or whether the truth labels should be changed to boss/context/background.

4. `transition_vs_nontransition_conflict`

   Do not auto-accept these. These likely need manual review because fillet/chamfer should remain independent first-paper target classes, and many of these may be face-role or truth-table issues.

For each row you review, add one value in a new `user_decision` column:

- `accept_family_match`
- `keep_strict`
- `truth_label_should_change`
- `model_prediction_wrong`
- `needs_face_level_review`

The next agent should not change inference labels until this review is complete.
```

- [ ] **Step 2: Confirm the review file exists**

Run:

```powershell
Test-Path output_real_step_s3_v5_19_semantic_audit\reports\user_semantic_review_instructions.md
```

Expected:

```text
True
```

## 9. Final Verification

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_cross_macro_audit tests.test_proposal_miss_atlas -v
```

Expected:

```text
OK
```

## 10. What the Implementing Agent Must Report

The final response must include:

```text
1. Test command and result.
2. cross_macro_compatible_count.
3. by_category counts.
4. by_pair top 10.
5. policy simulation table.
6. exact recommendation from v5_19_decision.md.
7. whether any behavior/inference/training files were modified.
8. whether user review is needed before code-changing repairs.
```

Behavior/inference/training files that should not be modified:

```text
utils/object_candidate_proposer.py
utils/instance_candidate_features.py
utils/instance_candidate_postprocess.py
utils/topology_postprocess.py
train_gnn_baseline.py
train_candidate_topology_classifier.py
object_proposal_recall_report.py
proposal_miss_atlas.py
```

## 11. How I Will Review This Work

I will check:

```text
tests/test_semantic_cross_macro_audit.py
semantic_cross_macro_audit.py
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.md
output_real_step_s3_v5_19_semantic_audit/reports/cross_macro_semantic_audit.csv
output_real_step_s3_v5_19_semantic_audit/reports/semantic_policy_simulation.md
output_real_step_s3_v5_19_semantic_audit/reports/v5_19_decision.md
output_real_step_s3_v5_19_semantic_audit/reports/user_semantic_review_instructions.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_semantic_cross_macro_audit tests.test_proposal_miss_atlas -v
```

I will not accept a code-changing proposer repair until this semantic audit says whether the dominant cross-macro compatible rows should be counted as valid CAE-family matches, corrected in truth labels, or treated as model/proposer errors.

## 12. English Handoff Summary

v5.18 showed the largest persistent no-proposal failure class is `cross_macro_area_compatible`, not a pure absence of candidates. Implement v5.19 as a diagnostic semantic audit. Group exact/near exact cross-macro matches, classify them into CAE-semantic categories, simulate strict and family-aware evaluation policies, and produce a user review CSV. Do not change inference, training, proposer, or topology code in this iteration.
