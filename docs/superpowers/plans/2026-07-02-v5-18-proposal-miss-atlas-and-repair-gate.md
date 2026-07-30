# v5.18 Proposal Miss Atlas and Repair Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the persistent `no_area_compatible_proposal = 65` failures on real S3 and decide whether the next code change should repair proposal generation, evaluation/truth matching, or final export interaction.

**Architecture:** Do not modify model training, GNN inference, STEP generation, topology postprocess, or candidate classifiers. Add a diagnostic script that reads existing proposal-recall JSON reports, inspects every `no_area_compatible_proposal` truth entry, finds nearest same-macro and cross-macro candidates by area, classifies the miss cause, and writes a macro/part/source repair gate report.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, JSON, Markdown, existing `utils.area_truth_normalization.normalize_truth_macro`, existing v5.13/v5.17 proposal recall reports.

---

## 0. Review of v5.17 and Project-Level Direction

v5.17 was implemented correctly:

- `utils/instance_candidate_features.py` now exposes role-aware loop/cylinder wall features and topology reason/evidence features.
- `tests/test_instance_candidate_features.py` includes inner/outer cylindrical wall tests and topology reason tests.
- `compare_candidate_classifier_recall_reports.py` supports repeated named reports.
- Verification passed:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_build_instance_candidate_dataset tests.test_train_candidate_topology_classifier tests.test_compare_candidate_classifier_recall_reports -v
```

Observed:

```text
Ran 26 tests
OK
```

The method was genuinely implemented. Evidence:

- `feature_schema.json` contains `boundary_unique_circle_count`, `boundary_loop_role_inner_count`, `boundary_cylinder_role_inner_count`, `candidate_boundary_inner_wall_face_fraction`, `topology_reason_count`, and `topology_evidence_count`.
- v5.17 feature importance includes role/topology features such as `topology_evidence_count`, `candidate_boundary_inner_loop_face_count`, `internal_cylinder_role_outer_ratio`, and `candidate_boundary_inner_wall_face_fraction`.

But v5.17 should not be promoted:

```text
Version           Proposal  Classified  Accepted  Final-survival  No proposal  Classifier wrong/rejected
v5.13             0.6814    0.5343      0.4804    0.5196          65           32
v5.14             0.6814    0.4069      0.3971    0.4020          65           56
v5.16             0.6814    0.3971      0.3873    0.3922          65           58
v5.17_plain       0.6814    0.4902      0.4804    0.4657          65           39
v5.17_rebalanced 0.6814    0.4755      0.4755    0.4510          65           42
```

Interpretation:

- v5.17 is a real partial gain over v5.14/v5.16.
- v5.17 does not beat v5.13.
- The proposal-stage recall is unchanged at `0.6814`.
- `no_area_compatible_proposal = 65` has remained unchanged across v5.13/v5.14/v5.16/v5.17.
- The next iteration must stop candidate-classifier-only tuning and inspect proposal misses directly.

## 1. Scope Control

### Create

- `proposal_miss_atlas.py`
- `tests/test_proposal_miss_atlas.py`
- `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.json`
- `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.md`
- `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.json`
- `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.md`
- `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_repair_gate.md`

### Do Not Modify

- `utils/object_candidate_proposer.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- `utils/topology_postprocess.py`
- `train_gnn_baseline.py`
- `train_candidate_topology_classifier.py`
- any model checkpoint
- any existing v5.13/v5.14/v5.16/v5.17 report directory

This is diagnostic and planning only. It must not change model behavior.

## 2. Fixed Inputs

Use these two reports:

```text
v5.13 current-best report:
output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.json

v5.17 latest role-aware report:
output_real_step_s3_v5_17_candidate_features_plain/reports/proposal_recall_s3_v5_17_plain.json
```

The script must read the `best_by_final_survival_recall` section from each report.

## 3. Failure Taxonomy

Every `truth_failures` entry with `failure_stage == "no_area_compatible_proposal"` must receive exactly one `miss_cause`:

```text
no_same_macro_candidates
same_macro_area_near_miss
same_macro_split_or_merge_suspected
cross_macro_area_compatible
cross_macro_area_near_miss
no_related_candidate
```

Definitions:

- `no_same_macro_candidates`: no candidate has the same normalized macro as the truth entry.
- `same_macro_area_near_miss`: same-macro candidate exists, but the closest same-macro candidate is outside tolerance and has relative area error <= 0.25.
- `same_macro_split_or_merge_suspected`: same-macro candidates exist and the sum of the closest same-macro candidate areas can match the truth within tolerance, but no single candidate matches.
- `cross_macro_area_compatible`: a different-macro candidate has area compatible with the truth. This suggests macro confusion or truth/evaluation semantic ambiguity.
- `cross_macro_area_near_miss`: no compatible candidate exists, but a different-macro candidate has relative area error <= 0.25.
- `no_related_candidate`: none of the above applies.

Compatibility uses the same tolerances as the proposal recall reports:

```text
absolute_area_tolerance = report["absolute_area_tolerance"]
relative_area_tolerance = report["relative_area_tolerance"]
```

## 4. Task 1: Add Unit Tests for Proposal Miss Classification

**Files:**

- Create: `tests/test_proposal_miss_atlas.py`
- Create: `proposal_miss_atlas.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_proposal_miss_atlas.py`:

```python
from __future__ import annotations

import unittest

from proposal_miss_atlas import (
    area_compatible,
    classify_no_proposal_failure,
    nearest_candidates,
    summarize_misses,
)


class ProposalMissAtlasTest(unittest.TestCase):
    def test_area_compatible_uses_absolute_or_relative_tolerance(self):
        self.assertTrue(area_compatible(100.0, 109.0, absolute_tolerance=10.0, relative_tolerance=0.02))
        self.assertTrue(area_compatible(1000.0, 1015.0, absolute_tolerance=10.0, relative_tolerance=0.02))
        self.assertFalse(area_compatible(100.0, 115.0, absolute_tolerance=10.0, relative_tolerance=0.02))

    def test_nearest_candidates_sorts_by_relative_error(self):
        candidates = [
            {"proposal_macro": "rib", "area": 80.0, "source": "same_instance", "face_indices": [1]},
            {"proposal_macro": "rib", "area": 105.0, "source": "topology_rule", "face_indices": [2]},
            {"proposal_macro": "fillet", "area": 101.0, "source": "topology_rule", "face_indices": [3]},
        ]

        nearest = nearest_candidates(100.0, candidates, macro="rib", limit=2)

        self.assertEqual([item["area"] for item in nearest], [105.0, 80.0])
        self.assertAlmostEqual(nearest[0]["relative_error"], 0.05)

    def test_classifies_no_same_macro_candidates(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "fillet", "area": 10.0, "source": "topology_rule", "face_indices": [7]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "no_same_macro_candidates")
        self.assertEqual(result["same_macro_candidate_count"], 0)

    def test_classifies_same_macro_area_near_miss(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "rib", "area": 120.0, "source": "same_instance", "face_indices": [1, 2]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "same_macro_area_near_miss")
        self.assertAlmostEqual(result["nearest_same_macro"][0]["relative_error"], 0.20)

    def test_classifies_same_macro_split_or_merge_suspected(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [
            {"proposal_macro": "rib", "area": 45.0, "source": "same_instance", "face_indices": [1]},
            {"proposal_macro": "rib", "area": 55.0, "source": "same_instance", "face_indices": [2]},
        ]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "same_macro_split_or_merge_suspected")
        self.assertAlmostEqual(result["same_macro_area_sum"], 100.0)

    def test_classifies_cross_macro_area_compatible(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "web", "area": 101.0, "source": "same_instance", "face_indices": [5]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "cross_macro_area_compatible")
        self.assertEqual(result["nearest_cross_macro"][0]["proposal_macro"], "web")

    def test_summarize_misses_counts_macro_part_and_cause(self):
        misses = [
            {"sample": "part1", "macro": "rib", "miss_cause": "no_same_macro_candidates"},
            {"sample": "part1", "macro": "rib", "miss_cause": "same_macro_area_near_miss"},
            {"sample": "part2", "macro": "fillet", "miss_cause": "no_same_macro_candidates"},
        ]

        summary = summarize_misses(misses)

        self.assertEqual(summary["miss_count"], 3)
        self.assertEqual(summary["by_macro"]["rib"], 2)
        self.assertEqual(summary["by_part"]["part1"], 2)
        self.assertEqual(summary["by_cause"]["no_same_macro_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_proposal_miss_atlas -v
```

Expected before implementation:

```text
ERROR
```

The failure should mention missing `proposal_miss_atlas`.

## 5. Task 2: Implement Proposal Miss Atlas Helpers

**Files:**

- Modify: `proposal_miss_atlas.py`

- [ ] **Step 1: Add helper implementation**

Create `proposal_miss_atlas.py` with this content:

```python
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from utils.area_truth_normalization import normalize_truth_macro


MISS_CAUSES = (
    "no_same_macro_candidates",
    "same_macro_area_near_miss",
    "same_macro_split_or_merge_suspected",
    "cross_macro_area_compatible",
    "cross_macro_area_near_miss",
    "no_related_candidate",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def normalized_macro(value: Any, *, sample: str | None = None, area: float | None = None, face_count: int | None = None) -> str:
    return normalize_truth_macro(
        str(value or ""),
        sample_id=sample,
        area=area,
        face_count=face_count,
        truth_source_name="Feature_Comparison_Table.xlsx",
    )


def area_compatible(
    truth_area: float,
    candidate_area: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    truth_area = finite_float(truth_area)
    candidate_area = finite_float(candidate_area)
    diff = abs(truth_area - candidate_area)
    if diff <= absolute_tolerance:
        return True
    denominator = max(abs(truth_area), 1e-9)
    return diff / denominator <= relative_tolerance


def relative_error(truth_area: float, candidate_area: float) -> float:
    truth_area = finite_float(truth_area)
    candidate_area = finite_float(candidate_area)
    return abs(truth_area - candidate_area) / max(abs(truth_area), 1e-9)


def candidate_macro(candidate: dict[str, Any]) -> str:
    for key in ("proposal_macro", "classified_macro", "macro"):
        if candidate.get(key) is not None:
            return normalized_macro(candidate[key])
    return "unknown"


def candidate_area(candidate: dict[str, Any]) -> float:
    return finite_float(candidate.get("area", 0.0))


def compact_candidate(candidate: dict[str, Any], truth_area: float) -> dict[str, Any]:
    area = candidate_area(candidate)
    return {
        "source": str(candidate.get("source", "unknown")),
        "proposal_macro": candidate_macro(candidate),
        "classified_macro": normalized_macro(candidate.get("classified_macro", candidate.get("proposal_macro", ""))),
        "action": str(candidate.get("action", "")),
        "confidence": candidate.get("confidence"),
        "face_indices": list(candidate.get("face_indices", [])),
        "face_count": int(candidate.get("face_count", len(candidate.get("face_indices", [])))),
        "area": area,
        "absolute_error": abs(finite_float(truth_area) - area),
        "relative_error": relative_error(truth_area, area),
    }


def nearest_candidates(
    truth_area: float,
    candidates: list[dict[str, Any]],
    *,
    macro: str | None = None,
    exclude_macro: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        macro_value = candidate_macro(candidate)
        if macro is not None and macro_value != macro:
            continue
        if exclude_macro is not None and macro_value == exclude_macro:
            continue
        rows.append(compact_candidate(candidate, truth_area))
    rows.sort(key=lambda item: (item["relative_error"], item["absolute_error"], item["proposal_macro"]))
    return rows[:limit]


def closest_same_macro_area_sum(
    truth_area: float,
    same_macro_candidates: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_group_size: int = 4,
) -> tuple[float, list[dict[str, Any]]]:
    best_sum = 0.0
    best_group: list[dict[str, Any]] = []
    candidates = sorted(
        same_macro_candidates,
        key=lambda item: relative_error(truth_area, candidate_area(item)),
    )[:10]
    for group_size in range(2, min(max_group_size, len(candidates)) + 1):
        for group in combinations(candidates, group_size):
            area_sum = sum(candidate_area(item) for item in group)
            if not best_group or relative_error(truth_area, area_sum) < relative_error(truth_area, best_sum):
                best_sum = area_sum
                best_group = list(group)
            if area_compatible(
                truth_area,
                area_sum,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ):
                return area_sum, [compact_candidate(item, truth_area) for item in group]
    return best_sum, [compact_candidate(item, truth_area) for item in best_group]


def classify_no_proposal_failure(
    failure: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    sample = str(failure.get("sample", ""))
    truth_area = finite_float(failure.get("area", 0.0))
    face_count = int(failure.get("face_count", 0) or 0)
    macro = normalized_macro(failure.get("macro", ""), sample=sample, area=truth_area, face_count=face_count)
    same_macro_candidates = [candidate for candidate in candidates if candidate_macro(candidate) == macro]
    cross_macro_candidates = [candidate for candidate in candidates if candidate_macro(candidate) != macro]
    nearest_same = nearest_candidates(truth_area, same_macro_candidates, limit=5)
    nearest_cross = nearest_candidates(truth_area, cross_macro_candidates, limit=5)
    compatible_cross = [
        item for item in nearest_cross
        if area_compatible(
            truth_area,
            item["area"],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    ]
    same_area_sum, same_area_group = closest_same_macro_area_sum(
        truth_area,
        same_macro_candidates,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    if not same_macro_candidates:
        miss_cause = "no_same_macro_candidates"
    elif nearest_same and nearest_same[0]["relative_error"] <= 0.25:
        miss_cause = "same_macro_area_near_miss"
    elif same_area_group and area_compatible(
        truth_area,
        same_area_sum,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        miss_cause = "same_macro_split_or_merge_suspected"
    elif compatible_cross:
        miss_cause = "cross_macro_area_compatible"
    elif nearest_cross and nearest_cross[0]["relative_error"] <= 0.25:
        miss_cause = "cross_macro_area_near_miss"
    else:
        miss_cause = "no_related_candidate"

    return {
        "sample": sample,
        "truth_index": failure.get("truth_index"),
        "macro": macro,
        "area": truth_area,
        "face_count": face_count,
        "failure_stage": failure.get("failure_stage"),
        "miss_cause": miss_cause,
        "candidate_count": len(candidates),
        "same_macro_candidate_count": len(same_macro_candidates),
        "cross_macro_candidate_count": len(cross_macro_candidates),
        "nearest_same_macro": nearest_same,
        "nearest_cross_macro": nearest_cross,
        "same_macro_area_sum": same_area_sum,
        "same_macro_area_group": same_area_group,
    }


def summarize_misses(misses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "miss_count": len(misses),
        "by_macro": dict(Counter(item["macro"] for item in misses).most_common()),
        "by_part": dict(Counter(item["sample"] for item in misses).most_common()),
        "by_cause": dict(Counter(item["miss_cause"] for item in misses).most_common()),
        "by_macro_cause": dict(
            Counter(f"{item['macro']}|{item['miss_cause']}" for item in misses).most_common()
        ),
        "by_part_macro": dict(
            Counter(f"{item['sample']}|{item['macro']}" for item in misses).most_common()
        ),
    }


def build_atlas(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    absolute_tolerance = finite_float(report.get("absolute_area_tolerance", 10.0), 10.0)
    relative_tolerance = finite_float(report.get("relative_area_tolerance", 0.02), 0.02)
    misses: list[dict[str, Any]] = []
    for sample in best.get("samples", []):
        sample_name = str(sample.get("sample", ""))
        candidates = list(sample.get("candidates", []))
        for failure in sample.get("truth_failures", []):
            if failure.get("failure_stage") != "no_area_compatible_proposal":
                continue
            enriched_failure = dict(failure)
            enriched_failure["sample"] = sample_name
            misses.append(
                classify_no_proposal_failure(
                    enriched_failure,
                    candidates,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )

    return {
        "schema": "proposal_miss_atlas_v1",
        "source_report": report.get("classifier", report.get("schema", "unknown")),
        "absolute_area_tolerance": absolute_tolerance,
        "relative_area_tolerance": relative_tolerance,
        "summary": summarize_misses(misses),
        "misses": misses,
    }


def recommend_next_action(summary: dict[str, Any]) -> tuple[str, str]:
    miss_count = int(summary.get("miss_count", 0))
    by_cause = summary.get("by_cause", {})
    no_same = int(by_cause.get("no_same_macro_candidates", 0))
    split_merge = int(by_cause.get("same_macro_split_or_merge_suspected", 0))
    cross_compatible = int(by_cause.get("cross_macro_area_compatible", 0))
    near_miss = int(by_cause.get("same_macro_area_near_miss", 0))
    if miss_count == 0:
        return "no_proposal_repair_needed", "No no-area-compatible proposal failures remain."
    if no_same >= max(split_merge, cross_compatible, near_miss) and no_same / miss_count >= 0.35:
        return "targeted_proposal_generation_repair", "Most misses have no same-macro candidate, so repair proposal generators for dominant missed macros."
    if split_merge >= max(no_same, cross_compatible, near_miss) and split_merge / miss_count >= 0.25:
        return "candidate_merge_split_repair", "Many same-macro candidates exist only as split/merged fragments, so repair candidate assembly before adding new feature types."
    if cross_compatible >= max(no_same, split_merge, near_miss) and cross_compatible / miss_count >= 0.25:
        return "truth_or_macro_semantic_audit", "Many misses have area-compatible cross-macro candidates, so inspect macro semantics and truth labels before coding proposal repairs."
    if near_miss / miss_count >= 0.25:
        return "area_tolerance_or_boundary_refinement_audit", "Many misses are close in area but outside tolerance, so inspect area truth precision and boundary grouping."
    return "manual_review_then_targeted_repair", "No single miss cause dominates; manually review the top macro/part clusters before changing proposal code."


def write_markdown(path: Path, atlas: dict[str, Any], title: str) -> None:
    summary = atlas["summary"]
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- miss_count: `{summary['miss_count']}`",
        f"- absolute_area_tolerance: `{atlas['absolute_area_tolerance']}`",
        f"- relative_area_tolerance: `{atlas['relative_area_tolerance']}`",
        "",
        "## Miss Cause Counts",
        "",
        "| Cause | Count |",
        "|-------|-------|",
    ]
    for cause, count in summary["by_cause"].items():
        lines.append(f"| {cause} | {count} |")
    lines.extend(["", "## Macro Counts", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Part Counts", "", "| Part | Count |", "|------|-------|"])
    for part, count in summary["by_part"].items():
        lines.append(f"| {part} | {count} |")
    lines.extend(["", "## Top Miss Details", "", "| Sample | Macro | Area | Faces | Cause | Same Macro Candidates | Nearest Same | Nearest Cross |", "|--------|-------|------|-------|-------|-----------------------|--------------|---------------|"])
    for miss in atlas["misses"][:80]:
        nearest_same = miss["nearest_same_macro"][0] if miss["nearest_same_macro"] else {}
        nearest_cross = miss["nearest_cross_macro"][0] if miss["nearest_cross_macro"] else {}
        nearest_same_text = f"{nearest_same.get('proposal_macro', '')}:{nearest_same.get('area', ''):.2f}:{nearest_same.get('relative_error', ''):.3f}" if nearest_same else ""
        nearest_cross_text = f"{nearest_cross.get('proposal_macro', '')}:{nearest_cross.get('area', ''):.2f}:{nearest_cross.get('relative_error', ''):.3f}" if nearest_cross else ""
        lines.append(
            f"| {miss['sample']} | {miss['macro']} | {miss['area']:.2f} | {miss['face_count']} | "
            f"{miss['miss_cause']} | {miss['same_macro_candidate_count']} | {nearest_same_text} | {nearest_cross_text} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repair_gate(path: Path, named_atlases: list[tuple[str, dict[str, Any]]]) -> None:
    lines = [
        "# v5.18 Repair Gate",
        "",
        "## Recommendation By Report",
        "",
        "| Report | Miss Count | Recommendation | Reason |",
        "|--------|------------|----------------|--------|",
    ]
    for name, atlas in named_atlases:
        recommendation, reason = recommend_next_action(atlas["summary"])
        lines.append(f"| {name} | {atlas['summary']['miss_count']} | {recommendation} | {reason} |")
    lines.extend([
        "",
        "## Required Next Step",
        "",
        "Use the v5.13 atlas as the primary decision source because v5.13 is still the current best real-S3 baseline. Use v5.17_plain only as supporting evidence for whether role-aware features changed the failure mix.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report must use name=path format, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"report must use name=path format, got {value!r}")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-compatible-proposal failure atlas.")
    parser.add_argument("--report", action="append", required=True, help="Named report in name=path format.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_atlases: list[tuple[str, dict[str, Any]]] = []
    for name, report_path in [parse_named_report(item) for item in args.report]:
        atlas = build_atlas(load_json(report_path))
        named_atlases.append((name, atlas))
        write_json(args.output_dir / f"{name}_proposal_miss_atlas.json", atlas)
        write_markdown(args.output_dir / f"{name}_proposal_miss_atlas.md", atlas, f"{name} Proposal Miss Atlas")
    write_repair_gate(args.output_dir / "v5_18_repair_gate.md", named_atlases)
    print(json.dumps({name: atlas["summary"] for name, atlas in named_atlases}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run helper tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_proposal_miss_atlas -v
```

Expected:

```text
OK
```

## 6. Task 3: Generate v5.18 Proposal Miss Atlases

**Files:**

- Generate: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.json`
- Generate: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.md`
- Generate: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.json`
- Generate: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.md`
- Generate: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_repair_gate.md`

- [ ] **Step 1: Run atlas script**

Run:

```powershell
W:\python\envs\feagpt\python.exe proposal_miss_atlas.py `
  --report v5_13=output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json `
  --report v5_17_plain=output_real_step_s3_v5_17_candidate_features_plain\reports\proposal_recall_s3_v5_17_plain.json `
  --output-dir output_real_step_s3_v5_18_proposal_miss_atlas\reports
```

Expected console output:

```text
"v5_13"
"v5_17_plain"
"miss_count"
```

- [ ] **Step 2: Inspect repair gate**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_18_proposal_miss_atlas\reports\v5_18_repair_gate.md
```

Expected:

```text
The report contains one recommendation row for v5_13 and one row for v5_17_plain.
```

- [ ] **Step 3: Inspect v5.13 atlas**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_18_proposal_miss_atlas\reports\v5_13_proposal_miss_atlas.md -TotalCount 160
```

Expected:

```text
The report lists Miss Cause Counts, Macro Counts, Part Counts, and Top Miss Details.
```

## 7. Task 4: Write Human-Readable Next-Step Recommendation

**Files:**

- Create: `output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_next_step_recommendation.md`

- [ ] **Step 1: Create recommendation writer**

Append this function to `proposal_miss_atlas.py` before `parse_args()`:

```python
def write_next_step_recommendation(path: Path, primary_name: str, atlas: dict[str, Any]) -> None:
    recommendation, reason = recommend_next_action(atlas["summary"])
    summary = atlas["summary"]
    top_macros = list(summary["by_macro"].items())[:8]
    top_parts = list(summary["by_part"].items())[:8]
    lines = [
        "# v5.18 Next Step Recommendation",
        "",
        f"Primary report: `{primary_name}`",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Top Missed Macros",
        "",
        "| Macro | Count |",
        "|-------|-------|",
    ]
    for macro, count in top_macros:
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Top Missed Parts", "", "| Part | Count |", "|------|-------|"])
    for part, count in top_parts:
        lines.append(f"| {part} | {count} |")
    lines.extend([
        "",
        "## Execution Rule For Next Agent",
        "",
        "If the recommendation is `targeted_proposal_generation_repair`, write the next plan against `utils/object_candidate_proposer.py` and the topology candidate functions it imports. If the recommendation is `truth_or_macro_semantic_audit`, do not change proposer code until the area truth table or macro semantics are reviewed. If the recommendation is `candidate_merge_split_repair`, write the next plan against candidate assembly and same-instance grouping.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
```

In `main()`, after `write_repair_gate(...)`, add:

```python
    if named_atlases:
        write_next_step_recommendation(
            args.output_dir / "v5_18_next_step_recommendation.md",
            named_atlases[0][0],
            named_atlases[0][1],
        )
```

- [ ] **Step 2: Re-run atlas script**

Run:

```powershell
W:\python\envs\feagpt\python.exe proposal_miss_atlas.py `
  --report v5_13=output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json `
  --report v5_17_plain=output_real_step_s3_v5_17_candidate_features_plain\reports\proposal_recall_s3_v5_17_plain.json `
  --output-dir output_real_step_s3_v5_18_proposal_miss_atlas\reports
```

Expected:

```text
output_real_step_s3_v5_18_proposal_miss_atlas\reports\v5_18_next_step_recommendation.md exists
```

- [ ] **Step 3: Read recommendation**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_18_proposal_miss_atlas\reports\v5_18_next_step_recommendation.md
```

Expected:

```text
The file contains Recommendation, Reasoning, Top Missed Macros, Top Missed Parts, and Execution Rule For Next Agent.
```

## 8. Task 5: Final Verification

- [ ] **Step 1: Run all relevant tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_proposal_miss_atlas tests.test_object_proposal_recall_report tests.test_compare_candidate_classifier_recall_reports -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Confirm no behavior files were modified**

The final response must state whether any of these files were modified:

```text
utils/object_candidate_proposer.py
utils/instance_candidate_features.py
utils/instance_candidate_postprocess.py
utils/topology_postprocess.py
train_gnn_baseline.py
train_candidate_topology_classifier.py
```

Expected:

```text
None of them modified.
```

If one of them was modified, explain why and show the exact test evidence. This plan does not require modifying them.

## 9. What The Implementing Agent Must Report

The final response must include:

```text
1. Test command and result.
2. v5_13 miss_count and by_cause counts.
3. v5_13 top missed macros.
4. v5_13 top missed parts.
5. v5_17_plain miss_count and whether its failure mix differs from v5_13.
6. Exact recommendation from v5_18_repair_gate.md.
7. Exact recommendation from v5_18_next_step_recommendation.md.
8. Whether any behavior files were modified.
```

## 10. How I Will Review This Work

I will check:

```text
tests/test_proposal_miss_atlas.py
proposal_miss_atlas.py
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_13_proposal_miss_atlas.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_17_plain_proposal_miss_atlas.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_repair_gate.md
output_real_step_s3_v5_18_proposal_miss_atlas/reports/v5_18_next_step_recommendation.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_proposal_miss_atlas tests.test_object_proposal_recall_report tests.test_compare_candidate_classifier_recall_reports -v
```

I will accept the work only if it clearly separates no-proposal failures into same-macro absence, same-macro area near miss, split/merge suspicion, cross-macro compatible candidate, and no-related-candidate groups. The next code-changing plan should be based on the dominant v5.13 miss cause, not on intuition.

## 11. English Handoff Summary

v5.17 was real but not enough. Do not continue candidate-classifier-only tuning. Build a v5.18 proposal miss atlas from the existing v5.13 and v5.17 proposal recall JSON reports. Classify every `no_area_compatible_proposal` failure by nearby same-macro/cross-macro candidates and output a repair gate. This will decide whether the next implementation should repair proposal generation, candidate split/merge assembly, or truth/evaluation semantics.
