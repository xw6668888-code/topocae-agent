# v5.11.1 Diagnostic Macro Normalization Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the v5.11 proposal-recall diagnostic so area-truth macros and prediction macros are compared in the same canonical label space, then regenerate the s3 diagnostic report before deciding the next model/data change.

**Architecture:** This is an evaluation repair, not a model repair. Reuse `utils.area_truth_normalization.normalize_truth_macro` inside `object_proposal_recall_report.py`, add regression tests for all known s3 macro aliases, explicitly report skipped invalid threshold pairs, and write a corrected comparison report under a new output directory.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, existing JSON prediction/graph artifacts, existing `utils.area_truth_normalization`.

---

## 0. Review Context

The previous v5.11 agent created the expected files and reports:

- `object_proposal_recall_report.py`
- `tests/test_object_proposal_recall_report.py`
- `output_real_step_s3_v5_11_proposal_diagnostics/reports/proposal_recall_s3.json`
- `output_real_step_s3_v5_11_proposal_diagnostics/reports/proposal_recall_s3.md`
- `output_real_step_s3_v5_11_proposal_diagnostics/reports/v5_11_proposal_recall_decision.md`

Verification already performed:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Observed result:

```text
Ran 26 tests
OK
```

The code compiles.

However, v5.11 has two review findings:

1. **Major finding:** s3 truth macros are not normalized to the project prediction vocabulary.
   - Truth file contains `through_hole`, `blind_hole`, `stepped_hole`, `cylindrical_boss`, `rectangle_boss`.
   - Prediction files contain `through_hole_round`, `blind_hole_round`, `stepped_hole_round`, `cylinder_boss`, `rectangular_boss`.
   - Existing helper `utils.area_truth_normalization.normalize_truth_macro` already maps these aliases.
   - v5.11 `object_proposal_recall_report.py` currently does not use that helper.
   - Therefore the reported `no_area_compatible_proposal = 167` may be inflated by label-space mismatch.

2. **Minor finding:** the threshold grid input is 6 strong thresholds by 5 weak thresholds, which is 30 requested pairs. The implementation evaluates only 23 pairs because it skips `weak >= strong`, which is a valid constraint, but the report does not explicitly state skipped pairs.

Do not accept v5.11's `proposal_generation_fix` recommendation until this plan is executed.

## 1. Scope Control

### Files to Modify

- Modify: `object_proposal_recall_report.py`
- Modify: `tests/test_object_proposal_recall_report.py`

### Files to Create

- Create generated output directory: `output_real_step_s3_v5_11_1_macrofix_diagnostics`
- Create report: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.json`
- Create report: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.md`
- Create report: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/v5_11_1_macrofix_decision.md`
- Create report: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/v5_11_vs_v5_11_1_comparison.md`

### Files Not to Modify

Do not modify:

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `build_instance_candidate_dataset.py`
- `utils/instance_candidate_assembly.py`
- `utils/instance_candidate_postprocess.py`
- `run_gnn_inference.py`
- `prediction_to_feature_objects.py`
- any generator file under `features/`

This task fixes the evaluator only. If model or generator changes seem necessary, write that as the final recommendation after corrected evaluation.

## 2. Task 1: Add Macro Alias Normalization Tests

**Files:**

- Modify: `tests/test_object_proposal_recall_report.py`

- [ ] **Step 1: Add import-free tests for truth macro alias normalization**

Append this test class to `tests/test_object_proposal_recall_report.py` before the `if __name__ == "__main__":` block:

```python
class TestMacroNormalization(unittest.TestCase):
    """v5.11.1: area truth macros must be normalized to prediction vocabulary."""

    def test_truth_macro_aliases_are_normalized_to_prediction_vocabulary(self):
        from object_proposal_recall_report import normalize_truth_entries

        payload = {
            "schema": "cae_area_truth_table_v1",
            "source_xlsx": "W:\\python\\FEA\\dataset_generator\\test_part_s3\\Feature_Comparison_Table_s3_mm2.xlsx",
            "samples": {
                "part1": [
                    {"area": 100.0, "macro": "through_hole", "face_count": 1},
                    {"area": 101.0, "macro": "blind_hole", "face_count": 2},
                    {"area": 102.0, "macro": "stepped_hole", "face_count": 4},
                    {"area": 103.0, "macro": "cylindrical_boss", "face_count": 2},
                    {"area": 104.0, "macro": "rectangle_boss", "face_count": 5},
                    {"area": 105.0, "macro": "poctet_rect", "face_count": 5},
                ]
            },
        }

        normalized = normalize_truth_entries(payload)["part1"]
        macros = [entry["macro"] for entry in normalized]

        self.assertEqual(
            macros,
            [
                "through_hole_round",
                "blind_hole_round",
                "stepped_hole_round",
                "cylinder_boss",
                "rectangular_boss",
                "pocket_rect",
            ],
        )

    def test_candidate_and_final_prediction_macros_are_normalized_before_matching(self):
        from object_proposal_recall_report import candidate_matches_truth

        truth = {"macro": "through_hole_round", "area": 200.0, "face_count": 1}
        candidate = {
            "proposal_macro": "through_hole",
            "classified_macro": "through_hole",
            "action": "accepted_remap",
            "face_indices": [7],
            "face_count": 1,
            "area": 200.0,
        }
        final_prediction = {
            "faces": [
                {"idx": 7, "macro_feature": "through_hole"},
            ]
        }

        self.assertTrue(
            candidate_matches_truth(
                candidate,
                truth,
                "proposal",
                absolute_area_tolerance=10.0,
                relative_area_tolerance=0.02,
                final_prediction=final_prediction,
            )
        )
        self.assertTrue(
            candidate_matches_truth(
                candidate,
                truth,
                "final_survival",
                absolute_area_tolerance=10.0,
                relative_area_tolerance=0.02,
                final_prediction=final_prediction,
            )
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestMacroNormalization -v
```

Expected result before implementation:

```text
FAIL
```

The first test should show that `through_hole` remains `through_hole` instead of becoming `through_hole_round`.

## 3. Task 2: Normalize Truth and Candidate Macros in the Diagnostic

**Files:**

- Modify: `object_proposal_recall_report.py`

- [ ] **Step 1: Import the existing area-truth normalization helper**

Near the top of `object_proposal_recall_report.py`, after standard imports, add:

```python
from utils.area_truth_normalization import normalize_truth_macro
```

- [ ] **Step 2: Add a local wrapper for all diagnostic macro labels**

Add this function after `FAILURE_STAGES`:

```python
def normalize_macro_label(
    value: object,
    *,
    sample_id: str | None = None,
    area: float | None = None,
    face_count: int | None = None,
    truth_source_name: str | None = None,
) -> str:
    """Normalize macro labels into the project prediction vocabulary."""
    text = str(value).strip()
    if not text:
        return "unknown"
    return normalize_truth_macro(
        text,
        sample_id=sample_id,
        area=area,
        face_count=face_count,
        truth_source_name=truth_source_name,
    )
```

- [ ] **Step 3: Update `normalize_truth_entries` to use the wrapper**

Inside `normalize_truth_entries`, before iterating samples, read the source name:

```python
truth_source_name = None
if isinstance(payload, dict):
    truth_source_name = payload.get("source_xlsx") or payload.get("truth_source_name") or payload.get("source")
    if truth_source_name:
        truth_source_name = Path(str(truth_source_name)).name
```

Replace the current normalized append block with:

```python
area = _first_float(entry, ("area", "Area", "area_sum"), 0.0)
face_count = _first_int(entry, ("face_count", "number_of_faces", "number of faces", "num_faces"))
raw_macro = _first_str(entry, ("macro", "Macro", "feature", "feature_type"), "unknown")
normalized.append(
    {
        "sample": str(sample_name),
        "macro": normalize_macro_label(
            raw_macro,
            sample_id=str(sample_name),
            area=area,
            face_count=face_count,
            truth_source_name=truth_source_name,
        ),
        "raw_macro": str(raw_macro),
        "area": area,
        "face_count": face_count,
        "truth_index": idx,
    }
)
```

- [ ] **Step 4: Normalize candidate macros by stage**

Replace `candidate_macro_for_stage` with:

```python
def candidate_macro_for_stage(candidate: dict, stage: str) -> str | None:
    """Return the normalized macro label to use for a given pipeline stage."""
    if stage == "proposal":
        val = candidate.get("proposal_macro")
        return normalize_macro_label(val) if val is not None else None
    if stage in ("classified", "accepted", "final_survival"):
        val = candidate.get("classified_macro")
        if val is not None:
            label = normalize_macro_label(val)
            if label.lower() in (r.lower() for r in REJECT_LABELS):
                return None
            return label
        return None
    return None
```

- [ ] **Step 5: Normalize final prediction macro labels**

Inside `final_survives`, replace:

```python
face_macro = _first_str(face, ("macro_feature", "predicted_macro_feature", "predicted_macro", "macro"), "")
face_macro_by_idx[idx] = face_macro
```

with:

```python
face_macro = _first_str(face, ("macro_feature", "predicted_macro_feature", "predicted_macro", "macro"), "")
face_macro_by_idx[idx] = normalize_macro_label(face_macro)
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestMacroNormalization -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Run full diagnostic unit tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report -v
```

Expected:

```text
OK
```

## 4. Task 3: Make Threshold Pair Accounting Explicit

**Files:**

- Modify: `object_proposal_recall_report.py`
- Modify: `tests/test_object_proposal_recall_report.py`

- [ ] **Step 1: Add a threshold accounting test**

Append this method to `TestDatasetLevelResultBuilder`:

```python
    def test_threshold_pair_accounting_records_skipped_invalid_pairs(self):
        from object_proposal_recall_report import build_threshold_pairs

        pairs, skipped = build_threshold_pairs(
            strong_thresholds=[0.3, 0.4],
            weak_thresholds=[0.2, 0.3, 0.5],
        )

        self.assertEqual(
            pairs,
            [
                {"strong_threshold": 0.3, "weak_threshold": 0.2},
                {"strong_threshold": 0.4, "weak_threshold": 0.2},
                {"strong_threshold": 0.4, "weak_threshold": 0.3},
            ],
        )
        self.assertEqual(
            skipped,
            [
                {"strong_threshold": 0.3, "weak_threshold": 0.3, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
                {"strong_threshold": 0.3, "weak_threshold": 0.5, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
                {"strong_threshold": 0.4, "weak_threshold": 0.5, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
            ],
        )
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_threshold_pair_accounting_records_skipped_invalid_pairs -v
```

Expected before implementation:

```text
ERROR
```

because `build_threshold_pairs` does not exist.

- [ ] **Step 3: Implement threshold pair accounting**

Add this function near `parse_thresholds`:

```python
def build_threshold_pairs(
    strong_thresholds: list[float],
    weak_thresholds: list[float],
) -> tuple[list[dict], list[dict]]:
    """Return valid threshold pairs and skipped invalid pairs."""
    pairs: list[dict] = []
    skipped: list[dict] = []
    for strong in strong_thresholds:
        for weak in weak_thresholds:
            record = {"strong_threshold": strong, "weak_threshold": weak}
            if weak >= strong:
                skipped.append(
                    {
                        **record,
                        "reason": "weak_threshold_must_be_less_than_strong_threshold",
                    }
                )
                continue
            pairs.append(record)
    return pairs, skipped
```

- [ ] **Step 4: Use threshold pairs in `build_report`**

Replace the nested threshold loop in `build_report` with:

```python
threshold_pairs, skipped_threshold_pairs = build_threshold_pairs(strong_thresholds, weak_thresholds)

threshold_results = []
for pair in threshold_pairs:
    strong = pair["strong_threshold"]
    weak = pair["weak_threshold"]
    result = build_threshold_result(
        graph_by_sample=graph_by_sample,
        raw_prediction_by_sample=raw_prediction_by_sample,
        truth_by_sample=truth_by_sample,
        final_prediction_by_sample=final_prediction_by_sample,
        classifier_bundle=classifier_bundle,
        strong_threshold=strong,
        weak_threshold=weak,
        min_confidence=args.min_confidence,
        absolute_area_tolerance=args.absolute_area_tolerance,
        relative_area_tolerance=args.relative_area_tolerance,
    )
    threshold_results.append(result)
```

Add these fields to the report dict:

```python
"requested_threshold_pair_count": len(strong_thresholds) * len(weak_thresholds),
"evaluated_threshold_pair_count": len(threshold_results),
"skipped_threshold_pairs": skipped_threshold_pairs,
```

- [ ] **Step 5: Update markdown output with threshold accounting**

In `write_markdown`, after tolerance lines, add:

```python
lines.append("## Threshold Pair Accounting")
lines.append("")
lines.append(f"- Requested threshold pairs: {report.get('requested_threshold_pair_count', 0)}")
lines.append(f"- Evaluated valid threshold pairs: {report.get('evaluated_threshold_pair_count', len(report.get('threshold_results', [])))}")
lines.append(f"- Skipped invalid threshold pairs: {len(report.get('skipped_threshold_pairs', []))}")
lines.append("")
```

- [ ] **Step 6: Replace non-ASCII arrows and symbols in markdown**

In `write_markdown`, replace any literal arrow or warning symbol strings with ASCII-only text:

```python
lines.append("- Absolute area tolerance: " + str(report.get("absolute_area_tolerance")) + " mm^2")
lines.append("> Important: This validation is area-level only, not face IoU. Area truth cannot prove face-level correspondence.")
```

For best-threshold lines, use:

```python
lines.append(
    f"- Best by proposal recall: strong={bp.get('strong_threshold')}, weak={bp.get('weak_threshold')} -> proposal recall={bp.get('proposal_stage', {}).get('recall', 0.0):.4f}"
)
lines.append(
    f"- Best by final-survival recall: strong={bf.get('strong_threshold')}, weak={bf.get('weak_threshold')} -> final-survival recall={bf.get('final_survival_stage', {}).get('recall', 0.0):.4f}"
)
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report.TestDatasetLevelResultBuilder.test_threshold_pair_accounting_records_skipped_invalid_pairs -v
```

Expected:

```text
OK
```

## 5. Task 4: Regenerate Corrected v5.11.1 Reports

**Files:**

- Use existing raw predictions: `output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions`
- Create reports under: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports`

- [ ] **Step 1: Run corrected diagnostic**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\proposal_recall_s3_macrofix.json `
  --output-md output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\proposal_recall_s3_macrofix.md `
  --strong-thresholds 0.30,0.40,0.50,0.55,0.60,0.70 `
  --weak-thresholds 0.20,0.30,0.35,0.40,0.50 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

- [ ] **Step 2: Verify generated files**

Run:

```powershell
Test-Path output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\proposal_recall_s3_macrofix.json
Test-Path output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\proposal_recall_s3_macrofix.md
Test-Path output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\v5_11_1_macrofix_decision.md
```

Expected:

```text
True
True
True
```

If the decision memo is still written with the old name `v5_11_proposal_recall_decision.md`, rename the new file to:

```text
output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/v5_11_1_macrofix_decision.md
```

Do not overwrite old v5.11 reports.

- [ ] **Step 3: Verify JSON fields**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("schema", data.get("schema"))
print("requested", data.get("requested_threshold_pair_count"))
print("evaluated", data.get("evaluated_threshold_pair_count"))
print("skipped", len(data.get("skipped_threshold_pairs", [])))
bp = data.get("best_by_proposal_recall", {})
bf = data.get("best_by_final_survival_recall", {})
print("best_proposal", bp.get("strong_threshold"), bp.get("weak_threshold"), bp.get("proposal_stage", {}).get("recall"))
print("best_final", bf.get("strong_threshold"), bf.get("weak_threshold"), bf.get("final_survival_stage", {}).get("recall"))
print("truth_macros", sorted(bp.get("proposal_stage", {}).get("truth_by_macro", {}).keys()))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
schema cae_object_proposal_recall_report_v1
requested 30
evaluated 23
skipped 7
```

Expected truth macro names must include canonical names:

```text
blind_hole_round
cylinder_boss
rectangular_boss
stepped_hole_round
through_hole_round
```

Expected truth macro names must not include alias names:

```text
blind_hole
cylindrical_boss
rectangle_boss
stepped_hole
through_hole
```

## 6. Task 5: Write v5.11 vs v5.11.1 Comparison Report

**Files:**

- Create: `output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/v5_11_vs_v5_11_1_comparison.md`

- [ ] **Step 1: Add comparison helper script section using inline Python**

Run:

```powershell
@'
import json
from pathlib import Path

old_path = Path("output_real_step_s3_v5_11_proposal_diagnostics/reports/proposal_recall_s3.json")
new_path = Path("output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.json")
out_path = Path("output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/v5_11_vs_v5_11_1_comparison.md")

old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))

def best(report, key):
    return report.get(key, {})

def stage(report, key, stage_key):
    return best(report, key).get(stage_key, {})

old_bp = best(old, "best_by_proposal_recall")
new_bp = best(new, "best_by_proposal_recall")
old_bf = best(old, "best_by_final_survival_recall")
new_bf = best(new, "best_by_final_survival_recall")

old_prop = stage(old, "best_by_proposal_recall", "proposal_stage")
new_prop = stage(new, "best_by_proposal_recall", "proposal_stage")
old_final = stage(old, "best_by_final_survival_recall", "final_survival_stage")
new_final = stage(new, "best_by_final_survival_recall", "final_survival_stage")

old_fail = old_bp.get("failure_stage_counts", {})
new_fail = new_bp.get("failure_stage_counts", {})

lines = [
    "# v5.11 vs v5.11.1 Diagnostic Comparison",
    "",
    "## Purpose",
    "",
    "This report checks whether macro alias normalization changes the v5.11 conclusion.",
    "",
    "## Summary",
    "",
    "| Metric | v5.11 old | v5.11.1 macrofix |",
    "|--------|-----------|------------------|",
    f"| Proposal recall | {old_prop.get('recall', 0.0):.4f} | {new_prop.get('recall', 0.0):.4f} |",
    f"| Final-survival recall | {old_final.get('recall', 0.0):.4f} | {new_final.get('recall', 0.0):.4f} |",
    f"| No compatible proposal | {old_fail.get('no_area_compatible_proposal', 0)} | {new_fail.get('no_area_compatible_proposal', 0)} |",
    f"| Classifier wrong/rejected | {old_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} | {new_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} |",
    f"| Accepted but final changes/removes | {old_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} | {new_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} |",
    "",
    "## Canonical Truth Macros After v5.11.1",
    "",
]

truth_by_macro = new_prop.get("truth_by_macro", {})
matched_by_macro = new_prop.get("matched_by_macro", {})
lines.extend(["| Macro | Truth | Proposal matched | Missed |", "|-------|-------|------------------|--------|"])
for macro in sorted(truth_by_macro):
    truth_count = truth_by_macro[macro]
    matched_count = matched_by_macro.get(macro, 0)
    lines.append(f"| {macro} | {truth_count} | {matched_count} | {truth_count - matched_count} |")

lines.extend([
    "",
    "## Interpretation Rule",
    "",
    "- If no-compatible-proposal remains the dominant failure after macro normalization, the next task should fix proposal generation.",
    "- If classifier wrong/rejected becomes dominant, the next task should retrain or recalibrate the instance candidate classifier.",
    "- If accepted-but-final-changed becomes dominant, the next task should repair topology/export interaction.",
    "",
])

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(out_path)
'@ | W:\python\envs\feagpt\python.exe -
```

- [ ] **Step 2: Inspect comparison report**

Run:

```powershell
Get-Content output_real_step_s3_v5_11_1_macrofix_diagnostics\reports\v5_11_vs_v5_11_1_comparison.md -TotalCount 120
```

Expected:

- It shows old and corrected proposal recall.
- It shows old and corrected final-survival recall.
- It shows old and corrected failure-stage counts.
- It lists canonical truth macro names, not aliases.

## 7. Task 6: Final Verification

**Files:**

- Verify all touched and generated files.

- [ ] **Step 1: Run full test suite used by v5.11**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run compile check**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m py_compile object_proposal_recall_report.py tests\test_object_proposal_recall_report.py utils\area_truth_normalization.py
```

Expected:

```text
no output
```

- [ ] **Step 3: Check no alias labels remain in corrected truth macro coverage**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_11_1_macrofix_diagnostics/reports/proposal_recall_s3_macrofix.json")
data = json.loads(p.read_text(encoding="utf-8"))
macros = set(data["best_by_proposal_recall"]["proposal_stage"]["truth_by_macro"])
bad = macros & {"through_hole", "blind_hole", "stepped_hole", "cylindrical_boss", "rectangle_boss"}
print("bad_aliases", sorted(bad))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
bad_aliases []
```

## 8. Required Final Response From Implementing Agent

The implementing agent must report these exact items:

1. Whether the previous v5.11 decision changed after macro normalization.
2. Old v5.11 proposal recall and corrected v5.11.1 proposal recall.
3. Old v5.11 final-survival recall and corrected v5.11.1 final-survival recall.
4. Corrected failure-stage counts.
5. Whether `no_area_compatible_proposal` remains the dominant failure.
6. Whether the threshold pair accounting reports `requested=30`, `evaluated=23`, `skipped=7`.
7. Paths to:
   - `proposal_recall_s3_macrofix.json`
   - `proposal_recall_s3_macrofix.md`
   - `v5_11_1_macrofix_decision.md`
   - `v5_11_vs_v5_11_1_comparison.md`
8. The single recommended next action.

## 9. Review Checklist for Codex

When reviewing the implementing agent's work, check:

1. `object_proposal_recall_report.py` imports and uses `utils.area_truth_normalization.normalize_truth_macro`.
2. `normalize_truth_entries` stores canonical `macro` and preserves `raw_macro`.
3. Candidate proposal, classified, and final prediction labels are normalized before comparison.
4. Corrected report truth macro coverage contains `through_hole_round`, not `through_hole`.
5. Corrected report truth macro coverage contains `blind_hole_round`, not `blind_hole`.
6. Corrected report truth macro coverage contains `stepped_hole_round`, not `stepped_hole`.
7. Corrected report truth macro coverage contains `cylinder_boss`, not `cylindrical_boss`.
8. Corrected report truth macro coverage contains `rectangular_boss`, not `rectangle_boss`.
9. Report says requested threshold pairs are `30`, evaluated valid pairs are `23`, and skipped invalid pairs are `7`.
10. Markdown report uses ASCII `mm^2`, `->`, and `Important:` strings, with no mojibake.
11. The comparison report explicitly states whether the v5.11 conclusion survives the macrofix.

Only after this review should the project decide whether to enter a v5.12 proposal-generation repair.
