# v5.11 Object Proposal Recall Diagnostics Implementation Plan

> **For implementer:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan step by step.

## 0. Purpose

v5.10 的 same-instance candidate assembly 在 `test_part_s3` 上没有带来净收益：

- v5.9-C t0.84: recall `0.6716`, precision `0.6650`
- v5.10 + v5.9 connected: recall `0.6520`, precision `0.6821`
- v5.10 + v5.9 assembled: recall `0.3529`, precision `0.8182`

这说明 v5.10 的候选级装配非常保守。当前不能继续直接改 GNN、重训或扩大数据集，因为我们还不知道真实件召回损失发生在哪个环节。

本计划要实现 v5.11 诊断工具，用同一批 `test_part_s3` 真实件，把每个真实目标按面积真值追踪到四个阶段：

1. raw GNN 后是否能提出候选对象。
2. 候选对象经过 instance candidate classifier 后是否仍是正确类别。
3. 候选对象是否被接受为可用对象。
4. 最终 topology postprocess 和 feature-object export 后是否仍然存活。

v5.11 的目标不是提高指标，而是输出可审查的 failure atlas，指导下一步到底应该修 proposal、修 classifier、修 topology/export，还是暂时放弃 v5.10 路线。

## 1. Scope Control

### 1.1 Files to Create

Create exactly these new files:

- `object_proposal_recall_report.py`
- `tests/test_object_proposal_recall_report.py`

### 1.2 Files That May Be Updated Only If Strictly Needed

Only update these files if a unit test proves a direct integration defect:

- `CODEX_CONTEXT.md`
- `docs/superpowers/plans/2026-07-01-v5-11-object-proposal-recall-diagnostics.md`

### 1.3 Files Not to Modify in This Task

Do not modify model architecture, training code, generator code, or existing inference behavior:

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `build_instance_candidate_dataset.py`
- `utils/instance_candidate_assembly.py`
- `utils/instance_candidate_postprocess.py`
- `run_gnn_inference.py`
- `prediction_to_feature_objects.py`
- any file under `features/`

If you believe one of these must be modified, stop and write the reason into the final report instead of changing it.

## 2. Existing Artifacts to Reuse

Use the current workspace:

```powershell
Set-Location W:\python\FEA\dataset_generator
```

Use the virtual environment:

```powershell
W:\python\envs\feagpt\python.exe
```

Use the s3 area truth file:

```text
configs/test_part_s3_area_truth_part20_mm2.json
```

Use the current best v5.10 plus v5.9 prediction directory as final-stage reference:

```text
output_real_step_s3_v5_10_instance_plus_v59/predictions
```

Use the current graph directory for s3:

```text
output_real_step_s3_v5_6_b_rib_boundary/graphs
```

Use the current v5.10 instance candidate classifier bundle:

```text
output_generation_v5_10_instance_candidate_dataset_v55/model_runs/random_forest_instance_candidate_v5_10_v55/model.joblib
```

Create all new diagnostics under:

```text
output_real_step_s3_v5_11_proposal_diagnostics
```

## 3. Required Output

The implementation must generate:

```text
output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions/*.json
output_real_step_s3_v5_11_proposal_diagnostics/reports/proposal_recall_s3.json
output_real_step_s3_v5_11_proposal_diagnostics/reports/proposal_recall_s3.md
output_real_step_s3_v5_11_proposal_diagnostics/reports/v5_11_proposal_recall_decision.md
```

The markdown report must include:

- one overall summary table;
- one threshold-grid table;
- macro-level coverage counts;
- per-part missed truth entries;
- top failure-stage counts;
- an explicit warning that area truth cannot prove face-level IoU.

The decision memo must recommend exactly one next action:

- `proposal_generation_fix`
- `classifier_recalibration_or_retraining`
- `topology_export_interaction_fix`
- `defer_v5_10_and_keep_v5_9_default`

The recommendation must be derived from the report statistics, not from subjective inspection.

## 4. Report Data Model

`proposal_recall_s3.json` must have this top-level structure:

```json
{
  "schema": "cae_object_proposal_recall_report_v1",
  "graph_dir": "output_real_step_s3_v5_6_b_rib_boundary/graphs",
  "raw_prediction_dir": "output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions",
  "truth_json": "configs/test_part_s3_area_truth_part20_mm2.json",
  "classifier": "output_generation_v5_10_instance_candidate_dataset_v55/model_runs/random_forest_instance_candidate_v5_10_v55/model.joblib",
  "final_prediction_dir": "output_real_step_s3_v5_10_instance_plus_v59/predictions",
  "absolute_area_tolerance": 10.0,
  "relative_area_tolerance": 0.02,
  "threshold_results": [],
  "best_by_proposal_recall": {},
  "best_by_final_survival_recall": {}
}
```

Each threshold result must include:

```json
{
  "strong_threshold": 0.55,
  "weak_threshold": 0.35,
  "sample_count": 21,
  "truth_count": 204,
  "proposal_candidate_count": 0,
  "proposal_stage": {},
  "classified_stage": {},
  "accepted_stage": {},
  "final_survival_stage": {},
  "failure_stage_counts": {},
  "samples": []
}
```

Each stage summary must include:

```json
{
  "matched_truth_count": 0,
  "recall": 0.0,
  "matched_by_macro": {},
  "truth_by_macro": {},
  "missed_by_macro": {}
}
```

Each sample entry must include:

```json
{
  "sample": "part1",
  "truth_count": 0,
  "candidate_count": 0,
  "proposal_stage": {},
  "classified_stage": {},
  "accepted_stage": {},
  "final_survival_stage": {},
  "truth_failures": [],
  "candidates": []
}
```

Each candidate summary must include:

```json
{
  "source": "assembled_instance_candidate",
  "proposal_macro": "rib",
  "classified_macro": "rib",
  "action": "accepted_remap",
  "confidence": 0.91,
  "face_indices": [1, 2, 3],
  "face_count": 3,
  "area": 100.0,
  "mean_same_instance_probability": 0.8,
  "min_same_instance_probability": 0.6
}
```

## 5. Matching Rules

### 5.1 Truth Normalization

Truth entries come from area truth. Normalize each entry into:

```python
{
    "sample": "part1",
    "macro": "rib",
    "area": 1234.56,
    "face_count": 3,
    "truth_index": 0,
}
```

Accept common aliases from the truth JSON if present:

- sample key: `sample`, `part`, `name`, or parent map key
- macro key: `macro`, `Macro`, `feature`, or `feature_type`
- area key: `area`, `Area`, or `area_sum`
- face-count key: `face_count`, `number_of_faces`, `number of faces`, or `num_faces`

### 5.2 Area Matching

A candidate matches a truth entry when:

```python
same_macro = candidate_macro == truth_macro
area_close = abs(candidate_area - truth_area) <= max(
    absolute_area_tolerance,
    relative_area_tolerance * max(abs(truth_area), 1.0),
)
face_count_close = truth_face_count is missing or candidate_face_count == truth_face_count
```

The report must state that this is area-level matching only. Do not call it face IoU.

### 5.3 Stage Semantics

Use these candidate macro values for each stage:

- `proposal_stage`: use `proposal_macro`.
- `classified_stage`: use `classified_macro`, but exclude rejected labels.
- `accepted_stage`: use `classified_macro` only when `action == "accepted_remap"`.
- `final_survival_stage`: candidate faces must still exist in final predictions and all final macro labels on those faces must equal `classified_macro`.

Rejected labels are:

- `reject`
- `stock`
- `background`
- `none`
- `mixed`

### 5.4 Failure Stage Assignment

For each truth entry, assign exactly one failure or survival stage:

1. `survives_final_prediction`
   - A matching candidate exists at final-survival stage.
2. `classifier_accepts_but_final_changes_or_removes`
   - A matching candidate exists at accepted stage, but not final-survival stage.
3. `proposal_exists_classifier_wrong_or_rejected`
   - A matching candidate exists at proposal stage, but not accepted stage.
4. `no_area_compatible_proposal`
   - No matching candidate exists at proposal stage.

This classification is the central output of v5.11.

## 6. Implementation Tasks

### Task 1: Add Failing Tests for Pure Matching

Create `tests/test_object_proposal_recall_report.py`.

Add `test_area_truth_matching_separates_proposal_classification_acceptance_and_final_survival`.

The test must build:

- one `rib` truth object with area `100.0`, face count `3`;
- one candidate with proposal macro `rib`, classified macro `rib`, action `accepted_remap`, faces `[1, 2, 3]`, area `100.5`;
- final predictions where faces `[1, 2, 3]` remain `rib`.

Expected:

- matched at proposal stage;
- matched at classified stage;
- matched at accepted stage;
- matched at final-survival stage;
- failure stage is `survives_final_prediction`.

Add `test_failure_stage_distinguishes_rejected_classifier_from_missing_proposal`.

The test must build:

- truth A: `rib`, area `100.0`;
- truth B: `slot_u`, area `200.0`;
- one proposal candidate for truth A, but classified as `reject`;
- no proposal candidate for truth B.

Expected:

- truth A failure stage: `proposal_exists_classifier_wrong_or_rejected`;
- truth B failure stage: `no_area_compatible_proposal`.

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report -v
```

Expected result at this point: import failure because `object_proposal_recall_report.py` does not exist.

### Task 2: Implement Pure Matching Functions

Create `object_proposal_recall_report.py`.

Implement these functions:

```python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REJECT_LABELS = {"reject", "stock", "background", "none", "mixed"}

FAILURE_STAGES = (
    "survives_final_prediction",
    "classifier_accepts_but_final_changes_or_removes",
    "proposal_exists_classifier_wrong_or_rejected",
    "no_area_compatible_proposal",
)
```

Required functions:

```python
def load_json(path: Path) -> object: ...
def write_json(path: Path, payload: object) -> None: ...
def node_area(node: dict) -> float: ...
def candidate_area(candidate: dict, graph: dict | None = None) -> float: ...
def normalize_truth_entries(payload: object) -> dict[str, list[dict]]: ...
def candidate_macro_for_stage(candidate: dict, stage: str) -> str | None: ...
def final_survives(candidate: dict, final_prediction: dict, macro: str) -> bool: ...
def candidate_matches_truth(candidate: dict, truth: dict, stage: str, absolute_area_tolerance: float, relative_area_tolerance: float, final_prediction: dict | None = None) -> bool: ...
def matched_truth_indices_for_stage(truth_entries: list[dict], candidates: list[dict], stage: str, absolute_area_tolerance: float, relative_area_tolerance: float, final_prediction: dict | None = None) -> set[int]: ...
def summarize_stage_matches(truth_entries: list[dict], matched_indices: set[int]) -> dict: ...
def classify_truth_failure_stage(truth: dict, candidates: list[dict], absolute_area_tolerance: float, relative_area_tolerance: float, final_prediction: dict | None = None) -> str: ...
def match_truth_by_stage(truth_entries: list[dict], candidates: list[dict], absolute_area_tolerance: float, relative_area_tolerance: float, final_prediction: dict | None = None) -> dict: ...
```

Implementation details:

- `node_area` must read `node["attributes"]["physical"]["area"]` first.
- If that key is missing, fallback to `node["area"]`.
- If both are missing, return `0.0`.
- `candidate_area` must use `candidate["area"]` if present.
- If candidate area is missing and graph is provided, sum graph node areas for candidate face ids.
- Face ids may be stored as `face_indices`, `faces`, or `face_ids`.
- Preserve input face ids as integers.

After implementing, rerun:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report -v
```

Expected result: Task 1 tests pass.

### Task 3: Add Candidate Assembly Replay Tests

Add `test_build_candidate_summaries_aligns_classifier_records_by_face_set_and_macro`.

Test setup:

- Create two proposal dicts:
  - proposal 1: macro `rib`, faces `[1, 2, 3]`, area `100.0`
  - proposal 2: macro `slot_u`, faces `[4, 5]`, area `50.0`
- Create classifier records:
  - one accepted record for faces `[3, 2, 1]`, proposal macro `rib`, classified macro `rib`, confidence `0.9`;
  - one rejected record for faces `[4, 5]`, proposal macro `slot_u`, classified macro `reject`, confidence `0.7`.

Expected:

- candidates are matched by sorted face tuple plus proposal macro;
- rib candidate action is `accepted_remap`;
- slot candidate action is `rejected_label`;
- face order does not affect matching.

Add helper:

```python
def proposal_key(record: dict) -> tuple[str, tuple[int, ...]]: ...
```

Implement:

```python
def build_candidate_summaries(
    proposals: list[dict],
    classifier_records: list[dict],
    graph: dict | None = None,
) -> list[dict]: ...
```

Action mapping:

- if classified macro is in `REJECT_LABELS`: `rejected_label`
- if classifier record has `accepted == True` or `action == "accepted_remap"`: `accepted_remap`
- if classifier record has `action`: reuse the provided action
- if no classifier record: `unclassified`

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report -v
```

Expected result: all current tests pass.

### Task 4: Add Assembly Replay Implementation

Import and reuse existing code:

```python
from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_postprocess import (
    classify_instance_candidate_proposals,
    load_instance_candidate_classifier_bundle,
)
```

Implement:

```python
def replay_candidates_for_sample(
    graph: dict,
    raw_prediction: dict,
    classifier_bundle: object | None,
    strong_threshold: float,
    weak_threshold: float,
    min_confidence: float,
) -> list[dict]: ...
```

Required behavior:

1. Call `assemble_instance_candidates` using raw predictions and thresholds.
2. Convert proposals into candidate summaries.
3. If classifier bundle is provided, call `classify_instance_candidate_proposals`.
4. Merge classifier results back into candidate summaries by `proposal_key`.
5. Return candidate summaries sorted by:
   - macro name;
   - minimum face id;
   - face count.

Do not call topology postprocess here. This script diagnoses candidate stages; it must not mutate inference predictions.

### Task 5: Add Dataset-Level Result Builder

Add `test_build_threshold_result_reports_sample_and_macro_counts`.

The test must build two samples:

- `part1` has one truth `rib`, one matching accepted candidate.
- `part2` has one truth `slot_u`, no candidate.

Expected:

- `truth_count == 2`
- `proposal_stage.matched_truth_count == 1`
- `failure_stage_counts["survives_final_prediction"] == 1`
- `failure_stage_counts["no_area_compatible_proposal"] == 1`
- `proposal_stage.truth_by_macro == {"rib": 1, "slot_u": 1}`
- `proposal_stage.matched_by_macro == {"rib": 1}`
- `proposal_stage.missed_by_macro == {"slot_u": 1}`

Implement:

```python
def merge_stage_summary(accumulator: dict, sample_summary: dict) -> None: ...
def merge_failure_counts(accumulator: Counter, sample_failures: list[dict]) -> None: ...
def build_threshold_result(
    graph_by_sample: dict[str, dict],
    raw_prediction_by_sample: dict[str, dict],
    truth_by_sample: dict[str, list[dict]],
    final_prediction_by_sample: dict[str, dict],
    classifier_bundle: object | None,
    strong_threshold: float,
    weak_threshold: float,
    min_confidence: float,
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> dict: ...
```

Each `truth_failures` entry must include:

```json
{
  "truth_index": 0,
  "macro": "rib",
  "area": 100.0,
  "face_count": 3,
  "failure_stage": "no_area_compatible_proposal"
}
```

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report -v
```

Expected result: all current tests pass.

### Task 6: Add CLI

Implement CLI arguments:

```text
--graph-dir
--raw-prediction-dir
--truth-json
--classifier
--final-prediction-dir
--output-json
--output-md
--strong-thresholds
--weak-thresholds
--min-confidence
--absolute-area-tolerance
--relative-area-tolerance
```

Defaults:

```text
--min-confidence 0.50
--absolute-area-tolerance 10.0
--relative-area-tolerance 0.02
```

Implement helpers:

```python
def parse_thresholds(value: str) -> list[float]: ...
def collect_json_by_stem(directory: Path) -> dict[str, dict]: ...
def load_truth_by_sample(path: Path) -> dict[str, list[dict]]: ...
def best_result(results: list[dict], stage_name: str) -> dict: ...
def build_report(args: argparse.Namespace) -> dict: ...
def write_markdown(path: Path, report: dict) -> None: ...
def write_decision_memo(path: Path, report: dict) -> None: ...
def parse_args(argv: list[str] | None = None) -> argparse.Namespace: ...
def main(argv: list[str] | None = None) -> int: ...
```

`write_markdown` must include:

- inputs;
- tolerance;
- best threshold by proposal recall;
- best threshold by final-survival recall;
- threshold-grid table;
- macro coverage for best proposal-recall threshold;
- failure-stage counts;
- top missed truth entries by sample and macro.

`write_decision_memo` must choose exactly one recommendation:

- If `no_area_compatible_proposal` is the largest non-survival failure count: recommend `proposal_generation_fix`.
- If `proposal_exists_classifier_wrong_or_rejected` is the largest non-survival failure count: recommend `classifier_recalibration_or_retraining`.
- If `classifier_accepts_but_final_changes_or_removes` is the largest non-survival failure count: recommend `topology_export_interaction_fix`.
- If final-survival recall is below v5.9-C recall and no single failure dominates by at least 40 percent of all non-survival failures: recommend `defer_v5_10_and_keep_v5_9_default`.

The decision memo must be generated from computed report values. It must not contain unresolved marker strings.

### Task 7: Generate Raw Predictions for s3

Run raw inference without topology postprocess:

```powershell
W:\python\envs\feagpt\python.exe run_gnn_inference.py `
  --checkpoint output_generation_v5_5_v51dist_looprole_5400\model_runs\edge_aware_v5_5_expanded_v51_compat_24dim\best.pt `
  --input-path output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --output-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --disable-topology-postprocess
```

Verify that raw predictions exist:

```powershell
Get-ChildItem output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions -Filter *.json | Measure-Object
```

Expected count: `21`.

If the count is not 21, stop and report which graph/prediction files are missing.

### Task 8: Run Threshold Grid Diagnostic

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_11_proposal_diagnostics\reports\proposal_recall_s3.json `
  --output-md output_real_step_s3_v5_11_proposal_diagnostics\reports\proposal_recall_s3.md `
  --strong-thresholds 0.30,0.40,0.50,0.55,0.60,0.70 `
  --weak-thresholds 0.20,0.30,0.35,0.40,0.50 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Expected:

- JSON report exists.
- Markdown report exists.
- Decision memo exists.
- Threshold grid contains `30` combinations.

### Task 9: Final Verification

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Run:

```powershell
W:\python\envs\feagpt\python.exe -m py_compile object_proposal_recall_report.py tests\test_object_proposal_recall_report.py utils\instance_candidate_postprocess.py utils\topology_postprocess.py run_gnn_inference.py prediction_to_feature_objects.py
```

Run:

```powershell
Test-Path output_real_step_s3_v5_11_proposal_diagnostics\reports\proposal_recall_s3.json
Test-Path output_real_step_s3_v5_11_proposal_diagnostics\reports\proposal_recall_s3.md
Test-Path output_real_step_s3_v5_11_proposal_diagnostics\reports\v5_11_proposal_recall_decision.md
```

All three `Test-Path` commands must print `True`.

## 7. Review Checklist for Codex

After another agent executes this plan, review these points first:

1. Scope:
   - Only `object_proposal_recall_report.py`, `tests/test_object_proposal_recall_report.py`, generated reports, and context docs were changed.
   - No model/training/generator files were modified.

2. Tests:
   - The new test file covers pure area matching.
   - It covers classifier-record alignment by unordered face set plus macro.
   - It covers dataset-level failure-stage counting.

3. Report correctness:
   - `proposal_recall_s3.json` contains `schema == "cae_object_proposal_recall_report_v1"`.
   - It has four stages: proposal, classified, accepted, final survival.
   - It has failure-stage counts.
   - It has 30 threshold combinations.
   - The markdown report explicitly says the validation is area-level, not face IoU.

4. Decision quality:
   - `v5_11_proposal_recall_decision.md` recommends exactly one next action.
   - The recommendation follows the failure-stage counts.
   - It does not claim model improvement.

5. Scientific interpretation:
   - If `no_area_compatible_proposal` dominates, next work is proposal generation or raw GNN recall.
   - If `proposal_exists_classifier_wrong_or_rejected` dominates, next work is classifier calibration or candidate classifier training data.
   - If `classifier_accepts_but_final_changes_or_removes` dominates, next work is topology/export interaction.
   - If no stage dominates and v5.10 remains below v5.9-C, keep v5.9-C as current best and treat v5.10 as a research branch.

## 8. Final Response Required From Implementing Agent

The implementing agent must report:

- tests run and pass/fail status;
- generated report paths;
- best threshold by proposal recall;
- best threshold by final-survival recall;
- failure-stage count summary;
- the single recommended next action from the decision memo;
- any deviation from this plan.

Do not summarize the whole codebase. The final response should focus on whether v5.11 diagnosis was implemented and what it says about the next technical move.
