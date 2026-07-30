# v5.9-B Rib Hard-Case Proposer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve rib candidate proposal by mining existing colored real STEP rib truth into focused positive and negative hard cases, then retrain and validate the proposer.

**Architecture:** Add a hard-case record builder that reads colored rib diagnostics, graph JSON, and current predictions. Merge these hard cases with the existing v5.9-A face-level proposer dataset, train with explicit sample weights, and evaluate the complete inference chain at the current recommended proposer threshold.

**Tech Stack:** Python 3 in `W:/python/envs/feagpt`, JSON/JSONL datasets, scikit-learn candidate classifier, existing STEP graph/prediction/feature-object scripts.

---

### Task 1: Hard-Case Dataset Builder

**Files:**
- Create: `utils/rib_candidate_hardcases.py`
- Modify: `build_rib_candidate_proposer_dataset.py`
- Test: `tests/test_rib_candidate_hardcase_dataset.py`

- [ ] **Step 1: Write failing tests**

Test that colored rib faces become high-weight positive records, adjacent colored background faces become negative records, and duplicate base records are merged with hard-case records.

- [ ] **Step 2: Verify RED**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_rib_candidate_hardcase_dataset -v
```

Expected: import/function missing failure.

- [ ] **Step 3: Implement hard-case builder**

Implement functions:

```python
build_colored_rib_hardcase_records(...)
merge_rib_candidate_records(...)
```

Records must preserve `sample_id`, `graph_path`, `candidate_face_idx`, `label`, `features`, `source`, `sample_weight`, and `hard_case_types`.

- [ ] **Step 4: Verify GREEN**

Run the same unittest and confirm pass.

### Task 2: Weighted Training Support

**Files:**
- Modify: `train_candidate_topology_classifier.py`
- Test: `tests/test_train_candidate_topology_classifier.py`

- [ ] **Step 1: Write/extend failing test**

Add a test for `sample_weights_for_records` where an explicit `sample_weight` overrides source-based weighting.

- [ ] **Step 2: Verify RED**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_train_candidate_topology_classifier -v
```

Expected: explicit weight behavior missing.

- [ ] **Step 3: Implement support**

Update `sample_weights_for_records` so `record["sample_weight"]` is used when present; otherwise keep existing source-based behavior.

- [ ] **Step 4: Verify GREEN**

Run the same unittest and confirm pass.

### Task 3: Build and Train v5.9-B Proposer

**Files/Artifacts:**
- Dataset: `output_generation_v5_9_b_rib_candidate_hardcase_dataset`
- Model: `output_generation_v5_9_b_rib_candidate_hardcase_dataset/model_runs/random_forest_rib_candidate_proposer_v5_9_b/model.joblib`

- [ ] **Step 1: Build dataset**

Use existing reviewed colored truth and diagnostics from v5.8-B plus v5.9-A predictions as hard-case context.

- [ ] **Step 2: Train classifier**

Train random forest with explicit hard-case sample weights and record metrics.

### Task 4: Full Pipeline Validation

**Files/Artifacts:**
- s3 output: `output_real_step_s3_v5_9_b_rib_hardcase_proposer_t095`
- optional legacy outputs:
  - `output_real_step_s_v5_9_b_rib_hardcase_proposer_t095`
  - `output_real_step_s1_v5_9_b_rib_hardcase_proposer_t095`

- [ ] **Step 1: Run full s3 inference**

Use the same checkpoint, candidate classifier, and rib boundary classifier as v5.9-A, replacing only the rib candidate proposer.

- [ ] **Step 2: Convert predictions to feature objects**

Run `prediction_to_feature_objects.py`.

- [ ] **Step 3: Generate area truth report**

Run `generate_area_truth_review.py` against `configs/test_part_s3_area_truth_part20_mm2.json`.

- [ ] **Step 4: Generate compact reviews**

Run `generate_compact_real_step_review.py` for s3 and legacy sets when graph inputs exist.

### Task 5: Report and Context

**Files:**
- Create: `output_real_step_s3_v5_9_b_rib_hardcase_proposer_t095/reports/v5_9_b_rib_hardcase_comparison_report.md`
- Modify: `CODEX_CONTEXT.md`

- [ ] **Step 1: Write comparison report**

Compare v5.8-C, v5.9-A, and v5.9-B on overall matched objects, precision/recall, predicted rib, and matched rib.

- [ ] **Step 2: Final verification**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m py_compile build_rib_candidate_proposer_dataset.py utils\rib_candidate_hardcases.py train_candidate_topology_classifier.py
W:\python\envs\feagpt\python.exe -m unittest tests.test_rib_candidate_hardcase_dataset tests.test_rib_candidate_proposer_dataset tests.test_rib_candidate_proposer_postprocess tests.test_train_candidate_topology_classifier -v
```

Expected: all pass.
