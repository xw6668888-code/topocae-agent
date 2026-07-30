# v5.24 Reason-Aware Candidate Classifier Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the instance candidate classifier so v5.23's improved topology-rule proposals survive classification and acceptance, instead of improving proposal recall without improving final recall.

**Architecture:** Keep the v5.23 proposer changes. Add assembly-reason features to the candidate feature vector, rebuild a hybrid instance-candidate dataset using the v5.23 proposer, add reason-specific positive clones for the new targeted proposer reasons, retrain the candidate classifier, and rerun the same s3 strict/semantic/review-aware evaluation chain. Do not change GNN, graph extraction, STEP generation, or area truth.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, scikit-learn/joblib, existing `build_instance_candidate_dataset.py`, existing `train_candidate_topology_classifier.py`, existing v5.21-v5.23 evaluation scripts.

---

## 0. Review of v5.23 Output

v5.23 satisfies the previous plan.

Verification command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_topology_postprocess_v5_23 `
  tests.test_object_candidate_proposer_v5_23 `
  tests.test_compare_review_aware_residual_atlases `
  tests.test_object_candidate_proposer `
  tests.test_topology_postprocess_v5_8 `
  tests.test_review_aware_residual_failure_atlas -v
```

Observed:

```text
Ran 24 tests
OK
```

v5.23 changed the right thing:

```text
v5.21 proposal recall: 0.6814
v5.23 proposal recall: 0.7402
v5.21 classified recall: 0.5343
v5.23 classified recall: 0.5539
v5.21 accepted recall: 0.4804
v5.23 accepted recall: 0.5000
v5.21 final-survival recall: 0.5196
v5.23 final-survival recall: 0.5196
```

v5.23 review-aware residual comparison:

```text
miss_count_delta: -12
chamfer_delta: -4
fillet_delta: -7
rib_delta: -1
```

Interpretation:

- v5.23 proposer repair is useful.
- The new proposals reduce residual proposal misses.
- The final result does not improve, so the bottleneck has moved from "candidate not proposed" to "candidate not accepted/finalized".
- The next step should repair the candidate classifier and its feature representation.

Minor v5.23 report issue:

- `review_aware_residual_atlas.md` in the v5.23 output still has a static title saying `v5.22`. This is cosmetic and should be fixed while touching the report script, but it does not invalidate the numbers.

## 1. Human Task

No human intervention is required for v5.24.

The user should inspect only the final comparison report:

```text
output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_vs_v5_23_review_aware_comparison.md
output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_decision.md
```

If v5.24 improves final-survival recall, continue with this classifier path. If v5.24 only improves training/test metrics but not s3 final recall, stop classifier-only tuning and plan model/data redesign.

## 2. Scope Control

### Modify

- `utils/instance_candidate_features.py`
- `build_instance_candidate_dataset.py`
- `review_aware_residual_failure_atlas.py`

### Create

- `tests/test_instance_candidate_features_v5_24.py`
- `tests/test_build_instance_candidate_dataset_v5_24.py`
- `compare_recall_stage_reports.py`
- `tests/test_compare_recall_stage_reports.py`
- `output_generation_v5_24_reason_aware_instance_candidate_dataset/instance_candidate_dataset`
- `output_generation_v5_24_reason_aware_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_24_reasonaware/model.joblib`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/proposal_recall_s3_v5_24_reasonaware.json`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/proposal_recall_s3_v5_24_reasonaware.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/semantic_policy_recall_s3_v5_24.json`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/semantic_policy_recall_s3_v5_24.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_proposal_miss_atlas.json`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_proposal_miss_atlas.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/review_aware_residual_atlas.json`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/review_aware_residual_atlas.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_vs_v5_23_review_aware_comparison.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_stage_comparison.md`
- `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_decision.md`

### Do Not Modify

- `utils/object_candidate_proposer.py`
- `utils/topology_postprocess.py`
- graph extraction
- STEP generation
- GNN model/training code
- area truth JSON/XLSX
- existing v5.20-v5.23 reports
- existing v5.10 model checkpoint

v5.24 only changes candidate classifier features/dataset/model and reporting.

## 3. Why v5.24 Must Add Assembly-Reason Features

Current `utils/instance_candidate_features.py` has proposal source features:

```text
proposal_source_topology_rule
proposal_source_raw_macro_component
proposal_source_same_instance
...
```

But all v5.23 targeted proposer candidates are still just `topology_rule` at the classifier feature level. The classifier cannot distinguish:

```text
topology_fillet_stock_transition
topology_chamfer_stock_transition
topology_review_targeted_fillet_residual
topology_review_targeted_chamfer_residual
topology_review_targeted_rib_residual
```

Therefore v5.24 must encode `assembly_reasons` as explicit binary features.

## 4. Task 1: Add Assembly-Reason Candidate Features

**Files:**

- Modify: `utils/instance_candidate_features.py`
- Create: `tests/test_instance_candidate_features_v5_24.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_instance_candidate_features_v5_24.py`:

```python
from __future__ import annotations

import unittest

from utils.instance_candidate_features import extract_instance_candidate_features


def node(idx: int, primitive: str = "plane", area: float = 10.0):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "area": area,
        "attributes": {
            "physical": {"area": area},
            "shape_descriptors": {
                "projected_long_extent": 10.0,
                "projected_short_extent": 2.0,
                "projected_aspect_ratio": 5.0,
            },
        },
    }


class InstanceCandidateFeaturesV524Test(unittest.TestCase):
    def test_targeted_assembly_reasons_are_encoded_as_binary_features(self):
        graph_data = {"nodes": [node(1), node(2), node(3)], "edges": [], "edge_attributes": []}
        candidate = {
            "proposal_macro": "rib",
            "face_indices": [1, 2, 3],
            "source": "topology_rule",
            "proposal_sources": ["topology_rule"],
            "assembly_reasons": ["topology_review_targeted_rib_residual"],
        }

        features = extract_instance_candidate_features(graph_data, candidate, prediction_faces=[])

        self.assertEqual(features["assembly_reason_count"], 1.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_rib_residual"], 1.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_fillet_residual"], 0.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_chamfer_residual"], 0.0)

    def test_unknown_assembly_reason_sets_other_bucket(self):
        graph_data = {"nodes": [node(4, primitive="cylinder")], "edges": [], "edge_attributes": []}
        candidate = {
            "proposal_macro": "fillet",
            "face_indices": [4],
            "source": "topology_rule",
            "proposal_sources": ["topology_rule"],
            "assembly_reasons": ["new_reason_not_in_schema"],
        }

        features = extract_instance_candidate_features(graph_data, candidate, prediction_faces=[])

        self.assertEqual(features["assembly_reason_count"], 1.0)
        self.assertEqual(features["assembly_reason_other"], 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features_v5_24 -v
```

Expected before implementation:

```text
KeyError or AssertionError for missing assembly_reason_* features
```

- [ ] **Step 3: Add reason buckets**

In `utils/instance_candidate_features.py`, near `PROPOSAL_SOURCE_BUCKETS`, add:

```python
ASSEMBLY_REASON_BUCKETS = (
    "topology_review_targeted_rib_residual",
    "topology_review_targeted_fillet_residual",
    "topology_review_targeted_chamfer_residual",
    "topology_rib_stock",
    "topology_mixed_rib",
    "topology_existing_rib_cap_completion",
    "topology_transition_cap_rib",
    "topology_pocket_embedded_rib",
    "topology_existing_rib_transition_absorption",
    "topology_fillet_stock_transition",
    "topology_chamfer_stock_transition",
    "topology_through_hole_opening",
    "topology_counterbore",
    "topology_countersink",
    "topology_cylinder_boss",
    "topology_rectangular_slot",
    "topology_u_slot",
    "topology_o_slot",
    "topology_v_slot",
    "source_macro_positive_clone",
    "assembly_reason_positive_clone",
    "other",
)
```

- [ ] **Step 4: Add helper functions**

In `utils/instance_candidate_features.py`, near `_candidate_sources`, add:

```python
def _candidate_assembly_reasons(candidate: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    raw_reasons = candidate.get("assembly_reasons")
    if isinstance(raw_reasons, Iterable) and not isinstance(raw_reasons, (str, bytes, Mapping)):
        for reason in raw_reasons:
            token = str(reason).strip()
            if token:
                reasons.add(token)
    augmentation = candidate.get("augmentation")
    if augmentation is not None:
        token = str(augmentation).strip()
        if token:
            reasons.add(token)
    return reasons


def _add_assembly_reason_features(features: dict[str, float], candidate: Mapping[str, Any]) -> None:
    reasons = _candidate_assembly_reasons(candidate)
    features["assembly_reason_count"] = float(len(reasons))
    known = set(ASSEMBLY_REASON_BUCKETS[:-1])
    for bucket in ASSEMBLY_REASON_BUCKETS:
        if bucket == "other":
            value = 1.0 if reasons and not any(reason in known for reason in reasons) else 0.0
        else:
            value = 1.0 if bucket in reasons else 0.0
        features[f"assembly_reason_{bucket}"] = value
```

- [ ] **Step 5: Call the new feature injector**

In `extract_instance_candidate_features`, after:

```python
    _add_proposal_source_features(features, candidate)
```

add:

```python
    _add_assembly_reason_features(features, candidate)
```

- [ ] **Step 6: Run feature tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_instance_candidate_features_v5_24 -v
```

Expected:

```text
OK
```

## 5. Task 2: Add Assembly-Reason Positive Clones to Dataset Builder

**Files:**

- Modify: `build_instance_candidate_dataset.py`
- Create: `tests/test_build_instance_candidate_dataset_v5_24.py`

- [ ] **Step 1: Create failing tests**

Create `tests/test_build_instance_candidate_dataset_v5_24.py`:

```python
from __future__ import annotations

import unittest

from build_instance_candidate_dataset import (
    assign_candidate_sample_weight,
    build_assembly_reason_positive_clone_proposals,
    parse_assembly_reason_macro_pairs,
)


class BuildInstanceCandidateDatasetV524Test(unittest.TestCase):
    def test_parse_assembly_reason_macro_pairs(self):
        pairs = parse_assembly_reason_macro_pairs(
            "topology_review_targeted_rib_residual:rib,topology_review_targeted_fillet_residual:fillet"
        )

        self.assertEqual(
            pairs,
            {
                ("topology_review_targeted_rib_residual", "rib"),
                ("topology_review_targeted_fillet_residual", "fillet"),
            },
        )

    def test_build_assembly_reason_positive_clone_proposals(self):
        truth_instances = [
            {"macro_feature": "rib", "face_indices": [1, 2, 3]},
            {"macro_feature": "fillet", "face_indices": [4]},
        ]

        clones = build_assembly_reason_positive_clone_proposals(
            truth_instances,
            {("topology_review_targeted_rib_residual", "rib")},
        )

        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0]["proposal_macro"], "rib")
        self.assertEqual(clones[0]["source"], "topology_rule")
        self.assertEqual(clones[0]["face_indices"], [1, 2, 3])
        self.assertEqual(clones[0]["assembly_reasons"], ["topology_review_targeted_rib_residual"])
        self.assertEqual(clones[0]["augmentation"], "assembly_reason_positive_clone")

    def test_assembly_reason_positive_clone_gets_clone_weight(self):
        weight, action = assign_candidate_sample_weight(
            source="topology_rule",
            proposal_macro="rib",
            label="rib",
            augmentation="assembly_reason_positive_clone",
            source_macro_positive_clone_pairs=set(),
            reject_downweight_source_macro_pairs=set(),
            default_sample_weight=1.0,
            positive_nontruth_weight=3.0,
            positive_clone_weight=8.0,
            downweighted_reject_weight=0.35,
        )

        self.assertEqual(weight, 8.0)
        self.assertEqual(action, "positive_assembly_reason_clone")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset_v5_24 -v
```

Expected before implementation:

```text
ImportError
```

- [ ] **Step 3: Add parser argument**

In `parse_args`, add:

```python
    parser.add_argument("--assembly-reason-positive-clones", default="")
```

- [ ] **Step 4: Add parser and clone builder**

In `build_instance_candidate_dataset.py`, after `parse_source_macro_pairs`, add:

```python
def parse_assembly_reason_macro_pairs(value: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"assembly reason/macro item must use reason:macro format, got {item!r}")
        reason, macro = (part.strip() for part in item.split(":", 1))
        if not reason or not macro:
            raise ValueError(f"assembly reason/macro item must use reason:macro format, got {item!r}")
        pairs.add((reason, macro))
    return pairs
```

After `build_source_macro_positive_clone_proposals`, add:

```python
def build_assembly_reason_positive_clone_proposals(
    truth_instances: list[dict[str, Any]],
    reason_macro_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for instance in truth_instances:
        macro = str(instance["macro_feature"])
        face_indices = list(instance["face_indices"])
        for reason, pair_macro in sorted(reason_macro_pairs):
            if pair_macro != macro:
                continue
            proposals.append(
                {
                    "source": "topology_rule",
                    "proposal_sources": ["topology_rule"],
                    "proposal_macro": macro,
                    "face_indices": face_indices,
                    "augmentation": "assembly_reason_positive_clone",
                    "assembly_reasons": [reason],
                }
            )
    return proposals
```

- [ ] **Step 5: Wire clone builder into `build_proposals`**

Extend `build_proposals` signature with:

```python
    assembly_reason_positive_clone_pairs: set[tuple[str, str]] | None = None,
```

After source-macro clone extension, add:

```python
    proposals.extend(
        build_assembly_reason_positive_clone_proposals(
            truth_instances,
            assembly_reason_positive_clone_pairs or set(),
        )
    )
```

- [ ] **Step 6: Wire clone pairs through `build_candidate_records_for_graph` and `build_dataset_from_paths`**

Add the same optional parameter to both functions:

```python
    assembly_reason_positive_clone_pairs: set[tuple[str, str]] | None = None,
```

Pass it into `build_proposals`.

In `main`, pass:

```python
        assembly_reason_positive_clone_pairs=parse_assembly_reason_macro_pairs(args.assembly_reason_positive_clones),
```

- [ ] **Step 7: Update sample weighting**

In `assign_candidate_sample_weight`, before the source-macro clone branch, add:

```python
    if label != REJECT_LABEL and augmentation == "assembly_reason_positive_clone":
        return float(positive_clone_weight), "positive_assembly_reason_clone"
```

Keep the existing `source_macro_positive_clone` behavior unchanged.

- [ ] **Step 8: Update dataset summary**

In `write_dataset_outputs`, compute:

```python
    assembly_reason_positive_clone_count = sum(
        1 for record in enriched_records
        if record.get("augmentation") == "assembly_reason_positive_clone"
    )
```

Add to `summary`:

```python
        "assembly_reason_positive_clone_count": int(assembly_reason_positive_clone_count),
```

- [ ] **Step 9: Run dataset builder tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset_v5_24 -v
```

Expected:

```text
OK
```

## 6. Task 3: Rebuild Hybrid Reason-Aware Candidate Dataset

**Files:**

- Generate: `output_generation_v5_24_reason_aware_instance_candidate_dataset/instance_candidate_dataset`

- [ ] **Step 1: Build dataset**

Run:

```powershell
W:\python\envs\feagpt\python.exe build_instance_candidate_dataset.py `
  --graph-dir output_generation_v5_5_v51dist_looprole_5400\graphs `
  --prediction-dir output_generation_v5_10_instance_candidate_dataset_v55\predictions_raw `
  --output-dir output_generation_v5_24_reason_aware_instance_candidate_dataset\instance_candidate_dataset `
  --recursive `
  --proposal-mode hybrid `
  --source-macro-positive-clones topology_rule:rib,topology_rule:fillet,topology_rule:chamfer `
  --assembly-reason-positive-clones topology_review_targeted_rib_residual:rib,topology_review_targeted_fillet_residual:fillet,topology_review_targeted_chamfer_residual:chamfer `
  --positive-nontruth-weight 4.0 `
  --positive-clone-weight 8.0 `
  --reject-downweight-source-macros topology_rule:rib,topology_rule:fillet,topology_rule:chamfer `
  --downweighted-reject-weight 0.5 `
  --seed 20260706
```

Expected:

```text
"record_count" is greater than 23722
```

The old v5.10 dataset had `23722` records and was mostly `same_instance`; v5.24 must include hybrid topology-rule candidates and reason-aware positive clones.

- [ ] **Step 2: Inspect dataset summary**

Run:

```powershell
Get-Content -LiteralPath output_generation_v5_24_reason_aware_instance_candidate_dataset\instance_candidate_dataset\dataset_summary.json -TotalCount 160
```

Expected:

```text
"proposal_mode": "hybrid"
"assembly_reason_positive_clone_count" greater than 0
"source_counts" includes "topology_rule"
```

- [ ] **Step 3: Verify feature schema includes reason features**

Run:

```powershell
Select-String -LiteralPath output_generation_v5_24_reason_aware_instance_candidate_dataset\instance_candidate_dataset\feature_schema.json -Pattern "assembly_reason_topology_review_targeted"
```

Expected:

```text
assembly_reason_topology_review_targeted_rib_residual
assembly_reason_topology_review_targeted_fillet_residual
assembly_reason_topology_review_targeted_chamfer_residual
```

## 7. Task 4: Train Reason-Aware Candidate Classifier

**Files:**

- Generate: `output_generation_v5_24_reason_aware_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_24_reasonaware/model.joblib`
- Generate: `output_generation_v5_24_reason_aware_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_24_reasonaware/metrics.json`
- Generate: `output_generation_v5_24_reason_aware_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_24_reasonaware/report.md`

- [ ] **Step 1: Train random forest classifier**

Run:

```powershell
W:\python\envs\feagpt\python.exe train_candidate_topology_classifier.py `
  --dataset-dir output_generation_v5_24_reason_aware_instance_candidate_dataset\instance_candidate_dataset `
  --output-dir output_generation_v5_24_reason_aware_instance_candidate_dataset\model_runs `
  --run-name random_forest_instance_candidate_v5_24_reasonaware `
  --model-kind random_forest `
  --seed 20260706 `
  --n-estimators 700 `
  --max-depth 24 `
  --hard-source-weight 4.0
```

Expected:

```text
"test_macro_f1" is printed
```

- [ ] **Step 2: Inspect training report**

Run:

```powershell
Get-Content -LiteralPath output_generation_v5_24_reason_aware_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_24_reasonaware\report.md -TotalCount 120
```

Expected:

```text
The test macro-F1 should be reported.
Top features should include at least one assembly_reason_* or proposal_source_topology_rule feature if the model uses the new information.
```

If no assembly-reason feature appears in top features, continue evaluation anyway; do not tune yet.

## 8. Task 5: Evaluate v5.24 on s3 Real Parts

**Files:**

- Generate v5.24 reports under `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports`.

- [ ] **Step 1: Run strict recall report with new classifier**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_v5_21_reviewed.json `
  --classifier output_generation_v5_24_reason_aware_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_24_reasonaware\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --proposal-mode hybrid `
  --strong-thresholds 0.3,0.4,0.5,0.55,0.6,0.7 `
  --weak-thresholds 0.2,0.3,0.35,0.4,0.5 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02 `
  --output-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.json `
  --output-md output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.md
```

Expected:

```text
Report written to output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.json
```

- [ ] **Step 2: Apply semantic policy**

Run:

```powershell
W:\python\envs\feagpt\python.exe apply_semantic_review_policy_to_recall_report.py `
  --report-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --output-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\semantic_policy_recall_s3_v5_24.json `
  --output-md output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\semantic_policy_recall_s3_v5_24.md `
  --area-tolerance 0.02
```

- [ ] **Step 3: Build proposal miss atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe proposal_miss_atlas.py `
  --report v5_24=output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.json `
  --output-dir output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports
```

- [ ] **Step 4: Build review-aware residual atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe review_aware_residual_failure_atlas.py `
  --atlas-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_proposal_miss_atlas.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --semantic-policy-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\semantic_policy_recall_s3_v5_24.json `
  --output-dir output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports
```

If `proposal_miss_atlas.py` writes a different atlas filename, use the actual generated filename in this command and report it.

- [ ] **Step 5: Compare v5.23 and v5.24 review-aware residual atlases**

Run:

```powershell
W:\python\envs\feagpt\python.exe compare_review_aware_residual_atlases.py `
  --old-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\review_aware_residual_atlas.json `
  --new-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\review_aware_residual_atlas.json `
  --output-md output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_vs_v5_23_review_aware_comparison.md
```

## 9. Task 6: Add Stage Comparison Tool

**Files:**

- Create: `compare_recall_stage_reports.py`
- Create: `tests/test_compare_recall_stage_reports.py`

- [ ] **Step 1: Create tests**

Create `tests/test_compare_recall_stage_reports.py`:

```python
from __future__ import annotations

import unittest

from compare_recall_stage_reports import compare_stage_reports, extract_stage_metrics


class CompareRecallStageReportsTest(unittest.TestCase):
    def test_extract_stage_metrics_reads_best_final_survival_result(self):
        report = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.8, "matched_truth_count": 8},
                "classified_stage": {"recall": 0.7, "matched_truth_count": 7},
                "accepted_stage": {"recall": 0.6, "matched_truth_count": 6},
                "final_survival_stage": {"recall": 0.5, "matched_truth_count": 5},
            }
        }

        metrics = extract_stage_metrics(report)

        self.assertEqual(metrics["truth_count"], 10)
        self.assertEqual(metrics["proposal_recall"], 0.8)
        self.assertEqual(metrics["classified_recall"], 0.7)
        self.assertEqual(metrics["accepted_recall"], 0.6)
        self.assertEqual(metrics["final_survival_recall"], 0.5)

    def test_compare_stage_reports_computes_deltas(self):
        old = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.8, "matched_truth_count": 8},
                "classified_stage": {"recall": 0.6, "matched_truth_count": 6},
                "accepted_stage": {"recall": 0.5, "matched_truth_count": 5},
                "final_survival_stage": {"recall": 0.4, "matched_truth_count": 4},
            }
        }
        new = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.9, "matched_truth_count": 9},
                "classified_stage": {"recall": 0.7, "matched_truth_count": 7},
                "accepted_stage": {"recall": 0.6, "matched_truth_count": 6},
                "final_survival_stage": {"recall": 0.5, "matched_truth_count": 5},
            }
        }

        comparison = compare_stage_reports(old, new)

        self.assertAlmostEqual(comparison["delta"]["proposal_recall"], 0.1)
        self.assertAlmostEqual(comparison["delta"]["final_survival_recall"], 0.1)
        self.assertEqual(comparison["delta"]["final_survival_matched"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement tool**

Create `compare_recall_stage_reports.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_stage_metrics(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    proposal = best.get("proposal_stage") or {}
    classified = best.get("classified_stage") or {}
    accepted = best.get("accepted_stage") or {}
    final = best.get("final_survival_stage") or {}
    return {
        "truth_count": int(best.get("truth_count", report.get("truth_count", 0))),
        "proposal_recall": float(proposal.get("recall", 0.0)),
        "proposal_matched": int(proposal.get("matched_truth_count", 0)),
        "classified_recall": float(classified.get("recall", 0.0)),
        "classified_matched": int(classified.get("matched_truth_count", 0)),
        "accepted_recall": float(accepted.get("recall", 0.0)),
        "accepted_matched": int(accepted.get("matched_truth_count", 0)),
        "final_survival_recall": float(final.get("recall", 0.0)),
        "final_survival_matched": int(final.get("matched_truth_count", 0)),
    }


def compare_stage_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_metrics = extract_stage_metrics(old)
    new_metrics = extract_stage_metrics(new)
    delta: dict[str, Any] = {}
    for key, old_value in old_metrics.items():
        if key == "truth_count":
            delta[key] = int(new_metrics[key]) - int(old_value)
        elif key.endswith("_matched"):
            delta[key] = int(new_metrics[key]) - int(old_value)
        else:
            delta[key] = float(new_metrics[key]) - float(old_value)
    return {"old": old_metrics, "new": new_metrics, "delta": delta}


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# v5.24 vs v5.23 Recall Stage Comparison",
        "",
        "| Metric | v5.23 | v5.24 | Delta |",
        "|--------|-------|-------|-------|",
    ]
    keys = [
        "proposal_recall",
        "classified_recall",
        "accepted_recall",
        "final_survival_recall",
        "proposal_matched",
        "classified_matched",
        "accepted_matched",
        "final_survival_matched",
    ]
    for key in keys:
        old_value = comparison["old"][key]
        new_value = comparison["new"][key]
        delta = comparison["delta"][key]
        if isinstance(old_value, float):
            lines.append(f"| {key} | {old_value:.4f} | {new_value:.4f} | {delta:+.4f} |")
        else:
            lines.append(f"| {key} | {old_value} | {new_value} | {delta:+d} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare object proposal recall stages between two reports.")
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--new-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_stage_reports(load_json(args.old_json), load_json(args.new_json))
    write_markdown(args.output_md, comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_compare_recall_stage_reports -v
```

Expected:

```text
OK
```

- [ ] **Step 4: Generate stage comparison**

Run:

```powershell
W:\python\envs\feagpt\python.exe compare_recall_stage_reports.py `
  --old-json output_real_step_s3_v5_23_targeted_proposer_repair\reports\proposal_recall_s3_v5_23_targeted_proposer.json `
  --new-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\proposal_recall_s3_v5_24_reasonaware.json `
  --output-md output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_stage_comparison.md
```

## 10. Task 7: Fix Review-Aware Report Title

**Files:**

- Modify: `review_aware_residual_failure_atlas.py`

- [ ] **Step 1: Add optional title argument**

In `parse_args`, add:

```python
    parser.add_argument("--title", default="Review-Aware Residual Failure Atlas")
```

Change `write_markdown` signature:

```python
def write_markdown(path: Path, payload: dict[str, Any], *, title: str = "Review-Aware Residual Failure Atlas") -> None:
```

Change first markdown line from the static v5.22 title to:

```python
        f"# {title}",
```

In `main`, call:

```python
    write_markdown(args.output_dir / "review_aware_residual_atlas.md", payload, title=args.title)
```

- [ ] **Step 2: Regenerate v5.24 review-aware markdown title**

Run:

```powershell
W:\python\envs\feagpt\python.exe review_aware_residual_failure_atlas.py `
  --atlas-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_proposal_miss_atlas.json `
  --review-csv output_real_step_s3_v5_20_user_semantic_review\reports\semantic_review_template.csv `
  --semantic-policy-json output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\semantic_policy_recall_s3_v5_24.json `
  --output-dir output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports `
  --title "v5.24 Review-Aware Residual Failure Atlas"
```

Expected:

```text
review_aware_residual_atlas.md starts with # v5.24 Review-Aware Residual Failure Atlas
```

## 11. Task 8: Write v5.24 Decision

**Files:**

- Generate: `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_decision.md`

- [ ] **Step 1: Inspect comparison reports**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_stage_comparison.md
Get-Content -LiteralPath output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\v5_24_vs_v5_23_review_aware_comparison.md
Get-Content -LiteralPath output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports\semantic_policy_recall_s3_v5_24.md
```

- [ ] **Step 2: Create decision memo**

Create `output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_decision.md` with this exact structure and filled numeric values from reports:

```markdown
# v5.24 Decision

## Recommendation

**<one of: accept_reason_aware_classifier | tune_classifier_thresholds | switch_to_postprocess_finalization | stop_classifier_tuning_and_plan_model_data_redesign>**

## Evidence

- v5.23 proposal_recall:
- v5.24 proposal_recall:
- v5.23 classified_recall:
- v5.24 classified_recall:
- v5.23 accepted_recall:
- v5.24 accepted_recall:
- v5.23 final_survival_recall:
- v5.24 final_survival_recall:
- v5.23 semantic_policy_recall:
- v5.24 semantic_policy_recall:
- review_aware_miss_count_delta:
- rib_delta:
- fillet_delta:
- chamfer_delta:

## Reasoning

<short explanation grounded in v5_24_stage_comparison.md and v5_24_vs_v5_23_review_aware_comparison.md>
```

Decision rule:

- Choose `accept_reason_aware_classifier` if final-survival recall improves by at least `+0.03` absolute and review-aware residual miss count decreases.
- Choose `tune_classifier_thresholds` if classified recall improves but accepted/final recall barely changes.
- Choose `switch_to_postprocess_finalization` if accepted recall improves but final-survival recall barely changes.
- Choose `stop_classifier_tuning_and_plan_model_data_redesign` if proposal/classified/accepted/final recalls do not improve meaningfully.

## 12. Task 9: Final Verification

**Files:**

- Verify code and reports.

- [ ] **Step 1: Run all relevant tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_instance_candidate_features `
  tests.test_instance_candidate_features_v5_24 `
  tests.test_build_instance_candidate_dataset_v5_24 `
  tests.test_compare_recall_stage_reports `
  tests.test_compare_review_aware_residual_atlases `
  tests.test_topology_postprocess_v5_23 `
  tests.test_object_candidate_proposer_v5_23 `
  tests.test_review_aware_residual_failure_atlas -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Verify report files**

Run:

```powershell
Get-ChildItem -LiteralPath output_real_step_s3_v5_24_reason_aware_candidate_classifier\reports | Select-Object Name,Length,LastWriteTime
```

Expected files:

```text
proposal_recall_s3_v5_24_reasonaware.json
proposal_recall_s3_v5_24_reasonaware.md
semantic_policy_recall_s3_v5_24.json
semantic_policy_recall_s3_v5_24.md
v5_24_proposal_miss_atlas.json
v5_24_proposal_miss_atlas.md
review_aware_residual_atlas.json
review_aware_residual_atlas.md
v5_24_vs_v5_23_review_aware_comparison.md
v5_24_stage_comparison.md
v5_24_decision.md
```

## 13. Final Response Requirements

The implementing agent must report:

```text
1. Test commands and results.
2. Files modified.
3. Dataset record_count, source_counts, assembly_reason_positive_clone_count.
4. Candidate classifier test macro-F1 and top 10 features.
5. v5.23 vs v5.24 proposal/classified/accepted/final recalls.
6. v5.23 vs v5.24 semantic-policy recall.
7. v5.23 vs v5.24 review-aware residual deltas for rib, fillet, chamfer, through_hole_round, slot_rect, cylinder_boss.
8. v5.24 recommendation.
9. Confirmation that GNN, graph extraction, STEP generation, area truth, and existing model checkpoints were not modified.
```

## 14. How I Will Review This Work

I will inspect:

```text
utils/instance_candidate_features.py
build_instance_candidate_dataset.py
review_aware_residual_failure_atlas.py
tests/test_instance_candidate_features_v5_24.py
tests/test_build_instance_candidate_dataset_v5_24.py
compare_recall_stage_reports.py
tests/test_compare_recall_stage_reports.py
output_generation_v5_24_reason_aware_instance_candidate_dataset/instance_candidate_dataset/dataset_summary.json
output_generation_v5_24_reason_aware_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_24_reasonaware/report.md
output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_stage_comparison.md
output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_vs_v5_23_review_aware_comparison.md
output_real_step_s3_v5_24_reason_aware_candidate_classifier/reports/v5_24_decision.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest `
  tests.test_instance_candidate_features `
  tests.test_instance_candidate_features_v5_24 `
  tests.test_build_instance_candidate_dataset_v5_24 `
  tests.test_compare_recall_stage_reports `
  tests.test_compare_review_aware_residual_atlases `
  tests.test_topology_postprocess_v5_23 `
  tests.test_object_candidate_proposer_v5_23 `
  tests.test_review_aware_residual_failure_atlas -v
```

I will reject the implementation if:

- It trains on real s3 area truth as if it were face-level truth.
- It changes area truth to make metrics look better.
- It changes the v5.23 proposer again while claiming to test classifier repair.
- It does not add assembly-reason features.
- It only improves synthetic test macro-F1 but does not rerun s3 final evaluation.
- It overwrites v5.23 outputs instead of creating v5.24 output folders.

## 15. Next Plan After v5.24

Use `v5_24_decision.md`:

- If `accept_reason_aware_classifier`, freeze v5.24 as the current best pipeline and run a broader report for the paper.
- If `tune_classifier_thresholds`, write v5.25 threshold calibration over classifier confidence and overlap policy.
- If `switch_to_postprocess_finalization`, write v5.25 final-prediction merge/remap repair plan.
- If `stop_classifier_tuning_and_plan_model_data_redesign`, stop local classifier tweaks and plan either richer synthetic data or a stronger graph/candidate model.

## 16. English Handoff Summary

v5.23 improved proposal-stage recall but not final-survival recall. The likely reason is that the v5.10 candidate classifier was trained without v5.23 targeted topology-rule candidates and without assembly-reason features. v5.24 should add assembly-reason features, rebuild a hybrid reason-aware candidate dataset, retrain the candidate classifier, and rerun the same s3 evaluation chain. Do not touch GNN, STEP generation, truth tables, or proposer rules in v5.24.
