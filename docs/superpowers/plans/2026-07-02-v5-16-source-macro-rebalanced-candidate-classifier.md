# v5.16 Source/Macro Rebalanced Candidate Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the hybrid candidate classifier training set so valid `fallback_rule` and `topology_rule` candidates are not systematically learned as `reject`, then verify whether real S3 accepted/final-survival recall recovers toward the v5.13 baseline.

**Architecture:** Keep the current GNN, graph extraction, and hybrid proposer unchanged. Add controlled source/macro positive cloning and source/macro sample weighting in `build_instance_candidate_dataset.py`, retrain only the candidate-level classifier, and compare v5.13 old classifier, v5.14 hybrid classifier, and v5.16 rebalanced classifier on the same S3 proposal recall protocol.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, JSON/JSONL, scikit-learn RandomForest via `train_candidate_topology_classifier.py`, existing `object_proposal_recall_report.py`, existing `candidate_classifier_error_atlas.py`.

---

## 0. Review of Previous Agent Output

The v5.15 agent satisfies the requested diagnostic plan.

Verified command:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_candidate_classifier_error_atlas -v
```

Observed result:

```text
Ran 3 tests in 0.117s
OK
```

Important v5.15 outputs:

```text
output_real_step_s3_v5_15_classifier_error_atlas/reports/classifier_error_atlas_s3.json
output_real_step_s3_v5_15_classifier_error_atlas/reports/classifier_error_atlas_s3.md
output_real_step_s3_v5_15_classifier_error_atlas/reports/training_label_audit_v5_14.json
output_real_step_s3_v5_15_classifier_error_atlas/reports/training_label_audit_v5_14.md
output_real_step_s3_v5_15_classifier_error_atlas/reports/v5_15_decision.md
```

The recommendation is reasonable:

```text
keep_v5_13_classifier_and_build_source_macro_label_rebalance
```

Evidence:

```text
both_correct_accept: 76
both_wrong_on_matched_proposal: 52
old_correct_accept_new_reject: 16
unmatched_proposal: 356

fallback_rule / through_hole_round:
  lost old-correct count: 14
  train records: 4914
  train positive rate: 0.0000
  train reject rate: 1.0000

topology_rule / through_hole_round:
  lost old-correct count: 13
  train records: 18
  train positive rate: 0.0000
  train reject rate: 1.0000

topology_rule / rectangular_boss:
  lost old-correct count: 2
  train records: 364
  train positive rate: 0.0549
  train reject rate: 0.9451
```

Interpretation:

- v5.14 did not improve real S3 results because it learned that some proposal sources are almost always negative.
- The strongest signal is source/macro label bias, not a need to change the GNN yet.
- v5.16 must keep v5.13 old classifier as baseline and prove the rebalanced classifier improves real S3 accepted-stage recall before it becomes the new default.

## 1. Scope Control

### Modify

- `build_instance_candidate_dataset.py`
- `tests/test_build_instance_candidate_dataset.py`

### Use Existing Scripts Without Changing

- `train_candidate_topology_classifier.py`
- `object_proposal_recall_report.py`
- `candidate_classifier_error_atlas.py`

### Create Generated Outputs

- `output_generation_v5_16_rebalanced_instance_candidate_dataset/instance_candidate_dataset/`
- `output_generation_v5_16_rebalanced_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_16_rebalanced/`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/proposal_recall_s3_v5_16.json`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/proposal_recall_s3_v5_16.md`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/classifier_error_atlas_s3.json`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/classifier_error_atlas_s3.md`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/training_label_audit_v5_14.json`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/training_label_audit_v5_14.md`
- `output_real_step_s3_v5_16_rebalanced_classifier/reports/v5_16_decision.md`

### Do Not Modify

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `utils/object_candidate_proposer.py`
- `utils/instance_candidate_features.py`
- `utils/instance_candidate_postprocess.py`
- `run_gnn_inference.py`
- STEP generation scripts
- existing v5.13/v5.14/v5.15 report directories

## 2. Fixed Inputs

Use the same synthetic training inputs as v5.14:

```text
Synthetic graph dir:
output_generation_v5_5_v51dist_looprole_5400/graphs

Synthetic raw prediction dir:
output_generation_v5_10_instance_candidate_dataset_v55/predictions_raw
```

Use the same real S3 evaluation inputs as v5.13/v5.14/v5.15:

```text
S3 graph dir:
output_real_step_s3_v5_6_b_rib_boundary/graphs

S3 raw prediction dir:
output_real_step_s3_v5_11_proposal_diagnostics/raw_predictions

S3 area truth:
configs/test_part_s3_area_truth_part20_mm2.json

Reference final prediction dir:
output_real_step_s3_v5_10_instance_plus_v59/predictions

v5.13 old classifier:
output_generation_v5_10_instance_candidate_dataset_v55/model_runs/random_forest_instance_candidate_v5_10_v55/model.joblib

v5.14 regressed classifier:
output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid/model.joblib
```

## 3. Success Criteria

The implementation is acceptable only if all checks pass:

```text
tests/test_build_instance_candidate_dataset.py passes
tests/test_train_candidate_topology_classifier.py passes
tests/test_candidate_classifier_error_atlas.py passes
```

The v5.16 dataset summary must show:

```text
proposal_mode: hybrid
source_macro_positive_clone_count > 0
rebalance_action_counts.positive_source_macro_clone > 0
rebalance_action_counts.downweighted_reject > 0
```

The v5.16 real S3 report must show:

```text
proposal recall remains 0.6814 or higher
accepted-stage recall is higher than v5.14 value 0.3971
classifier wrong/rejected is lower than v5.14 value 56
```

If v5.16 accepted-stage recall is not higher than v5.14, stop and recommend `candidate_feature_revision` instead of further weighting.

## 4. Rebalance Policy

Use these source/macro pairs for v5.16:

```text
fallback_rule:through_hole_round
topology_rule:through_hole_round
topology_rule:rectangular_boss
```

Use these weights:

```text
default_sample_weight = 1.0
positive_nontruth_weight = 3.0
positive_clone_weight = 5.0
downweighted_reject_weight = 0.35
```

Meaning:

- Clone truth-positive instances as if they came from selected proposal sources.
- Upweight non-truth positives so the classifier learns valid object candidates, not just exact truth candidates.
- Downweight rejects from the selected source/macro pairs so source tags do not become a near-hard negative rule.
- Do not relabel existing rejected candidates. Rejected candidates remain rejected; their weight is reduced only for diagnosed biased source/macro pairs.

## 5. Task 1: Add Dataset Rebalance Tests

**Files:**

- Modify: `tests/test_build_instance_candidate_dataset.py`
- Modify: `build_instance_candidate_dataset.py`

- [ ] **Step 1: Add failing tests for source/macro parsing and weighting**

Append these tests inside `BuildInstanceCandidateDatasetTest` in `tests/test_build_instance_candidate_dataset.py`, before the `if __name__ == "__main__":` block:

```python
    def test_parse_source_macro_pairs_accepts_csv_and_ignores_empty_items(self):
        from build_instance_candidate_dataset import parse_source_macro_pairs

        pairs = parse_source_macro_pairs(
            "fallback_rule:through_hole_round, topology_rule:rectangular_boss, "
        )

        self.assertEqual(
            pairs,
            {
                ("fallback_rule", "through_hole_round"),
                ("topology_rule", "rectangular_boss"),
            },
        )

    def test_parse_source_macro_pairs_rejects_malformed_items(self):
        from build_instance_candidate_dataset import parse_source_macro_pairs

        with self.assertRaisesRegex(ValueError, "source:macro"):
            parse_source_macro_pairs("fallback_rule-through_hole_round")

    def test_hybrid_rebalance_adds_positive_source_clone_and_downweights_reject(self):
        from unittest.mock import patch
        from pathlib import Path

        from build_instance_candidate_dataset import build_candidate_records_for_graph

        graph_data = {
            "nodes": [
                node(0, primitive="cylinder", area=20.0),
                node(1, primitive="cylinder", area=18.0),
            ],
            "edges": [],
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "through_hole_round",
                    "node_indices": [0],
                }
            ],
        }
        prediction_data = {
            "sample_id": "toy",
            "faces": [
                {"idx": 0, "macro_feature": "stock"},
                {"idx": 1, "macro_feature": "stock"},
            ],
        }
        bad_fallback_candidate = {
            "source": "fallback_rule",
            "proposal_sources": ["fallback_rule"],
            "proposal_macro": "through_hole_round",
            "face_indices": [1],
            "seed_face_indices": [1],
            "assembly_reasons": ["fallback_two_circular_plane_boundaries"],
        }

        with patch("build_instance_candidate_dataset.assemble_instance_candidates", return_value=[]), \
             patch("build_instance_candidate_dataset.generate_hybrid_object_proposals", return_value=[bad_fallback_candidate]):
            records = build_candidate_records_for_graph(
                sample_id="toy",
                graph_data=graph_data,
                prediction_data=prediction_data,
                graph_path=Path("toy.json"),
                prediction_path=Path("toy_pred.json"),
                target_macros={"through_hole_round"},
                min_iou=0.5,
                min_candidate_coverage=0.7,
                min_truth_coverage=0.7,
                proposal_mode="hybrid",
                source_macro_positive_clone_pairs={("fallback_rule", "through_hole_round")},
                reject_downweight_source_macro_pairs={("fallback_rule", "through_hole_round")},
                default_sample_weight=1.0,
                positive_nontruth_weight=3.0,
                positive_clone_weight=5.0,
                downweighted_reject_weight=0.35,
            )

        by_source_faces = {
            (record["source"], tuple(record["face_indices"])): record
            for record in records
        }

        clone = by_source_faces[("fallback_rule", (0,))]
        self.assertEqual(clone["label"], "through_hole_round")
        self.assertEqual(clone["augmentation"], "source_macro_positive_clone")
        self.assertEqual(clone["rebalance_action"], "positive_source_macro_clone")
        self.assertAlmostEqual(clone["sample_weight"], 5.0)
        self.assertEqual(clone["features"]["proposal_source_fallback_rule"], 1.0)

        reject = by_source_faces[("fallback_rule", (1,))]
        self.assertEqual(reject["label"], "reject")
        self.assertEqual(reject["rebalance_action"], "downweighted_reject")
        self.assertAlmostEqual(reject["sample_weight"], 0.35)

    def test_write_outputs_reports_rebalance_action_and_weight_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            records = [
                {
                    "sample_id": "a",
                    "source": "truth",
                    "proposal_macro": "rib",
                    "label": "rib",
                    "features": {"face_count": 1.0},
                    "sample_weight": 1.0,
                    "rebalance_action": "default",
                },
                {
                    "sample_id": "b",
                    "source": "fallback_rule",
                    "proposal_macro": "through_hole_round",
                    "label": "through_hole_round",
                    "features": {"face_count": 1.0},
                    "sample_weight": 5.0,
                    "rebalance_action": "positive_source_macro_clone",
                    "augmentation": "source_macro_positive_clone",
                },
                {
                    "sample_id": "c",
                    "source": "fallback_rule",
                    "proposal_macro": "through_hole_round",
                    "label": "reject",
                    "features": {"face_count": 1.0},
                    "sample_weight": 0.35,
                    "rebalance_action": "downweighted_reject",
                },
            ]

            summary = write_dataset_outputs(
                records,
                output_dir,
                seed=1,
                train_ratio=0.5,
                val_ratio=0.0,
                target_macros={"rib", "through_hole_round"},
                skipped_prediction_count=0,
                proposal_mode="hybrid",
            )

            self.assertEqual(summary["rebalance_action_counts"]["default"], 1)
            self.assertEqual(summary["rebalance_action_counts"]["positive_source_macro_clone"], 1)
            self.assertEqual(summary["rebalance_action_counts"]["downweighted_reject"], 1)
            self.assertEqual(summary["source_macro_positive_clone_count"], 1)
            self.assertAlmostEqual(summary["sample_weight_stats"]["min"], 0.35)
            self.assertAlmostEqual(summary["sample_weight_stats"]["max"], 5.0)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset -v
```

Expected result before implementation:

```text
ERROR
```

The failure should mention missing `parse_source_macro_pairs` or unexpected keyword arguments for `build_candidate_records_for_graph`.

## 6. Task 2: Implement Source/Macro Rebalance in Dataset Builder

**Files:**

- Modify: `build_instance_candidate_dataset.py`

- [ ] **Step 1: Add CLI arguments**

In `parse_args()`, after `--proposal-mode`, add:

```python
    parser.add_argument("--source-macro-positive-clones", default="")
    parser.add_argument("--reject-downweight-source-macros", default="")
    parser.add_argument("--default-sample-weight", type=float, default=1.0)
    parser.add_argument("--positive-nontruth-weight", type=float, default=3.0)
    parser.add_argument("--positive-clone-weight", type=float, default=5.0)
    parser.add_argument("--downweighted-reject-weight", type=float, default=0.35)
```

- [ ] **Step 2: Add source/macro parsing helper**

Add this function after `parse_macro_set()`:

```python
def parse_source_macro_pairs(value: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"source/macro item must use source:macro format, got {item!r}")
        source, macro = (part.strip() for part in item.split(":", 1))
        if not source or not macro:
            raise ValueError(f"source/macro item must use source:macro format, got {item!r}")
        pairs.add((source, macro))
    return pairs
```

- [ ] **Step 3: Add clone proposal builder**

Add this function after `build_truth_instances()`:

```python
def build_source_macro_positive_clone_proposals(
    truth_instances: list[dict[str, Any]],
    source_macro_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for instance in truth_instances:
        macro = str(instance["macro_feature"])
        face_indices = list(instance["face_indices"])
        for source, pair_macro in sorted(source_macro_pairs):
            if pair_macro != macro:
                continue
            proposals.append(
                {
                    "source": source,
                    "proposal_sources": [source],
                    "proposal_macro": macro,
                    "face_indices": face_indices,
                    "augmentation": "source_macro_positive_clone",
                    "assembly_reasons": [f"source_macro_positive_clone:{source}:{macro}"],
                }
            )
    return proposals
```

- [ ] **Step 4: Add sample-weight helper**

Add this function after `label_candidate()`:

```python
def assign_candidate_sample_weight(
    *,
    source: str,
    proposal_macro: str,
    label: str,
    augmentation: str | None,
    source_macro_positive_clone_pairs: set[tuple[str, str]],
    reject_downweight_source_macro_pairs: set[tuple[str, str]],
    default_sample_weight: float,
    positive_nontruth_weight: float,
    positive_clone_weight: float,
    downweighted_reject_weight: float,
) -> tuple[float, str]:
    source_macro = (source, proposal_macro)
    if label == REJECT_LABEL and source_macro in reject_downweight_source_macro_pairs:
        return float(downweighted_reject_weight), "downweighted_reject"
    if label != REJECT_LABEL and augmentation == "source_macro_positive_clone":
        return float(positive_clone_weight), "positive_source_macro_clone"
    if label != REJECT_LABEL and source != "truth":
        if source_macro in source_macro_positive_clone_pairs:
            return float(positive_clone_weight), "positive_source_macro_pair"
        return float(positive_nontruth_weight), "positive_nontruth"
    return float(default_sample_weight), "default"
```

- [ ] **Step 5: Thread rebalance parameters through `build_proposals()`**

Change the `build_proposals()` signature to include:

```python
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
```

Inside `build_proposals()`, after truth proposals are appended and before prediction candidates are appended, add:

```python
    clone_pairs = source_macro_positive_clone_pairs or set()
    proposals.extend(build_source_macro_positive_clone_proposals(truth_instances, clone_pairs))
```

- [ ] **Step 6: Thread rebalance parameters through `build_candidate_records_for_graph()`**

Change the `build_candidate_records_for_graph()` signature to include these keyword-only arguments:

```python
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
    reject_downweight_source_macro_pairs: set[tuple[str, str]] | None = None,
    default_sample_weight: float = 1.0,
    positive_nontruth_weight: float = 3.0,
    positive_clone_weight: float = 5.0,
    downweighted_reject_weight: float = 0.35,
```

When calling `build_proposals()`, pass:

```python
        source_macro_positive_clone_pairs=source_macro_positive_clone_pairs or set(),
```

After `label, match = label_candidate(...)`, add:

```python
        augmentation = proposal.get("augmentation")
        sample_weight, rebalance_action = assign_candidate_sample_weight(
            source=source,
            proposal_macro=proposal_macro,
            label=label,
            augmentation=str(augmentation) if augmentation is not None else None,
            source_macro_positive_clone_pairs=source_macro_positive_clone_pairs or set(),
            reject_downweight_source_macro_pairs=reject_downweight_source_macro_pairs or set(),
            default_sample_weight=default_sample_weight,
            positive_nontruth_weight=positive_nontruth_weight,
            positive_clone_weight=positive_clone_weight,
            downweighted_reject_weight=downweighted_reject_weight,
        )
```

In the `record = { ... }` dictionary, add:

```python
            "sample_weight": sample_weight,
            "rebalance_action": rebalance_action,
```

After the existing optional `assembly_reasons` block, add:

```python
        if augmentation is not None:
            record["augmentation"] = str(augmentation)
```

- [ ] **Step 7: Add summary statistics**

Add this helper before `write_dataset_outputs()`:

```python
def summarize_sample_weights(records: list[dict[str, Any]]) -> dict[str, float]:
    weights = [_finite_float(record.get("sample_weight", 1.0)) for record in records]
    if not weights:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(weights)),
        "max": float(max(weights)),
        "mean": float(sum(weights) / len(weights)),
    }
```

In `write_dataset_outputs()`, after `source_counts = Counter(...)`, add:

```python
    rebalance_action_counts = Counter(record.get("rebalance_action", "default") for record in enriched_records)
    source_macro_positive_clone_count = sum(
        1 for record in enriched_records
        if record.get("augmentation") == "source_macro_positive_clone"
    )
```

In `summary = { ... }`, add:

```python
        "rebalance_action_counts": dict(sorted(rebalance_action_counts.items())),
        "source_macro_positive_clone_count": int(source_macro_positive_clone_count),
        "sample_weight_stats": summarize_sample_weights(enriched_records),
```

- [ ] **Step 8: Thread rebalance parameters through `build_dataset_from_paths()`**

Change the `build_dataset_from_paths()` signature to include:

```python
    source_macro_positive_clone_pairs: set[tuple[str, str]] | None = None,
    reject_downweight_source_macro_pairs: set[tuple[str, str]] | None = None,
    default_sample_weight: float = 1.0,
    positive_nontruth_weight: float = 3.0,
    positive_clone_weight: float = 5.0,
    downweighted_reject_weight: float = 0.35,
```

When calling `build_candidate_records_for_graph()`, pass these values through.

- [ ] **Step 9: Thread CLI args into `main()`**

In `main()`, pass:

```python
        source_macro_positive_clone_pairs=parse_source_macro_pairs(args.source_macro_positive_clones),
        reject_downweight_source_macro_pairs=parse_source_macro_pairs(args.reject_downweight_source_macros),
        default_sample_weight=args.default_sample_weight,
        positive_nontruth_weight=args.positive_nontruth_weight,
        positive_clone_weight=args.positive_clone_weight,
        downweighted_reject_weight=args.downweighted_reject_weight,
```

- [ ] **Step 10: Run tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset -v
```

Expected:

```text
OK
```

- [ ] **Step 11: Run training-script tests to confirm `sample_weight` remains supported**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_train_candidate_topology_classifier -v
```

Expected:

```text
OK
```

## 7. Task 3: Build v5.16 Rebalanced Hybrid Candidate Dataset

**Files:**

- Generate: `output_generation_v5_16_rebalanced_instance_candidate_dataset/instance_candidate_dataset/`

- [ ] **Step 1: Build dataset**

Run:

```powershell
W:\python\envs\feagpt\python.exe build_instance_candidate_dataset.py `
  --graph-dir output_generation_v5_5_v51dist_looprole_5400\graphs `
  --prediction-dir output_generation_v5_10_instance_candidate_dataset_v55\predictions_raw `
  --output-dir output_generation_v5_16_rebalanced_instance_candidate_dataset\instance_candidate_dataset `
  --recursive `
  --proposal-mode hybrid `
  --source-macro-positive-clones "fallback_rule:through_hole_round,topology_rule:through_hole_round,topology_rule:rectangular_boss" `
  --reject-downweight-source-macros "fallback_rule:through_hole_round,topology_rule:through_hole_round,topology_rule:rectangular_boss" `
  --default-sample-weight 1.0 `
  --positive-nontruth-weight 3.0 `
  --positive-clone-weight 5.0 `
  --downweighted-reject-weight 0.35 `
  --seed 20260702 `
  --train-ratio 0.80 `
  --val-ratio 0.10
```

Expected console output:

```text
"record_count"
"split_counts"
"label_counts"
```

- [ ] **Step 2: Inspect dataset summary**

Run:

```powershell
Get-Content -LiteralPath output_generation_v5_16_rebalanced_instance_candidate_dataset\instance_candidate_dataset\dataset_summary.json
```

Expected conditions:

```text
"proposal_mode": "hybrid"
"source_macro_positive_clone_count" is greater than 0
"positive_source_macro_clone" appears under "rebalance_action_counts"
"downweighted_reject" appears under "rebalance_action_counts"
"sample_weight_stats"."min" equals 0.35
"sample_weight_stats"."max" equals 5.0
```

- [ ] **Step 3: If conditions fail, stop**

If any expected condition in Step 2 fails, do not train. Fix Task 2 first.

## 8. Task 4: Train v5.16 Candidate Classifier

**Files:**

- Generate: `output_generation_v5_16_rebalanced_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_16_rebalanced/`

- [ ] **Step 1: Train RandomForest**

Run:

```powershell
W:\python\envs\feagpt\python.exe train_candidate_topology_classifier.py `
  --dataset-dir output_generation_v5_16_rebalanced_instance_candidate_dataset\instance_candidate_dataset `
  --output-dir output_generation_v5_16_rebalanced_instance_candidate_dataset\model_runs `
  --run-name random_forest_instance_candidate_v5_16_rebalanced `
  --model-kind random_forest `
  --seed 20260702 `
  --n-estimators 300 `
  --max-depth 18 `
  --hard-source-weight 1.0
```

Expected console output:

```text
"run_dir": "output_generation_v5_16_rebalanced_instance_candidate_dataset\\model_runs\\random_forest_instance_candidate_v5_16_rebalanced"
"test_macro_f1"
```

- [ ] **Step 2: Confirm model exists**

Run:

```powershell
Test-Path output_generation_v5_16_rebalanced_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_16_rebalanced\model.joblib
```

Expected:

```text
True
```

- [ ] **Step 3: Inspect synthetic metrics**

Run:

```powershell
Get-Content -LiteralPath output_generation_v5_16_rebalanced_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_16_rebalanced\report.md
```

Expected conditions:

```text
test macro-F1 is at least 0.93
test weighted-F1 is at least 0.93
```

If synthetic test macro-F1 falls below 0.93, keep the artifact but mark v5.16 as failed in the decision report.

## 9. Task 5: Run S3 Proposal Recall with v5.16 Classifier

**Files:**

- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/proposal_recall_s3_v5_16.json`
- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/proposal_recall_s3_v5_16.md`

- [ ] **Step 1: Run S3 evaluation**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_16_rebalanced_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_16_rebalanced\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_16_rebalanced_classifier\reports\proposal_recall_s3_v5_16.json `
  --output-md output_real_step_s3_v5_16_rebalanced_classifier\reports\proposal_recall_s3_v5_16.md `
  --proposal-mode hybrid `
  --strong-thresholds "0.3,0.4,0.5,0.55,0.6,0.7" `
  --weak-thresholds "0.2,0.3,0.35,0.4,0.5" `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Expected console output:

```text
"best_by_proposal_recall"
"best_by_final_survival_recall"
```

- [ ] **Step 2: Inspect report**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_16_rebalanced_classifier\reports\proposal_recall_s3_v5_16.md -TotalCount 80
```

Expected conditions:

```text
Proposal recall is 0.6814 or higher.
Accepted-stage recall is higher than 0.3971.
Classifier wrong/rejected is lower than 56.
```

## 10. Task 6: Compare v5.13, v5.14, and v5.16

**Files:**

- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/v5_13_v5_14_v5_16_comparison.md`
- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/v5_16_decision.md`

- [ ] **Step 1: Add a short comparison script**

Create `compare_candidate_classifier_recall_reports.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_final(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("best_by_final_survival_recall") or {})


def metric_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    best = best_final(report)
    failure_counts = best.get("failure_stage_counts", {})
    return {
        "name": name,
        "proposal_recall": float(best.get("proposal_stage", {}).get("recall", 0.0)),
        "classified_recall": float(best.get("classified_stage", {}).get("recall", 0.0)),
        "accepted_recall": float(best.get("accepted_stage", {}).get("recall", 0.0)),
        "final_survival_recall": float(best.get("final_survival_stage", {}).get("recall", 0.0)),
        "no_compatible_proposal": int(failure_counts.get("no_area_compatible_proposal", 0)),
        "classifier_wrong_or_rejected": int(failure_counts.get("proposal_exists_classifier_wrong_or_rejected", 0)),
        "accepted_but_final_changes": int(failure_counts.get("classifier_accepts_but_final_changes_or_removes", 0)),
        "survives_final": int(failure_counts.get("survives_final_prediction", 0)),
    }


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v5.13 vs v5.14 vs v5.16 Candidate Classifier Comparison",
        "",
        "| Version | Proposal recall | Classified recall | Accepted recall | Final-survival recall | No proposal | Classifier wrong/rejected | Final changes | Survives final |",
        "|---------|-----------------|-------------------|-----------------|-----------------------|-------------|---------------------------|---------------|----------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | "
            f"{row['proposal_recall']:.4f} | "
            f"{row['classified_recall']:.4f} | "
            f"{row['accepted_recall']:.4f} | "
            f"{row['final_survival_recall']:.4f} | "
            f"{row['no_compatible_proposal']} | "
            f"{row['classifier_wrong_or_rejected']} | "
            f"{row['accepted_but_final_changes']} | "
            f"{row['survives_final']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    v514 = by_name["v5.14"]
    v516 = by_name["v5.16"]
    if (
        v516["accepted_recall"] > v514["accepted_recall"]
        and v516["classifier_wrong_or_rejected"] < v514["classifier_wrong_or_rejected"]
    ):
        recommendation = "promote_v5_16_for_candidate_classifier_testing"
        reason = "v5.16 improves accepted-stage recall and reduces classifier wrong/rejected failures compared with v5.14."
    else:
        recommendation = "do_not_promote_v5_16_move_to_candidate_feature_revision"
        reason = "v5.16 does not beat v5.14 on both accepted-stage recall and classifier wrong/rejected failures."
    lines = [
        "# v5.16 Decision",
        "",
        f"## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Required Next Step",
        "",
        "If promoted, rerun full feature-object export with the v5.16 classifier. If not promoted, keep v5.13 as current best and implement candidate feature revision.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate classifier recall reports.")
    parser.add_argument("--v513", type=Path, required=True)
    parser.add_argument("--v514", type=Path, required=True)
    parser.add_argument("--v516", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--decision-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        metric_row("v5.13", load_json(args.v513)),
        metric_row("v5.14", load_json(args.v514)),
        metric_row("v5.16", load_json(args.v516)),
    ]
    write_comparison(args.output_md, rows)
    write_decision(args.decision_md, rows)
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run comparison**

Run:

```powershell
W:\python\envs\feagpt\python.exe compare_candidate_classifier_recall_reports.py `
  --v513 output_real_step_s3_v5_13_transition_opening_proposals\reports\proposal_recall_s3_v5_13.json `
  --v514 output_real_step_s3_v5_14_hybrid_classifier\reports\proposal_recall_s3_v5_14.json `
  --v516 output_real_step_s3_v5_16_rebalanced_classifier\reports\proposal_recall_s3_v5_16.json `
  --output-md output_real_step_s3_v5_16_rebalanced_classifier\reports\v5_13_v5_14_v5_16_comparison.md `
  --decision-md output_real_step_s3_v5_16_rebalanced_classifier\reports\v5_16_decision.md
```

Expected output:

```text
"name": "v5.13"
"name": "v5.14"
"name": "v5.16"
```

- [ ] **Step 3: Inspect decision**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_16_rebalanced_classifier\reports\v5_16_decision.md
```

Expected:

```text
Either promote_v5_16_for_candidate_classifier_testing
or do_not_promote_v5_16_move_to_candidate_feature_revision
```

## 11. Task 7: Run v5.15 Error Atlas Against v5.16

**Files:**

- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/classifier_error_atlas_s3.json`
- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/classifier_error_atlas_s3.md`
- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/training_label_audit_v5_14.json`
- Generate: `output_real_step_s3_v5_16_rebalanced_classifier/reports/training_label_audit_v5_14.md`

- [ ] **Step 1: Compare v5.13 old classifier against v5.16**

Run:

```powershell
W:\python\envs\feagpt\python.exe candidate_classifier_error_atlas.py `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --old-classifier output_generation_v5_10_instance_candidate_dataset_v55\model_runs\random_forest_instance_candidate_v5_10_v55\model.joblib `
  --new-classifier output_generation_v5_16_rebalanced_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_16_rebalanced\model.joblib `
  --training-records output_generation_v5_16_rebalanced_instance_candidate_dataset\instance_candidate_dataset\records.jsonl `
  --output-dir output_real_step_s3_v5_16_rebalanced_classifier\reports `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

Expected output:

```text
"decision"
"old_correct_accept_new_reject"
```

- [ ] **Step 2: Inspect atlas**

Run:

```powershell
Get-Content -LiteralPath output_real_step_s3_v5_16_rebalanced_classifier\reports\classifier_error_atlas_s3.md -TotalCount 120
```

Expected conditions:

```text
old_correct_accept_new_reject is lower than 16
fallback_rule / through_hole_round no longer has train positive rate 0.0000 in training_label_audit_v5_14.md
topology_rule / through_hole_round no longer has train positive rate 0.0000 in training_label_audit_v5_14.md
```

## 12. Final Review Checklist for the Implementing Agent

- [ ] `tests.test_build_instance_candidate_dataset` passes.
- [ ] `tests.test_train_candidate_topology_classifier` passes.
- [ ] `tests.test_candidate_classifier_error_atlas` passes.
- [ ] `dataset_summary.json` contains nonzero positive clone and downweighted reject counts.
- [ ] `model.joblib` exists under the v5.16 model run directory.
- [ ] `proposal_recall_s3_v5_16.md` exists.
- [ ] `v5_13_v5_14_v5_16_comparison.md` exists.
- [ ] `v5_16_decision.md` exists.
- [ ] The final response reports whether v5.16 beats v5.14 on accepted-stage recall and classifier wrong/rejected count.
- [ ] The final response states whether v5.16 should be promoted, rejected, or followed by candidate feature revision.

## 13. How I Will Review This Work

I will check these files first:

```text
output_generation_v5_16_rebalanced_instance_candidate_dataset/instance_candidate_dataset/dataset_summary.json
output_generation_v5_16_rebalanced_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_16_rebalanced/report.md
output_real_step_s3_v5_16_rebalanced_classifier/reports/proposal_recall_s3_v5_16.md
output_real_step_s3_v5_16_rebalanced_classifier/reports/v5_13_v5_14_v5_16_comparison.md
output_real_step_s3_v5_16_rebalanced_classifier/reports/v5_16_decision.md
output_real_step_s3_v5_16_rebalanced_classifier/reports/training_label_audit_v5_14.md
```

I will run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset tests.test_train_candidate_topology_classifier tests.test_candidate_classifier_error_atlas -v
```

I will accept promotion only if v5.16 improves real S3 accepted-stage recall over v5.14 and reduces `proposal_exists_classifier_wrong_or_rejected` below 56. If it improves synthetic metrics but not S3, the next plan should be candidate feature revision rather than more source/macro weighting.

## 14. English Handoff Summary

v5.15 proved that v5.14 regressed because useful real candidates from `fallback_rule` and `topology_rule` were learned as `reject` in the synthetic hybrid candidate dataset. Implement v5.16 by adding source/macro positive clone augmentation and sample weighting in `build_instance_candidate_dataset.py`, retrain only the candidate-level RandomForest classifier, then compare v5.13, v5.14, and v5.16 on the same S3 proposal recall report. Do not change the GNN, graph extractor, STEP generator, or hybrid proposer in this iteration.
