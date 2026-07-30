# v5.14 Hybrid Candidate Classifier Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the instance candidate classifier on the same hybrid proposal sources used by v5.13, then verify whether accepted-stage recall on `test_part_s3` improves.

**Architecture:** Keep the GNN, graph extractor, data generator, and v5.13 proposal generator unchanged. Modify only the candidate-level dataset builder and feature extractor so the classifier is trained on `same_instance`, `raw_macro_component`, `topology_rule`, and `fallback_rule` candidates. Train a new random-forest classifier first, because the main issue is train/inference proposal-distribution mismatch, not proven model capacity.

**Tech Stack:** Python 3 from `W:/python/envs/feagpt/python.exe`, `unittest`, scikit-learn/joblib via existing `train_candidate_topology_classifier.py`, existing generated v5.5 graph dataset, existing v5.10 raw predictions, existing v5.13 s3 proposal diagnostic flow.

---

## 0. Current Review Result

v5.13 satisfies the previous plan.

Verified:

```text
tests: 39 OK
v5.13 proposal recall:       0.6814
v5.13 accepted-stage recall: 0.4804
v5.13 final-survival recall: 0.5196
```

v5.13 source coverage:

```text
same_instance        candidates 281, matched truth  77
raw_macro_component  candidates 219, matched truth  73
topology_rule        candidates 210, matched truth 127
fallback_rule        candidates 127, matched truth  49
```

v5.13 failure stages:

```text
survives_final_prediction                        106
no_area_compatible_proposal                       65
proposal_exists_classifier_wrong_or_rejected      32
classifier_accepts_but_final_changes_or_removes    1
```

Interpretation:

- Proposal recall is now above `0.65`.
- Accepted-stage recall is much lower than proposal recall.
- v5.13 decision `hybrid_candidate_classifier_retraining` is reasonable.
- The old classifier was trained mainly on truth plus assembled same-instance candidates. It was not trained on the new hybrid proposal-source distribution.

## 1. Scope Control

### Files to Modify

- `utils/instance_candidate_features.py`
- `tests/test_instance_candidate_features.py`
- `build_instance_candidate_dataset.py`
- `tests/test_build_instance_candidate_dataset.py`

### Files to Create

- `output_generation_v5_14_hybrid_instance_candidate_dataset/instance_candidate_dataset`
- `output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid`
- `output_real_step_s3_v5_14_hybrid_classifier/reports/proposal_recall_s3_v5_14.json`
- `output_real_step_s3_v5_14_hybrid_classifier/reports/proposal_recall_s3_v5_14.md`
- `output_real_step_s3_v5_14_hybrid_classifier/reports/v5_13_vs_v5_14_comparison.md`
- `output_real_step_s3_v5_14_hybrid_classifier/reports/v5_14_decision.md`

### Files Not to Modify

Do not modify:

- `models/gnn_baseline.py`
- `train_gnn_baseline.py`
- `utils/object_candidate_proposer.py`
- `utils/topology_postprocess.py`
- `run_gnn_inference.py`
- `prediction_to_feature_objects.py`
- any file under `features/`
- any model checkpoint

This task trains a candidate-level classifier for hybrid proposal records. It does not change the base GNN or geometry generation.

## 2. Task 1: Add Proposal-Source Features to Candidate Feature Extraction

**Files:**

- Modify: `utils/instance_candidate_features.py`
- Modify: `tests/test_instance_candidate_features.py`

- [ ] **Step 1: Add source-feature regression test**

Append this test method to `tests/test_instance_candidate_features.py`:

```python
    def test_extract_features_includes_hybrid_proposal_source_buckets(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "attributes": {
                        "physical": {"area": 12.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "shape_descriptors": {
                            "bbox_longest_extent": 4.0,
                            "bbox_middle_extent": 2.0,
                            "bbox_shortest_extent": 1.0,
                            "projected_long_extent": 4.0,
                            "projected_short_extent": 2.0,
                            "projected_aspect_ratio": 2.0,
                        },
                    },
                }
            ],
            "edges": [],
        }
        candidate = {
            "proposal_macro": "through_hole_round",
            "face_indices": [0],
            "proposal_sources": ["topology_rule", "fallback_rule"],
            "source": "hybrid_merged",
        }

        features = extract_instance_candidate_features(
            graph_data,
            candidate,
            prediction_faces=[{"idx": 0, "macro_feature": "stock"}],
            candidate_macro="through_hole_round",
        )

        self.assertEqual(features["proposal_source_count"], 2.0)
        self.assertEqual(features["proposal_source_same_instance"], 0.0)
        self.assertEqual(features["proposal_source_raw_macro_component"], 0.0)
        self.assertEqual(features["proposal_source_topology_rule"], 1.0)
        self.assertEqual(features["proposal_source_fallback_rule"], 1.0)
        self.assertEqual(features["proposal_source_hybrid_merged"], 1.0)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features.InstanceCandidateFeaturesTest.test_extract_features_includes_hybrid_proposal_source_buckets -v
```

Expected before implementation:

```text
FAIL
```

because proposal-source feature keys do not exist.

- [ ] **Step 3: Add source bucket constants**

In `utils/instance_candidate_features.py`, after `PREDICTION_MACRO_BUCKETS`, add:

```python
PROPOSAL_SOURCE_BUCKETS = (
    "same_instance",
    "raw_macro_component",
    "topology_rule",
    "fallback_rule",
    "hybrid_merged",
    "truth",
    "assembled_instance_candidate",
    "other",
)
```

- [ ] **Step 4: Add source extraction helper**

In `utils/instance_candidate_features.py`, after `_candidate_macro`, add:

```python
def _candidate_sources(candidate: Mapping[str, Any]) -> set[str]:
    sources: set[str] = set()
    raw_sources = candidate.get("proposal_sources")
    if isinstance(raw_sources, Iterable) and not isinstance(raw_sources, (str, bytes, Mapping)):
        for source in raw_sources:
            token = str(source).strip()
            if token:
                sources.add(token)
    source = candidate.get("source")
    if source is not None:
        token = str(source).strip()
        if token:
            sources.add(token)
    if not sources:
        sources.add("other")
    return sources
```

- [ ] **Step 5: Add source features**

In `utils/instance_candidate_features.py`, after `_add_prediction_vote_features`, add:

```python
def _add_proposal_source_features(features: dict[str, float], candidate: Mapping[str, Any]) -> None:
    sources = _candidate_sources(candidate)
    features["proposal_source_count"] = float(len(sources))
    for bucket in PROPOSAL_SOURCE_BUCKETS:
        if bucket == "other":
            value = 1.0 if not any(source in PROPOSAL_SOURCE_BUCKETS[:-1] for source in sources) else 0.0
        else:
            value = 1.0 if bucket in sources else 0.0
        features[f"proposal_source_{bucket}"] = value
```

Inside `extract_instance_candidate_features`, after `_add_prediction_vote_features(...)`, call:

```python
    _add_proposal_source_features(features, candidate)
```

- [ ] **Step 6: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features.InstanceCandidateFeaturesTest.test_extract_features_includes_hybrid_proposal_source_buckets -v
```

Expected:

```text
OK
```

## 3. Task 2: Add Hybrid Proposal Mode to Instance Candidate Dataset Builder

**Files:**

- Modify: `build_instance_candidate_dataset.py`
- Modify: `tests/test_build_instance_candidate_dataset.py`

- [ ] **Step 1: Add dataset-builder hybrid mode test**

Append this test method to `BuildInstanceCandidateDatasetTest` in `tests/test_build_instance_candidate_dataset.py`:

```python
    def test_build_records_can_use_hybrid_proposals_and_preserve_sources(self):
        from unittest.mock import patch

        from build_instance_candidate_dataset import build_candidate_records_for_graph

        graph_data = {
            "nodes": [
                node(0, primitive="cylinder", area=20.0),
                node(1, primitive="plane", area=5.0),
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
        hybrid_candidate = {
            "source": "fallback_rule",
            "proposal_sources": ["fallback_rule"],
            "proposal_macro": "through_hole_round",
            "face_indices": [0],
            "seed_face_indices": [0],
            "assembly_reasons": ["fallback_two_circular_plane_boundaries"],
        }

        with patch("build_instance_candidate_dataset.assemble_instance_candidates", return_value=[]), \
             patch("build_instance_candidate_dataset.generate_hybrid_object_proposals", return_value=[hybrid_candidate]):
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
            )

        by_source = {record["source"]: record for record in records}
        self.assertIn("truth", by_source)
        self.assertIn("fallback_rule", by_source)
        self.assertEqual(by_source["fallback_rule"]["label"], "through_hole_round")
        self.assertEqual(by_source["fallback_rule"]["proposal_sources"], ["fallback_rule"])
        self.assertEqual(by_source["fallback_rule"]["assembly_reasons"], ["fallback_two_circular_plane_boundaries"])
        self.assertEqual(by_source["fallback_rule"]["features"]["proposal_source_fallback_rule"], 1.0)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset.BuildInstanceCandidateDatasetTest.test_build_records_can_use_hybrid_proposals_and_preserve_sources -v
```

Expected before implementation:

```text
ERROR
```

because `build_candidate_records_for_graph` does not accept `proposal_mode` and `generate_hybrid_object_proposals` is not imported.

- [ ] **Step 3: Import hybrid proposer**

In `build_instance_candidate_dataset.py`, add:

```python
from utils.object_candidate_proposer import generate_hybrid_object_proposals
```

- [ ] **Step 4: Add CLI argument**

In `parse_args`, add:

```python
    parser.add_argument("--proposal-mode", choices=("same_instance", "hybrid"), default="same_instance")
```

- [ ] **Step 5: Update `build_proposals` signature and implementation**

Change signature to:

```python
def build_proposals(
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any] | None,
    truth_instances: list[dict[str, Any]],
    target_macros: set[str],
    *,
    proposal_mode: str = "same_instance",
) -> list[dict[str, Any]]:
```

Replace the prediction proposal block with:

```python
    if prediction_data:
        same_instance_candidates = list(
            assemble_instance_candidates(
                graph_data,
                prediction_data,
                target_macros=target_macros,
            )
        )
        if proposal_mode == "hybrid":
            candidate_iterable = generate_hybrid_object_proposals(
                graph_data,
                prediction_data,
                same_instance_proposals=same_instance_candidates,
            )
        else:
            candidate_iterable = same_instance_candidates

        for candidate in candidate_iterable:
            proposal_macro = str(candidate.get("proposal_macro", "unknown_feature"))
            if proposal_macro not in target_macros:
                continue
            proposal = dict(candidate)
            if proposal_mode == "hybrid":
                proposal["source"] = str(proposal.get("source", "hybrid_candidate"))
            else:
                proposal["source"] = "assembled_instance_candidate"
                proposal["assembly_source"] = candidate.get("source")
            proposal["proposal_macro"] = proposal_macro
            proposals.append(proposal)
```

- [ ] **Step 6: Update `build_candidate_records_for_graph` signature**

Add parameter:

```python
    proposal_mode: str = "same_instance",
```

Pass it into `build_proposals`:

```python
    proposals = build_proposals(
        graph_data,
        prediction_data,
        truth_instances,
        target_macros,
        proposal_mode=proposal_mode,
    )
```

When writing each record, preserve source metadata:

```python
        if proposal.get("proposal_sources") is not None:
            record["proposal_sources"] = list(proposal["proposal_sources"])
        if proposal.get("assembly_reasons") is not None:
            record["assembly_reasons"] = list(proposal["assembly_reasons"])
```

- [ ] **Step 7: Update `build_dataset_from_paths`**

Add parameter:

```python
    proposal_mode: str = "same_instance",
```

Pass it into `build_candidate_records_for_graph`.

Add to summary:

```python
    summary["proposal_mode"] = proposal_mode
```

- [ ] **Step 8: Update `main`**

Pass:

```python
        proposal_mode=args.proposal_mode,
```

into `build_dataset_from_paths`.

- [ ] **Step 9: Run focused test**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_build_instance_candidate_dataset.BuildInstanceCandidateDatasetTest.test_build_records_can_use_hybrid_proposals_and_preserve_sources -v
```

Expected:

```text
OK
```

## 4. Task 3: Build v5.14 Hybrid Instance Candidate Dataset

**Files:**

- Input graph dir: `output_generation_v5_5_v51dist_looprole_5400/graphs`
- Input raw prediction dir: `output_generation_v5_10_instance_candidate_dataset_v55/predictions_raw`
- Output dataset dir: `output_generation_v5_14_hybrid_instance_candidate_dataset/instance_candidate_dataset`

- [ ] **Step 1: Run full unit tests before dataset build**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_build_instance_candidate_dataset tests.test_object_candidate_proposer tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Build dataset**

Run:

```powershell
W:\python\envs\feagpt\python.exe build_instance_candidate_dataset.py `
  --graph-dir output_generation_v5_5_v51dist_looprole_5400\graphs `
  --prediction-dir output_generation_v5_10_instance_candidate_dataset_v55\predictions_raw `
  --output-dir output_generation_v5_14_hybrid_instance_candidate_dataset\instance_candidate_dataset `
  --proposal-mode hybrid `
  --target-macros blind_hole_round,chamfer,counterbore_hole,countersink_hole,cylinder_boss,fillet,obround_hole,pocket_circular,pocket_rect,rectangular_boss,rib,slot_o,slot_rect,slot_u,slot_v,through_hole_round,web `
  --min-iou 0.50 `
  --min-candidate-coverage 0.70 `
  --min-truth-coverage 0.70 `
  --seed 20260702 `
  --train-ratio 0.80 `
  --val-ratio 0.10
```

- [ ] **Step 3: Verify dataset summary**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_generation_v5_14_hybrid_instance_candidate_dataset/instance_candidate_dataset/dataset_summary.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("proposal_mode", data.get("proposal_mode"))
print("record_count", data.get("record_count"))
print("source_counts", data.get("source_counts"))
print("label_counts_has_reject", "reject" in data.get("label_counts", {}))
print("feature_schema_path", data.get("feature_schema_path"))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
proposal_mode hybrid
label_counts_has_reject True
```

Expected `source_counts` includes at least:

```text
truth
same_instance
raw_macro_component
topology_rule
fallback_rule
```

If one source is absent, inspect `records.jsonl` before continuing and report which source is absent.

- [ ] **Step 4: Verify source features exist in schema**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_generation_v5_14_hybrid_instance_candidate_dataset/instance_candidate_dataset/feature_schema.json")
schema = json.loads(p.read_text(encoding="utf-8"))["features"]
required = [
    "proposal_source_count",
    "proposal_source_same_instance",
    "proposal_source_raw_macro_component",
    "proposal_source_topology_rule",
    "proposal_source_fallback_rule",
    "proposal_source_hybrid_merged",
]
missing = [name for name in required if name not in schema]
print("missing", missing)
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

```text
missing []
```

## 5. Task 4: Train v5.14 Hybrid Candidate Classifier

**Files:**

- Dataset: `output_generation_v5_14_hybrid_instance_candidate_dataset/instance_candidate_dataset`
- Output run dir: `output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid`

- [ ] **Step 1: Train random forest classifier**

Run:

```powershell
W:\python\envs\feagpt\python.exe train_candidate_topology_classifier.py `
  --dataset-dir output_generation_v5_14_hybrid_instance_candidate_dataset\instance_candidate_dataset `
  --output-dir output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs `
  --run-name random_forest_instance_candidate_v5_14_hybrid `
  --model-kind random_forest `
  --seed 20260702 `
  --n-estimators 600 `
  --max-depth 24 `
  --hard-source-weight 4.0
```

- [ ] **Step 2: Verify model files**

Run:

```powershell
Test-Path output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_14_hybrid\model.joblib
Test-Path output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_14_hybrid\metrics.json
Test-Path output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_14_hybrid\report.md
```

Expected:

```text
True
True
True
```

- [ ] **Step 3: Read test metrics**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid/metrics.json")
data = json.loads(p.read_text(encoding="utf-8"))
test = data["metrics"]["test"]
print("test_accuracy", test["accuracy"])
print("test_macro_f1", test["macro_f1"])
print("test_weighted_f1", test["weighted_f1"])
print("test_by_source_keys", sorted(data["metrics"].get("test_by_source", {}).keys()))
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

- `test_macro_f1` is printed.
- `test_by_source_keys` includes non-truth hybrid sources.

Do not require a fixed macro-F1 threshold here. Synthetic test-set performance is secondary; s3 accepted-stage recall is the actual decision metric.

## 6. Task 5: Evaluate New Classifier on s3 v5.13 Hybrid Proposals

**Files:**

- New classifier: `output_generation_v5_14_hybrid_instance_candidate_dataset/model_runs/random_forest_instance_candidate_v5_14_hybrid/model.joblib`
- Output reports: `output_real_step_s3_v5_14_hybrid_classifier/reports`

- [ ] **Step 1: Run s3 diagnostic with new classifier**

Run:

```powershell
W:\python\envs\feagpt\python.exe object_proposal_recall_report.py `
  --proposal-mode hybrid `
  --graph-dir output_real_step_s3_v5_6_b_rib_boundary\graphs `
  --raw-prediction-dir output_real_step_s3_v5_11_proposal_diagnostics\raw_predictions `
  --truth-json configs\test_part_s3_area_truth_part20_mm2.json `
  --classifier output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_14_hybrid\model.joblib `
  --final-prediction-dir output_real_step_s3_v5_10_instance_plus_v59\predictions `
  --output-json output_real_step_s3_v5_14_hybrid_classifier\reports\proposal_recall_s3_v5_14.json `
  --output-md output_real_step_s3_v5_14_hybrid_classifier\reports\proposal_recall_s3_v5_14.md `
  --strong-thresholds 0.30,0.40,0.50,0.55,0.60,0.70 `
  --weak-thresholds 0.20,0.30,0.35,0.40,0.50 `
  --min-confidence 0.50 `
  --absolute-area-tolerance 10.0 `
  --relative-area-tolerance 0.02
```

- [ ] **Step 2: Verify v5.14 metrics**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("output_real_step_s3_v5_14_hybrid_classifier/reports/proposal_recall_s3_v5_14.json")
data = json.loads(p.read_text(encoding="utf-8"))
bp = data["best_by_proposal_recall"]
print("proposal_recall", bp["proposal_stage"]["recall"])
print("classified_recall", bp["classified_stage"]["recall"])
print("accepted_recall", bp["accepted_stage"]["recall"])
print("final_survival_recall", bp["final_survival_stage"]["recall"])
print("failure_stage_counts", bp["failure_stage_counts"])
'@ | W:\python\envs\feagpt\python.exe -
```

Expected:

- `proposal_recall` remains close to v5.13 proposal recall `0.6814`.
- `accepted_recall` is printed and compared against v5.13 accepted recall `0.4804`.

## 7. Task 6: Write v5.13 vs v5.14 Comparison and Decision

**Files:**

- Create: `output_real_step_s3_v5_14_hybrid_classifier/reports/v5_13_vs_v5_14_comparison.md`
- Create: `output_real_step_s3_v5_14_hybrid_classifier/reports/v5_14_decision.md`

- [ ] **Step 1: Generate comparison and decision**

Run:

```powershell
@'
import json
from pathlib import Path

old_path = Path("output_real_step_s3_v5_13_transition_opening_proposals/reports/proposal_recall_s3_v5_13.json")
new_path = Path("output_real_step_s3_v5_14_hybrid_classifier/reports/proposal_recall_s3_v5_14.json")
comparison_path = Path("output_real_step_s3_v5_14_hybrid_classifier/reports/v5_13_vs_v5_14_comparison.md")
decision_path = Path("output_real_step_s3_v5_14_hybrid_classifier/reports/v5_14_decision.md")

old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))
old_bp = old["best_by_proposal_recall"]
new_bp = new["best_by_proposal_recall"]

old_prop = old_bp["proposal_stage"]["recall"]
new_prop = new_bp["proposal_stage"]["recall"]
old_acc = old_bp["accepted_stage"]["recall"]
new_acc = new_bp["accepted_stage"]["recall"]
old_final = old_bp["final_survival_stage"]["recall"]
new_final = new_bp["final_survival_stage"]["recall"]
old_fail = old_bp["failure_stage_counts"]
new_fail = new_bp["failure_stage_counts"]

comparison_lines = [
    "# v5.13 vs v5.14 Hybrid Candidate Classifier Comparison",
    "",
    "## Summary",
    "",
    "| Metric | v5.13 old classifier | v5.14 hybrid-trained classifier |",
    "|--------|----------------------|--------------------------------|",
    f"| Proposal recall | {old_prop:.4f} | {new_prop:.4f} |",
    f"| Accepted-stage recall | {old_acc:.4f} | {new_acc:.4f} |",
    f"| Final-survival recall | {old_final:.4f} | {new_final:.4f} |",
    f"| No compatible proposal | {old_fail.get('no_area_compatible_proposal', 0)} | {new_fail.get('no_area_compatible_proposal', 0)} |",
    f"| Classifier wrong/rejected | {old_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} | {new_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)} |",
    f"| Accepted but final changes/removes | {old_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} | {new_fail.get('classifier_accepts_but_final_changes_or_removes', 0)} |",
    f"| Survives final | {old_fail.get('survives_final_prediction', 0)} | {new_fail.get('survives_final_prediction', 0)} |",
    "",
]
comparison_path.parent.mkdir(parents=True, exist_ok=True)
comparison_path.write_text("\\n".join(comparison_lines) + "\\n", encoding="utf-8")

accepted_gain = new_acc - old_acc
final_gain = new_final - old_final

if accepted_gain >= 0.08 and new_acc >= 0.58:
    recommendation = "hybrid_pipeline_integration_trial"
    reason = "Hybrid-trained classifier substantially improves accepted-stage recall. Run a controlled end-to-end inference export trial next."
elif accepted_gain >= 0.04:
    recommendation = "classifier_threshold_scan"
    reason = "Hybrid-trained classifier improves accepted-stage recall, but not enough for integration. Scan min-confidence thresholds before changing model family."
elif new_bp["failure_stage_counts"].get("proposal_exists_classifier_wrong_or_rejected", 0) >= old_bp["failure_stage_counts"].get("proposal_exists_classifier_wrong_or_rejected", 0):
    recommendation = "classifier_error_atlas_and_feature_revision"
    reason = "Hybrid-trained classifier does not reduce classifier wrong/rejected failures. Inspect per-source confusion and revise candidate features or labeling."
else:
    recommendation = "rib_focused_proposal_diagnosis"
    reason = "Classifier improves somewhat, but proposal recall gap remains dominated by no-compatible-proposal cases."

decision_lines = [
    "# v5.14 Hybrid Candidate Classifier Decision",
    "",
    "## Recommendation",
    "",
    f"**{recommendation}**",
    "",
    "## Reasoning",
    "",
    reason,
    "",
    "## Supporting Data",
    "",
    f"- v5.13 accepted-stage recall: {old_acc:.4f}",
    f"- v5.14 accepted-stage recall: {new_acc:.4f}",
    f"- accepted-stage recall gain: {accepted_gain:.4f}",
    f"- v5.13 final-survival recall: {old_final:.4f}",
    f"- v5.14 final-survival recall: {new_final:.4f}",
    f"- final-survival recall gain: {final_gain:.4f}",
    f"- v5.14 classifier wrong/rejected: {new_fail.get('proposal_exists_classifier_wrong_or_rejected', 0)}",
    f"- v5.14 no compatible proposal: {new_fail.get('no_area_compatible_proposal', 0)}",
    "",
]
decision_path.write_text("\\n".join(decision_lines) + "\\n", encoding="utf-8")

print(comparison_path)
print(decision_path)
'@ | W:\python\envs\feagpt\python.exe -
```

- [ ] **Step 2: Inspect decision**

Run:

```powershell
Get-Content output_real_step_s3_v5_14_hybrid_classifier\reports\v5_14_decision.md
```

Expected:

- Exactly one recommendation appears in bold.
- It is one of:
  - `hybrid_pipeline_integration_trial`
  - `classifier_threshold_scan`
  - `classifier_error_atlas_and_feature_revision`
  - `rib_focused_proposal_diagnosis`

## 8. Final Verification

- [ ] **Step 1: Run full tests**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_instance_candidate_features tests.test_build_instance_candidate_dataset tests.test_object_candidate_proposer tests.test_object_proposal_recall_report tests.test_instance_candidate_postprocess tests.test_prediction_to_feature_objects tests.test_run_gnn_inference_edge_aware tests.test_selectable_regions -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run compile check**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m py_compile utils\instance_candidate_features.py build_instance_candidate_dataset.py tests\test_instance_candidate_features.py tests\test_build_instance_candidate_dataset.py
```

Expected:

```text
no output
```

- [ ] **Step 3: Verify output files**

Run:

```powershell
Test-Path output_generation_v5_14_hybrid_instance_candidate_dataset\instance_candidate_dataset\dataset_summary.json
Test-Path output_generation_v5_14_hybrid_instance_candidate_dataset\model_runs\random_forest_instance_candidate_v5_14_hybrid\model.joblib
Test-Path output_real_step_s3_v5_14_hybrid_classifier\reports\proposal_recall_s3_v5_14.json
Test-Path output_real_step_s3_v5_14_hybrid_classifier\reports\v5_13_vs_v5_14_comparison.md
Test-Path output_real_step_s3_v5_14_hybrid_classifier\reports\v5_14_decision.md
```

Expected:

```text
True
True
True
True
True
```

## 9. Required Final Response From Implementing Agent

The implementing agent must report:

1. Tests run and whether they passed.
2. v5.14 dataset record count.
3. v5.14 dataset source counts.
4. v5.14 classifier synthetic test macro-F1.
5. v5.13 accepted-stage recall and v5.14 accepted-stage recall.
6. v5.13 final-survival recall and v5.14 final-survival recall.
7. v5.14 failure-stage counts.
8. The single recommendation from `v5_14_decision.md`.
9. Paths to:
   - `dataset_summary.json`
   - `model.joblib`
   - `proposal_recall_s3_v5_14.json`
   - `v5_13_vs_v5_14_comparison.md`
   - `v5_14_decision.md`
10. Any deviation from this plan.

## 10. Review Checklist for Codex

When reviewing implementation:

1. `utils/instance_candidate_features.py` contains proposal-source feature buckets.
2. `build_instance_candidate_dataset.py` supports `--proposal-mode hybrid`.
3. v5.14 dataset summary reports `proposal_mode hybrid`.
4. v5.14 source counts include `truth`, `same_instance`, `raw_macro_component`, `topology_rule`, and `fallback_rule`.
5. v5.14 feature schema includes proposal-source features.
6. v5.14 classifier model exists.
7. s3 v5.14 diagnostic uses the new v5.14 classifier path, not the v5.10 classifier.
8. If accepted-stage recall improves strongly, next task should be controlled pipeline integration.
9. If accepted-stage recall does not improve, next task should be classifier error atlas and feature/label revision, not more proposal expansion.
