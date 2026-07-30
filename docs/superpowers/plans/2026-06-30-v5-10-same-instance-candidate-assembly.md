# v5.10 Same-Instance Candidate Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile “same macro connected component” object grouping with an edge same-instance classifier and candidate object assembly stage, so CAE feature objects are formed by learned/topology-aware instance boundaries rather than face labels alone.

**Architecture:** Keep the current edge-aware GNN face classifier, but elevate its `edge_same_instance_logits` branch into a first-class object proposal stage. The pipeline becomes: STEP graph -> face/edge prediction -> edge calibration -> instance graph assembly -> candidate object classifier -> topology legality repair -> feature objects. The work is validated by separate edge-level, object-level, and area-truth metrics.

**Tech Stack:** Python 3 in `W:/python/envs/feagpt`, PyTorch GNN in `models/gnn_baseline.py`, JSON graph records, scikit-learn candidate classifiers, existing STEP graph extraction and reporting scripts.

---

## 1. Research Motivation And Hypothesis

Current failure mode:

- Face-level macro classification can be locally correct but object grouping fails.
- Existing `prediction_to_feature_objects.py` groups by connected components with the same macro feature.
- That grouping cannot solve cases where:
  - a rib is split into several low-confidence faces,
  - a U/O slot is confused with fillet or obround because individual faces look similar,
  - a hole/cylinder role depends on boundary context,
  - a transition face should be absorbed into a parent feature.

Scientific hypothesis:

- Adjacent faces belonging to the same CAE feature instance can be predicted more robustly from B-Rep topology and edge attributes than from face macro labels alone.
- A learned same-instance edge boundary, combined with rule-constrained candidate assembly, should reduce false object splits and false merges.
- This gives a clearer paper contribution than only increasing model size: “CAE-oriented B-Rep feature recognition via edge-supervised instance assembly and topology-aware candidate repair.”

---

## 2. Target Pipeline

Current pipeline:

```text
STEP/STP
-> graph extraction
-> edge-aware GNN face prediction
-> candidate classifier / rib proposer / rib boundary repair
-> topology postprocess
-> connected_components_same_macro_feature
-> feature objects
```

v5.10 pipeline:

```text
STEP/STP
-> graph extraction
-> edge-aware GNN face + edge same-instance prediction
-> edge same-instance calibration
-> candidate object assembly
-> candidate-level macro classifier
-> topology legality repair
-> feature objects
-> area-truth and failure-atlas reports
```

The key change is that object boundaries are no longer determined only by face macro equality. They are determined by learned same-instance edges plus topology constraints.

---

## 3. Files To Create Or Modify

### New Files

- `utils/instance_edge_metrics.py`
  - Computes edge-level same-instance precision/recall/F1/AUC and threshold curves on labeled generated graphs.

- `utils/instance_candidate_assembly.py`
  - Builds feature-object candidate components from predicted same-instance edge probabilities.
  - Supports macro-compatible merging, weak-face expansion, and strict merge guards.

- `build_instance_candidate_dataset.py`
  - Builds candidate-object training data from generated graphs, GNN predictions, and truth `feature_sets`.
  - Labels assembled candidates as true macro / reject using IoU and coverage against truth instances.

- `utils/instance_candidate_features.py`
  - Extracts object-level features for candidate assembly: face histogram, edge boundary histogram, concave/convex ratios, area ratios, primitive types, same-instance probability statistics, topology validity flags.

- `utils/instance_candidate_postprocess.py`
  - Applies trained object classifier to assembled candidates during inference.
  - Resolves overlapping accepted candidates by confidence, topology validity, and area coverage.

- `tests/test_instance_candidate_assembly.py`
  - Unit tests for same-instance connected components, weak-face expansion, protected boundary stopping, and singleton rejection.

- `tests/test_instance_candidate_features.py`
  - Unit tests for object-level feature extraction.

- `tests/test_instance_candidate_postprocess.py`
  - Unit tests for overlap resolution and candidate-to-face remapping.

### Modified Files

- `train_gnn_baseline.py`
  - Add edge metrics reporting during train/validation/test.
  - Add configurable edge threshold calibration output.

- `run_gnn_inference.py`
  - Add CLI options:
    - `--instance-candidate-classifier`
    - `--instance-candidate-min-confidence`
    - `--same-instance-assembly-threshold`
    - `--same-instance-weak-threshold`
  - Insert instance assembly after raw GNN prediction and before existing candidate/rib repair.

- `prediction_to_feature_objects.py`
  - Add grouping mode option:
    - existing: `connected_components_same_macro_feature`
    - new: `assembled_instance_candidates`
  - Preserve source assembly records in feature object JSON.

- `build_candidate_classifier_dataset.py`
  - Reuse or delegate proposal generation to `utils/instance_candidate_assembly.py` so training and inference use the same proposal logic.

- `generate_area_truth_review.py`
  - Add optional reporting of split errors and merge errors by comparing predicted object face sets against area truth / generated truth instances.

- `CODEX_CONTEXT.md`
  - Append v5.10 decisions and best model artifacts after execution.

---

## 4. Implementation Tasks

### Task 1: Establish Baseline Edge Same-Instance Diagnostics

Question answered: “Does the current GNN already learn useful same-instance edges?”

- [ ] Add `utils/instance_edge_metrics.py`.
- [ ] Compute edge metrics from validation/test JSONL records:
  - edge precision
  - edge recall
  - edge F1
  - ROC-AUC if both classes exist
  - PR-AUC if both classes exist
  - threshold scan from `0.30` to `0.90`
- [ ] Modify `train_gnn_baseline.py` to save an `edge_same_instance_metrics.json` report in model output.
- [ ] Run this on the current v5.x training output.

Expected result:

- If edge F1 is high on generated validation but poor on real-like samples, the gap is data/domain.
- If edge F1 is poor even on generated validation, the model/loss is underpowered or badly weighted.

Acceptance criteria:

- Report contains edge metrics independent of face macro accuracy.
- We can choose an assembly threshold from validation data instead of by intuition.

---

### Task 2: Build Same-Instance Candidate Assembly

Question answered: “Can predicted edge boundaries assemble more reasonable feature objects than same-macro connected components?”

- [ ] Create `utils/instance_candidate_assembly.py`.
- [ ] Implement `build_edge_probability_lookup(graph_data, prediction_data)`.
- [ ] Implement `assemble_seed_components(...)`:
  - nodes are faces,
  - edges are accepted when `same_instance_probability >= strong_threshold`,
  - ignore edges where either face is predicted stock unless weak expansion is active,
  - do not cross protected class boundaries such as hole-to-stock outer cylinder unless topology says internal loop.
- [ ] Implement `expand_weak_candidate_faces(...)`:
  - allow nearby low-confidence rib/slot faces to join if edge probability is above weak threshold,
  - require primitive/topology compatibility,
  - prevent large stock faces from being absorbed into rib by area-ratio guard.
- [ ] Implement `deduplicate_candidates(...)`.

Initial recommended thresholds:

- `strong_threshold`: chosen from validation edge F1 curve, initially `0.55` for testing.
- `weak_threshold`: chosen from validation recall curve, initially `0.35` for rib/slot expansion only.

Acceptance criteria:

- Rib faces like `[0, 6, 11]` can become one candidate if connected by same-instance or topology-compatible weak edges.
- Single isolated rib-like faces are not accepted as final rib objects unless object classifier approves them.
- U/O/rect/V slot candidates remain separate from ordinary fillet/chamfer unless topology supports merging.

---

### Task 3: Build Candidate-Object Feature Extractor

Question answered: “Can object-level topology distinguish rib/stock, slot/fillet, boss/hole better than face-level labels?”

- [ ] Create `utils/instance_candidate_features.py`.
- [ ] Extract these feature groups:
  - face count and total area ratio,
  - primitive histogram: plane/cylinder/cone/torus/other,
  - internal edge histogram: line/circle/ellipse/spline,
  - boundary edge histogram,
  - concave/convex/smooth/unknown ratios,
  - same-instance probability min/mean/max on internal edges,
  - boundary same-instance probability mean,
  - candidate macro face vote histogram,
  - semantic subtype vote histogram,
  - topology legality flags from `utils/topology_rules.py`,
  - shape descriptors: bbox ratio, thickness ratio, cylindrical radius stats.
- [ ] Add tests with synthetic graph fragments for:
  - rectangular pocket with five faces,
  - rib with three faces,
  - slot_u with bottom + side faces,
  - single fillet face,
  - blind hole cylinder + bottom.

Acceptance criteria:

- Features are rotation-invariant.
- Features use only data available from generated and real STEP graphs.
- No feature relies on true labels at inference time.

---

### Task 4: Build Instance Candidate Dataset

Question answered: “Can we train a candidate classifier using generated truth and current prediction errors?”

- [ ] Create `build_instance_candidate_dataset.py`.
- [ ] Inputs:
  - labeled generated graph directory,
  - prediction directory from current GNN,
  - optional real colored rib truth diagnostics,
  - output directory.
- [ ] Generate candidate records from:
  - truth feature sets,
  - same-instance assembled components,
  - old same-macro components,
  - topology rule proposals,
  - hard-case proposals from real colored rib truth.
- [ ] Label each candidate with:
  - true macro if IoU and both-coverages pass thresholds,
  - `reject` if it overlaps truth weakly or violates topology,
  - `partial_<macro>` optional diagnostic only, not a final class.
- [ ] Save:
  - `records.jsonl`,
  - `train.jsonl`,
  - `val.jsonl`,
  - `test.jsonl`,
  - `feature_schema.json`,
  - `dataset_summary.json`.

Recommended label thresholds:

- `min_iou = 0.50`
- `min_candidate_coverage = 0.70`
- `min_truth_coverage = 0.70`

Reason:

- Candidate objects may have transition faces absorbed or omitted, so strict IoU `0.80+` would over-penalize useful CAE selections.
- Both-coverage prevents a huge stock region from matching a small rib by area/overlap accident.

Acceptance criteria:

- Dataset contains both positive and hard negative samples for:
  - rib vs stock/pocket,
  - slot_o vs fillet,
  - slot_u vs obround_hole,
  - cylinder_boss vs blind_hole/through_hole/outer cylinder,
  - counterbore vs stepped_hole.

---

### Task 5: Train Candidate Object Classifier

Question answered: “Does object-level classification improve real STEP feature objects?”

- [ ] Reuse `train_candidate_topology_classifier.py` if feature-schema and labels are compatible.
- [ ] If not compatible, add a small wrapper script `train_instance_candidate_classifier.py` that uses the same model format as current `model.joblib` bundles.
- [ ] Train first with RandomForest / ExtraTrees because:
  - dataset is tabular candidate features,
  - feature importance can be inspected,
  - training is stable on small hard-case sets,
  - easier to explain in the paper.
- [ ] Save model bundle with:
  - estimator,
  - feature schema,
  - label map,
  - threshold/calibration metadata,
  - validation confusion matrix.

Acceptance criteria:

- Candidate validation report explicitly lists confusion pairs:
  - rib/stock,
  - rib/pocket,
  - slot_o/fillet,
  - slot_u/obround,
  - cylinder_boss/blind_hole,
  - through_hole/outer-cylinder.

---

### Task 6: Integrate Into Inference Pipeline

Question answered: “Does the complete pipeline improve feature-object output, not only intermediate metrics?”

- [ ] Create `utils/instance_candidate_postprocess.py`.
- [ ] Modify `run_gnn_inference.py` so the new stage runs after raw GNN face/edge prediction.
- [ ] Output diagnostic fields:
  - `instance_candidate_assembly.enabled`,
  - `candidate_count`,
  - `accepted_count`,
  - `rejected_count`,
  - `records` with face indices, label, confidence, source, and overlap-resolution reason.
- [ ] Modify `prediction_to_feature_objects.py` so accepted candidate objects can be emitted directly as feature objects.
- [ ] Keep the old grouping mode available for ablation.

Acceptance criteria:

- Feature object JSON records contain `grouping_method = assembled_instance_candidates` for new objects.
- The old `connected_components_same_macro_feature` path still works for comparison.

---

### Task 7: Evaluation Protocol

Question answered: “Is v5.10 scientifically better, and where does it still fail?”

Run four systems:

1. Face-only GNN + same macro connected components.
2. Current v5.9-B/v5.9-C best pipeline.
3. v5.10 same-instance assembly + candidate object classifier.
4. v5.10 plus hard-case real calibration, if enough truth exists.

Evaluate on:

- generated validation/test set,
- `test_part_s`,
- `test_part_s1`,
- `test_part_s3`,
- later, a held-out real/simple-real set not used for threshold choice.

Metrics:

- Face macro accuracy/F1.
- Edge same-instance precision/recall/F1/AUC.
- Object-level precision/recall/F1 by macro class.
- Split error count: one truth object split into multiple predicted objects.
- Merge error count: multiple truth objects merged into one predicted object.
- Area-truth matched objects and class-wise recall.
- Rotation consistency for rotated variants.

Acceptance criteria for v5.10 to be considered useful:

- Object recall improves over v5.9-C without a larger precision drop than `2-3` points.
- Rib matched count improves on s3 and does not create many large-stock rib false positives.
- Slot_u/slot_o errors reduce on known hard parts.
- Failure atlas shows fewer split/merge errors than same-macro connected components.

---

## 5. User / Codex Division Of Work

### Codex Responsibilities

Codex should do all code and experiment work:

- inspect and modify Python scripts,
- add same-instance metrics,
- implement candidate assembly,
- implement candidate-object features,
- build datasets,
- train classifiers,
- run inference,
- generate reports,
- maintain `CODEX_CONTEXT.md`,
- produce comparison tables and failure atlases.

Codex should also make engineering decisions when they are internal implementation details, for example:

- exact JSON field names,
- class names,
- CLI parameter names,
- model artifact layout,
- unit test structure,
- threshold scan mechanics.

### User Responsibilities

The user should only intervene where CAE/geometric semantics matter:

1. Confirm semantic boundary definitions:
   - When a rib has fillets/chamfers, whether transitions are absorbed or independent.
   - Whether a pocket containing a rib should still output the pocket as an independent target.
   - Whether a boss/shaft/context cylinder is target or background in a given part.

2. Provide or correct small real validation truth:
   - Not thousands of faces manually.
   - Prefer area-level truth tables, colored faces for rib-like structures, or short notes such as “face group A is a rib, this outer cylinder is background”.

3. Inspect small STEP batches only at milestone checkpoints:
   - generated hard cases around rib/slot/hole,
   - not full datasets.
   - Codex should provide a short list of 10-30 files where visual inspection matters.

4. Decide paper-scope rules:
   - Which features are official target classes.
   - Which features are context/background.
   - Which ambiguous cases can be excluded from the first paper.

---

## 6. Scientific Controls For Paper Credibility

To avoid unscientific threshold tuning:

- Split real/simple-real data into:
  - calibration set: used for thresholds and error-driven hard cases,
  - held-out test set: never used for model/threshold decisions.
- Report threshold selection from validation PR curves, not from final test results.
- Keep ablation experiments:
  - without edge attributes,
  - without same-instance edge loss,
  - without candidate object classifier,
  - without real hard-case calibration.
- Report failure modes honestly:
  - unseen complex assemblies,
  - ambiguous context cylinders,
  - user-defined CAE semantics that cannot be inferred from geometry alone.

---

## 7. Recommended Execution Order

1. Edge same-instance diagnostics.
2. Same-instance candidate assembly without training a new classifier.
3. Candidate-object feature extractor and dataset builder.
4. Candidate-object classifier training.
5. Inference integration.
6. Full comparison on `test_part_s`, `test_part_s1`, and `test_part_s3`.
7. Decide whether GAT/Transformer model upgrade is still necessary.

Rationale:

- If Task 1 shows weak edge prediction, fix the GNN/loss/data first.
- If Task 1 shows useful edge prediction but Task 2 fails, fix assembly rules.
- If Task 2 generates good candidates but wrong labels, train object-level classifier.
- Only after these are measured should we decide whether a heavier GAT/Transformer is worth the complexity.

---

## 8. Expected Outputs

- Plan document:
  - `docs/superpowers/plans/2026-06-30-v5-10-same-instance-candidate-assembly.md`
- New tests:
  - `tests/test_instance_candidate_assembly.py`
  - `tests/test_instance_candidate_features.py`
  - `tests/test_instance_candidate_postprocess.py`
- New reports:
  - `reports/edge_same_instance_metrics_v5_10.md`
  - `reports/v5_10_candidate_assembly_ablation.md`
  - `reports/v5_10_real_step_comparison.md`
  - `reports/v5_10_failure_atlas.md`
- New model artifact, if classifier helps:
  - `output_generation_v5_10_instance_candidate_dataset/model_runs/.../model.joblib`

---

## 9. Stop / Continue Criteria

Continue v5.10 if:

- edge same-instance validation F1 is meaningful, or
- assembled candidates improve split/merge diagnostics even before retraining.

Stop and redirect to data/model repair if:

- edge same-instance prediction is near random on generated validation,
- candidate assembly creates mostly invalid objects,
- failures are dominated by missing graph attributes rather than model decisions.

If stopped, next branch should be:

- improve edge labels and graph attributes,
- add edge-aware GATv2 / transformer ablation,
- strengthen generated data for the failing feature families.
