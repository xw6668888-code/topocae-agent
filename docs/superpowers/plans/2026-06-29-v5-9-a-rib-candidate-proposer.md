# v5.9-A Rib Candidate Proposer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rib candidate proposer that can create rib seed faces/components from B-Rep graph evidence before rib boundary pruning/expansion.

**Architecture:** Use the colored real rib face truth in `rib_face_truth_template_v5_8_b_colored_reviewed.json` to train a face-level binary classifier. During inference, run it after topology postprocess and before rib boundary classifier; accepted candidate faces are temporarily promoted to `rib` seeds, then the existing rib boundary classifier repairs object boundaries.

**Tech Stack:** Python, scikit-learn/joblib, existing graph JSON, existing `train_candidate_topology_classifier.py`, existing `run_gnn_inference.py`.

---

### Task 1: Face-Level Rib Candidate Feature Dataset

**Files:**
- Create: `build_rib_candidate_proposer_dataset.py`
- Create: `utils/rib_candidate_features.py`
- Create: `tests/test_rib_candidate_proposer_dataset.py`

- [ ] Write RED tests for extracting a stable face-level feature vector from a graph node and its neighborhood.
- [ ] Implement feature extraction: primitive flags, area ratio, bbox/projected aspect, neighbor counts, line/convex/concave edge counts, large-support adjacency counts.
- [ ] Write RED tests that a reviewed colored template becomes positive/negative records.
- [ ] Implement dataset builder that reads reviewed template + graph paths and writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `metadata.json`.
- [ ] Verify tests pass.

### Task 2: Rib Candidate Proposer Postprocess

**Files:**
- Create: `utils/rib_candidate_proposer_postprocess.py`
- Modify: `run_gnn_inference.py`
- Create: `tests/test_rib_candidate_proposer_postprocess.py`

- [ ] Write RED test: a high-confidence proposer result promotes a stock/thin plane face to `rib` before rib boundary repair.
- [ ] Write RED test: protected macro classes such as hole/slot/transition are not overwritten unless explicitly allowed.
- [ ] Implement bundle loading and `apply_rib_candidate_proposer_postprocess`.
- [ ] Add CLI args to `run_gnn_inference.py`: `--rib-candidate-proposer`, `--rib-candidate-proposer-min-confidence`.
- [ ] Insert proposer after topology postprocess and before rib boundary classifier.
- [ ] Verify postprocess tests pass.

### Task 3: Train And Evaluate v5.9-A

**Files/Outputs:**
- Output dataset: `output_generation_v5_9_a_rib_candidate_proposer_dataset`
- Output model: `output_generation_v5_9_a_rib_candidate_proposer_dataset/model_runs/random_forest_rib_candidate_proposer_v5_9_a/model.joblib`
- Output s3 run: `output_real_step_s3_v5_9_a_rib_candidate_proposer_t075`

- [ ] Build dataset from `rib_face_truth_template_v5_8_b_colored_reviewed.json`.
- [ ] Train with `train_candidate_topology_classifier.py`.
- [ ] Re-run `test_part_s3` with v5.8-C rib boundary classifier plus v5.9-A proposer.
- [ ] Convert predictions to feature objects.
- [ ] Generate area truth review and compact review.
- [ ] Compare against v5.8-A/v5.8-C metrics, especially matched rib, missed rib, false-positive rib.

### Task 4: Decision

- [ ] If matched rib improves without unacceptable FP rib increase, keep v5.9-A and ask the user for 5-10 more rib-containing colored real/simplified parts.
- [ ] If no matched rib improvement, inspect decision logs and determine whether the proposer needs component-level candidates instead of face-level candidates.
