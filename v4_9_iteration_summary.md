# v4.9 Iteration Summary

Last updated: 2026-06-17

## 1. Current Paper Logic

This project is a CAE-oriented STEP/B-Rep feature recognition workflow for single-part structural-static preprocessing. The central task is not generic CAD recognition or machining feature recognition. The target is to identify selectable CAE regions from STEP geometry, so downstream tools or agents can map user simulation requirements to concrete face sets.

The current pipeline is:

1. STEP model input.
2. B-Rep graph extraction.
   - Node = face.
   - Edge = adjacency between faces.
   - Node attributes include surface type, area, shape descriptors, axis/radius/normal when available.
   - Edge attributes include curve type, length, convexity/concavity, dihedral angle and related topology evidence.
3. Edge-aware GNN prediction.
   - Face-level macro feature label.
   - Face-level semantic subtype.
   - Derived CAE labels and idealization labels.
   - Same-instance edge prediction for grouping faces into one feature instance.
4. Topology-aware candidate repair.
   - Uses explicit B-Rep rules to reject impossible objects and repair common confusions.
   - Examples: two-face invalid pocket is demoted; U-slot can be recovered from one curved bottom plus two long side planes; exterior cylinder plus cap can be recovered as cylinder boss.
5. Feature-object export.
   - Converts face predictions into feature instances with face ids, roles, areas, centroids and selectable regions.
6. CAE requirement matching / Abaqus mapping.
   - Simulation requirements are matched to selectable feature regions.
   - Final output should be face sets usable for constraints, loads, idealization or preprocessing review.

The strongest paper framing is:

> A neuro-symbolic B-Rep graph method for CAE-oriented geometric feature recognition and selectable region extraction.

The paper should emphasize that the system combines learned graph perception with explicit topology validity rules. This is more defensible than a plain GNN classifier because CAE preprocessing needs exact selectable faces, not only visual or semantic labels.

## 2. v4.9 Work Completed

### 2.1 Candidate-Level Topology Repair

Two current real/simplified real errors were converted into regression tests and fixed.

Fixed cases:

- `test_part_s/part31.stp` to `part33.stp`
  - Problem: U-shaped slot was still recognized as `slot_rect` or swallowed by `web`.
  - Root cause: `find_u_slot_candidates()` rejected a U-slot bottom face when the raw model predicted the curved bottom as `web`.
  - Fix: allow `web` as a candidate macro for the curved/cylindrical U-slot bottom, while retaining the requirement of long concave line connections to two side planes.

- `test_part_s1/part31.stp`
  - Problem: one `cylinder_boss` was missing.
  - Root cause: raw model predicted the exterior two-face cylinder-plus-cap candidate as `stepped_hole_round`.
  - Fix: allow `stepped_hole_round` and stepped-hole semantic subtypes as candidate evidence for `cylinder_boss`, while keeping the exterior-cap check.

Changed files:

- `utils/topology_postprocess.py`
- `tests/test_topology_postprocess.py`

Verification:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_topology_rules tests.test_topology_postprocess tests.test_run_gnn_inference_edge_aware tests.test_compact_real_step_review -v
```

Result:

- `Ran 43 tests ... OK`

### 2.2 Real STEP Rulefix Recheck With v4.8 Checkpoint

Using the existing v4.8 model and the new v4.9 rulefix:

- `output_real_step_s_v4_9_rulefix_current`
- `output_real_step_s1_v4_9_rulefix_current`

Key improvements:

- `test_part_s/part31~part33` now include `slot_u`.
- `test_part_s1/part31` now includes two `cylinder_boss` objects.

Important review files:

- `output_real_step_s_v4_9_rulefix_current/reports/compact_review_edge_aware_v4_9_rulefix_topology_post.md`
- `output_real_step_s1_v4_9_rulefix_current/reports/compact_review_edge_aware_v4_9_rulefix_topology_post.md`

### 2.3 Focused Hard-Confusion Dataset

Generated a v4.9 focused dataset:

- root: `output_generation_v4_9_focused_hardconfusion_1500`
- samples: `1500`
- seed: `20260623`
- mode: `nonintersecting_multi`
- target classes:
  - `slot_u`
  - `slot_rect`
  - `slot_o`
  - `slot_v`
  - `web`
  - `rib`
  - `cylinder_boss`
  - `stepped_hole_round`
  - `blind_hole_round`
  - `pocket_circular`

Achieved instance counts:

- `blind_hole_round`: `411`
- `cylinder_boss`: `553`
- `pocket_circular`: `482`
- `rib`: `122`
- `slot_o`: `374`
- `slot_rect`: `483`
- `slot_u`: `391`
- `slot_v`: `276`
- `stepped_hole_round`: `386`
- `web`: `326`

### 2.4 v4.9 Augmented Full Dataset

Merged v4.8 full data with the v4.9 focused hard-confusion set:

- root: `output_generation_v4_9_augmented_full_7806_clean`
- full samples: `6306`
- focused hard-confusion samples: `1500`
- total samples: `7806`

Important combined class counts:

- `slot_u`: `1621`
- `slot_rect`: `1829`
- `slot_o`: `1549`
- `slot_v`: `1066`
- `web`: `942`
- `rib`: `589`
- `cylinder_boss`: `1469`
- `stepped_hole_round`: `1113`
- `blind_hole_round`: `1189`
- `pocket_circular`: `1338`

### 2.5 v4.9 Training

Trained model:

- run: `output_generation_v4_9_augmented_full_7806_clean/model_runs/edge_aware_v4_9_augmented_hardconfusion_70ep`
- checkpoint: `best.pt`
- best epoch: `62`
- model: edge-aware GNN
- epochs: `70`
- hidden dim: `176`
- layers: `4`
- batch size: `32`
- dropout: `0.12`
- edge loss weight: `0.6`
- seed: `20260624`

Training test metrics from `metrics.json`:

- test macro-F1 active: `0.9518`
- test semantic-F1 active: `0.9144`
- test derived micro-F1: `0.9963`
- test idealization micro-F1: `0.9961`
- test edge same-instance F1: `0.9984`

Generated-set raw prediction evaluation:

- macro accuracy: `0.9857`
- macro-F1: `0.9608`
- semantic accuracy: `0.9828`
- semantic-F1: `0.9286`

Generated-set topology-post evaluation:

- macro accuracy: `0.9295`
- macro-F1: `0.8905`
- semantic accuracy: `0.9035`
- semantic-F1: `0.7838`

Interpretation:

- Raw model metrics remain strong and are still the preferred synthetic benchmark metrics for the paper.
- Topology-post is useful for real STEP repair and review, but it remains too conservative for synthetic benchmark scoring because it demotes/rewrites some valid generated features.

Focused raw per-class results:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `slot_u` | 24570 | 0.9965 | 0.9991 | 0.9978 |
| `slot_rect` | 8235 | 0.9962 | 0.9987 | 0.9975 |
| `slot_o` | 7999 | 0.9925 | 0.9990 | 0.9958 |
| `slot_v` | 6310 | 0.9868 | 0.9979 | 0.9924 |
| `web` | 2831 | 0.9996 | 0.9993 | 0.9995 |
| `rib` | 1783 | 0.9978 | 1.0000 | 0.9989 |
| `cylinder_boss` | 2940 | 0.8171 | 0.6245 | 0.7079 |
| `stepped_hole_round` | 4465 | 0.9996 | 0.9989 | 0.9992 |
| `blind_hole_round` | 2395 | 0.9295 | 0.8589 | 0.8928 |
| `pocket_circular` | 2821 | 0.7312 | 0.9699 | 0.8338 |

Important conclusion:

- Slot classes, rib and web are now strong in generated-set evaluation.
- The remaining hard family is still `cylinder_boss / blind_hole_round / pocket_circular`, especially when the local topology is only one cylinder plus one plane.
- This is a candidate-instance ambiguity, not just a data volume problem.

### 2.6 v4.9 Real/Simplified Real STEP Recheck

Using the new v4.9 checkpoint:

- `output_real_step_s_v4_9_augmented_current`
- `output_real_step_s1_v4_9_augmented_current`

Key observations:

- `test_part_s/part31~part33` still include `slot_u`.
- `test_part_s1/part31` and `part32` include two `cylinder_boss` objects.
- The new model is more aggressive on some simplified real parts, e.g. some regions are converted into additional through-hole or single-face slot candidates. This confirms that face-level training alone is not enough for reliable real-part behavior.

Important review files:

- `output_real_step_s_v4_9_augmented_current/reports/compact_review_edge_aware_v4_9_topology_post.md`
- `output_real_step_s1_v4_9_augmented_current/reports/compact_review_edge_aware_v4_9_topology_post.md`

## 3. Current Progress Toward A Small Paper

Already completed:

- CAE target feature scope established.
- Synthetic STEP generator developed and iteratively improved.
- B-Rep graph extractor supports node and edge attributes.
- Rotation-invariant shape descriptors added.
- Edge-aware GNN trained.
- Same-instance edge supervision added.
- Topology rule library added.
- Candidate-level topology postprocess added.
- Controlled ablation datasets and full augmented datasets generated.
- Real/simplified real STEP testing workflow established.
- Compact review reports built for human inspection.

Still needed for a publishable paper:

1. Define the final task formally.
   - Input: STEP model and optional CAE requirement.
   - Output: feature instances and selectable face regions.
   - Scope: single-part structural-static preprocessing.

2. Freeze the feature taxonomy.
   - Holes, slots, pockets, bosses, ribs/webs, fillets/chamfers.
   - Explicitly state what is not included, such as full assembly semantics, full CAD command reconstruction and complex manufacturing process planning.

3. Add candidate-level model as the next methodological step.
   - Current GNN is face-level.
   - Current topology repair is rule-level.
   - Next model should classify candidate subgraphs:
     - candidate faces;
     - candidate internal/boundary edges;
     - pooled face/edge embeddings;
     - topology validity features.
   - This should target the remaining ambiguous families, especially `cylinder_boss / blind_hole_round / pocket_circular`.

4. Build a small manually reviewed real/simplified-real benchmark.
   - Not thousands of manually labeled real parts.
   - Recommended size: 15 to 30 simplified real models.
   - Use area truth tables only for selected ambiguous cases.
   - Report face-set IoU, instance precision/recall and CAE query success rate.

5. Decide final baselines.
   - Rule-only B-Rep recognition.
   - Face-level GNN.
   - Edge-aware GNN.
   - Edge-aware GNN + topology repair.
   - Candidate-level topology-aware model, once implemented.

6. Decide final paper metrics.
   - Face macro accuracy / macro-F1.
   - Semantic subtype macro-F1.
   - Same-instance edge F1.
   - Feature instance F1.
   - Face-set IoU.
   - Topology invalid rate.
   - CAE query success rate.

## 4. Recommended Next Version

The next major version should not be another random data increase.

Recommended v5.0 target:

> Candidate-level topology-aware feature classifier.

Proposed data flow:

1. Existing GNN predicts face labels and same-instance edge scores.
2. Rule library proposes candidate subgraphs.
3. Candidate encoder pools:
   - face embeddings;
   - face primitive histogram;
   - internal edge attributes;
   - boundary edge attributes;
   - topology rule pass/fail flags.
4. Candidate classifier predicts object class:
   - `slot_u`, `slot_rect`, `slot_o`, `slot_v`, `rib`, `web`;
   - `cylinder_boss`, `blind_hole_round`, `pocket_circular`, `stepped_hole_round`;
   - transition candidates.
5. Final repair selects non-overlapping candidates and outputs feature objects.

Why this matters:

- The current remaining errors are mostly not due to missing face evidence.
- The local evidence exists, but the system needs to decide at the object/candidate level.
- This is also a stronger paper contribution than only training a larger face-level GNN.

