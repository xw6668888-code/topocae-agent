# v5.4-G to v5.5 Validation Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the remaining real-part validation errors after v5.4-F, then produce a v5.5 model/report loop that is suitable for paper experiments.

**Architecture:** Keep the current pipeline unchanged: STEP -> invariant B-Rep graph -> edge-aware GNN -> candidate classifier -> topology postprocess -> feature objects -> area-truth review. v5.4-G only changes candidate-boundary/topology rules; v5.5 adds focused hard-case data and retraining after the rule layer is stable.

**Tech Stack:** Python, pythonOCC/OCC-derived graph extraction, PyTorch edge-aware GNN, scikit-learn candidate classifier, JSON graph/prediction artifacts, Markdown/JSON validation reports.

---

## File Structure

- Modify: `W:\python\FEA\dataset_generator\utils\topology_postprocess.py`
  - Add residual topology guards for large exterior through-hole false positives, slot_rect boundary matching, and fillet area mismatch cases.
- Modify: `W:\python\FEA\dataset_generator\tests\test_topology_postprocess_v5_4.py`
  - Add regression tests for part1/part2/part7 residual errors.
- Read: `W:\python\FEA\dataset_generator\output_real_step_s2_v5_4_f_part11_boundaryfix\reports\s2_area_truth_validation_v5_4_f_part11_boundaryfix_mergeeval.md`
  - Source of current residual errors.
- Read: `W:\python\FEA\dataset_generator\output_real_step_s2_v5_4_f_part11_boundaryfix\reports\failure_atlas_all_v5_4_f.md`
  - Face-level diagnosis source.
- Create: `W:\python\FEA\dataset_generator\output_real_step_s2_v5_4_g_residual_boundaryfix\`
  - New v5.4-G inference, feature object, and report output folder.
- Later modify/create: data generation configs/scripts for v5.5 focused hard cases after v5.4-G passes.

---

### Task 1: Freeze v5.4-F Residual Error List

- [ ] **Step 1: Read current area-truth report**

Run:

```powershell
Get-Content W:\python\FEA\dataset_generator\output_real_step_s2_v5_4_f_part11_boundaryfix\reports\s2_area_truth_validation_v5_4_f_part11_boundaryfix_mergeeval.md
```

Expected: report shows overall precision `0.9483`, recall `0.9823`, and part11 precision/recall `1.0000 / 1.0000`.

- [ ] **Step 2: Extract only residual samples**

The residual samples to focus on are:

```text
part1: one extra large through_hole_round false positive.
part2: missed slot_rect and one fillet; extra through_hole_round, slot_rect, fillet.
part7: inspect false positives from report before changing rules.
```

- [ ] **Step 3: Do not change part11 rules unless regression appears**

Part11 is now a passing regression target. Any later rule must preserve:

```text
part11 counterbore_hole: [7,8,14], [9,10,13]
part11 slot_u: [15,16,17]
part11 slot_v: [18,19]
part11 slot_o: [20]
part11 chamfer: [21], [22], [23], [24]
```

---

### Task 2: Add v5.4-G Regression Tests

- [ ] **Step 1: Add a large exterior cylinder false-positive guard test**

Modify `W:\python\FEA\dataset_generator\tests\test_topology_postprocess_v5_4.py`.

Add a test that builds a single large cylinder-like face with no two true circular opening loops and verifies it is not kept as `through_hole_round`.

Expected behavior:

```python
self.assertEqual(repaired_macro_for_large_exterior_cylinder, "stock")
```

- [ ] **Step 2: Add a slot_rect boundary test**

Add a test where a true rectangular slot has exactly three long concave-connected faces and a nearby unrelated plane. Verify the repair returns the three-face slot only.

Expected behavior:

```python
self.assertEqual(sorted(slot_faces), [expected_bottom, expected_side_a, expected_side_b])
```

- [ ] **Step 3: Add a fillet area-boundary test**

Add a test where a valid fillet and a nearby stock/slot face have similar topology but different support-edge patterns. Verify only the blend surface remains `fillet`.

Expected behavior:

```python
self.assertEqual(repaired_macro_for_true_blend, "fillet")
self.assertEqual(repaired_macro_for_wide_nonblend, "stock")
```

- [ ] **Step 4: Run the new tests and verify they fail before implementation**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_topology_postprocess_v5_4 -v
```

Expected: the newly added tests fail before the code changes.

---

### Task 3: Implement v5.4-G Residual Boundary Fixes

- [ ] **Step 1: Guard large exterior through-hole predictions**

Modify `W:\python\FEA\dataset_generator\utils\topology_postprocess.py`.

Add a helper near existing through-hole/transition guards:

```python
def _through_hole_shape_invalid_reason(face_indices: list[int], graph_data: dict) -> str | None:
    if len(face_indices) != 1:
        return None
    nodes = _nodes_by_idx(graph_data)
    idx = int(face_indices[0])
    node = nodes.get(idx, {})
    if node.get("primitive_surface") != "cylinder":
        return None
    if _node_descriptor(node, "uv_u_span") < 5.5:
        return "through_hole_requires_full_cylindrical_wall"
    circular_neighbors = _circular_edge_neighbor_count(idx, graph_data)
    if circular_neighbors < 2:
        return "through_hole_requires_two_circular_openings"
    return None
```

If `_circular_edge_neighbor_count` does not exist, implement it by counting adjacent edge records whose curve type is circular and whose neighbor face is not part of the same cylinder face.

- [ ] **Step 2: Apply the guard before accepting `through_hole_round` objects**

Inside `apply_topology_postprocess`, before finalizing a `through_hole_round` component:

```python
if macro_feature == "through_hole_round":
    hole_reason = _through_hole_shape_invalid_reason(component["face_indices"], graph_data)
    if hole_reason is not None:
        _demote_component(component, reason=hole_reason)
        continue
```

- [ ] **Step 3: Tighten slot_rect candidate completion**

Update the rectangular slot repair so that a slot candidate must satisfy:

```text
exactly one bottom-like plane
two side-like planes
side planes connected to bottom by concave line edges
all three faces have elongated projected aspect ratio or long-edge support
do not absorb unrelated adjacent stock or transition faces
```

- [ ] **Step 4: Preserve part11 U/V/O slot behavior**

After the slot_rect change, run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_topology_postprocess_v5_4.TopologyPostprocessV54Test.test_part11_like_overmerged_stepped_component_splits_to_counterbores_and_v_slot tests.test_topology_postprocess_v5_4.TopologyPostprocessV54Test.test_through_like_partial_cylinder_with_two_concave_side_planes_is_repaired_to_u_slot -v
```

Expected: both tests pass.

---

### Task 4: Re-run v5.4-G Real-Part Validation

- [ ] **Step 1: Run inference using unchanged trained models**

Run:

```powershell
W:\python\envs\feagpt\python.exe run_gnn_inference.py --checkpoint output_generation_v5_1_augmented_full_5400_clean\model_runs\edge_aware_v5_1_augmented_70ep\best.pt --input-path output_real_step_s2_v5_4_e_part11_refresh\graphs --output-dir output_real_step_s2_v5_4_g_residual_boundaryfix\predictions\edge_aware_v5_4_g_candidate_topology_post --recursive --device cpu --candidate-classifier output_generation_v5_1_candidate_topology_dataset\model_runs\random_forest_v5_1_candidate_topology_hardw8\model.joblib
```

- [ ] **Step 2: Convert predictions to feature objects**

Run:

```powershell
W:\python\envs\feagpt\python.exe prediction_to_feature_objects.py --prediction-path output_real_step_s2_v5_4_g_residual_boundaryfix\predictions\edge_aware_v5_4_g_candidate_topology_post --graph-dir output_real_step_s2_v5_4_e_part11_refresh\graphs --output-dir output_real_step_s2_v5_4_g_residual_boundaryfix\feature_objects\edge_aware_v5_4_g_candidate_topology_post --recursive
```

- [ ] **Step 3: Generate area-truth report**

Run:

```powershell
W:\python\envs\feagpt\python.exe generate_area_truth_review.py --feature-objects-path output_real_step_s2_v5_4_g_residual_boundaryfix\feature_objects\edge_aware_v5_4_g_candidate_topology_post --truth-json configs\test_part_s2_area_truth_v5_4_e_part11_refresh.json --output-json output_real_step_s2_v5_4_g_residual_boundaryfix\reports\s2_area_truth_validation_v5_4_g_residual_boundaryfix_mergeeval.json --output-md output_real_step_s2_v5_4_g_residual_boundaryfix\reports\s2_area_truth_validation_v5_4_g_residual_boundaryfix_mergeeval.md
```

- [ ] **Step 4: Generate failure atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe generate_failure_atlas.py --graph-dir output_real_step_s2_v5_4_e_part11_refresh\graphs --truth-json configs\test_part_s2_area_truth_v5_4_e_part11_refresh.json --prediction-stage candidate_topology_post=output_real_step_s2_v5_4_g_residual_boundaryfix\predictions\edge_aware_v5_4_g_candidate_topology_post --feature-object-stage candidate_topology_post=output_real_step_s2_v5_4_g_residual_boundaryfix\feature_objects\edge_aware_v5_4_g_candidate_topology_post --output-json output_real_step_s2_v5_4_g_residual_boundaryfix\reports\failure_atlas_all_v5_4_g.json --output-md output_real_step_s2_v5_4_g_residual_boundaryfix\reports\failure_atlas_all_v5_4_g.md
```

- [ ] **Step 5: Acceptance criteria**

Accept v5.4-G only if:

```text
part11 precision = 1.0000 and recall = 1.0000
part10 precision = 1.0000 and recall = 1.0000
overall recall >= 0.9823
overall precision > 0.9483
no new missed counterbore_hole, countersink_hole, slot_u, slot_v, rib
```

---

### Task 5: Build v5.5 Focused Hard-Case Dataset

- [ ] **Step 1: Add focused generated samples only after v5.4-G is accepted**

Generate hard cases for:

```text
large exterior cylinders that are stock/context, not through_hole_round
rectangular slots near fillets/chamfers
through holes near chamfered plate edges
slot_u / slot_o / fillet confusion cases
counterbore_hole and countersink_hole in thin plates
rib/web inside pockets with transitions absorbed into rib where semantically correct
```

- [ ] **Step 2: Keep hard-case labels object-level clean**

Each hard-case sample must have:

```text
valid STEP solid
valid graph JSON
macro feature object labels
same-instance groups
edge_attr preserved
no intentional destructive feature interference
```

- [ ] **Step 3: Train v5.5**

Use the current v5.1 model and candidate-classifier route as baseline. Retrain only after hard-case data are merged into the class-balanced distribution.

Expected output:

```text
output_generation_v5_5_hardcase_full\model_runs\...
output_generation_v5_5_hardcase_full\reports\...
```

---

### Task 6: Paper-Ready Experiment Packaging

- [ ] **Step 1: Freeze validation sets**

Define:

```text
Synthetic test set: generated, held-out, class-balanced.
Simplified real-part set: test_part_s2 with area truth.
Failure-focused set: hard cases derived from observed confusion modes.
```

- [ ] **Step 2: Export final paper tables**

Minimum tables:

```text
Table 1: target CAE feature categories and semantic use.
Table 2: graph attributes: face descriptors, edge descriptors, topology descriptors.
Table 3: ablation: GNN only / GNN + candidate classifier / GNN + classifier + topology postprocess.
Table 4: synthetic test metrics.
Table 5: simplified real-part area-truth metrics.
Table 6: hard-case confusion reduction.
```

- [ ] **Step 3: Export representative visual reports**

Use existing HTML/Markdown reports and selected STEP screenshots from UG/manual inspection. The paper should show:

```text
one hole/counterbore/countersink example
one slot example
one rib/web example
one transition example
one failure-to-success repair example
```

---

## Division of Work

**Codex should do:**

```text
write tests
modify topology rules
run inference/report scripts
compare metrics
generate hard-case data
train and evaluate v5.5
prepare report tables
```

**User should do:**

```text
confirm whether remaining false positives are true annotation/model errors
spot-check a small set of generated hard-case STEP files in UG
confirm final CAE semantic boundaries, especially for slot/transition/rib
provide or correct area-truth rows only for final validation parts, not for thousands of generated parts
```

