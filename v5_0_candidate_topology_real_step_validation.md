# v5.0 Candidate Topology Real STEP Validation

## Pipeline

Validated pipeline:

```text
raw edge-aware GNN
-> candidate proposal
-> candidate topology classifier
-> topology postprocess
-> feature objects
-> compact review
```

GNN checkpoint:

`output_generation_v4_9_augmented_full_7806_clean/model_runs/edge_aware_v4_9_augmented_hardconfusion_70ep/best.pt`

Candidate classifier:

`output_generation_v5_0_candidate_topology_dataset/model_runs/random_forest_v5_0_candidate_topology_hardw4/model.joblib`

## Outputs

`test_part_s`:

- Predictions: `output_real_step_s_v5_0_candidate_topology_hardw4_current/predictions/edge_aware_v5_0_candidate_topology_post`
- Feature objects: `output_real_step_s_v5_0_candidate_topology_hardw4_current/feature_objects/edge_aware_v5_0_candidate_topology_post`
- Review: `output_real_step_s_v5_0_candidate_topology_hardw4_current/reports/compact_review_edge_aware_v5_0_candidate_topology_post.md`

`test_part_s1`:

- Predictions: `output_real_step_s1_v5_0_candidate_topology_hardw4_current/predictions/edge_aware_v5_0_candidate_topology_post`
- Feature objects: `output_real_step_s1_v5_0_candidate_topology_hardw4_current/feature_objects/edge_aware_v5_0_candidate_topology_post`
- Review: `output_real_step_s1_v5_0_candidate_topology_hardw4_current/reports/compact_review_edge_aware_v5_0_candidate_topology_post.md`

## Observations

### `test_part_s`

The v5.0 candidate layer keeps the v4.9.1 corrected behavior:

- `part11~part13`: base plate faces `[0, 1, 9]` remain `stock`, not `rib`.
- `part11~part13`: rectangular boss `[15, 16, 17, 18, 19]` remains detected.
- `part31~part33`: slot and transition split remains stable:
  - `slot_o`: one object;
  - `slot_u`: one object;
  - `slot_rect`: one object;
  - `slot_v`: one object;
  - two `fillet` and two `chamfer` transition faces.

### `test_part_s1`

The important hard cases are improved or preserved:

- `part31/part32`: obround hole `[10, 11, 12, 13]` is preserved as one 4-face `obround_hole`.
- `part31/part32`: both cylinder bosses are detected:
  - `[21, 22]`
  - `[23, 24]`
- `part31/part32`: both U-slots are detected:
  - `[25, 26, 27]`
  - `[28, 29, 30]`
- `part21~part23`: rib, through holes, fillets, chamfers, O-slot and V-slot are retained.

## Fix Made During Validation

Initial v5.0 validation exposed one regression:

- multi-face `obround_hole` components could be split by later slot repair rules into two `slot_o` end cylinders plus stock side planes.

Fix:

- `utils/topology_postprocess.py` now protects multi-face obround predictions with at least two cylindrical end faces and two planar side faces from being consumed by `slot_u` or `slot_o` candidate repairs.

Regression test:

- `tests/test_topology_postprocess.py::TopologyPostprocessTest.test_multi_face_obround_hole_is_not_split_into_slot_o_end_cylinders`

## Interpretation

The candidate-level classifier is useful, but its best role is not to replace topology rules. It works best as a second-stage object verifier:

- reject impossible prediction components;
- accept or relabel candidate objects when topology descriptors are strong;
- then let topology postprocess enforce hard B-Rep validity.

For the current real-step samples, the most visible gain is in object-level consistency:

- preserving complete obround holes;
- keeping U-slot instances together;
- recovering cylinder bosses as two-face objects;
- preventing large stock-like components from becoming rib.
