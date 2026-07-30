# v5.0 Candidate Topology Classifier Results

## Dataset

Source data:

- Graphs: `output_generation_v4_9_augmented_full_7806_clean/graphs`
- Raw GNN predictions: `output_generation_v4_9_augmented_full_7806_clean/predictions/edge_aware_v4_9_augmented_hardconfusion_70ep_raw`
- Candidate dataset: `output_generation_v5_0_candidate_topology_dataset`

Build command:

```powershell
W:\python\envs\feagpt\python.exe build_candidate_classifier_dataset.py `
  --graph-dir output_generation_v4_9_augmented_full_7806_clean\graphs `
  --prediction-dir output_generation_v4_9_augmented_full_7806_clean\predictions\edge_aware_v4_9_augmented_hardconfusion_70ep_raw `
  --output-dir output_generation_v5_0_candidate_topology_dataset `
  --recursive `
  --seed 20260625 `
  --max-records-per-label 12000
```

Candidate count:

- Total: `19352`
- Train: `15417`
- Val: `1984`
- Test: `1951`

Label counts:

- `reject`: `3692`
- `cylinder_boss`: `2003`
- `slot_rect`: `1840`
- `slot_u`: `1750`
- `slot_o`: `1597`
- `pocket_circular`: `1401`
- `blind_hole_round`: `1363`
- `through_hole_round`: `1201`
- `obround_hole`: `1197`
- `slot_v`: `1114`
- `web`: `951`
- `fillet`: `652`
- `rib`: `591`

## Models

All runs use a RandomForest candidate-level classifier over B-Rep topology features.

| Run | hard-source weight | Test Acc | Test Macro-F1 | Prediction-component Acc |
|---|---:|---:|---:|---:|
| `random_forest_v5_0_candidate_topology_hardw1` | 1.0 | 0.9539 | 0.9585 | 0.6407 |
| `random_forest_v5_0_candidate_topology_hardw4` | 4.0 | 0.9400 | 0.9465 | 0.7056 |
| `random_forest_v5_0_candidate_topology_hardw8` | 8.0 | 0.9262 | 0.9326 | 0.7576 |

Recommended first integration model:

- `output_generation_v5_0_candidate_topology_dataset/model_runs/random_forest_v5_0_candidate_topology_hardw4/model.joblib`

Rationale:

- `hardw1` is best for clean synthetic candidate scoring.
- `hardw4` is a better compromise for real inference because it pays more attention to model/rule-generated hard candidates.
- `hardw8` improves prediction-component accuracy further, but starts to damage clean candidate performance and cylinder-boss/blind-hole separation.

## Key Test F1 for Recommended Run

For `hardw4`:

- `reject`: `0.9735`
- `rib`: `1.0000`
- `slot_o`: `0.9937`
- `slot_u`: `0.9938`
- `fillet`: `0.9531`
- `cylinder_boss`: `0.7667`
- `blind_hole_round`: `0.8764`

## Interpretation

The classifier is strong for:

- `rib/stock` rejection;
- `slot_o/fillet` separation;
- `slot_u/obround_hole` style slot-object recognition.

The weaker area remains:

- `cylinder_boss / blind_hole_round / pocket_circular`, because these share similar cylinder-plus-plane local topology.

Next step:

- plug the candidate classifier into real inference as a second-stage repair layer;
- evaluate `hardw1` and `hardw4` on `test_part_s` and `test_part_s1`;
- if cylinder-boss/blind-hole remains unstable, add candidate features for exterior/interior direction, cap-plane side, and opening count.
