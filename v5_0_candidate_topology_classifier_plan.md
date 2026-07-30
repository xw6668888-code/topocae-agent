# v5.0 Candidate-Level Topology-Aware Classifier Plan

## Purpose

v4.x already proves that an edge-aware face-level GNN can recognize many CAE feature faces, but the remaining real-part errors are mostly object-level ambiguities:

- large base plate faces can locally look like a rib/web candidate;
- large swept slot surfaces can look like fillet cylinders;
- U/O/rect/V slots can share local face primitives but differ in boundary context;
- cylinder boss, blind hole and circular pocket can share a cylinder-plus-plane pattern.

v5.0 therefore adds a second-stage classifier at the candidate/instance level. It should classify a proposed face set, not a single face.

## Data Flow

1. Run the existing edge-aware GNN on a graph.
2. Use topology rules to produce candidate face sets.
3. Extract candidate-level topology features with `utils/candidate_topology_features.py`.
4. Classify each candidate as one target macro or `reject/stock`.
5. Apply accepted candidate labels back to faces.
6. Run final topology validation and feature-object export.

## Candidate Features

The first implemented feature extractor outputs:

- candidate size: face count, area, model area ratio;
- primitive composition: plane/cylinder/cone/curved counts;
- internal and boundary edge statistics: line/circle counts, concave/convex/smooth counts, long-edge counts;
- parent context: number of external neighbor faces and parent-plane parallel/orthogonal/intersecting pair counts;
- cylinder span estimate: area divided by radius and sweep length;
- prediction evidence: raw stock fraction and candidate macro face fraction;
- topology validity flag for the proposed macro.

These directly target the latest observed failures:

- `rib` false positives are filtered by large candidate area ratio and raw-stock evidence;
- `slot_o` vs `fillet` uses cylindrical span plus parent-plane relation;
- `slot_u` vs `obround_hole` uses bottom cylinder plus two side planes and concave boundary evidence.

## Training Dataset

Build candidate samples from existing generated graphs and predictions:

- positive candidates: ground-truth `feature_sets` from graph JSON;
- hard negative candidates: candidates proposed by current rules whose face set does not match a ground-truth instance;
- hard confusion candidates: rule/model outputs that overlap a true instance but have the wrong macro.

Label policy:

- exact or high-overlap match with same macro: target macro;
- high-overlap match with different macro: true macro;
- low-overlap or mostly stock: `reject`.

## Model

Start with a small tabular classifier rather than a larger GNN:

- input: candidate topology feature vector;
- output: macro class plus `reject`;
- recommended first model: MLP or GradientBoosting/RandomForest baseline;
- metric: candidate-level macro-F1, reject precision, and per-class confusion matrix.

This is intentionally separate from the face-level GNN so we can explain it clearly in the paper as a neuro-symbolic repair layer.

## Next Implementation Steps

1. Add `build_candidate_classifier_dataset.py`.
2. Generate candidate JSONL from v4.9 full dataset and v4.9 raw predictions.
3. Add `train_candidate_topology_classifier.py`.
4. Evaluate on:
   - generated validation/test split;
   - `test_part_s`;
   - `test_part_s1`.
5. Add optional `--candidate-classifier` flag to `run_gnn_inference.py`.
6. Compare three outputs in the report:
   - raw GNN;
   - GNN + rules;
   - GNN + rules + candidate classifier.

## Paper Position

v5.0 should be described as a candidate-level topology-aware verification layer for CAE feature selection. Its contribution is not replacing the GNN, but correcting candidate instances using interpretable B-Rep object descriptors.
