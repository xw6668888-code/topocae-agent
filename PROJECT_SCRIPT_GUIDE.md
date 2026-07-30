# Project Script Guide

Last updated: 2026-06-17

This guide explains the main Python scripts in `W:\python\FEA\dataset_generator` from a CAE engineer's point of view.

## 1. Big Picture Workflow

The project has five practical stages:

1. Generate STEP samples and labeled B-Rep graphs.
2. Export graph data into training format.
3. Train a GNN model.
4. Run inference on generated or real STEP-derived graphs.
5. Convert predictions into feature objects and CAE/Abaqus selections.

The important idea is:

- STEP geometry is first converted into a graph.
- Each graph node is a face.
- Each graph edge is a face adjacency relation.
- The neural network predicts face labels and same-instance relationships.
- The topology postprocess repairs or rejects impossible feature instances.
- Feature objects are then used for review or downstream CAE mapping.

## 2. Most Important Scripts

### `main.py`

Purpose:

- General dataset generator.
- Creates synthetic STEP models, extracts labeled graphs, writes label maps and exports training data.

Use when:

- You want to generate a normal configured dataset from a YAML config.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe main.py --config configs\static_structural_v3_1_balanced_3000.yaml --output-dir output_xxx
```

What it does:

- Samples a stock/body.
- Applies feature generators such as holes, slots, bosses, pockets, ribs, fillets and chamfers.
- Writes STEP files under `steps`.
- Writes graph JSON files under `graphs`.
- Writes label maps under `label_maps`.
- Exports training data under `training/gnn` and `training/llm`.

### `generate_ablation_dataset.py`

Purpose:

- Controlled dataset generator for experiments.
- More useful than `main.py` when you want a clean experimental dataset with specific constraints.

Use when:

- You want single-feature data.
- You want non-intersecting multi-feature data.
- You want focused hard cases for selected feature classes.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe generate_ablation_dataset.py `
  --mode nonintersecting_multi `
  --output-dir output_generation_v4_9_focused_hardconfusion_1500 `
  --sample-count 1500 `
  --seed 20260623 `
  --target-macros slot_u,slot_rect,slot_o,slot_v,web,rib,cylinder_boss,stepped_hole_round,blind_hole_round,pocket_circular `
  --max-features-per-sample 4 `
  --min-features-per-sample 2
```

Important modes:

- `single_feature`: one feature per model.
- `nonintersecting_multi`: multiple features per model, with footprint conflict checks.

### `build_augmented_dataset.py`

Purpose:

- Merge a full dataset and a focused hard-case dataset.

Use when:

- You already have a broad full distribution.
- You generate extra hard cases.
- You want one combined training set.

Example from v4.9:

```powershell
W:\python\envs\feagpt\python.exe build_augmented_dataset.py `
  --full-root output_generation_v4_8_augmented_full_6306_clean `
  --hardcase-root output_generation_v4_9_focused_hardconfusion_1500 `
  --output-dir output_generation_v4_9_augmented_full_7806_clean `
  --hardcase-prefix hardconfusion `
  --seed 20260624
```

Output:

- Combined `steps`, `graphs`, `training`, `label_maps`, `reports`.

### `train_gnn_baseline.py`

Purpose:

- Train the graph neural network.

Despite the name `baseline`, this now supports the edge-aware model.

Use when:

- You have exported graph training data.
- You want to train or retrain a model.

Example from v4.9:

```powershell
W:\python\envs\feagpt\python.exe train_gnn_baseline.py `
  --training-dir output_generation_v4_9_augmented_full_7806_clean\training\gnn `
  --label-map-dir output_generation_v4_9_augmented_full_7806_clean\label_maps `
  --output-dir output_generation_v4_9_augmented_full_7806_clean\model_runs `
  --run-name edge_aware_v4_9_augmented_hardconfusion_70ep `
  --model-kind edge_aware `
  --epochs 70 `
  --hidden-dim 176 `
  --num-layers 4 `
  --batch-size 32 `
  --dropout 0.12 `
  --edge-loss-weight 0.6 `
  --seed 20260624 `
  --device cuda
```

Important outputs:

- `best.pt`: best model checkpoint.
- `last.pt`: last epoch checkpoint.
- `metrics.json`: training/validation/test metrics.

### `step_to_inference_graph.py`

Purpose:

- Convert real `.step` or `.stp` files into unlabeled inference graphs.

Use when:

- You have STEP files from UG or another CAD tool.
- You want the trained model to recognize features in them.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe step_to_inference_graph.py `
  --input-path test_part_s1 `
  --output-dir output_real_step_s1_current\inference_graphs `
  --recursive `
  --overwrite
```

Output:

- One graph JSON per STEP file.
- These graphs do not contain ground-truth feature labels.

### `run_gnn_inference.py`

Purpose:

- Run a trained GNN checkpoint on graph JSON files.

Use when:

- You want predictions on generated graphs or real STEP-derived graphs.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe run_gnn_inference.py `
  --checkpoint output_generation_v4_9_augmented_full_7806_clean\model_runs\edge_aware_v4_9_augmented_hardconfusion_70ep\best.pt `
  --input-path output_real_step_s1_v4_8_augmented_current\inference_graphs `
  --output-dir output_real_step_s1_v4_9_augmented_current\predictions\edge_aware_v4_9_topology_post `
  --recursive `
  --device cpu
```

Important option:

- Default behavior includes topology postprocess.
- Add `--disable-topology-postprocess` if you want pure raw model predictions.

### `prediction_to_feature_objects.py`

Purpose:

- Convert face-level predictions into feature-object records.

Use when:

- You want to inspect recognized features as instances rather than face-by-face labels.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe prediction_to_feature_objects.py `
  --prediction-path output_real_step_s1_v4_9_augmented_current\predictions\edge_aware_v4_9_topology_post `
  --graph-dir output_real_step_s1_v4_8_augmented_current\inference_graphs `
  --output-dir output_real_step_s1_v4_9_augmented_current\feature_objects\edge_aware_v4_9_topology_post `
  --recursive
```

Output contains:

- macro feature type;
- face ids;
- area;
- centroid;
- bounding box;
- selectable region roles such as slot side faces, boss top face, transition faces.

### `generate_compact_real_step_review.py`

Purpose:

- Create a compact human-readable review table from feature objects.

Use when:

- You want to inspect real STEP inference results without opening large JSON files.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe generate_compact_real_step_review.py `
  --feature-objects-path output_real_step_s1_v4_9_augmented_current\feature_objects\edge_aware_v4_9_topology_post `
  --output-json output_real_step_s1_v4_9_augmented_current\reports\compact_review_edge_aware_v4_9_topology_post.json `
  --output-md output_real_step_s1_v4_9_augmented_current\reports\compact_review_edge_aware_v4_9_topology_post.md `
  --recursive
```

This is one of the most useful files for manual checking.

## 3. Evaluation And Report Scripts

### `evaluate_gnn_predictions.py`

Purpose:

- Compare prediction JSON files against labeled generated graph JSON files.

Use when:

- You are evaluating generated validation/test data.
- Do not use it directly for unlabeled real STEP parts unless ground truth exists.

Typical command:

```powershell
W:\python\envs\feagpt\python.exe evaluate_gnn_predictions.py `
  --prediction-path output_generation_v4_9_augmented_full_7806_clean\predictions\edge_aware_v4_9_augmented_hardconfusion_70ep_raw `
  --graph-dir output_generation_v4_9_augmented_full_7806_clean\graphs `
  --output-path output_generation_v4_9_augmented_full_7806_clean\reports\edge_aware_v4_9_augmented_hardconfusion_70ep_raw_prediction_eval.json `
  --recursive
```

### `generate_experiment_report.py`

Purpose:

- Convert metrics into a self-contained HTML report.

Use when:

- You want a visual report for inspection or paper figures.

Example:

```powershell
W:\python\envs\feagpt\python.exe generate_experiment_report.py `
  --dataset-root output_generation_v4_9_augmented_full_7806_clean `
  --run-dir output_generation_v4_9_augmented_full_7806_clean\model_runs\edge_aware_v4_9_augmented_hardconfusion_70ep `
  --prediction-eval-path output_generation_v4_9_augmented_full_7806_clean\reports\edge_aware_v4_9_augmented_hardconfusion_70ep_raw_prediction_eval.json `
  --output-path output_generation_v4_9_augmented_full_7806_clean\reports\edge_aware_v4_9_augmented_hardconfusion_70ep_raw_experiment_report.html
```

### `generate_area_truth_review.py`

Purpose:

- Compare predicted feature objects with manual area-based truth tables.

Use when:

- You do not know face ids in a real part.
- You can provide approximate total area and expected number of faces for each feature.

This is useful for your manually made simplified real parts.

### `compare_ablation_results.py`

Purpose:

- Compare two experiment outputs.

Use when:

- You want to compare v4.8 vs v4.9, or raw vs topology-post.

### `utils/rotation_prediction_consistency.py`

Purpose:

- Summarize consistency across rotated variants such as `part11`, `part12`, `part13`.

Use when:

- You want to test whether recognition is rotation-invariant.

## 4. Abaqus / Downstream Scripts

### `match_simulation_requirements.py`

Purpose:

- Match simulation requirement text to selectable regions.

Current note:

- Earlier versions supported Chinese matching.
- The long-term plan is to let an LLM/agent parse requirements, then call feature/region matching tools.

### `prediction_to_abaqus_mapping.py`

Purpose:

- Map recognized feature face ids onto Abaqus face ids using geometry fingerprints.

Use when:

- STEP face ids and Abaqus face ids are not identical.
- You need to bridge recognition results to an Abaqus model.

### `abaqus_mapping_to_action_plan.py`

Purpose:

- Convert Abaqus face mapping output into an action plan that can be consumed by script generation.

### `real_abaqus_minimal_pipeline.py`

Purpose:

- Run a minimal end-to-end pipeline from STEP to Abaqus action plan and script.

Use when:

- You want to smoke-test the downstream CAE integration.

### `mapping_eval.py`

Purpose:

- Evaluate Abaqus face mapping outputs if reference mapping or action-plan truth is available.

### `build_mock_abaqus_topology_smoke.py`

Purpose:

- Build a mock Abaqus topology snapshot from a graph.

Use when:

- You want to test mapping logic without launching Abaqus.

## 5. Validation And Audit Utilities

### `utils/topology_rules.py`

Purpose:

- Defines what is topologically valid for each feature.

Examples:

- A rectangular pocket should have planar bottom/side faces and concave internal edges.
- A cylinder boss should have cylinder side plus top plane.
- A through hole should be a single cylindrical wall with two circular openings.

### `utils/topology_postprocess.py`

Purpose:

- Applies topology-aware postprocess to GNN predictions.

This is where v4.9 fixed:

- U-slot recovery from a web-predicted curved bottom.
- Cylinder-boss recovery from a stepped-hole-predicted two-face exterior cylinder.

### `utils/topology_rule_audit.py`

Purpose:

- Audit generated graph feature sets using topology rules.

Use when:

- You want to check if a generated dataset contains invalid labels.

### `utils/step_solid_validation.py`

Purpose:

- Check whether generated STEP files are valid readable solids.

Useful for:

- Detecting bad geometry generation.
- Catching shell/sheet-body problems.

### `utils/slot_generation_audit.py`

Purpose:

- Audit swept slot generation, especially profile/path statistics and empty slot labels.

### `utils/hole_generation_audit.py`

Purpose:

- Audit hole axis distribution and internal-entry risk.

### `utils/transition_generation_audit.py`

Purpose:

- Audit fillet/chamfer generation.

### `utils/verify_dataset.py` and `utils/verify_labels.py`

Purpose:

- General consistency checks for generated graph labels and label maps.

## 6. Core Code Folders

### `features/`

Contains individual feature generators.

Examples:

- `through_hole.py`: through holes.
- `blind_hole.py`: blind holes.
- `stepped_hole.py`: stepped holes.
- `swept_channel_slot.py`: swept U/V/O slot generation.
- `slot_rect.py`: rectangular slot.
- `rect_pocket.py`: rectangular pocket.
- `circular_pocket.py`: circular pocket.
- `boss_feature.py`: cylinder boss.
- `pad.py`: rectangular boss.
- `rib.py`: rib/gusset and cavity rib.
- `web.py`: web/thin-wall candidate.
- `fillet_feature.py`: fillets.
- `chamfer_feature.py`: chamfers.

### `core/AAGExtractor.py`

Purpose:

- Extracts the attributed adjacency graph from OCC B-Rep topology.

This is where raw STEP geometry becomes:

- face nodes;
- adjacency edges;
- surface types;
- shape descriptors;
- edge attributes.

### `models/gnn_baseline.py`

Purpose:

- Defines the neural network models.

Important model:

- `EdgeAwareMultiHeadGNN`

This model uses:

- face/node features;
- primitive-surface embeddings;
- edge attributes;
- message passing;
- multiple output heads.

### `utils/training_export.py`

Purpose:

- Converts graph JSON files into `train.jsonl`, `val.jsonl`, `test.jsonl`.
- Defines node feature schemas and edge feature schemas.

This is the bridge between generated graph JSON and neural network training.

## 7. Recommended Commands For Reproducing v4.9

Generate focused hard-confusion data:

```powershell
W:\python\envs\feagpt\python.exe generate_ablation_dataset.py --mode nonintersecting_multi --output-dir output_generation_v4_9_focused_hardconfusion_1500 --sample-count 1500 --seed 20260623 --target-macros slot_u,slot_rect,slot_o,slot_v,web,rib,cylinder_boss,stepped_hole_round,blind_hole_round,pocket_circular --max-features-per-sample 4 --min-features-per-sample 2
```

Merge with v4.8 full dataset:

```powershell
W:\python\envs\feagpt\python.exe build_augmented_dataset.py --full-root output_generation_v4_8_augmented_full_6306_clean --hardcase-root output_generation_v4_9_focused_hardconfusion_1500 --output-dir output_generation_v4_9_augmented_full_7806_clean --hardcase-prefix hardconfusion --seed 20260624
```

Train:

```powershell
W:\python\envs\feagpt\python.exe train_gnn_baseline.py --training-dir output_generation_v4_9_augmented_full_7806_clean\training\gnn --label-map-dir output_generation_v4_9_augmented_full_7806_clean\label_maps --output-dir output_generation_v4_9_augmented_full_7806_clean\model_runs --run-name edge_aware_v4_9_augmented_hardconfusion_70ep --model-kind edge_aware --epochs 70 --hidden-dim 176 --num-layers 4 --batch-size 32 --dropout 0.12 --edge-loss-weight 0.6 --seed 20260624 --device cuda
```

Run real STEP inference:

```powershell
W:\python\envs\feagpt\python.exe run_gnn_inference.py --checkpoint output_generation_v4_9_augmented_full_7806_clean\model_runs\edge_aware_v4_9_augmented_hardconfusion_70ep\best.pt --input-path output_real_step_s1_v4_8_augmented_current\inference_graphs --output-dir output_real_step_s1_v4_9_augmented_current\predictions\edge_aware_v4_9_topology_post --recursive --device cpu
```

Convert predictions to feature objects:

```powershell
W:\python\envs\feagpt\python.exe prediction_to_feature_objects.py --prediction-path output_real_step_s1_v4_9_augmented_current\predictions\edge_aware_v4_9_topology_post --graph-dir output_real_step_s1_v4_8_augmented_current\inference_graphs --output-dir output_real_step_s1_v4_9_augmented_current\feature_objects\edge_aware_v4_9_topology_post --recursive
```

Generate compact review:

```powershell
W:\python\envs\feagpt\python.exe generate_compact_real_step_review.py --feature-objects-path output_real_step_s1_v4_9_augmented_current\feature_objects\edge_aware_v4_9_topology_post --output-json output_real_step_s1_v4_9_augmented_current\reports\compact_review_edge_aware_v4_9_topology_post.json --output-md output_real_step_s1_v4_9_augmented_current\reports\compact_review_edge_aware_v4_9_topology_post.md --recursive
```

## 8. How To Read The Project As A Beginner

Recommended reading order:

1. `PROJECT_SCRIPT_GUIDE.md`
2. `v4_9_iteration_summary.md`
3. `core/AAGExtractor.py`
4. `utils/training_export.py`
5. `models/gnn_baseline.py`
6. `utils/topology_rules.py`
7. `utils/topology_postprocess.py`
8. One simple feature generator, such as `features/through_hole.py`
9. One complex feature generator, such as `features/swept_channel_slot.py`

You do not need to understand all neural-network details first. For this project, the engineering intuition is:

- `AAGExtractor.py` decides what geometric information the model can see.
- `training_export.py` decides how that information becomes numbers.
- `gnn_baseline.py` learns face and edge labels from those numbers.
- `topology_rules.py` and `topology_postprocess.py` force the final output to obey B-Rep feature logic.

