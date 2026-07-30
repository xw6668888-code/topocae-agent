# TopoCAE-Agent

TopoCAE-Agent is a research system for grounding natural-language structural FEA requirements to exact B-Rep face sets in controlled, single-part STEP models. It combines topology-aware geometry recognition, typed CAE intent parsing, constrained face-set grounding, validation, and Abaqus action generation.

## Current Scope and Status

The first paper targets single-part structural-static FEA pre-processing. The supported workflow is:

```text
STEP/B-Rep
-> attributed topology graph
-> frozen geometry recognizer and feature-object memory

natural-language CAE requirement
-> typed CAE-IR
-> topology-constrained grounding
-> exact face set
-> validation
-> solver-neutral action plan
-> Abaqus execution adapter
```

Current research status:

- P1 geometry recognizer: bounded pass in a controlled domain.
- P2 CAE-IR: bounded pass.
- P3 CAE-QueryGround data: pass.
- P4 language and grounding baselines: LLM runtime preparation is in progress; formal test data remains locked.

This is an active research repository, not a production-ready CAE product.

## Repository Contents

- `core/`, `features/`, `models/`, `utils/`, `CAE/`: implementation.
- `configs/`, `schemas/`, `prompts/`: reproducible configuration and data contracts.
- `tests/`: regression and validation tests.
- `docs/`, `migration/`: design, research, and portability documentation.
- `environment.yml`: Conda environment definition.

Generated datasets, checkpoints, CAE files, logs, and experiment reports are intentionally excluded by `.gitignore`. Keep selected lightweight conclusions in `docs/`; store large artifacts separately with a versioned manifest and checksum.

## Environment

```bash
conda env create -f environment.yml
conda activate aagnet
```
## Legacy Dataset Generator Guide

面向 `single_part_structural_cae` 的程序化数据集生成项目。当前项目可以直接生成：

- `STEP` 几何文件
- `AAG/B-Rep graph` 图数据
- 真实 `STEP -> inference graph` 无标签推理输入
- 统一 ontology label maps
- `train/val/test` split
- 直接可用于 `GNN` 的 JSONL
- 直接可用于 `LLM SFT` 的 messages JSONL
- 规则推断的 `thin_wall_region / thickness_transition / midsurface_candidate`
- 规则推断的 `idealization_tag`

## V1 默认支持的宏特征

当前默认配置聚焦单零件结构静力前处理中的局部、可查询特征。非核心生成器仍保留在 `features/` 中，但默认数据集不会启用。

- `through_hole_round`
- `blind_hole_round`
- `stepped_hole_round`
- `counterbore_hole`
- `countersink_hole`
- `obround_hole`
- `slot_u`
- `slot_rect`
- `pocket_rect`
- `pocket_circular`
- `cylinder_boss`
- `rectangular_boss`
- `rib`
- `web`
- `fillet`
- `chamfer`

孔组、法兰、支撑垫、加载垫不作为 V1 宏特征：孔组由多个孔对象后处理聚类得到；支撑/加载含义由仿真工况决定；法兰属于更高层的零件/连接语义。

## 目录结构

```text
dataset_generator/
├── CAE_feature_ontology.yaml
├── face_semantic_subtype_ontology.yaml
├── step_to_inference_graph.py
├── configs/default_dataset_config.yaml
├── features/
├── core/
├── utils/
└── output/
    ├── steps/
    ├── graphs/
    ├── feature_catalog/
    ├── label_maps/
    ├── manifests/
    ├── splits/
    ├── reports/
    └── training/
        ├── gnn/
        ├── llm/
        └── metadata/
```

## 运行

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/main.py
```

自定义输出目录、样本数、随机种子：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/main.py \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output_v2 \
  --num-samples 1000 \
  --seed 20260420
```

默认启用 `coverage_control`，会在基础样本数生成完成后，继续追加单特征补样，直到每个启用宏特征达到最小 `feature_set` 实例数阈值，或达到 `max_extra_samples`。

只生成“单特征目录”，每个 `STEP` 只包含一种宏特征：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/main.py \
  --build-feature-catalog \
  --feature-catalog-only \
  --feature-catalog-variants 1 \
  --seed 20260421
```

输出目录结构：

```text
output/feature_catalog/
├── blind_hole_round/
│   ├── steps/
│   └── graphs/
├── chamfer/
│   ├── steps/
│   └── graphs/
├── ...
└── feature_catalog_meta.json
```

只重新导出训练文件：

```bash
python /home/ywx/ywx/FEA/dataset_generator/utils/training_export.py \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output
```

校验 graph 数据：

```bash
python /home/ywx/ywx/FEA/dataset_generator/utils/verify_dataset.py
```

覆盖率摘要输出在 `output/reports/coverage_report.json`。

训练 GNN baseline：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/train_gnn_baseline.py \
  --epochs 20 \
  --batch-size 16
```

训练输出会写到 `output/model_runs/<run_name>/`，至少包含：

- `config.json`
- `best.pt`
- `last.pt`
- `metrics.json`

对无标签 inference graph 运行推理：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/run_gnn_inference.py \
  --checkpoint /home/ywx/ywx/FEA/dataset_generator/output/model_runs/<run_name>/best.pt \
  --input-path /home/ywx/ywx/FEA/dataset_generator/output/inference_graphs \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/predictions \
  --recursive
```

将真实 `STEP` 转成无标签推理图：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/step_to_inference_graph.py \
  --input-path /home/ywx/ywx/FEA/dataset_generator/output/steps/composite_feature_000.step \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/inference_graphs \
  --overwrite
```

批量转换目录：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/step_to_inference_graph.py \
  --input-path /home/ywx/ywx/FEA/dataset_generator/output/steps \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/inference_graphs \
  --recursive \
  --overwrite
```

输出的 inference graph 只保留模型输入字段：

- `idx`
- `primitive_surface`
- `primitive_surface_id`
- `attributes`
- `edges`

不包含监督标签字段：

- `macro_feature`
- `semantic_label`
- `instance_id`

对应 schema 在 [schemas/inference_graph.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/inference_graph.schema.json:1)。

## 派生 CAE 标签

当前在不增加新几何生成器的前提下，基于 face subtype 和邻接关系额外推断：

- `thin_wall_region`
- `thickness_transition`
- `midsurface_candidate`

同时写入节点级 `idealization_tag`：

- `shell_candidate`
- `mesh_sensitive_region`

这些标签会出现在每个 graph 节点的以下字段中：

- `derived_macro_feature_ids`
- `derived_macro_features`
- `idealization_tag_ids`
- `idealization_tags`

图级摘要写在 `derived_label_summary`。

## 训练对接

### GNN

`output/training/gnn/{train,val,test}.jsonl` 每行一个样本，包含：

- `node_features`
- `edge_index`
- `primitive_surface_ids`
- `macro_feature_ids`
- `semantic_subtype_ids`
- `derived_macro_feature_ids_per_node`
- `idealization_tag_ids_per_node`
- `instance_ids`
- `feature_sets`

配套 schema 在 [schemas/gnn_record.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/gnn_record.schema.json:1)。

当前 baseline 监督头：

- `macro_feature_ids`
- `semantic_subtype_ids`
- `derived_macro_feature_ids_per_node`
- `idealization_tag_ids_per_node`

当前 baseline 推理输出：

- `macro_feature`
- `semantic_subtype`
- `derived_macro_features`
- `idealization_tags`

将 face-level prediction 组装成 feature objects：

```bash
python /home/ywx/ywx/FEA/dataset_generator/prediction_to_feature_objects.py \
  --prediction-path /home/ywx/ywx/FEA/dataset_generator/output/predictions \
  --graph-dir /home/ywx/ywx/FEA/dataset_generator/output/inference_graphs \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/feature_objects \
  --recursive \
  --fingerprint-round-ndigits 6
```

当前 feature object 采用启发式组装：

- 按 `macro_feature_id`
- 按图邻接连通分量

每个 feature object 会输出：

- `face_indices`
- `fingerprint.schema_version`
- `fingerprint.software_agnostic_signature`
- `fingerprint.area_weighted_centroid`
- `fingerprint.center_bbox_min/max/size`
- `fingerprint.radius_stats`
- `fingerprint.face_descriptors`
- `primitive_surface_histogram`
- `semantic_subtype_histogram`
- `derived_macro_feature_histogram`
- `idealization_tag_histogram`
- `adjacent_macro_feature_histogram`
- `matching_hints`

对应 schema 在 [schemas/feature_objects.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/feature_objects.schema.json:1)。
fingerprint 独立 schema 在 [schemas/feature_fingerprint.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/feature_fingerprint.schema.json:1)。

`software_agnostic_signature` 不包含原始 face id，用于跨软件候选匹配；真正落到 Abaqus、UG/NX 等软件时，仍应结合 `face_descriptors` 里的面积、中心、法向/轴向、半径和局部拓扑信息做二次确认。

### Abaqus Mapping

将 `feature objects` 映射到 Abaqus 拓扑面编号：

```bash
python /home/ywx/ywx/FEA/dataset_generator/prediction_to_abaqus_mapping.py \
  --feature-object-path /home/ywx/ywx/FEA/dataset_generator/output/feature_objects \
  --topology-path /home/ywx/ywx/FEA/cae/topology_data.json \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/abaqus_mapping
```

如果 `--topology-path` 指向目录，脚本会按文件名匹配对应的 topology snapshot。

映射输出包含：

- `feature_mappings`
- `mapped_abaqus_face_ids`
- `mapping_score_mean`
- `mapping_status`
- `suggested_abaqus_regions`

对应 schema 在 [schemas/abaqus_feature_mapping.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/abaqus_feature_mapping.schema.json:1)。

### Abaqus Smoke Case

没有真实 Abaqus 导出的 topology snapshot 时，可以先用同源 inference graph 构造一份 mock topology，验证“编号打乱后是否还能正确回配”：

```bash
python /home/ywx/ywx/FEA/dataset_generator/build_mock_abaqus_topology_smoke.py \
  --graph-path /home/ywx/ywx/FEA/dataset_generator/output/inference_graph_smoke/composite_feature_000.json \
  --output-path /home/ywx/ywx/FEA/dataset_generator/output/mock_abaqus_topology_smoke/composite_feature_000.json \
  --seed 42 \
  --part-name MockCompositeFeature000
```

再执行映射：

```bash
python /home/ywx/ywx/FEA/dataset_generator/prediction_to_abaqus_mapping.py \
  --feature-object-path /home/ywx/ywx/FEA/dataset_generator/output/feature_objects_smoke \
  --topology-path /home/ywx/ywx/FEA/dataset_generator/output/mock_abaqus_topology_smoke \
  --output-dir /home/ywx/ywx/FEA/dataset_generator/output/abaqus_mapping_smoke \
  --recursive
```

这个 smoke case 用来验证流程和匹配逻辑，不代表真实 Abaqus 精度上限。由于 mock topology 与源 graph 同源，几何字段一致，只是 face id 被打乱，所以理想结果会接近满分匹配。

### Action Plan Conversion

将 `prediction_to_abaqus_mapping.py` 的输出转换成 [step5_script_gen.py](/home/ywx/ywx/FEA/cae/step5_script_gen.py:1) 可直接消费的 action plan：

```bash
python /home/ywx/ywx/FEA/dataset_generator/abaqus_mapping_to_action_plan.py \
  --mapping-path /home/ywx/ywx/FEA/dataset_generator/output/abaqus_mapping_smoke/composite_feature_000.json \
  --output-path /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke/composite_feature_000.json \
  --simulation-name AI_Job_Smoke \
  --part-name MockCompositeFeature000 \
  --instance-name MockCompositeFeature000-1 \
  --bc-mode largest_stock_planar \
  --load-mode first_feature_region
```

默认会生成：

- `Material`
- `Mesh`
- `MeshRefinement`
- `BoundaryCondition`
- `Load`

其中：

- `BoundaryCondition` 默认取最大 `stock_planar_region`
- `Load` 默认取第一个达到最小置信度阈值的 `feature_object_region`
- `MeshRefinement` 默认对所有达到最小置信度阈值的 feature faces 做局部加密

对应 schema 在 [schemas/abaqus_action_plan.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/abaqus_action_plan.schema.json:1)。

### Step5 Script Generation

`step5_script_gen.py` 现在支持显式指定输入和输出路径：

```bash
python /home/ywx/ywx/FEA/cae/step5_script_gen.py \
  --input-action-file /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke/composite_feature_000.json \
  --output-script-file /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke/abaqus_run_script_smoke.py
```

如果 action plan 顶层包含：

- `model_name`
- `part_name`
- `instance_name`

脚本会优先使用这些值；CLI 参数会覆盖 JSON 顶层配置。

### Real Abaqus Minimal Pipeline

对真实 Abaqus topology 的最小闭环，建议直接使用模板脚本：

```bash
conda run -n aagnet python /home/ywx/ywx/FEA/dataset_generator/real_abaqus_minimal_pipeline.py \
  --step-path /abs/path/your_part.step \
  --topology-path /abs/path/topology_data.json \
  --checkpoint /home/ywx/ywx/FEA/dataset_generator/output/model_runs/smoke_gnn_baseline/best.pt \
  --output-root /abs/path/real_pipeline_run \
  --part-name YourPart \
  --instance-name YourPart-1 \
  --global-size 2.0 \
  --local-size 0.5 \
  --bc-mode largest_stock_planar \
  --load-mode first_feature_region \
  --overwrite
```

输出目录会包含：

- `inference_graphs`
- `predictions`
- `feature_objects`
- `abaqus_mapping`
- `action_plans`
- `abaqus_scripts`
- `real_pipeline_manifest.json`

前提条件：

1. `STEP` 和 `topology_data.json` 必须来自同一个真实零件。
2. `topology_data.json` 需要先在 Abaqus 中运行 [step1_topology.py](/home/ywx/ywx/FEA/cae/step1_topology.py:1) 导出。
3. `part_name / instance_name` 必须和 Abaqus 模型中的名字一致。

### Mapping Evaluation

对映射结果做结构审计，或和参考映射/参考 action plan 对比：

```bash
python /home/ywx/ywx/FEA/dataset_generator/mapping_eval.py \
  --mapping-path /home/ywx/ywx/FEA/dataset_generator/output/abaqus_mapping_smoke \
  --reference-mapping-path /home/ywx/ywx/FEA/dataset_generator/output/abaqus_mapping_smoke \
  --action-plan-path /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke \
  --reference-action-plan-path /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke \
  --output-path /home/ywx/ywx/FEA/dataset_generator/output/action_plan_smoke/mapping_eval_report.json \
  --recursive
```

如果只提供 `--mapping-path`，脚本会输出：

- `feature_mapping_count`
- `mapped_face_count`
- `unique_mapped_face_count`
- `reused_mapped_face_count`
- `mean_mapping_score`
- `status_counts`

如果同时提供参考映射，会额外输出：

- `exact_feature_ratio`
- `mean_feature_iou`
- `mean_feature_precision`
- `mean_feature_recall`

如果同时提供 action plan 和参考 action plan，还会输出：

- `boundary_condition` 的 `IoU / precision / recall`
- `load` 的 `IoU / precision / recall`

### 推理输入

模型推理建议走：

`STEP -> inference graph -> model -> per-face labels / instance grouping -> feature objects`

当前项目不建议直接把 `STEP` 文本作为监督输入。训练和推理都以 graph 为中心。

### LLM SFT

`output/training/llm/{train,val,test}.jsonl` 采用 `messages` 格式：

- `system`: 任务规则
- `user`: 未标注的图结构输入
- `assistant`: 标准答案 JSON

配套 schema 在 [schemas/llm_sft_record.schema.json](/home/ywx/ywx/FEA/dataset_generator/schemas/llm_sft_record.schema.json:1)。

## 现阶段边界

- 这是 `single-part` 数据集，不包含装配接触语义。
- `analysis_role` 还未启用，当前重点是几何与 face-level recognition。
- `match_face` 仍然是后处理策略，后续建议继续改成生成即标注。
- `thin_wall_region / thickness_transition / midsurface_candidate` 当前是规则推断，不是独立 procedural geometry。

## 下一步建议

优先顺序：

1. 引入基于规则的 `idealization_tag`
2. 将 `match_face` 迁移到生成时同步标注
3. 增加大样本量生成和 coverage threshold 控制
4. 为 `thin_wall_region` 增加更严格的厚度/长细比判定

