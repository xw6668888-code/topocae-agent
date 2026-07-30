# CAE STEP 识别路线图实施计划

> **面向代理执行者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项执行本计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 将当前的 STEP/AAG 数据集生成器发展为一个面向 CAE 特征识别的、论证充分的研究流程；该流程应包含边属性感知的图输入、特征实例分组、CAE 角色标签、真实 Abaqus 验证，以及可直接支撑论文写作的证据。

**架构：** 保持当前流程形态：`STEP -> AAG graph -> training records -> GNN prediction -> feature objects -> Abaqus mapping -> action plan`。通过增加边属性、显式实例亲和性监督、合成 CAE 分析工况标签和真实拓扑验证来补强薄弱环节。将 ontology 和 schema 作为数据生成、训练、推理和论文实验之间的契约。

**技术栈：** Python 3.10、pythonocc-core/OpenCASCADE、PyTorch、PyYAML、JSON Schema 风格记录、来自 `W:\python\FEA\cae` 的 Abaqus 拓扑导出脚本，以及 Markdown/CSV 实验报告。

---

## 文件结构

- 修改：`W:\python\FEA\dataset_generator\core\AAGExtractor.py`
  - 增加由共享 B-Rep 边和相邻面几何派生的边级描述符。
- 修改：`W:\python\FEA\dataset_generator\utils\training_export.py`
  - 导出 `edge_attr`、`edge_instance_affinity`，后续再导出 `analysis_role_ids_per_node`。
- 修改：`W:\python\FEA\dataset_generator\schemas\gnn_record.schema.json`
  - 记录新的边字段和 CAE 角色字段。
- 修改：`W:\python\FEA\dataset_generator\models\gnn_baseline.py`
  - 增加可选的边特征投影和边亲和性预测头。
- 修改：`W:\python\FEA\dataset_generator\train_gnn_baseline.py`
  - 训练并报告 face 标签指标和实例亲和性指标。
- 修改：`W:\python\FEA\dataset_generator\run_gnn_inference.py`
  - 解码新的模型输出。
- 新建：`W:\python\FEA\dataset_generator\utils\analysis_case_labels.py`
  - 生成确定性的合成支撑/载荷/网格区域标签，并与纯几何标签分离。
- 新建：`W:\python\FEA\dataset_generator\utils\label_audit.py`
  - 审计标签一致性，并导出样本级 CSV 供人工检查。
- 新建：`W:\python\FEA\dataset_generator\utils\experiment_report.py`
  - 汇总指标、覆盖率、映射结果和数据集事实，形成论文可用表格。
- 新建：`W:\python\FEA\dataset_generator\docs\research\problem_scope.md`
  - 固化论文主张：面向 CAE 的 STEP/B-Rep 识别，而不是通用 CAD/CAM 特征识别。
- 新建：`W:\python\FEA\dataset_generator\docs\research\experiment_matrix.md`
  - 定义数据集、划分方式、基线、指标和预期图表。

---

## 任务 1：固化研究范围和实验矩阵

**文件：**
- 新建：`W:\python\FEA\dataset_generator\docs\research\problem_scope.md`
- 新建：`W:\python\FEA\dataset_generator\docs\research\experiment_matrix.md`
- 修改：`W:\python\FEA\dataset_generator\README.md`

- [ ] **步骤 1：创建研究文档目录**

运行：

```powershell
New-Item -ItemType Directory -Force -Path 'W:\python\FEA\dataset_generator\docs\research'
```

预期结果：目录存在。

- [ ] **步骤 2：编写 `problem_scope.md`**

使用以下内容：

```markdown
# 问题范围

本项目研究用于单零件结构 CAE 前处理的 STEP/B-Rep 图识别。

输入是由 STEP 模型转换得到的属性邻接图，其中每个节点对应一个 B-Rep 面，每条边表示面之间的邻接关系。主要输出包括面级特征类别、特征实例、理想化标签，以及用于边界条件、载荷和网格加密的候选 CAE 区域。

本工作不同于加工特征识别，因为其下游目标是 CAE 模型准备，而不是工艺规划。当制造风格特征有助于识别承载、支撑、薄壁、孔、槽、口袋、肋、凸台和过渡区域时，本工作保留这些特征。

Version 1 的主张：

1. 面向 CAE 的 STEP/B-Rep 面识别特征 ontology。
2. 带有面级和实例级标签的合成 STEP/AAG 数据集。
3. 用于宏特征、语义子类型、理想化标签和实例分组的多任务图基线模型。
4. 从识别到的 feature objects 到 Abaqus 面区域的下游桥接。

Version 1 不声称能够从任意真实零件中完全自动推断工程意图。真实边界和载荷语义需要分析工况元数据。
```

- [ ] **步骤 3：编写 `experiment_matrix.md`**

使用以下内容：

```markdown
# 实验矩阵

## 数据集

| 数据集 | 来源 | 用途 |
| --- | --- | --- |
| Synthetic-Random | 由 `main.py` 生成的 `output/graphs` | 标准监督训练和测试 |
| Synthetic-Composition | 保留未见的特征组合 | 测试特征组合泛化能力 |
| Feature-Catalog | `output/feature_catalog` | 定性单特征验证 |
| Real-Abaqus-Small | 20 个手动选择的 STEP 零件及 Abaqus 拓扑快照 | 测试真实拓扑映射 |

## 数据划分

| 划分 | 规则 |
| --- | --- |
| Random | 当前 80/10/10 样本划分 |
| Composition | 保留特定特征对，例如 rib+hole、boss+fillet、slot+chamfer |
| Real | 不参与训练；仅用于推理和映射 |

## 基线方法

| 名称 | 实现 | 用途 |
| --- | --- | --- |
| Rule-only | 生成器标签或启发式分组 | 合理性上界/下界 |
| Node-MLP | 不使用图边 | 展示拓扑的价值 |
| GraphSAGE-no-edge-attr | 当前 `models/gnn_baseline.py` | 当前基线 |
| Edge-aware GraphSAGE | 带 `edge_attr` 的计划模型 | 主方法 |

## 指标

| 任务 | 指标 |
| --- | --- |
| 面级宏特征 | accuracy、macro-F1、per-class F1 |
| 面级语义子类型 | accuracy、macro-F1、per-class F1 |
| 实例分组 | edge-affinity F1、feature-instance IoU |
| 理想化 / CAE 标签 | micro-F1、macro-F1、exact match |
| Abaqus 映射 | mean feature IoU、exact feature ratio、mapped-face reuse |
| Action plan | boundary/load face IoU、precision、recall |

## 必要图表

1. 流程总览：STEP -> AAG -> multi-task GNN -> feature objects -> Abaqus action plan。
2. Ontology 图：primitive surface、macro feature、subtype、idealization tag、analysis role。
3. 数据集覆盖率表和特征分布。
4. 模型消融表。
5. 真实 Abaqus 映射示例。
```

- [ ] **步骤 4：更新 README 中的边界声明**

在当前边界部分下增加一小段：

```markdown
论文主张应表述为面向 CAE 的特征识别与区域推荐。Boundary/load 标签只有在来自显式合成分析工况或导入的真实 CAE 元数据时才有效；不应将其表述为仅由几何确定性推断出的工程意图。
```

- [ ] **步骤 5：验证文档**

运行：

```powershell
Get-Content 'W:\python\FEA\dataset_generator\docs\research\problem_scope.md' -Encoding UTF8
Get-Content 'W:\python\FEA\dataset_generator\docs\research\experiment_matrix.md' -Encoding UTF8
```

预期结果：两个文件均以可读的中英文 UTF-8 文本显示，没有乱码。

---

## 任务 2：为 AAG 图增加边属性

**文件：**
- 修改：`W:\python\FEA\dataset_generator\core\AAGExtractor.py`
- 修改：`W:\python\FEA\dataset_generator\utils\training_export.py`
- 修改：`W:\python\FEA\dataset_generator\schemas\gnn_record.schema.json`

- [ ] **步骤 1：在 `AAGExtractor.py` 中增加边描述符字段**

为 `edges` 中的每一条有向边创建一行对应的 `edge_attrs`。使用 `extract_aag_graph` 中已经可用的 pythonOCC 对象。

最小描述符：

```python
edge_attr = {
    "src": idx1,
    "dst": idx2,
    "shared_edge_length": length,
    "src_dst_center_distance": center_distance,
    "normal_dot_abs": normal_dot_abs,
    "same_primitive_surface": int(nodes[idx1]["primitive_surface_id"] == nodes[idx2]["primitive_surface_id"]),
}
```

使用 `BRepGProp.brepgprop.LinearProperties(edge, props)` 计算 `shared_edge_length`。使用已有节点属性中的面心和法向计算距离与法向相似度。

- [ ] **步骤 2：保持旧图兼容性**

返回：

```python
return {
    "nodes": nodes,
    "edges": edges_list,
    "edge_attrs": edge_attrs,
    "feature_sets": feature_sets,
}
```

如果旧图没有 `edge_attrs`，下游代码必须将其视为零特征边。

- [ ] **步骤 3：在 `training_export.py` 中导出边属性**

增加：

```python
EDGE_FEATURE_SCHEMA = [
    "shared_edge_length",
    "src_dst_center_distance",
    "normal_dot_abs",
    "same_primitive_surface",
]
```

在每个 GNN record 中增加：

```python
"edge_attr": [
    [round_value(attr.get(name, 0.0), ndigits) for name in EDGE_FEATURE_SCHEMA]
    for attr in graph_data.get("edge_attrs", [])
],
"edge_feature_schema": EDGE_FEATURE_SCHEMA,
```

- [ ] **步骤 4：增加 schema 字段**

在 `schemas/gnn_record.schema.json` 中增加以下 properties：

```json
"edge_attr": {
  "type": "array",
  "items": {
    "type": "array",
    "items": { "type": "number" }
  }
},
"edge_feature_schema": {
  "type": "array",
  "items": { "type": "string" }
}
```

同时在 schema 的 `required` 列表中，将 `edge_attr` 和 `edge_feature_schema` 加到 `edge_index` 之后。

- [ ] **步骤 5：重新生成一个小型 smoke 数据集**

运行：

```powershell
python main.py --output-dir output_edge_smoke --num-samples 10 --seed 20260526 --skip-training-export
```

预期结果：`output_edge_smoke\graphs\*.json` 同时包含 `edges` 和 `edge_attrs`，且二者长度相等。

- [ ] **步骤 6：导出训练包**

运行：

```powershell
python utils\training_export.py --output-dir output_edge_smoke
```

预期结果：`output_edge_smoke\training\gnn\train.jsonl` 包含 `edge_attr`。

---

## 任务 3：增加实例亲和性监督

**文件：**
- 修改：`W:\python\FEA\dataset_generator\utils\training_export.py`
- 修改：`W:\python\FEA\dataset_generator\schemas\gnn_record.schema.json`
- 修改：`W:\python\FEA\dataset_generator\models\gnn_baseline.py`
- 修改：`W:\python\FEA\dataset_generator\train_gnn_baseline.py`

- [ ] **步骤 1：导出边实例标签**

在 `build_gnn_record` 中，在收集 `instance_ids` 之后增加：

```python
instance_ids = [node.get("instance_id", 0) for node in nodes]
edge_instance_affinity = []
for src, dst in graph_data.get("edges", []):
    src_inst = int(instance_ids[src])
    dst_inst = int(instance_ids[dst])
    edge_instance_affinity.append(1 if src_inst > 0 and src_inst == dst_inst else 0)
```

将以下字段加入 record：

```python
"edge_instance_affinity": edge_instance_affinity,
```

- [ ] **步骤 2：增加 schema 字段**

在 `schemas/gnn_record.schema.json` 中增加：

```json
"edge_instance_affinity": {
  "type": "array",
  "items": { "type": "integer", "enum": [0, 1] }
}
```

并将其加入 `required`，位置在 `edge_attr` 之后。

- [ ] **步骤 3：为模型增加边亲和性预测头**

在 `MultiHeadGraphSAGE.__init__` 中增加：

```python
self.edge_affinity_head = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

在 `forward` 中，在节点嵌入计算完成后增加：

```python
if edge_index.numel() > 0:
    src, dst = edge_index[0], edge_index[1]
    edge_pair = torch.cat([x[src], x[dst]], dim=-1)
    outputs["edge_affinity_logits"] = self.edge_affinity_head(edge_pair).squeeze(-1)
else:
    outputs["edge_affinity_logits"] = x.new_zeros((0,))
```

- [ ] **步骤 4：整理 edge-affinity targets**

在 `GraphBatchCollator.__call__` 中增加：

```python
edge_instance_affinity = []
```

对每个 record：

```python
edge_instance_affinity.append(torch.tensor(record.get("edge_instance_affinity", []), dtype=torch.float32))
```

返回：

```python
"edge_instance_affinity": torch.cat(edge_instance_affinity, dim=0) if edge_instance_affinity else torch.zeros((0,), dtype=torch.float32),
```

- [ ] **步骤 5：训练边亲和性损失**

在 `run_epoch` 中读取：

```python
edge_affinity_targets = batch["edge_instance_affinity"].to(device)
```

增加损失：

```python
if edge_affinity_targets.numel() > 0:
    edge_affinity_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["edge_affinity_logits"],
        edge_affinity_targets,
    )
    loss = loss + 0.5 * edge_affinity_loss
```

- [ ] **步骤 6：报告边亲和性指标**

增加一个使用 0.5 阈值的指标函数：

```python
def compute_binary_f1(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits.cpu()) >= 0.5).to(torch.int64)
    tgt = targets.cpu().to(torch.int64)
    tp = int(((pred == 1) & (tgt == 1)).sum().item())
    fp = int(((pred == 1) & (tgt == 0)).sum().item())
    fn = int(((pred == 0) & (tgt == 1)).sum().item())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
    }
```

- [ ] **步骤 7：训练 smoke 模型**

运行：

```powershell
python train_gnn_baseline.py --training-dir output_edge_smoke\training\gnn --label-map-dir output_edge_smoke\label_maps --epochs 3 --batch-size 4 --run-name edge_affinity_smoke
```

预期结果：`output_edge_smoke\model_runs\edge_affinity_smoke\metrics.json` 包含 `edge_affinity_f1`。

---

## 任务 4：增加合成 CAE 分析工况标签

**文件：**
- 修改：`W:\python\FEA\dataset_generator\CAE_feature_ontology.yaml`
- 新建：`W:\python\FEA\dataset_generator\utils\analysis_case_labels.py`
- 修改：`W:\python\FEA\dataset_generator\main.py`
- 修改：`W:\python\FEA\dataset_generator\utils\training_export.py`
- 修改：`W:\python\FEA\dataset_generator\schemas\gnn_record.schema.json`

- [ ] **步骤 1：保持分析角色与几何标签分离**

在 `CAE_feature_ontology.yaml` 中，保持 `analysis_role` 作为独立层。使用以下 v2 active labels：

```yaml
analysis_role:
  labels:
    - key: support_candidate_face
    - key: load_candidate_face
    - key: symmetry_candidate_face
    - key: mesh_refinement_region
```

不要将这些标签合并进 `macro_feature`。

- [ ] **步骤 2：创建确定性工况标签模块**

创建 `utils\analysis_case_labels.py`，包含函数：

```python
def annotate_synthetic_analysis_roles(graph_data: dict, analysis_role_index: dict) -> dict:
    nodes = graph_data.get("nodes", [])
    key_to_id = analysis_role_index["key_to_id"]
    support_id = key_to_id["support_candidate_face"]
    load_id = key_to_id["load_candidate_face"]
    mesh_id = key_to_id["mesh_refinement_region"]

    stock_planar = [
        node for node in nodes
        if node.get("macro_feature") == "stock" and node.get("primitive_surface") == "plane"
    ]
    feature_faces = [
        node for node in nodes
        if int(node.get("macro_feature_id", 0)) > 0
    ]

    support_nodes = sorted(stock_planar, key=lambda n: n["attributes"]["physical"]["area"], reverse=True)[:1]
    load_nodes = sorted(feature_faces, key=lambda n: n["idx"])[:2]

    support_idx = {node["idx"] for node in support_nodes}
    load_idx = {node["idx"] for node in load_nodes}

    for node in nodes:
        ids = []
        keys = []
        if node["idx"] in support_idx:
            ids.append(support_id)
            keys.append("support_candidate_face")
        if node["idx"] in load_idx:
            ids.append(load_id)
            keys.append("load_candidate_face")
        if node.get("idealization_tags"):
            ids.append(mesh_id)
            keys.append("mesh_refinement_region")
        node["analysis_role_ids"] = ids
        node["analysis_roles"] = keys

    graph_data["analysis_case_summary"] = {
        "support_candidate_face_count": len(support_idx),
        "load_candidate_face_count": len(load_idx),
    }
    return graph_data
```

- [ ] **步骤 3：接入 `main.py`**

使用以下方式加载 `analysis_role` 层：

```python
analysis_role_index = load_layer_index(macro_ontology_path, "analysis_role")
```

在 `annotate_derived_cae_labels` 之后调用：

```python
annotate_synthetic_analysis_roles(graph_data, analysis_role_index)
```

- [ ] **步骤 4：导出分析角色**

在 GNN records 中增加：

```python
"analysis_role_ids_per_node": [
    node.get("analysis_role_ids", []) for node in nodes
],
"analysis_case_summary": graph_data.get("analysis_case_summary", {}),
```

- [ ] **步骤 5：验证生成标签**

运行：

```powershell
python main.py --output-dir output_analysis_smoke --num-samples 20 --seed 20260527
```

预期结果：每个 graph 都有 `analysis_role_ids` 和 `analysis_case_summary`；没有 graph 将 load/support 表述为纯几何真值。

---

## 任务 5：构建真实 Abaqus 验证集

**文件：**
- 新建：`W:\python\FEA\dataset_generator\docs\research\real_validation_set.md`
- 使用现有文件：`W:\python\FEA\dataset_generator\real_abaqus_minimal_pipeline.py`
- 使用现有文件：`W:\python\FEA\dataset_generator\mapping_eval.py`
- 使用外部文件：`W:\python\FEA\cae\step1_topology.py`

- [ ] **步骤 1：选择 20 个真实或近真实 STEP 零件**

创建 `docs\research\real_validation_set.md`，包含如下表格：

```markdown
# 真实 Abaqus 验证集

| ID | STEP path | Feature types expected | Topology JSON path | Notes |
| --- | --- | --- | --- | --- |
| R001 |  | hole, fillet, chamfer |  | simple sanity case |
| R002 |  | slot, pocket |  |  |
| R003 |  | rib, pad |  |  |
```

在运行最终实验前补齐全部 20 行。

- [ ] **步骤 2：导出 Abaqus 拓扑**

对每个 STEP 文件，使用 Abaqus 运行：

```powershell
abaqus cae noGUI=W:\python\FEA\cae\step1_topology.py -- --step-path <part.step> --output-path <topology_data.json>
```

预期结果：每个 STEP 文件对应一个 topology JSON，包含 face id、type、area、center，以及可用的 normal/axis/radius。

- [ ] **步骤 3：运行最小真实流程**

对每一对 STEP/topology：

```powershell
python real_abaqus_minimal_pipeline.py --step-path <part.step> --topology-path <topology_data.json> --checkpoint output\model_runs\<formal_run>\best.pt --output-root output\real_validation\<part_id> --part-name <part_name> --instance-name <part_name>-1 --global-size 2.0 --local-size 0.5 --overwrite
```

预期结果：输出文件夹包含 `inference_graphs`、`predictions`、`feature_objects`、`abaqus_mapping`、`action_plans` 和 `abaqus_scripts`。

- [ ] **步骤 4：评估映射结构**

运行：

```powershell
python mapping_eval.py --mapping-path output\real_validation --output-path output\real_validation\mapping_eval_report.json --recursive
```

预期结果：报告包含样本数、已映射面数、复用面数和平均映射分数。

- [ ] **步骤 5：人工检查 5 个案例**

打开生成的 Abaqus 脚本并确认：

```text
1. fixed/support faces are plausible,
2. load faces are not empty,
3. local mesh refinement targets recognized feature regions,
4. no face id is reused across incompatible feature objects.
```

将失败情况记录在 `docs\research\real_validation_set.md` 中。

---

## 任务 6：正式训练和论文可用报告

**文件：**
- 新建：`W:\python\FEA\dataset_generator\utils\experiment_report.py`
- 修改：`W:\python\FEA\dataset_generator\train_gnn_baseline.py`
- 读取：`W:\python\FEA\dataset_generator\output\reports\dataset_report.json`
- 读取：`W:\python\FEA\dataset_generator\output\reports\coverage_report.json`
- 读取：`W:\python\FEA\dataset_generator\output\model_runs\*\metrics.json`

- [ ] **步骤 1：生成正式数据集**

运行：

```powershell
python main.py --output-dir output_formal_v1 --num-samples 3000 --seed 20260528
```

预期结果：coverage report 的 status 为 `satisfied`，且没有宏特征低于配置阈值。

- [ ] **步骤 2：训练当前基线**

运行：

```powershell
python train_gnn_baseline.py --training-dir output_formal_v1\training\gnn --label-map-dir output_formal_v1\label_maps --output-dir output_formal_v1\model_runs --run-name graphsage_no_edge_attr --epochs 50 --batch-size 16
```

预期结果：`metrics.json` 包含 macro、semantic、derived 和 idealization heads 的最终 test metrics。

- [ ] **步骤 3：训练边感知模型**

运行：

```powershell
python train_gnn_baseline.py --training-dir output_formal_v1\training\gnn --label-map-dir output_formal_v1\label_maps --output-dir output_formal_v1\model_runs --run-name edge_aware_graphsage --epochs 50 --batch-size 16
```

预期结果：`edge_aware_graphsage` 优于 `graphsage_no_edge_attr`，或者能够清楚诊断其失败原因。

- [ ] **步骤 4：创建实验报告脚本**

创建 `utils\experiment_report.py`，读取 JSON 报告并写出：

```text
output_formal_v1\reports\paper_dataset_table.csv
output_formal_v1\reports\paper_model_table.csv
output_formal_v1\reports\paper_mapping_table.csv
```

最小列：

```text
paper_dataset_table.csv:
total_samples,node_count,feature_set_count,macro_feature,min_instance_count,max_instance_count

paper_model_table.csv:
run_name,macro_accuracy,macro_macro_f1_active,semantic_accuracy,semantic_macro_f1_active,derived_micro_f1,idealization_micro_f1,edge_affinity_f1

paper_mapping_table.csv:
sample_count,mean_mapping_score,mean_feature_iou,mean_bc_iou,mean_load_iou
```

- [ ] **步骤 5：运行报告脚本**

运行：

```powershell
python utils\experiment_report.py --output-dir output_formal_v1
```

预期结果：三个 CSV 文件存在，并可直接粘贴到论文表格中。

---

## 推荐执行顺序

1. 任务 1：固化研究范围和实验矩阵。
2. 任务 2：增加边属性。
3. 任务 3：增加实例亲和性监督。
4. 任务 6 的 smoke 版本：在 500-1000 个样本上训练，验证代码正确性。
5. 任务 4：增加合成 CAE 分析工况标签。
6. 任务 5：构建真实 Abaqus 验证集。
7. 任务 6 的正式版本：生成 3000+ 样本，训练正式基线，导出论文表格。

## 完成定义

下一阶段完成的标志是：

1. 数据集包含边属性、面标签、实例亲和性标签、理想化标签和分析角色标签。
2. 模型报告 macro-F1、semantic-F1、CAE-tag F1 和 instance-affinity F1。
3. 至少 20 个真实 STEP/Abaqus 案例能够走通真实流程。
4. `docs\research\experiment_matrix.md` 能将每个论文主张映射到一个指标或定性图。
5. `utils\experiment_report.py` 能从生成输出中产生论文可用 CSV 表格。

## 自检

- 规格覆盖：本计划覆盖了研究定位、数据集表示、模型监督、下游 Abaqus 验证和论文证据。
- 占位符检查：没有残留 `TBD`、泛泛 TODO，或未具体说明的“处理边界情况”。
- 类型一致性：新字段命名保持一致，分别为 `edge_attr`、`edge_feature_schema`、`edge_instance_affinity`、`analysis_role_ids_per_node` 和 `analysis_case_summary`。
