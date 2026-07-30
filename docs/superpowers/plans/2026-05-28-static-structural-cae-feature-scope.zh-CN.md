# 单零件结构静力 CAE 特征范围实施计划

> **给后续 agentic worker 的说明：** 若继续执行该计划，应按任务逐项推进，并在每个阶段后运行对应验证命令。当前仓库不是 git 仓库，因此本计划不要求提交 commit。

**目标：** 将数据集生成器收敛到 V1 单零件结构静力 CAE 特征范围，同时保留旧生成器模块，方便后续 V2 复用。

**架构：** 本轮改动以配置和本体为核心：保留现有 feature generator 类，但让默认 V1 路径只使用局部、可查询、可映射到 Abaqus 选面的特征。将 `pad` 规范化为 `rectangular_boss`，将长圆孔 `obround_hole` 保持在孔类中，并将孔组、法兰、支撑垫等从默认训练范围移出。

**技术栈：** Python、YAML ontology、现有 OCC 几何生成器、轻量 Python 回归测试、`feagpt` 虚拟环境。

**当前状态：** 任务 1-8 已在当前工作区完成并验证。`W:\python\envs\feagpt` 环境已能导入 `OCC`、`PyYAML`、`scikit-learn` 和 `torch`；V1 smoke 数据集生成与数据集校验均已通过。

---

### 任务 1：冻结 V1 研究范围文档

**文件：**

- 新建：`docs/superpowers/specs/2026-05-28-static-structural-cae-feature-scope-design.md`
- 新建：`docs/superpowers/specs/2026-05-28-static-structural-cae-feature-scope-design.zh-CN.md`
- 新建：`docs/superpowers/plans/2026-05-28-static-structural-cae-feature-scope.md`
- 新建：`docs/superpowers/plans/2026-05-28-static-structural-cae-feature-scope.zh-CN.md`

- [x] **步骤 1：写入 V1 设计说明**

记录已确认范围：

```text
single-part static structural preprocessing
local queryable features only
obround_hole remains a hole feature
hole groups are derived after single-hole recognition
flange/support_pad/load_pad are not V1 macro features
```

中文表述为：

```text
单零件结构静力前处理
仅保留局部、可查询、可选面的特征
长圆孔保留为孔类特征
孔组由单孔识别后的后处理聚类得到
法兰/支撑垫/加载垫不作为 V1 宏特征
```

- [x] **步骤 2：写入实施计划**

使用本计划文件记录第一轮实施任务和验证命令，保证后续修改可审查、可复现。

### 任务 2：规范化宏特征 ontology

**文件：**

- 修改：`CAE_feature_ontology.yaml`

- [x] **步骤 1：保留 V1 核心特征为 enabled**

V1 启用的宏特征为：

```text
stock
through_hole_round
blind_hole_round
stepped_hole_round
counterbore_hole
countersink_hole
obround_hole
slot_u
slot_rect
pocket_rect
pocket_circular
cylinder_boss
rectangular_boss
rib
web
fillet
chamfer
```

- [x] **步骤 2：禁用默认 coverage 中的非核心标签**

将以下宏特征设为 `enabled: false`：

```text
triangular_cutout
rectangular_cutout
notch_relief
cone_boss
triangle_prism_boss
rectangle_prism_boss
triangle_frustum_boss
rectangle_frustum_boss
flange
gusset
bolt_hole_pattern
bearing_seat
support_pad
```

- [x] **步骤 3：规范化 pad**

将原 id `24` 的 `pad` 规范化为：

```yaml
key: rectangular_boss
current_code_aliases:
  - pad
  - rectangle_boss
  - rectangular_boss
```

这样旧的 `PadGenerator` 不需要删除，但其宏特征会解析到 `rectangular_boss`。

### 任务 3：对齐 face semantic subtype

**文件：**

- 修改：`face_semantic_subtype_ontology.yaml`

- [x] **步骤 1：将 pad 子面归到 rectangular_boss**

将：

```yaml
parent_macro_feature: pad
```

改为：

```yaml
parent_macro_feature: rectangular_boss
```

影响的子面：

```text
pad_side_plane
pad_top_plane
```

- [x] **步骤 2：收窄 recommended_v1_label_subset**

保留孔、槽、口袋、圆柱凸台、矩形凸台、肋/腹板、圆角、倒角相关子面。移除法兰、支撑垫、孔组、轴承座、加强筋、缺口、锥形凸台、多边形切口、多边形/棱台凸台等非核心子面。

### 任务 4：收窄默认生成配置

**文件：**

- 修改：`main.py`
- 修改：`configs/default_dataset_config.yaml`
- 新建：`configs/static_structural_v1_dataset.yaml`
- 新建：`configs/static_structural_v1_smoke.yaml`

- [x] **步骤 1：添加 V1 生成器列表**

在 `main.py` 中定义：

```python
STATIC_STRUCTURAL_V1_GENERATORS = [
    "SteppedHoleGenerator",
    "ThroughHoleGenerator",
    "BlindHoleGenerator",
    "CounterboreHoleGenerator",
    "CountersinkHoleGenerator",
    "ObroundHoleGenerator",
    "USlotGenerator",
    "SlotRectGenerator",
    "RectPocketGenerator",
    "CircularPocketGenerator",
    "BossFeatureGenerator",
    "PadGenerator",
    "RibGenerator",
    "WebGenerator",
    "FilletGenerator",
    "ChamferGenerator",
]
```

- [x] **步骤 2：使用 V1 列表作为默认 enabled generators**

将 `DEFAULT_CONFIG["generation"]["enabled_generators"]` 改为使用 `STATIC_STRUCTURAL_V1_GENERATORS`。

- [x] **步骤 3：更新 feature catalog targets**

移除非核心 catalog 项，并将：

```text
rectangular_boss -> PadGenerator
```

- [x] **步骤 4：添加命名配置**

新增：

```text
configs/static_structural_v1_dataset.yaml
configs/static_structural_v1_smoke.yaml
```

其中 `static_structural_v1_dataset.yaml` 用于正式 V1 数据集生成，`static_structural_v1_smoke.yaml` 用于快速环境和流程验证。

### 任务 5：稳定 V1 凸台生成

**文件：**

- 修改：`features/boss_feature.py`

- [x] **步骤 1：限制默认凸台类型**

加入：

```python
self.allowed_boss_types = ["cylinder"]
```

并使用：

```python
boss_type = random.choice(self.allowed_boss_types)
```

这样 `cone_boss` 可以作为兼容旧本体的 V2 备用项保留，但不会进入 V1 默认数据。

### 任务 6：更新派生 CAE 标签

**文件：**

- 修改：`utils/derived_labels.py`

- [x] **步骤 1：移除非核心薄壁语义键**

保留：

```python
THIN_WALL_SEMANTIC_KEYS = {"rib_side_plane", "web_side_plane"}
```

- [x] **步骤 2：使用 rectangular_boss 作为厚度过渡相关宏特征**

保留 `pad_side_plane` 和 `pad_top_plane`，因为它们现在表示 `rectangular_boss` 的子面，并设置：

```python
THICKNESS_TRANSITION_MACROS = {"rectangular_boss"}
```

### 任务 7：添加回归检查

**文件：**

- 新建：`tests/__init__.py`
- 新建：`tests/test_static_structural_v1_scope.py`

- [x] **步骤 1：检查 ontology 和默认配置范围**

测试内容包括：

```python
assert "obround_hole" in enabled macro keys
assert "rectangular_boss" in enabled macro keys
assert "bolt_hole_pattern" not in enabled macro keys
assert "flange" not in default enabled generators
assert "PadGenerator" in default enabled generators
```

- [x] **步骤 2：运行回归检查**

运行：

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_static_structural_v1_scope -v
```

已验证结果：

```text
Ran 3 tests
OK
```

### 任务 8：Smoke 验证数据集基础设施

**文件：**

- 新建：`configs/static_structural_v1_smoke.yaml`

- [x] **步骤 1：验证环境导入**

运行：

```powershell
W:\python\envs\feagpt\python.exe -c "import OCC, yaml, sklearn, torch, numpy; print('OCC ok')"
```

已验证：

```text
OCC ok
yaml 6.0.3
sklearn 1.8.0
torch 2.10.0
numpy 2.4.6
cuda available True
```

- [x] **步骤 2：生成最小 smoke 数据集**

运行：

```powershell
W:\python\envs\feagpt\python.exe main.py --config configs/static_structural_v1_smoke.yaml --output-dir output_smoke_static_v1 --skip-training-export
```

已验证生成：

```text
生成样本完成: features=5 -> output_smoke_static_v1\graphs\composite_feature_000.json
生成样本完成: features=7 -> output_smoke_static_v1\graphs\composite_feature_001.json
```

- [x] **步骤 3：校验 smoke 数据集**

运行：

```powershell
W:\python\envs\feagpt\python.exe utils\verify_dataset.py `
  --graphs-dir output_smoke_static_v1\graphs `
  --dataset-meta output_smoke_static_v1\dataset_meta.json `
  --primitive-label-map output_smoke_static_v1\label_maps\primitive_surface_label_map.json `
  --label-map output_smoke_static_v1\label_maps\macro_feature_label_map.json `
  --idealization-label-map output_smoke_static_v1\label_maps\idealization_tag_label_map.json
```

已验证：

```text
checked_files: 2
checked_nodes: 51
checked_feature_sets: 12
verification_status: OK
```

### 后续任务建议

下一阶段不应继续扩充宏特征类别，而应推进 feature object 到 selectable region 的闭环：

1. 为每个 feature object 提取可选子区域，例如孔壁、孔底、凸台顶面、口袋底面、肋侧面。
2. 为 feature object 增加几何参数，例如直径、深度、高度、厚度、轴线、中心、法向。
3. 增加结构化仿真需求查询，例如“固定直径为 5 的盲孔圆柱面”。
4. 将查询结果映射到 Abaqus set/surface/action plan。
5. 增加指标：query success rate、selected face IoU、Abaqus set creation success rate。
