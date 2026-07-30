# v5.6-A Real-Part Sanity And Candidate Topology Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不急于重新训练的前提下，先确认真实件评估是否可信，再用候选对象级拓扑修复处理 s3 中暴露出的孔、槽、rib、transition 边界错误。

**Architecture:** 保持当前主流程不变：STEP -> invariant B-Rep graph -> edge-aware GNN -> candidate classifier -> topology postprocess -> feature objects -> area-truth review。v5.6-A 增加一个真实件评估 sanity checker，并在候选对象后处理层增加可回归测试的拓扑边界修复；只有当评估可信且规则层稳定后，才进入 v5.6-B 数据补强和重训。

**Tech Stack:** Python, pythonOCC/OCC-derived B-Rep graph JSON, PyTorch edge-aware GNN, scikit-learn candidate classifier, JSON/Markdown reports, unittest.

---

## 背景与问题

当前 s3 真实件验证使用：

- STEP 输入：`W:\python\FEA\dataset_generator\test_part_s3\part1.stp` 到 `part21.stp`
- 面积真值：`W:\python\FEA\dataset_generator\configs\test_part_s3_area_truth_part20_mm2.json`
- 预测对象：`W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\feature_objects\edge_aware_v5_5_expanded_v51_compat_candidate_topology_post`
- 当前报告：`W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\reports\s3_area_truth_validation_v5_5_expanded_v51_compat_part20mm2_mergeeval.md`

part20 单位从 `in^2` 修正为 `mm^2` 后，整体仍为：

```text
Truth objects = 204
Predicted objects = 186
Matched truth = 108
Recall = 0.5294
Precision = 0.5806
```

这说明问题不是单一的“模型太弱”。当前至少混合了三类问题：

1. 评估可信度问题：真值面积单位、面积合并口径、重复/连通槽合并口径、特征命名别名都可能影响指标。
2. 候选对象边界问题：模型可能局部面预测正确，但 feature object 组合错、拆分错或被规则层误 demote。
3. 分布偏移问题：真实件的上下文、尺度、孔密度、rib 与 pocket 的组合关系仍超过训练数据覆盖。

v5.6-A 只解决前两类，为后续补数据和重训提供干净基线。

---

## 文件结构

- Create: `W:\python\FEA\dataset_generator\generate_real_part_sanity_report.py`
  - 读取 graph、面积真值、feature objects，检查单位/面积/对象口径是否可信。
- Create: `W:\python\FEA\dataset_generator\tests\test_real_part_sanity_report.py`
  - 覆盖 sanity checker 的核心判断：可匹配面积、疑似单位错误、缺失 graph。
- Modify later: `W:\python\FEA\dataset_generator\utils\topology_postprocess.py`
  - 增加候选级拓扑修复规则，不改 GNN 主干。
- Modify later: `W:\python\FEA\dataset_generator\tests\test_topology_postprocess_v5_6.py`
  - 为 s3 暴露出的 hard cases 写最小回归测试。
- Output: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\reports\s3_real_part_sanity_part20mm2.md`
  - v5.6-A 第一阶段真实件评估 sanity 报告。
- Output later: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_6_a_sanity_topology_repair\`
  - v5.6-A 修复后推理、feature object、面积真值报告和 failure atlas。

---

### Task 1: 真实件评估 Sanity Checker

**Files:**

- Create: `W:\python\FEA\dataset_generator\generate_real_part_sanity_report.py`
- Create: `W:\python\FEA\dataset_generator\tests\test_real_part_sanity_report.py`

- [x] **Step 1: 写失败测试**

在 `tests\test_real_part_sanity_report.py` 中测试三个行为：

```python
def test_truth_object_with_matching_feature_object_is_plausible():
    truth = {"sample": "part1", "macro": "fillet", "area": 100.0, "face_count": 1}
    feature_objects = [{"macro": "fillet", "area": 100.2, "face_indices": [3]}]
    result = evaluate_truth_plausibility(truth, [90.0, 100.1], feature_objects)
    assert result["status"] == "plausible"

def test_truth_area_without_nearby_graph_or_object_area_is_suspicious():
    truth = {"sample": "part20", "macro": "counterbore_hole", "area": 3.0, "face_count": 3}
    result = evaluate_truth_plausibility(truth, [1900.0, 2450.0], [])
    assert result["status"] == "suspicious"

def test_missing_graph_is_reported():
    summary = build_sanity_report({"part1": []}, {}, {})
    assert summary["samples"]["part1"]["flags"] == ["missing_graph"]
```

- [x] **Step 2: 运行测试，确认失败**

Run:

```powershell
W:\python\envs\feagpt\python.exe -m unittest tests.test_real_part_sanity_report -v
```

Expected before implementation: import 或函数未定义导致失败。

- [x] **Step 3: 实现 checker**

`generate_real_part_sanity_report.py` 需要提供：

```python
def evaluate_truth_plausibility(truth_object, face_areas, feature_objects, rel_tol=0.02, abs_tol=5.0):
    ...

def build_sanity_report(truth_by_sample, graph_by_sample, feature_objects_by_sample):
    ...
```

核心判断：

```text
1. 真值对象面积是否能在同 macro + 同 face_count 的预测对象中找到近似匹配。
2. 单面真值是否能在 graph face area 中找到近似匹配。
3. 如果大量真值面积都比 graph/feature object 小约 645.16 倍，则提示 in^2/mm^2 单位风险。
4. 如果 graph 缺失、feature object 缺失、truth 缺失，需要显式打 flag。
5. 输出 markdown/json，列出每个 part 的 suspicious_truth_count、plausible_truth_count、possible_unit_ratio。
```

- [x] **Step 4: 跑 s3 修正单位后的 sanity 报告**

Run:

```powershell
W:\python\envs\feagpt\python.exe generate_real_part_sanity_report.py --graph-dir output_real_step_s3_v5_5_looprole\graphs --truth-json configs\test_part_s3_area_truth_part20_mm2.json --feature-objects-path output_real_step_s3_v5_5_looprole\feature_objects\edge_aware_v5_5_expanded_v51_compat_candidate_topology_post --output-json output_real_step_s3_v5_5_looprole\reports\s3_real_part_sanity_part20mm2.json --output-md output_real_step_s3_v5_5_looprole\reports\s3_real_part_sanity_part20mm2.md
```

Expected: 生成每个 part 的评估可信度摘要，并指出需要人工优先检查的 part。

---

### Task 2: s3 Failure Atlas 分类

**Files:**

- Read: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\reports\failure_atlas_all_v5_5_expanded_v51_compat_part20mm2.md`
- Read: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\reports\s3_real_part_sanity_part20mm2.md`
- Create: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_5_looprole\reports\v5_6_a_error_taxonomy.md`

- [x] **Step 1: 把错误分成四类**

分类标准：

```text
A. Truth/evaluation issue: 面积单位、真值口径、同一连通槽是否拆分等。
B. Candidate boundary issue: 面预测局部正确，但 feature object 合并/拆分错误。
C. Topology role issue: 内孔壁/外圆柱、counterbore/stepped/blind、fillet/slot_o 角色混淆。
D. Distribution issue: 真实件结构超出训练数据，规则层不应硬修。
```

- [x] **Step 2: 输出每个 part 的主导错误类型**

Markdown 表格字段：

```text
part | dominant issue | affected macros | evidence | proposed action
```

- [x] **Step 3: 只把 B/C 类推进 topology repair**

如果某个 part 被 sanity checker 标记为 A 类，先不把它当作模型错误修规则。

---

### Task 3: 候选对象级 Topology Repair 回归测试

**Files:**

- Modify: `W:\python\FEA\dataset_generator\tests\test_topology_postprocess_v5_6.py`
- Modify later: `W:\python\FEA\dataset_generator\utils\topology_postprocess.py`

- [ ] **Step 1: rib/stock 大面误检测试**

测试一个候选对象包含大 stock 基面时，若候选总面积占模型总面积过大且缺少薄壁厚度证据，必须 demote 为 `stock`。

Expected:

```python
assert repaired_macro == "stock"
```

- [ ] **Step 2: slot_o/fillet 角色测试**

测试一个圆柱/圆弧面若连接两个近似平行的槽侧面，应偏向 `slot_o`；若连接两个相交主面，应偏向 `fillet`。

Expected:

```python
assert slot_face_macro == "slot_o"
assert blend_face_macro == "fillet"
```

- [ ] **Step 3: slot_u/obround_hole 测试**

测试 U 形槽底圆弧面 + 两个侧面通过凹边连接时，不能被吸收到 `obround_hole`。

Expected:

```python
assert repaired_macro == "slot_u"
assert sorted(face_indices) == [side_a, bottom_arc, side_b]
```

- [ ] **Step 4: cylinder_boss/blind_hole 测试**

测试圆柱面 + 底面如果是凹边内孔上下文，应修为 `blind_hole_round`；如果是凸边外凸台上下文，应保留 `cylinder_boss`。

Expected:

```python
assert inner_candidate_macro == "blind_hole_round"
assert outer_candidate_macro == "cylinder_boss"
```

---

### Task 4: 实现候选对象修复规则

**Files:**

- Modify: `W:\python\FEA\dataset_generator\utils\topology_postprocess.py`

- [ ] **Step 1: 增加对象级几何上下文摘要**

为每个 candidate/component 计算：

```text
component_area
component_area / model_total_area
primitive_surface histogram
concave/convex adjacent edge counts
inner_loop/outer_loop edge-role counts
neighbor major-plane count
thin-wall evidence
slot-like long-direction evidence
```

- [ ] **Step 2: 修 rib/stock**

规则：

```text
如果 rib candidate 的面积占模型总面积过大，且没有薄厚方向证据，则 demote 为 stock。
如果 pocket 内细长薄板被拆成多个 rib 面，且这些面共享薄壁厚度和相邻 cavity wall，则合并为 rib。
```

- [ ] **Step 3: 修 slot_o/fillet**

规则：

```text
slot_o: 圆弧/圆柱面与两个近似平行的槽侧面相连，且连接边多为凹边。
fillet: 圆弧面连接两个相交主面，常为 transition，面积/宽度较小。
```

- [ ] **Step 4: 修 slot_u/obround_hole**

规则：

```text
slot_u: 底部圆弧面 + 两个平面侧壁，侧壁到底面为凹边，轨迹方向长度显著大于截面尺度。
obround_hole: 跑道状贯通开口，侧壁围成封闭孔边界，通常不具有槽底。
```

- [ ] **Step 5: 修 cylinder_boss/blind_hole**

规则：

```text
blind_hole_round: 内孔圆柱面 + 底面，圆柱壁连接入口面为内环角色，底部为凹边。
cylinder_boss: 外凸圆柱面 + 顶面，圆柱壁连接基体为外凸角色，顶部为凸边。
```

---

### Task 5: v5.6-A 真实件回归验证

**Files:**

- Create output dir: `W:\python\FEA\dataset_generator\output_real_step_s3_v5_6_a_sanity_topology_repair`

- [ ] **Step 1: 重新跑 s3 推理**

Run:

```powershell
W:\python\envs\feagpt\python.exe run_gnn_inference.py --checkpoint output_generation_v5_5_v51dist_looprole_5400\model_runs\edge_aware_v5_5_expanded_v51_compat_24dim\best.pt --input-path output_real_step_s3_v5_5_looprole\graphs --output-dir output_real_step_s3_v5_6_a_sanity_topology_repair\predictions\edge_aware_v5_6_a_candidate_topology_post --recursive --device cpu --candidate-classifier output_generation_v5_1_candidate_topology_dataset\model_runs\random_forest_v5_1_candidate_topology_hardw8\model.joblib
```

- [ ] **Step 2: 转 feature objects**

Run:

```powershell
W:\python\envs\feagpt\python.exe prediction_to_feature_objects.py --prediction-path output_real_step_s3_v5_6_a_sanity_topology_repair\predictions\edge_aware_v5_6_a_candidate_topology_post --graph-dir output_real_step_s3_v5_5_looprole\graphs --output-dir output_real_step_s3_v5_6_a_sanity_topology_repair\feature_objects\edge_aware_v5_6_a_candidate_topology_post --recursive
```

- [ ] **Step 3: 生成面积真值报告**

Run:

```powershell
W:\python\envs\feagpt\python.exe generate_area_truth_review.py --feature-objects-path output_real_step_s3_v5_6_a_sanity_topology_repair\feature_objects\edge_aware_v5_6_a_candidate_topology_post --truth-json configs\test_part_s3_area_truth_part20_mm2.json --output-json output_real_step_s3_v5_6_a_sanity_topology_repair\reports\s3_area_truth_validation_v5_6_a_mergeeval.json --output-md output_real_step_s3_v5_6_a_sanity_topology_repair\reports\s3_area_truth_validation_v5_6_a_mergeeval.md
```

- [ ] **Step 4: 生成 failure atlas**

Run:

```powershell
W:\python\envs\feagpt\python.exe generate_failure_atlas.py --graph-dir output_real_step_s3_v5_5_looprole\graphs --truth-json configs\test_part_s3_area_truth_part20_mm2.json --prediction-stage candidate_topology_post=output_real_step_s3_v5_6_a_sanity_topology_repair\predictions\edge_aware_v5_6_a_candidate_topology_post --feature-object-stage candidate_topology_post=output_real_step_s3_v5_6_a_sanity_topology_repair\feature_objects\edge_aware_v5_6_a_candidate_topology_post --output-json output_real_step_s3_v5_6_a_sanity_topology_repair\reports\failure_atlas_all_v5_6_a.json --output-md output_real_step_s3_v5_6_a_sanity_topology_repair\reports\failure_atlas_all_v5_6_a.md
```

- [ ] **Step 5: 验收标准**

v5.6-A 不要求“一步到论文最终效果”，但必须满足：

```text
1. s2 不回退：保持 v5.5 在 s2 上的高分结果，特别是已修好的 counterbore/countersink/slot/rib。
2. s3 有明确改善：Recall 或 Precision 至少有一项提升 0.08 以上，且不能靠删除预测对象虚假提高 Precision。
3. sanity checker 明确指出哪些 part 是评估口径问题，哪些是模型/拓扑问题。
4. 每条新增 topology repair 都有 unittest 覆盖。
```

---

## 用户与 Codex 分工

### Codex 负责

- 写 `generate_real_part_sanity_report.py`、测试、报告。
- 根据 sanity report 和 failure atlas 归纳错误类型。
- 写候选级 topology repair 规则和回归测试。
- 重新跑 s2/s3 验证，生成 Markdown/JSON 报告。
- 判断哪些错误应进入 v5.6-B 数据补强，哪些属于真实件真值口径问题。

### 用户负责

- 对 sanity checker 标出的少数高风险 part 做人工确认，不需要重新标注全部真实件。
- 当同一个连通几何既可拆分也可合并时，确认论文口径，例如连通 O 槽是否作为一个对象输出。
- 对 v5.6-A 仍无法修复的高频错误，确认其是否属于第一篇论文目标类别。

---

## v5.6-A 之后

如果 v5.6-A 证明主要错误来自数据覆盖不足，则进入 v5.6-B：

```text
1. 按错误类型生成 focused hard-case 数据。
2. 用 v5.5 兼容 checkpoint warm-start，而不是从零训练。
3. 训练 edge-aware GNN + candidate classifier。
4. 在 s2、s3 和新增真实件上做三套验证。
```

如果 v5.6-A 证明主要错误来自对象边界和评估口径，则优先完善 candidate-level topology-aware classifier，而不是继续扩大合成数据。

