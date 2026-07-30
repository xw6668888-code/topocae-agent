# v5.4 Problem Consolidation And Implementation Plan / v5.4 问题规整与实施计划

## 中文版

### 1. 当前阶段目标

当前阶段的目标不是单纯提高生成数据集上的指标，而是让模型在简化真实 STEP 零件上稳定识别 CAE 前处理所需的特征区域。当前完整流程为：

```text
STEP/STP
-> B-Rep 图提取
-> edge-aware GNN 面级预测
-> candidate proposal / candidate classifier
-> topology postprocess
-> feature objects
-> 面积真值评估与 failure atlas
```

当前采用的主报告为：

- `output_real_step_s2_v5_3_rulefix/reports/s2_area_truth_validation_v5_3_rulefix_mergeeval.md`
- `output_real_step_s2_v5_3_rulefix/reports/remaining_true_error_analysis_v5_3_mergeeval.md`

已确认：连通的 `slot_o` 可以作为一个 CAE 可选择区域输出，因此评估规则已经允许一个预测 `slot_o` 匹配多个连通的真值 `slot_o`。后续不再为了满足人工真值表而强行拆分连通槽。

### 2. 当前可见问题

#### 问题 A：圆柱面 / 圆锥面的功能角色混淆

表现：

- 外部大圆柱面被误识别为 `through_hole_round` 或 `slot_o`。
- 外部大圆柱面加倒角锥面被误修复为 `countersink_hole`。
- 真正的小孔圆柱面有时被保留为 `stock`。

典型样本：

- `part1`, `part2`, `part3`, `part7`, `part8`

本质：

图中已经有 `cylinder` / `cone` / 邻接边 / 面积 / 半径等信息，但当前规则和模型没有稳定区分：

- 外部 stock 圆柱面；
- 孔内壁；
- O 槽底面；
- 倒圆角圆柱面；
- countersink 的圆锥面；
- 普通 chamfer 的圆锥面。

#### 问题 B：通孔实例召回不完整

表现：

- 同一模型里一部分小孔识别成功，另一部分同尺寸小孔仍是 `stock`。
- 小圆柱 transition 有时被误识别为 `through_hole_round`。

典型样本：

- `part4`, `part7`, `part8`, `part10`

本质：

通孔不应只依赖单个圆柱面的面级标签，而应检查它是否具有“贯通开口”的拓扑证据。

#### 问题 C：rib / web 的对象边界不稳定

表现：

- rib 漏检；
- rib 吸收了孔壁圆柱面；
- rib、web、pocket_rect 在复杂局部区域互相混淆。

典型样本：

- `part3`, `part4`, `part10`

本质：

rib 的定义应以薄壁平面组为主，transition 可以被吸收，但孔壁圆柱面不应被吸收为 rib。

#### 问题 D：slot / pocket / transition 边界仍不稳

表现：

- `slot_rect` 面数过多或面积过大；
- `pocket_rect` 被过度合并成很大的平面组；
- `slot_v` 被 chamfer-like 面吞掉。

典型样本：

- `part2`, `part10`, `part11`

本质：

槽类识别不仅需要面级类别，还需要对象级尺度、凹边比例、长宽比、开口边界和邻域约束。

### 3. 分阶段实施计划

#### v5.4-A：圆柱 / 圆锥功能角色约束修复

目标：

- 降低 `countersink_hole`、`slot_o`、`through_hole_round` 的误修复。
- 防止外部大圆柱面或外部大锥面被当作孔 / 槽。
- 让真正有贯通开口证据的小孔更容易恢复为 `through_hole_round`。

Codex 负责：

- 在 `utils/topology_postprocess.py` 中增加候选级角色约束。
- 为以下行为写回归测试：
  - 大外圆柱 + chamfer cone 不应修成 `countersink_hole`；
  - 大外圆柱面不应修成 `slot_o`；
  - 有两条圆形开口边的 stock cylinder 可以恢复为 `through_hole_round`；
  - 没有开口证据的小圆柱不应被强行提升为 through hole。
- 运行 `test_part_s2` 推理或至少重新生成 feature-object 评估，检查 `part1/part2/part3/part7/part8`。

用户负责：

- 重点抽查 `part1/part2/part7` 中被判断为 chamfer/countersink 的锥面是否符合工程语义。
- 对于外部圆柱、孔内壁、槽底这类视觉上有歧义的区域，必要时给出人工确认。

完成标准：

- `part1/part2/part7` 的大圆柱 + 倒角锥面不再误报为 `countersink_hole`。
- `part3` 的巨大外圆柱面不再误报为 `slot_o`。
- `part8` 的真实 through-hole cylinder 不再被保留为 `stock`，前提是图中确实有贯通开口证据。

#### v5.4-B：rib / web 对象边界修复

目标：

- rib 候选以平面薄壁组为主体；
- rib 可以吸收合理 transition，但不能吸收 through-hole 圆柱面；
- rib-in-pocket、rib-with-hole 等真实场景更稳定。

Codex 负责：

- 增加 rib plane-dominance 检查。
- 修改 rib transition absorption 规则。
- 构造 rib-with-hole 和 rib-in-pocket 的 focused hard cases。
- 重新训练 candidate-level classifier。

用户负责：

- 确认 rib 带倒角 / 倒圆角时，transition 是否继续吸收到 rib 中。
- 抽查少量生成样本，判断 rib 形态是否符合 CAE 前处理语义。

完成标准：

- `part3/part4/part10` 的 rib 漏检和过度吸收明显减少。

#### v5.4-C：slot / pocket 对象边界修复

目标：

- 避免巨大 `pocket_rect` 误合并；
- 改善 `slot_v` 与 chamfer-like 平面的混淆；
- 稳定 `slot_rect` 的三面对象边界。

Codex 负责：

- 增加 slot/pocket candidate object descriptors。
- 加入最大对象尺寸和邻域复杂度约束。
- 生成 slot_v、slot_rect、pocket_rect hard cases。

用户负责：

- 对真实件中的槽类区域确认“一个连通槽是否作为一个对象输出”。

完成标准：

- `part2/part10/part11` 的 slot/pocket/transition 边界错误减少。

#### v5.4-D：focused hard-case 数据补强与重训

目标：

- 不盲目扩大随机数据，而是围绕当前失败模式补强。

Codex 负责：

- 生成 focused hard cases：
  - 外圆柱 + chamfer cone，作为 countersink 负样本；
  - 外部大圆柱，作为 slot_o / through-hole 负样本；
  - 真实 through-hole on curved carriers；
  - rib with holes；
  - rib in pocket；
  - V-slot near chamfers；
  - rectangular slot near fillet/chamfer。
- 重新训练 edge-aware GNN 和 candidate-level classifier。
- 重新输出 `test_part_s2` 验证报告。

用户负责：

- 抽查小批量 focused hard-case STEP。
- 只对关键争议样本补充面积真值，不需要大规模人工标注。

完成标准：

- 形成一个可用于论文的失败分析 -> 规则补强 -> 数据补强 -> 重训对比闭环。

### 4. 论文层面的意义

这一阶段的核心创新点可以表述为：

- 面向 CAE 的 STEP 特征识别不应只做面级分类，而应进行“B-Rep 面功能角色”的对象级判别。
- edge-aware GNN 给出基础预测，topology-aware candidate repair 提供物理/拓扑约束。
- 对真实简化件的失败分析驱动 focused hard-case 数据补强，形成可复现实验闭环。

---

## English Version

### 1. Current Objective

The current objective is not merely to improve metrics on generated data, but to make the recognizer stable on simplified real STEP parts for CAE preprocessing. The current pipeline is:

```text
STEP/STP
-> B-Rep graph extraction
-> edge-aware GNN face-level prediction
-> candidate proposal / candidate classifier
-> topology postprocess
-> feature objects
-> area-truth review and failure atlas
```

The current primary reports are:

- `output_real_step_s2_v5_3_rulefix/reports/s2_area_truth_validation_v5_3_rulefix_mergeeval.md`
- `output_real_step_s2_v5_3_rulefix/reports/remaining_true_error_analysis_v5_3_mergeeval.md`

Connected `slot_o` regions are accepted as one CAE-selectable region. Therefore, evaluation now allows one predicted `slot_o` object to match multiple connected truth `slot_o` entries. Future work should not split connected slots only to satisfy the manual area table.

### 2. Visible Problems

#### Problem A: Functional Role Confusion Of Cylindrical And Conical Faces

Symptoms:

- Exterior cylindrical stock faces are misclassified as `through_hole_round` or `slot_o`.
- Exterior cylinder plus chamfer cone is repaired as `countersink_hole`.
- Some true small through-hole cylinders remain `stock`.

Representative samples:

- `part1`, `part2`, `part3`, `part7`, `part8`

Root cause:

The graph already contains primitive type, adjacency, area, radius, and edge information, but the pipeline still struggles to distinguish:

- exterior stock cylinder;
- hole wall;
- O-slot bottom;
- fillet-like cylindrical transition;
- countersink cone;
- ordinary chamfer cone.

#### Problem B: Incomplete Through-Hole Instance Recall

Symptoms:

- Some small holes in the same part are detected, while equivalent holes remain `stock`.
- Small transition cylinders are sometimes predicted as `through_hole_round`.

Representative samples:

- `part4`, `part7`, `part8`, `part10`

Root cause:

A through hole should not be inferred only from a cylindrical face label. It should require explicit opening evidence.

#### Problem C: Unstable Rib / Web Object Boundaries

Symptoms:

- Ribs are missed.
- Rib objects absorb through-hole cylindrical walls.
- Rib, web, and pocket regions become confused in complex local geometry.

Representative samples:

- `part3`, `part4`, `part10`

Root cause:

Rib recognition should be plane-dominant. Transition faces may be absorbed, but through-hole cylinders should not become part of a rib object.

#### Problem D: Slot / Pocket / Transition Boundary Instability

Symptoms:

- `slot_rect` objects have too many faces or too much area.
- `pocket_rect` is over-merged into a large planar component.
- `slot_v` is absorbed into chamfer-like regions.

Representative samples:

- `part2`, `part10`, `part11`

Root cause:

Slot recognition needs object-level evidence: scale, concave-edge ratio, aspect ratio, open-boundary context, and neighborhood constraints.

### 3. Stepwise Implementation Plan

#### v5.4-A: Cylindrical / Conical Functional-Role Constraints

Goal:

- Reduce false repairs to `countersink_hole`, `slot_o`, and `through_hole_round`.
- Prevent exterior large cylinders and cones from being interpreted as holes or slots.
- Recover true through holes when opening evidence exists.

Codex responsibilities:

- Add candidate-level role constraints in `utils/topology_postprocess.py`.
- Add regression tests for:
  - large exterior cylinder + chamfer cone must not become `countersink_hole`;
  - large exterior cylinder must not become `slot_o`;
  - stock cylinder with two circular openings can recover as `through_hole_round`;
  - cylinder without opening evidence must not be promoted to through hole.
- Re-run `test_part_s2` evaluation, or at minimum regenerate feature-object reports for `part1/part2/part3/part7/part8`.

User responsibilities:

- Check whether ambiguous cones in `part1/part2/part7` should be ordinary chamfers or countersinks.
- Confirm ambiguous exterior-cylinder / hole-wall / slot-bottom regions when visual inspection is required.

Completion criteria:

- Large cylinder + chamfer cone in `part1/part2/part7` is no longer reported as `countersink_hole`.
- Huge exterior cylinder in `part3` is no longer reported as `slot_o`.
- True through-hole cylinders in `part8` are recovered if the graph contains valid opening evidence.

#### v5.4-B: Rib / Web Object-Boundary Repair

Goal:

- Make rib candidates plane-dominant.
- Allow reasonable transition absorption, but forbid absorbing through-hole cylinders.
- Improve rib-in-pocket and rib-with-hole cases.

Codex responsibilities:

- Add rib plane-dominance checks.
- Refine rib transition absorption.
- Generate focused hard cases for rib-with-hole and rib-in-pocket.
- Retrain the candidate-level classifier.

User responsibilities:

- Confirm whether fillet/chamfer transitions attached to ribs should be absorbed into rib objects.
- Inspect a small generated batch for realistic rib geometry.

Completion criteria:

- Rib misses and over-absorption decrease on `part3/part4/part10`.

#### v5.4-C: Slot / Pocket Object-Boundary Repair

Goal:

- Avoid huge false `pocket_rect` merges.
- Improve `slot_v` versus chamfer-like plane confusion.
- Stabilize three-face `slot_rect` objects.

Codex responsibilities:

- Add slot/pocket candidate object descriptors.
- Add maximum object-size and neighborhood-complexity guards.
- Generate hard cases for `slot_v`, `slot_rect`, and `pocket_rect`.

User responsibilities:

- Confirm whether connected slot regions should be output as one object in ambiguous real cases.

Completion criteria:

- Slot/pocket/transition boundary errors decrease on `part2/part10/part11`.

#### v5.4-D: Focused Hard-Case Data And Retraining

Goal:

- Improve generalization through failure-driven data, not blind random expansion.

Codex responsibilities:

- Generate focused hard cases:
  - exterior cylinder + chamfer cone as countersink negatives;
  - exterior large cylinders as slot/through-hole negatives;
  - through holes on curved carriers;
  - ribs with holes;
  - ribs in pockets;
  - V-slots near chamfers;
  - rectangular slots near fillet/chamfer transitions.
- Retrain the edge-aware GNN and candidate-level classifier.
- Re-run `test_part_s2` validation.

User responsibilities:

- Inspect a small focused hard-case STEP batch.
- Provide only key ambiguous area-truth corrections when needed; large-scale manual labeling is not required.

Completion criteria:

- The project has a reproducible loop suitable for the paper: failure analysis -> topology repair -> focused data augmentation -> retraining -> real-part validation.

### 4. Paper-Level Significance

The contribution of this stage can be framed as:

- CAE-oriented STEP feature recognition should go beyond face-level classification and reason about functional roles of B-Rep faces.
- The edge-aware GNN provides base predictions, while topology-aware candidate repair adds geometric and physical constraints.
- Failure-driven focused hard-case generation creates a reproducible experimental loop for validating generalization on simplified real parts.
