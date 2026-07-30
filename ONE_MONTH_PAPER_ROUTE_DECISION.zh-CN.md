# TopoCAE-Agent 一个月成稿路线决策

> 日期：2026-07-20  
> 性质：在完整 P1-D 与 Agent 主线之间的监督决策  
> 目标：一个月内形成结构完整、实验可执行、投稿定位不低于 CCF-B 预期的论文初稿

## 1. 监督结论

不再把“跨任意 CAD 的通用特征识别”作为第一篇论文必须先解决的前置条件，也不完全放弃几何识别。

论文主问题固定为：

> 给定受控类别范围内的 STEP 单零件和自然语言结构静力需求，如何把 CAE 意图可靠地映射到精确 B-Rep 面集，并生成可验证的 Abaqus 前处理操作？

推荐路线为：

1. 用 3–4 天执行 P1-D Lite，只诊断并冻结一个可用的受控域识别器。
2. 将 `test_part`、`test_part_s`、`test_part_s1`、`test_part_s2` 明确作为 Real-Dev，不再声称它们是未见外推集。
3. 立即进入 CAE-IR、自然语言 grounding、topology-constrained pointer 和 Abaqus 验证闭环。
4. 模型冻结后，仅新建 6–10 个简单、未参与规则修改的 Real-Holdout 零件，作为最终一次性外部验证。
5. 论文同时报告 Oracle feature objects 和 predicted feature objects 两条端到端链路，量化几何感知误差如何传递到 CAE 操作。

这是当前一个月时限下风险最低、研究问题最完整、最能利用既有资产的路线。

## 2. 旧版高分为什么出现

`output_real_step_s2_v5_6_a_rolefix` 的面积真值报告给出 114/114 truth objects matched，Recall 和 Precision 均为 1.0。但它不能作为未见真实件泛化结果，原因如下：

1. `test_part_s2` 在 v5.3、v5.4-A 到 v5.4-G、v5.5、v5.6-A 中被反复诊断和回归，事实上已经成为开发集。
2. 从 v5.3 的 Recall 0.7080、Precision 0.7596 到 v5.4-G 的双 1.0，所用 GNN checkpoint 和 candidate classifier 没有变化；主要变化来自拓扑后处理规则。
3. 最终 `v5_6_a_rolefix` 的 310 个面中，227 个面的最终 macro 与 raw GNN macro 不同，占 73.2%。最终成功主要来自候选重组、规则修复和无效对象降级，而不是 raw GNN 单独达到 1.0。
4. `utils/topology_postprocess.py` 已超过 3300 行，并配有大量针对历史真实件失败模式发展出的回归测试。规则本身未必使用 part ID，但规则选择和阈值是在这些 part 上迭代形成的。
5. 旧面积评价允许相连同类槽使用 `merged_truth` 匹配，且面积匹配不等于 exact face-set 泛化。

因此，旧版结论应改写为：

> 在受控真实开发集上，学习模型与拓扑规则耦合能够恢复高质量 CAE 特征对象；该结果证明工程可行性，但不构成独立外推证据。

## 3. 是否继续增加一个基体上的特征

需要有限度补强，但不能简单增加每个零件的特征数量。

`v5.5` 的 5395 个训练样本已经有约 78% 包含 2–4 个不同特征类别，平均为 2.28 类。因此当前主要缺口不是“只有单特征”，而是：

- 同一类别的多个实例，例如多个孔、多个 rib、多个槽；
- 不同特征之间的上下文竞争，例如 rib 位于 pocket 内、孔穿过 rib；
- 与真实建模历史相近但不破坏目标对象的组合；
- 不同基体、特征密度、尺寸相对关系和重复阵列；
- 按生成 recipe 分组的严格 train/dev/test 划分。

若 P1-D Lite 证明候选提议器确实受组合影响，只做一个小型 Mixed-Context 补强集：

- 30% 单特征纯概念样本；
- 40% 2–4 个不相交的异类特征；
- 20% 2–6 个同类多实例；
- 10% 经过几何审计的 hard composition。

不得以“一个基体特征越多越好”为生成原则，也不得让布尔干涉破坏原始拓扑概念。

## 4. P1-D Lite 的范围和停止条件

完整 P1-D 暂缓。P1-D Lite 只执行以下内容：

### Day 1：旧版与当前版分层对照

- 在同一 Real-Dev 上输出 raw GNN、candidate classifier、topology postprocess、feature objects 四级指标。
- 报告 proposal recall、candidate conditional accuracy、verifier false veto 和最终 object F1。
- 明确旧 `rolefix` 的收益来自哪一层。

### Day 2：Oracle 瓶颈分解

- Oracle face labels；
- Oracle candidate membership；
- Oracle candidate macro；
- Oracle verifier/decoder。

只修复收益最高且能用旋转/尺度无关 B-Rep 关系表达的一项瓶颈。

### Day 3–4：最小修复和冻结

- 不生成大规模新数据；
- 不换大型网络；
- 不读取 `test_part_s3`；
- 不按零件、face ID 或面积常数写规则；
- 加入 abstention，允许系统对低置信候选请求人工确认。

满足以下任一条件即停止 P1-D Lite 并进入 Agent：

1. Real-Dev object F1 达到或超过 0.60，且孔、槽、rib 三个家族均有可用召回率；
2. 相比当前 P1-C，object F1 提升至少 0.08；
3. Oracle 分解证明主要瓶颈需要长期重做数据生成或候选提议器，无法在两天内可靠修复。

第三种情况不是论文失败。识别器以受控域模块冻结，并在论文中通过 Oracle/predicted 双轨实验诚实量化其影响。

## 5. Agent 论文的核心方法

### 5.1 几何侧

```text
STEP
-> invariant attributed B-Rep graph
-> edge-aware GNN
-> face labels + same-instance scores
-> candidate feature objects
-> topology legality repair + abstention
-> feature-object memory with selectable face roles
```

### 5.2 语言与 grounding 侧

```text
Natural-language CAE requirement
-> open-weight LLM
-> typed CAE-IR
-> topology-constrained object/face-role pointer
-> exact STEP face set
-> semantic + topology + unit verifier
```

### 5.3 Abaqus 侧

```text
verified face set + CAE operation
-> STEP-to-Abaqus topology fingerprint
-> set/surface/BC/load/material action plan
-> Abaqus Python compiler
-> creation/execution feedback
-> repair, clarification, or abstention
```

LLM 只负责语言理解和结构化规划，不直接读取 STEP、不自由猜 face ID，也不替代 B-Rep 图模型。

## 6. 论文真正的技术贡献

1. CAE-FeatureGraph：带 feature instance、semantic face role 和 selectable region 的 STEP-graph 配对数据。
2. 受控域 TopoCAE-Recognizer：edge-aware GNN 与拓扑合法性约束耦合，并支持不确定时拒识。
3. CAE-IR 与 topology-constrained pointer：把语言需求映射为候选对象及精确面角色，而不是让 LLM 生成 raw face ID。
4. 可验证 Abaqus 闭环：从 exact face set 到可执行边界/载荷操作，并使用结构化反馈修复。
5. 误差传播分析：分别在 Oracle geometry 和 predicted geometry 下评估语言、grounding 与执行，指出真正瓶颈。

单纯“接入一个开源 LLM”或“生成 Abaqus 脚本”不能作为创新点。

## 7. 最低可信评估规模

完全不做外部验证不适合 CCF-B 预期。可以不追求复杂工业件，但必须保留一个小型冻结外部集。

### 几何评估

- Synthetic-Test：冻结合成测试集；
- Real-Dev：现有 `test_part` 到 `test_part_s2`；
- Real-Holdout：模型冻结后新建 6–10 个简单零件；
- Rotation：每个 Holdout 至少一个旋转派生版本；
- 指标：face Macro-F1、object P/R/F1、instance IoU、exact face set、rotation consistency、coverage-risk。

### 语言与 grounding 评估

- 从合成图标签自动产生 1000+ 条结构化 query；
- 人工编写或审核 50–100 条工程表达；
- 每个 Real-Holdout 零件设计 3–5 条约束/加载需求；
- 指标：CAE-IR exact match、slot F1、object selection accuracy、face-set IoU、exact face-set accuracy。

### 端到端评估

- Baseline A：LLM 直接输出 face ID；
- Baseline B：关键词/规则匹配；
- Baseline C：LLM -> CAE-IR -> deterministic grounding；
- Full：LLM -> CAE-IR -> topology-constrained pointer -> verifier；
- Oracle：使用真值 feature objects 的完整流程上界；
- 指标：Abaqus region creation success、BC/load creation success、script execution success、clarification/abstention rate。

## 8. 四周成稿排期

### 第 1 周：冻结识别器，建立 Agent 契约

- 完成 P1-D Lite 和分层报告；
- 冻结 Recognizer-Dev；
- 定义 CAE-IR JSON Schema、feature role policy、错误码和 abstention；
- 生成第一批语言 query。

### 第 2 周：完成语言解析与 grounding baseline

- 接入开源 LLM 的结构化输出；
- 实现 deterministic grounder；
- 实现 direct-LLM 和 keyword baselines；
- 建立 exact face-set 自动评估。

### 第 3 周：实现 pointer 与 Abaqus 验证闭环

- 训练或实现轻量 language-conditioned pointer；
- 加入 topology mask、数量词和属性筛选；
- 打通 Abaqus set/surface、BC、pressure/traction、material 的脚本生成与验证；
- 完成主要消融。

### 第 4 周：冻结外部测试并写初稿

- 用户新建 6–10 个简单 Real-Holdout；
- 冻结后只运行一次正式测试；
- 完成主表、失败案例、架构图和误差传播图；
- 形成方法、数据、实验和讨论完整初稿。

## 9. 人与 Agent 的分工

### 用户负责

- 审核 30–50 条典型结构静力需求是否符合 CAE 表达；
- 在系统冻结后绘制 6–10 个简单 Holdout 零件；
- 对少量歧义目标确认工程语义和正确面角色；
- 抽查 Abaqus 中 set/surface、约束方向和载荷方向是否合理。

用户不需要为大量零件逐面填写 face ID。

### 实施 Agent 负责

- P1-D Lite 分层诊断、最小修复和报告；
- CAE-IR、query 数据生成和 LLM parser；
- candidate/face-role grounding 与 pointer；
- Abaqus 编译、验证和回归测试；
- 自动生成所有指标、图表、消融和论文表格。

## 10. 最终投稿判断

- 继续做完整通用识别：科学上合理，但一个月内风险过高，最可能耗尽时间仍停留在跨域识别修补。
- 只做一个无外部验证的 Agent demo：实现快，但论文证据不足。
- 推荐方案：受控域识别 + 小型冻结 Real-Holdout + topology-constrained language grounding + Abaqus 可验证闭环。

该路线不保证 CCF-B 录用，但它把现有数据集、GNN、拓扑规则、自然语言和 CAE 专长组织成一个明确的新研究问题，是一个月内最有希望形成完整投稿的选择。
