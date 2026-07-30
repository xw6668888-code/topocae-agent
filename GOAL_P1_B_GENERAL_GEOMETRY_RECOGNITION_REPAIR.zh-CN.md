# Goal 提示词：P1-B 通用几何识别修复闭环

你是 `TopoCAE-Agent` 项目的实施 Agent。工作目录固定为：

`W:\python\FEA\dataset_generator`

Python 环境固定为：

`W:\python\envs\feagpt`

不得把依赖、模型、缓存或数据安装/写入 C 盘。不要使用 superpower 及其相关 skills。除非出现无法从代码、数据或本文判断的工程语义问题，否则自主完成代码、测试、数据、训练、推理和报告，不要只给计划。

## 一、总目标

修复当前合成数据概念错误和真实 STEP 泛化不足，形成一条可用于论文的通用识别闭环：

```text
STEP
-> 有效 B-Rep
-> 旋转/尺度稳健的 attributed face graph
-> edge-aware face encoder
-> same-instance 实例候选组装
-> candidate-level topology classifier
-> topology legality verifier + global conflict resolution
-> feature objects + semantic face roles + CAE selectable regions
```

最终应在修正后的合成数据上取得稳定结果，并显著改善 `test_part_s`、`test_part_s1`、`test_part_s2` 这批受控简单真实件。目标不是识别任意复杂工业零件，也不是针对现有零件增加专用阈值。

## 二、开始前必须阅读

1. `TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `reports/p1_a2r_human_review_supervisor_gate.zh-CN.md`
3. `NEXT_AGENT_TASK_P1_B1_SLOT_GENERATOR.zh-CN.md`
4. `reports/paper_v1_p1_a2/human_review_package_v2/human_review_synthetic_20.csv`
5. `reports/paper_v1_p1_a2/human_review_package_v2/human_review_real_dev_10.csv`
6. `reports/paper_v1_p1_a2/human_review_package_v2/Feature_Comparison_Table_New.xlsx`
7. `models/gnn_baseline.py`
8. `p1_a2_protocol.py`
9. `prediction_to_feature_objects.py`
10. `utils/instance_candidate_assembly.py`
11. `utils/topology_rules.py`
12. `features/swept_channel_slot.py`
13. `features/swept_slot.py`
14. `features/slot_rect.py`
15. `features/rib.py`、`features/web.py` 及相关生成上下文代码

首先输出一份不超过 200 行的 preflight 审计，确认当前入口、数据、模型、环境、冻结文件和已有错误证据。然后直接实施，不要等待用户再次确认。

## 三、固定研究边界

目标 macro 保持为：

- `through_hole_round`
- `blind_hole_round`
- `stepped_hole_round`
- `counterbore_hole`
- `countersink_hole`
- `obround_hole`
- `slot_rect`
- `slot_u`
- `slot_v`
- `slot_o`
- `pocket_rect`
- `pocket_circular`
- `cylinder_boss`
- `rectangular_boss`
- `rib`
- `web`
- `fillet`
- `chamfer`

研究范围固定为单零件、单实体、结构静力前处理中的几何特征识别与区域选择。

明确排除：

- `test_part_s3` 及其任何派生数据、图、报告和规则。
- 任意复杂工业件的全特征识别。
- 新增法兰、导轨、凸耳、安装板等目标类别。
- 让 LLM 直接读取 STEP、预测 raw face ID 或替代 B-Rep/GNN 识别器。
- 针对某个 part、某个 face ID 或某个面积值编写专用阈值和 if 分支。
- 在数据概念未通过人工审查前扩大规模或正式训练。

## 四、保护现有证据

以下内容只读，不得覆盖或删除：

- `output_generation_v5_1_augmented_full_5400_clean/`
- `output_generation_paper_v1_synthetic_holdout_v1/`
- `reports/paper_v1_p1_a2/`
- `paper_v1_clean_candidate_v2_freeze_manifest.json`
- 现有 checkpoint、split、manifest 和正式报告

所有新输出使用 `p1_b` 或更明确的新目录名。任何重训都必须使用新 run name。

## 五、当前已知基线和错误证据

- Synthetic-Holdout-v1 的 A0 face Macro-F1 为约 `0.8641`，object Macro-F1 为约 `0.8486`，但人工审查发现四类槽及 rib/web 的测试几何概念无效。
- Real-Dev 人工审查为 `1 PASS / 9 FAIL`。
- 新面积真值对冻结 A0 的初步 macro+area 匹配约为 `P=0.472 / R=0.614 / F1=0.534`。
- `test_part_s/part21` 的面 `[11,12]` 共同组成 countersink，GNN 的双向 same-instance 概率约为 `0.983/0.997`，但最终 feature object 仍被拆开，证明 same-instance 没有真正进入最终对象构建。
- 主要混淆包括：`slot_o -> fillet`、`slot_u -> obround_hole`、`cylinder_boss -> blind_hole_round`、`through_hole -> countersink/stepped/fillet/boss`、`rib -> stock/pocket`。
- 历史约 0.982 的 Real-Dev 面积结果来自多轮规则修复，不是独立泛化结果。

## 六、阶段 1：槽生成概念修复

严格执行 `NEXT_AGENT_TASK_P1_B1_SLOT_GENERATOR.zh-CN.md`，并满足以下核心要求。

### 6.1 统一生成框架

把 `slot_rect/slot_u/slot_v/slot_o` 统一为一个参数化 sweep-cut 框架，macro 只由截面决定：

- rect：四条直线闭合
- U：精确半圆弧加三条直线闭合
- V：闭合三角切削截面
- O：精确圆形闭合截面

guide/end 条件与截面正交组合：

1. straight + open_open
2. straight + open_closed
3. curved_open + open_open
4. curved_open + open_closed
5. curved_closed + closed_loop

四种截面共 20 个组合。

### 6.2 连续 guide

- straight 使用宿主平面的局部主方向，禁止连续随机斜角。
- curved guide 必须是单条解析圆弧、圆、Bezier 或 C1/C2 B-spline edge。
- 禁止通过采样点和多段直线模拟正式曲线。
- open_open 两端越过实体边界；open_closed 仅开放端越过边界；closed_loop 完全位于宿主面内部。
- 本论文版本只要求已有实体的平面宿主面，非平面宿主面的 sweep 明确记为后续工作。

### 6.3 自动审计与 pilot

生成 80 个 pilot，每个组合 4 个；另生成 20 个固定种子 canonical STEP，每个组合 1 个。

必须验证：单 solid、正体积、B-Rep valid、guide 单 edge/C1、端部状态、槽面标签覆盖、无槽面误标 stock、无 16/32/48 固定分段面数峰值。

### Human Gate H-B1：必须暂停

生成：

- `output_generation_p1_b1_slot_pilot_80/human_review_canonical_20.csv`
- `output_generation_p1_b1_slot_pilot_80/human_review_canonical_20/`
- 一份中文 UG/NX 检查指南

然后停止并告诉用户逐个检查以下内容：

1. 模型树是否只有一个有效实体。
2. 截面是否分别为矩形、U、V、O，而不是折线近似。
3. 弯槽 guide 是否视觉连续、没有折角和分段面带。
4. 通槽是否两端无残余实体。
5. 半通槽是否恰好一端开放、一端有完整终止壁。
6. 闭槽是否为首尾连续的环形 guide，且不与基体边界相交。
7. 是否存在埋在实体内部、没有开放到表面的隧道。

用户只需在 CSV 中填写 `PASS/FAIL` 和简短原因，不要求 face ID。收到用户全部通过或明确的修正意见后才能继续。

## 七、阶段 2：rib/web 工程语义修复

保留 `rib` 和 `web` 两个 macro，但共享薄壁候选的基础规则。不得生成孤立薄板正样本。

### 7.1 rib

- 位于 L 形两板连接处，或位于矩形/圆形腔内。
- 必须连接两个明确的支承区域或腔壁。
- 允许三角肋、矩形肋及合理倒角/圆角，但 transition 作为附属面时不得切断 rib instance。

### 7.2 web

- 为连接两个主体区域或两条边界的连续薄壁板区。
- 具有成对的大侧面和稳定的小厚度方向。
- 不得把 stock 的大基面或普通板侧面作为 web。

### 7.3 标签

- 所有薄壁组成面必须归入同一 instance。
- cavity/pocket 作为独立目标保留，内部 rib 不得吞并 pocket 面。
- context 几何保持 stock 标签。

生成至少 48 个 pilot，覆盖 L-corner rib、rect/circular cavity rib、web，以及带孔和 transition 的合理变体；另外导出 12 个 canonical STEP。

### Human Gate H-B2：必须暂停

告诉用户在 UG/NX 中检查：

1. rib/web 是否真实连接了两个支承区域。
2. 是否存在孤立薄板。
3. 尺寸比例是否像加强结构，而不是主体 stock。
4. 腔和 rib 是否同时存在且各自标签保持独立。
5. 带孔或 transition 时，薄壁主体是否仍完整。

用户只填写 `PASS/FAIL` 和原因，不要求 face ID。收到结论后再继续。

## 八、阶段 3：Graph v2 通用表示修复

在不丢失原始物理尺寸的前提下，建立用于识别的无量纲、旋转和尺度稳健特征。

### 8.1 节点特征

至少增加：

- `face_area / part_total_area`
- `face_area / part_bbox_diagonal^2`
- `radius / part_bbox_diagonal`
- bbox/projected/UV 尺度除以 part bbox diagonal
- `face_area / neighbor_area_mean`
- local extent ratios
- primitive surface、loop 数量、inner/outer loop 统计

### 8.2 边特征

至少增加：

- `edge_length / part_bbox_diagonal`
- `circle_radius / part_bbox_diagonal`
- curve type、convexity、dihedral angle
- adjacent plane loop role
- cylindrical wall inner/outer/ambiguous role
- material-side 或等价的内外壁证据

### 8.3 数值稳定性

- 对 area、length、radius 使用适当的 `log1p` 或稳健变换。
- 对长宽比、fill ratio 等做 finite 检查和训练集分位数裁剪。
- 禁止 NaN/Inf 和 `1e8~1e12` 的病态比值直接进入模型。
- 原始绝对值保留在 graph/object metadata 中，供 CAE 查询使用。

### 8.4 canonical surface patches

在候选推理层增加非破坏性的 superface/canonical patch 聚合：

- 可合并切向连续且共面/同轴的面片用于候选描述。
- 保留原始 face ID，最终仍映射回精确 B-Rep 面集。
- 不跨越凹凸角色、不同 instance 或真实锐边合并。

该模块用于减弱不同 CAD 建模历史造成的 face split/merge 差异。

### 8.5 测试

自己生成简单模型，执行整体旋转、统一尺度变化和合法 face partition 变化，验证：

- graph 拓扑同构或 canonical signature 一致。
- 无量纲特征在容差内一致。
- 原始物理尺寸按预期变化。
- 模型预测的 macro、instance 和 role 保持一致。

## 九、阶段 4：same-instance 与候选级识别闭环

### 9.1 实例组装

- 最终 feature objects 不得再默认使用 `connected_components_same_macro_feature`。
- 使用 GNN 输出的 same-instance 概率进行跨 provisional macro 的候选组装。
- 双向 edge 概率要有明确的对称化方式。
- 阈值只能在 Synthetic-Val 上选择并全局固定，不得按类别、part 或面积手调。
- 组装必须受 adjacency 和基础拓扑兼容性约束。

### 9.2 候选生成

候选来源至少包括：

- same-instance 强连接分量
- 圆柱/圆锥同轴链
- inner/outer loop 引导的孔/凸台候选
- 凹边围成的 pocket/slot 候选
- elongated swept-channel 候选
- thin-wall rib/web 候选
- parent-surface transition 候选

候选生成不得依赖真实测试件的 face ID 和面积常数。

### 9.3 候选级模型

实现可训练的 candidate-level topology classifier。优先复用 face GNN embedding，并融合：

- mean/max/attention pooled node embedding
- internal/boundary edge embedding
- primitive/semantic/convexity histograms
- loop、入口/出口、材料侧和同轴关系
- 无量纲尺度与邻域关系
- 候选面数和 canonical patch 数

输出至少包括：

- candidate macro logits
- candidate validity score
- face membership/boundary refinement score
- 可选的 semantic face-role logits

可以使用小型 MLP、Set Transformer 或候选子图 GNN。不要因为有 A40 就直接扩大到大型网络；先证明候选级结构带来收益。

### 9.4 hard negatives

从合成真值自动构造：

- 真实例的缺面子集
- 多吸收邻接 stock/transition 的超集
- 两个相邻实例的错误合并
- 同轴但不同功能角色的 cylinder/cone 组合
- `slot_o/fillet`、`slot_u/obround`、`boss/blind hole`、`rib/stock/pocket` 混淆候选

hard negative 规则必须由通用拓扑关系定义，不得引用 Real-Dev 的样本名、face ID 或面积。

### 9.5 topology verifier

- topology rules 只做类别 mask、合法性否决、置信度修正和 abstention。
- 禁止用规则强制把一个预测类别硬改成另一个类别。
- 非法或低置信候选输出 `unknown/abstain`，不要制造 false positive。

### 9.6 全局解码

用统一的 score 和冲突规则选择候选：

- 主特征之间默认不能无解释地重叠。
- fillet/chamfer 可以作为主特征的附属 transition。
- pocket 内 rib 可以与 pocket 共存，但不能相互吞并面集。
- 输出必须能追溯到候选分数、topology evidence 和原始 face IDs。

## 十、阶段 5：修正数据集

### 10.1 数据策略

- 对人工确认正确的旧几何可复用 STEP，但必须重新提取 Graph v2。
- 任何包含旧的错误槽或孤立 rib/web 的样本不得进入新数据集。
- 槽和 rib/web 必须全部使用通过 Human Gate 的新生成器重建。
- 对旧多特征样本先按 feature_sets 和生成 provenance 过滤；无法证明不含错误概念的样本直接排除。

### 10.2 规模与平衡

建立新的 class-balanced 数据集，目标 18 类每类至少 300 个有效实例，总体约 5000-7000 个样本或达到等价的实例覆盖。

同时包含：

- 单特征、不同 stock 和不同尺度
- 互不干涉的多特征组合
- 整体刚体旋转派生
- 统一尺度变化
- 合理的 face split/merge 建模历史变化
- 受控的 hard-confusion 组合

不得为了增加数量允许严重特征干涉。

### 10.3 split

- 按 base geometry family 分组切分，旋转、缩放和建模历史派生必须留在同一 split。
- train/val/development-test 无 geometry signature 泄漏。
- Real-Dev 不参与训练、阈值选择、early stopping 和模型选择。

## 十一、阶段 6：训练与实验

### 11.1 环境

- 优先使用可用 CUDA；记录 GPU、CUDA、PyTorch、Python、依赖和运行时间。
- 若本机无法访问用户的 A40 服务器，生成完整可复现命令和配置，不得假装已在 A40 上运行。
- 所有随机种子、配置、归一化统计和 checkpoint 均写入 manifest。

### 11.2 必做消融

至少训练和比较：

1. node-only GNN
2. + edge_attr
3. + same-instance assembly
4. + candidate-level classifier
5. + topology verifier/global decode

如实现 canonical patch，再增加无 canonical patch 的消融。

最终模型至少运行 3 个固定随机种子，报告 mean/std。

### 11.3 指标

必须报告：

- face macro accuracy、Macro-F1、per-class P/R/F1
- semantic role Macro-F1
- same-instance edge P/R/F1
- candidate proposal oracle recall
- object micro/macro P/R/F1
- exact instance ratio、instance IoU
- boundary face precision/recall/F1
- topology invalid rate、abstention rate
- rotation consistency、scale consistency
- 参数量、训练时间、推理时间和显存

### 11.4 模型选择

- 只使用修正后的 Synthetic-Val 和 Synthetic-Rotation/Scale-Val。
- Real-Dev 只用于最终诊断报告，不据其改变阈值或规则。

## 十二、阶段 7：真实件评估

使用：

- `test_part_s`
- `test_part_s1`
- `test_part_s2`，排除用户明确认定建模错误的 `part10`
- `Feature_Comparison_Table_New.xlsx`
- `human_review_real_dev_10.csv`

生成统一 area+macro、face-set（有真值时）、候选边界和人工审查报告。至少比较：

- 冻结 A0 Raw-GNN
- 历史规则管线，仅作为开发集上界参考
- 新的完整通用管线

失败必须分为：proposal miss、macro role confusion、fragment、merge、boundary error、invalid topology、abstention。不得只给总体 accuracy。

开发目标：相对当前约 `0.534` 的 area+macro F1 有明显提升，优先达到 `>=0.75`；该值只作为工程开发 Gate，不作为未知真实件最终论文结论。

## 十三、阶段 8：重新冻结和新 Holdout

模型和阈值只根据 Synthetic-Val 选择后：

1. 写入 preholdout selection hash。
2. 生成从未用于开发的 Synthetic-Holdout-v2。
3. 检查与训练/验证集无 base、exact 和 near graph signature 碰撞。
4. 只运行一次正式 Holdout 推理。
5. 生成 20 个分层人工审查 STEP，覆盖全部 18 类并包含槽和 rib/web。
6. 不根据 Holdout-v2 结果回头改模型、规则或阈值。

Holdout 结果无论高低都如实保存。若未达到预期，只能进入下一论文版本，不能污染该 Holdout。

## 十四、自动验收指标

在进入最终冻结前至少满足：

- Human Gate H-B1 的 20/20 canonical slot 通过，或所有失败均已修复后重新通过。
- Human Gate H-B2 的 12/12 rib/web canonical 通过。
- 新数据集中 STEP 单实体/B-Rep 有效率 100%。
- 槽面和 rib/web 面标签覆盖率 100%。
- 无 NaN/Inf，病态描述符审计通过。
- split leakage 为 0。
- Synthetic-Val face Macro-F1 不低于 0.90。
- Synthetic-Val object Macro-F1 不低于 0.88。
- exact instance ratio 不低于 0.85。
- same-instance edge F1 不低于 0.95。
- rotation consistency 不低于 0.98。
- candidate proposal oracle recall 不低于 0.95。
- 新完整管线在 Synthetic-Val 上不得比 Raw-GNN 的 face/semantic 指标显著退化，同时 object/boundary 指标必须有可测收益。

若某项未达到，不得用隐藏类别、删除困难样本或真实件专用规则绕过。继续在通用数据、表示或候选模型层诊断，最多完成两轮有明确假设的迭代；仍未达到时，提交诚实的瓶颈报告并请求监督者决定是否收缩论文 claim。

## 十五、交付物

至少交付：

- 修复后的槽和 rib/web 生成代码及测试
- 两个人工审查包及用户回填结果
- Graph v2 schema 和 extractor
- rotation/scale/canonical-patch 不变性测试
- same-instance candidate assembly
- candidate-level topology classifier 训练与推理代码
- topology verifier/global decoder
- 修正后的完整数据集、splits、dataset card 和 manifest
- 3-seed checkpoint、config、metrics 和环境记录
- 消融总表及源 CSV/JSON
- Synthetic-Holdout-v2 一次性正式报告
- Real-Dev 对比报告和 failure atlas
- 单一可复现入口，例如 `run_paper_v1_p1_b_pipeline.py`
- 更新后的中文项目上下文和阶段报告

所有报告中英文术语统一，所有数字必须可追溯到 JSON/CSV，不允许只提供截图。

## 十六、进度和汇报规则

- 每完成一个阶段就更新计划状态、测试结果、修改文件和下一步。
- 每 30 秒左右提供一次简短、实质性的工作更新。
- 长任务不得留下未结束的训练或 shell session。
- 遇到 Human Gate 时必须明确暂停，并给用户中文说明：打开哪些 STEP、在 UG/NX 看什么、如何填写哪个 CSV、什么算通过。
- 人工审查不得要求用户查 raw face ID 或重新制作大量真实零件。
- 除 Human Gate 或真正无法从现有资料判断的工程语义外，不把可编码、可测试、可运行的工作交给用户。

## 十七、Goal 完成条件

只有在以下全部成立时才能将 Goal 标记为 complete：

1. 两个人工 Gate 已完成且通过。
2. 槽和 rib/web 数据概念已修复并有自动回归测试。
3. Graph v2、same-instance、候选级分类和 topology verifier 已接入单一推理流程。
4. 新数据、训练、消融、三随机种子和报告全部完成。
5. 新 Holdout-v2 已按冻结协议只推理一次。
6. Real-Dev 已按新真值重新评估并生成逐类 failure atlas。
7. 所有必要测试通过，冻结 manifest 和复现命令完整。
8. 未使用 `test_part_s3`，未增加真实件专用阈值，未让 LLM 替代几何识别。

完成后向监督者提交一份中文验收摘要，逐项对应本提示词的自动指标、人工 Gate、交付文件和未解决限制；不要自行进入自然语言 CAE Agent 的 P2 阶段。
