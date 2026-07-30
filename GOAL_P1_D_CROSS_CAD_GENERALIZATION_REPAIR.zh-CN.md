# Goal P1-D：跨 CAD 域瓶颈分解与通用对象识别修复

## 可直接用于新 Goal 的提示词

你正在 `W:\python\FEA\dataset_generator` 中继续 TopoCAE 第一篇论文的 Phase 1 工作。请先完整阅读以下文件，再执行任何代码修改：

- `TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
- `reports/P1_B_P1_C_SUPERVISOR_AUDIT.zh-CN.md`
- `GOAL_P1_B_GENERAL_GEOMETRY_RECOGNITION_REPAIR.zh-CN.md`
- `GOAL_P1_C_CLEAN_GEOMETRY_FULL_REGEN_RETRAIN_COMPARISON.zh-CN.md`
- `reports/p1_c_final_acceptance_v1/final_acceptance.zh-CN.md`
- `reports/p1_c_real_dev_v1/metrics.json`
- `reports/p1_b_real_dev_v1/metrics.json`
- `reports/p1_c_freeze_v1/preholdout_selection.json`
- `CODEX_CONTEXT_P1_C.md`

### 总目标

在不改变第一篇论文范围的前提下，解释并修复 P1-C “Synthetic-Holdout-v3 object Macro-F1=0.941987，但 Real-Dev F1=0.446215、rib recall=0、slot-family recall=0.166667”的跨域失败。

最终识别范围仍是：单零件结构静力前处理中的 CAE 相关几何特征对象和面集。目标类别维持当前 18 类。`test_part_s3` 永久排除。不得进入 LLM 直接 STEP 识别、CAE-IR、pointer 或 Abaqus 编译阶段。

本 Goal 不是“继续扩大同分布合成数据”，也不是“换更大的 GNN 试试看”。必须先完成可量化瓶颈分解，再根据证据做一次有针对性的修复。

### 不可违反的约束

1. 保护 P1-B/P1-C frozen checkpoint、报告、Holdout-v2/v3 和现有数据；不得覆盖或二次运行一次性 Holdout。
2. `test_part_s3` 不得读取、训练、调参、评估或出现在清单中。
3. Real-Dev 仅包括现有 `test_part`、`test_part_s`、`test_part_s1`、`test_part_s2`，继续排除已声明错误的 `test_part_s2/part10`。
4. Real-Dev 从本 Goal 起明确作为开发/域适配集；它的结果不得再包装成最终外推结论。最终外推需在模型冻结后新建 Real-Holdout。
5. 不得添加零件名、sample ID、face ID、真实件专用面积常数、per-part 阈值或逐件 if/else。
6. 所有几何规则必须由旋转/尺度无关的 B-Rep 关系表达。
7. 训练和推理必须调用同一个 label-free candidate proposer；所有 truth 字段必须在正式推理前剥离。
8. 不要使用 superpower skills。
9. 使用 `W:\python\envs\feagpt\python.exe`；依赖不得安装到 C 盘。
10. 不要进入 P2；完成 P1-D 验收后停止并等待监督者决定。

## Task 0：冻结、复现与入口审计

1. 建立 `reports/p1_d_preflight/`，记录 P1-B/P1-C frozen 文件 SHA-256、环境、GPU、git 状态和输入数据清单。
2. 验证当前 P1-C 单一入口在全新 Python 进程中可以复现一个 synthetic-dev 样本和一个 Real-Dev 样本。
3. 审计所有 import-time monkeypatch/global rebind。P1-D 正式入口不得依赖模块导入顺序；尽量以显式依赖对象或 scoped context 注入。不得重构无关模块。
4. 新增回归测试，覆盖 strict -> legacy、legacy -> strict、verifier-v2/v4/v5 乱序导入以及连续两次不同 pipeline 调用。

交付物：

- `reports/p1_d_preflight/preflight.json`
- `reports/p1_d_preflight/import_state_audit.md`
- 对应自动测试。

## Task 1：Real 真值 face-set 自动恢复与 Human Gate H-D1

基于现有面积真值表、Graph face area、face_count、连通性、primitive 和 loop/edge 关系，为每个 Real-Dev truth object 枚举候选 face set：

1. 先按宏类别、目标 face_count 和总面积容差生成组合；
2. 使用连通性、primitive 可行性和拓扑签名排序；
3. 唯一匹配自动标为 `auto_unique`；
4. 多解标为 `ambiguous`，不得擅自选真值；
5. 生成带 face ID 和高亮颜色的 HTML/STEP 审查包。

输出：

- `reports/p1_d_truth_reconciliation/real_object_truth_candidates.json`
- `reports/p1_d_truth_reconciliation/human_review_real_face_sets.csv`
- `reports/p1_d_truth_reconciliation/human_review_real_face_sets.html`
- `reports/p1_d_truth_reconciliation/README.zh-CN.md`

到此必须停止等待 H-D1。告诉用户：只检查 CSV 中 `status=ambiguous` 的行；打开 HTML 对照 UG 模型，填写 `review_decision=accept/reject`，若 reject 则填写 `correct_candidate_id`。用户不需要手工抄 face ID，也不需要新画零件。

用户确认 H-D1 后，冻结 face-set truth 清单及 SHA-256，后续不得修改。

## Task 2：四级 Oracle 瓶颈分解

在冻结 Real-Dev face-set truth 上运行以下诊断，不先训练：

1. `E0 current_full`：当前 P1-C 全流程；
2. `E1 oracle_face`：Oracle face macro/semantic role + 当前 same-instance/proposer/classifier/verifier/decoder；
3. `E2 oracle_candidate`：当前 GNN + Oracle candidate face set + 当前 classifier/verifier/decoder；
4. `E3 oracle_macro`：当前 proposer + Oracle candidate macro/membership + 当前 verifier/decoder；
5. `E4 oracle_verifier_decode`：当前 proposer/classifier + Oracle accept/conflict decision，用于量化规则和全局解码损失。

必须按类和整体报告：

- raw face Macro-F1；
- proposal recall@IoU 0.5/0.75/0.95；
- candidate conditional macro accuracy/F1；
- membership IoU 和 exact boundary ratio；
- verifier false-veto / false-accept；
- decoder conflict loss；
- final object P/R/F1；
- rib、slot family、hole family、transition family 单独指标。

输出：

- `reports/p1_d_oracle_decomposition/metrics.json`
- `reports/p1_d_oracle_decomposition/per_class.csv`
- `reports/p1_d_oracle_decomposition/report.zh-CN.md`

报告必须明确指出首要和次要瓶颈。没有 Oracle 分解，不得进入 Task 4 的训练。

## Task 3：Synthetic -> Real 域差异审计

对 P1-C Clean-Core、P1-B Diverse-Context 和 Real-Dev 比较：

- graph node/edge/canonical-patch 数量；
- 每模型 feature-instance 数；
- primitive、loop role、edge curve、convexity、material-side 分布；
- dimensionless size/aspect/thinness 描述符；
- proposal source 和 candidate face-count 分布；
- GNN embedding 和 candidate embedding 的距离。

训练一个只用于审计的 synthetic-vs-real domain discriminator，使用固定交叉验证并报告 AUC。该模型不得进入正式推理。AUC 很高时必须列出最能区分域的输入族，并检查是否属于生成器快捷特征。

增加以下输入消融诊断：

- 去掉 proposal-source one-hot；
- 去掉 canonical patch；
- 去掉 GNN embedding，仅保留关系型 B-Rep 特征；
- 去掉 engineered features，仅保留候选子图 embedding。

输出 `reports/p1_d_domain_shift/` 下的 JSON、CSV 和中文报告。

## Task 4：槽族语义恢复与 Human Gate H-D2

重构 P1-C 槽生成契约。四种截面 `slot_rect/slot_u/slot_v/slot_o` 均应覆盖：

1. `straight_through`：与 host 主轴平行，两个端部均贯通；
2. `straight_semi_open`：一端贯通，一端封闭；
3. `curved_through`：连续曲线引导线，两端贯通；
4. `curved_semi_open`：连续曲线，一端贯通；
5. `curved_closed`：连续闭合引导线。

要求：

- 引导线必须使用 OCC line/arc/B-spline 等连续几何，禁止短线段折线逼近曲线；
- 截面保持此前确认的真实封闭截面定义；
- sweep path 长度显著大于截面尺度；
- 通槽和半通槽可做受限端部延长，但须通过单实体、valid B-Rep、目标实例保护和布尔干涉审计；
- 构造元数据记录 `profile_family / guide_family / end_condition`，这些字段仅用于审计和 split，不作为模型输入。

生成 20 个 canonical 检查样本，每类截面覆盖上述五种状态，生成 `human_review_slot_20.csv` 和 HTML 预览。到此停止等待 H-D2。

告诉用户逐个填写：

- `geometry_ok`：截面和扫掠形态正确；
- `guide_continuous`：曲线连续，无折线逼近；
- `end_condition_ok`：通/半通/闭状态正确；
- `label_ok`：槽面标注正确；
- `solid_ok`：UG 中为一个实体。

全部通过后冻结槽生成 revision。

## Task 5：候选提议器和关系型拓扑规则修复

根据 Task 2 的瓶颈结论实现，不得按 part 定制。

最低要求：

### 5.1 圆柱/孔族

- 以内外 material-side、开口数、封闭端、同轴有序链和相邻 shoulder 为核心；
- 外圆柱 stock/boss 不得因“存在 cylinder”通过 hole 规则；
- through/blind/counterbore/countersink/stepped 使用同一个同轴链表示，再由出口、底面、锥面、肩面细分；
- 保留不确定状态并允许 abstain。

### 5.2 slot/pocket

- 候选按局部 swept-channel 连续性构造，不把整个凹连通区域直接当一个对象；
- 表示端部开闭、截面 primitive 序列、长向连续性、与 parent surface 的边界关系；
- 允许相连同类槽作为一个可接受对象输出，但评估采用 connected same-macro component 兼容规则；
- pocket 中的 rib 不得被 pocket 吸收，pocket 仍保留独立目标。

### 5.3 rib/web

- 以薄壁 patch 对、厚度一致性、至少两个不同 support parent、长高厚比例和局部桥接关系构造候选；
- 不以单个面积占比阈值作为充分条件；
- 成员扩展应覆盖端面/顶面，但不得吸收穿过 rib 的孔壁、pocket 底面或大 stock 面；
- rib 带倒角时 transition 可作为附属面，不独立输出。

### 5.4 fillet/chamfer

- 必须有两个 parent surface 或等效 parent-patch 证据；
- fillet 使用光滑/圆角曲面与 parent dihedral 关系；
- chamfer 使用平面/锥面、两侧 parent 和尺度比；
- countersink cone 通过 hole chain 归属，不作为普通独立 chamfer。

为每条关系规则建立正例、反例、旋转、缩放和面划分回归测试。规则仅允许 class mask、veto、confidence correction 和 abstention，不允许按零件硬重写。

## Task 6：Mixed-Curriculum 和候选训练修复

不得先重新生成 7200 个同分布样本。先建立小规模可验证训练矩阵：

- `Clean-Core`：P1-C 已冻结 clean 数据；
- `Diverse-Context`：P1-B 中审计通过、标签可靠、非旧错误 slot/rib 的多样背景样本；
- `Slot-Guide-Hard`：Task 4 新槽族；
- `Real-Adapter`：H-D1 冻结的 Real-Dev face-set truth，按 part 分组切分，禁止旋转派生跨 split。

候选训练要求：

- 所有 split 都先由同一 label-free proposer 产生候选，再与 truth 匹配产生 target；
- `teacher_truth` 只能作为 curriculum，扫描占比 `0 / 0.1 / 0.25`，不得默认全量加入；
- 增加 merge/split/missing-member/extra-stock/wrong-role/parent-transition hard negatives；
- 增加 proposal-source dropout 或移除 source one-hot 的正式消融；
- 训练中使用 part-grouped sampler，避免一个模型的大量候选支配 loss；
- 对 candidate validity/macro/membership 做全局温度缩放或 conformal calibration。

只允许以下有限模型矩阵：

1. P1-B frozen；
2. P1-C frozen；
3. P1-D mixed，无 Real-Adapter；
4. P1-D mixed + Real-Adapter；
5. P1-D 去 topology rules；
6. P1-D 去 proposal-source / teacher-truth 的关键消融。

不允许在没有证据时增加 Transformer、大语言模型或显著扩大 GNN 参数量。

## Task 7：P1-D 正式开发评估

在不读取 `test_part_s3` 的条件下报告：

- Synthetic composition-dev；
- Real-Dev：`test_part` 到 `test_part_s2`；
- Rotation-Dev；
- P1-B/P1-C/P1-D 对比；
- per-class/family 指标；
- object F1、instance IoU、exact face set；
- proposal recall、verifier 和 decoder 误差；
- ECE、accuracy-coverage、risk-coverage 和 abstention；
- 运行时间、显存和参数量。

生成 `reports/p1_d_final_acceptance_v1/`：

- `final_acceptance.zh-CN.md`
- `metrics.json`
- `per_class.csv`
- `oracle_decomposition.json`
- `domain_shift.json`
- `failure_atlas.json/csv`
- `human_review_real_predictions.html`
- `delivery_manifest.json`

## P1-D 自动 Gate

以下条件全部满足才可提交给监督者：

1. Synthetic composition-dev object Macro-F1 `>=0.90`；
2. Real-Dev object F1 `>=0.65`，且明确高于 P1-B `0.600`；
3. rib recall `>=0.50`；
4. slot-family recall `>=0.50`；
5. Rotation-Dev object consistency `>=0.98`；
6. proposal recall@IoU0.5、candidate conditional F1、verifier/decoder loss 均已报告；
7. 不存在 `min_candidate_confidence=0.0` 的冻结选择；
8. 无 part-specific rule、area constant、sample-name input 或 truth leakage；
9. `test_part_s3_accessed=false`；
10. H-D1/H-D2 均有用户签字并冻结 SHA-256；
11. 导入顺序测试、拓扑关系测试、rotation/scale/partition 测试全部通过；
12. 不进入 P2，不创建 Real-Holdout，不修改一次性 Holdout 结果。

若某项未通过，报告真实结果和剩余瓶颈，不得通过降低 Gate、隐藏类别或重复查看 Holdout 来完成 Goal。完成后停止，等待监督者决定是否冻结 P1-D 并进入 P2。

