# 下一 Agent 任务：P1-A1

## 任务名称

`P1-A1：统一复现、数据泄漏审计与 clean baseline 建立`

## 启动前必读

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\reports\p0_supervisor_gate_review_full.zh-CN.md`
3. `W:\python\FEA\dataset_generator\configs\paper_v1\README.md`

## 目标

把当前几何识别结果整理成可复现、无 S3 派生训练数据、可直接进入论文实验表格的 clean baseline，并完成 Phase 1 的数据与评估审计。

## 不可突破的边界

- 不读取或使用 `test_part_s3`、`test_part_s3_rib_colored`、`output_real_step_s3*`、`configs/test_part_s3*` 中的几何、真值或预测结果。
- 允许读取 P0 监督报告中已经记录的污染结论，也允许读取 legacy 模型自身的 metadata 以形成 provenance；不得把其 S3 记录重新输入训练或评价。
- 不覆盖 `configs/paper_v1/`、`paper_v1_freeze_manifest.json`、现有 checkpoint 或 P0 报告。
- 不使用 Real-Dev 真值训练模型、选择阈值或新增逐件规则。所有模型选择和阈值选择只允许使用 Synthetic-Train/Synthetic-Val。
- 不处理复杂件，不逐件修规则，不修改 GNN 架构。
- 不开始 CAE-IR、LLM、pointer、Abaqus agent 或 Real-Holdout。

## 必须实现的工作

### A. 数据和模型 provenance 审计

实现可重复运行的审计脚本，递归追踪以下组件的训练数据、划分、随机种子、配置和模型哈希：

- frozen GNN；
- candidate topology classifier；
- rib candidate proposer；
- rib boundary classifier。

输出机器可读表，至少包含：artifact、artifact hash、training dataset、source graph roots、truth source、split、seed、是否含 real data、是否含 S3 派生数据、是否允许进入 clean paper baseline、判定理由。来源无法证明时标记 `unknown`，不得猜测。

### B. 数据集审计

对正式使用的 Synthetic-Train/Val/Test 完成：

- 样本数、face 数、feature instance 数和每类分布；
- solid validity 与 graph/STEP/label 配对完整性；
- 标签 ID 与 ontology/schema 一致性；
- base geometry、随机种子、派生旋转和近重复样本的跨 split 泄漏检查；
- 数据损坏、空图、非法实例和类别缺失统计。

不得只凭文件名判断泄漏。应优先使用 generation provenance、几何/图指纹和 rotation-invariant graph signature。

### C. 统一评价入口

建立一个新的 Phase 1 评价入口，固定评估以下数据：

- Synthetic-Test；
- Real-Dev：`test_part`、`test_part_s`、`test_part_s1`、`test_part_s2`；
- Rotation-Dev：只使用上述 Real-Dev 的既有旋转派生组或明确可追溯的旋转副本。

至少报告：

- face macro accuracy、Macro-F1、per-class precision/recall/F1；
- semantic subtype Macro-F1；
- same-instance edge precision/recall/F1；
- object precision/recall/F1、exact instance ratio、instance IoU；
- rotation consistency：同一零件不同姿态的类别一致率、对象对应一致率和 face-role 一致率；
- 95% bootstrap confidence interval（适用于 object/query/part 聚合指标）。

若某数据缺少计算某指标所需真值，必须报告 coverage 和 `not computable`，不得用其他指标冒充。

### D. 建立两条 clean 流程

固定比较：

1. `Clean-Core`：GNN + candidate classifier + topology postprocess；禁用两个 S3 派生 rib 模型。
2. `Clean-Rib`：在 Clean-Core 上加入仅由 Synthetic-Train 及合成 hard cases 训练的 rib proposer/boundary classifier。阈值只在 Synthetic-Val 上选择。

P0 legacy 流程只作为复现参考，不得参与 clean 模型选择，也不得把其指标作为论文主结果。

如果无法从现有合成数据构建合法的 rib 训练集，保留 Clean-Core，并在报告中把 Clean-Rib 标记为阻塞；不要为了完成任务而引入 Real-Dev 或 S3 真值。

### E. clean 候选冻结

不得覆盖 P0 冻结物。新建：

- `configs/paper_v1_clean_candidate/`
- `paper_v1_clean_candidate_freeze_manifest.json`
- `reports/paper_v1_p1/`

只有当 Clean-Rib 相对 Clean-Core 在 Synthetic-Test 的 rib object F1 有稳定收益、总体 Macro-F1 不下降超过 0.005、旋转一致性不下降，才将 Clean-Rib 写入 clean candidate；否则使用 Clean-Core。

Real-Dev 只用于报告现状和失败分类，不作为是否选择 Clean-Rib 的判据。

### F. 人工审查包

为用户生成：

- 20 个分层抽样的合成 STEP 审查清单，覆盖主要类别、不同基体和姿态；
- 每个样本对应的 graph 标签摘要和可在 UG 中核对的 face 面积/角色信息；
- 10 个 Real-Dev 预测审查清单，附预测对象、置信度、目标面和错误风险；
- 一个简洁的中文审查说明，明确用户逐项看什么、如何填写通过/不通过。

不要要求用户手工标注大批 face ID。

## 必须交付

- `audit_paper_v1_provenance.py` 或等价可执行脚本；
- `evaluate_paper_v1_recognition.py` 或等价统一评价入口；
- 新增的必要单元测试和回归测试；
- `reports/paper_v1_dataset_audit.md`；
- `reports/paper_v1_recognition_baseline.md`；
- `reports/paper_v1_rotation_consistency.md`；
- `reports/paper_v1_p1/model_data_provenance.json`；
- `reports/paper_v1_p1/metrics.json` 和扁平 CSV；
- `reports/paper_v1_p1/human_review_guide.zh-CN.md` 及审查清单；
- clean candidate 配置与冻结 manifest。

## Gate P1 的 Agent 侧通过标准

- provenance 对所有启用模型均给出可追溯结论，clean 流程不含 S3/Real-Dev 训练记录；
- 无 base geometry 或旋转派生跨 split；若发现泄漏，重新建立合法 split 并重跑，不允许只在报告中提醒；
- 正式 Synthetic-Test STEP 全部通过既定 solid validity，或明确列出被质量门剔除的比例和原因；
- 同一 clean 配置重复运行两次，指标和逐样本预测一致；
- 完整测试通过；
- Real-Dev 失败只分类和统计，不新增逐件规则；
- 生成可由用户实际执行的 20+10 人工审查包。

## 完成后的回复格式

严格按主计划第 14.1 节汇报：任务编号、修改文件、实际命令、测试结果、核心指标、报告绝对路径、遗留问题、范围触碰声明和下一步建议。完成后停止，不得自行进入 P2。
