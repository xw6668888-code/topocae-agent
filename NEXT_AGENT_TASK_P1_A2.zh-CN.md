# 下一 Agent 任务：P1-A2

## 任务名称

`P1-A2：clean baseline 选择纠偏与阶段消融`

## 必读文件

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\reports\p1_supervisor_gate_review.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\paper_v1_recognition_baseline.md`
4. `W:\python\FEA\dataset_generator\reports\paper_v1_p1\model_data_provenance.json`

## 唯一目标

在不进入 P2、不改变研究范围的前提下，纠正 P1-A1 的测试集选择错误，完成已有识别阶段的严格消融，并冻结真正有效的 clean geometry recognizer。

## 禁止事项

- 不读取或使用任何 S3 几何、真值、预测或派生训练数据。
- 不使用 Real-Dev 训练、选模型、选阈值或新增规则。
- 不再使用原 Synthetic-Test 选择任何方案；将其重命名为逻辑上的 `Synthetic-Development-Test`，历史结果只作诊断。
- 不修改 GNN 架构，不新增大模型，不开始 CAE-IR、pointer、Abaqus agent 或 Real-Holdout。
- 不针对单个零件或单个类别增加经验规则。
- 不覆盖 P0/P1-A1 的配置、manifest、报告和模型。

## Task A：建立完整阶段消融

在完全相同的 Synthetic-Val 上评价：

- `A0 Raw-GNN`；
- `A1 Raw-GNN + topology postprocess`；
- `A2 Raw-GNN + clean candidate classifier`；
- `A3 Raw-GNN + clean candidate classifier + topology postprocess`；
- `A4 Val 最优基础方案 + clean rib proposer/boundary`。

所有 variant 必须走同一 feature-object exporter 和同一 metric 实现。至少报告 face Macro-F1、semantic Macro-F1、object micro/macro P/R/F1、exact instance ratio、instance IoU、per-class object P/R/F1、same-instance edge F1，以及合成旋转验证集的一致性。

为 candidate、topology 和 rib 每个阶段生成 decision delta 日志：修改的 face/object 数量、correct->incorrect、incorrect->correct、unchanged correct、unchanged incorrect，并按规则名和类别汇总。明确解释 stepped hole、web、through hole 和 fillet 的退化发生在哪个阶段。

## Task B：只在 Val 上冻结方案

选择协议必须在读取新 Holdout 前写入并哈希冻结：

1. 主指标：object Macro-F1。
2. 约束：相对 A0，face Macro-F1 和 semantic Macro-F1 各自下降不得超过 `0.005`。
3. 约束：任何 Val support >= 10 的目标类别，不允许从非零 recall 退化为零。
4. 约束：合成旋转一致性不得下降超过 `0.005`。
5. 并列时依次选择 exact instance ratio 更高、instance IoU 更高、阶段更少的方案。

在 `reports/paper_v1_p1_a2/preholdout_selection.json` 中记录所选 variant、全部指标、配置 SHA256、模型 SHA256、选择代码 SHA256 和选择理由。生成该文件后不得再改变方法或阈值。

如果 A1-A4 均不满足约束，必须选择 `A0 Raw-GNN`。这种情况下保留 topology rules 作为后续非改写式合法性 verifier，而不是强制重标分类器。

## Task C：最多一轮通用 contract 修复

仅当阶段日志证明存在通用接线错误时，允许一轮最小修复，例如：

- 不支持的 macro 不得因 candidate label space 缺失而被 demote/remap；
- semantic subtype 必须与最终 macro 的合法 subtype 集一致；
- 后处理不得在没有满足硬拓扑前提时覆盖高置信正确预测。

修复必须只依据 Synthetic-Val，必须有回归测试；禁止加入 part ID、面积常数或类别专用补丁。修复后重新运行 A0-A4 并立即冻结。若仍不满足选择约束，选择 A0，停止修复。

## Task D：创建全新 Synthetic-Holdout

在 `preholdout_selection.json` 写入并校验完成之后，生成 `1000-1500` 个 class-balanced 合成样本作为 `Synthetic-Holdout-v1`：

- 使用未出现在原 Train/Val/Test 的新 seed；
- 与全部原数据执行 base provenance、旋转不变 exact/near graph signature 检查，碰撞为 0；
- STEP 必须全部通过单实体 solid validity；失败样本由质量门重试替换，不能进入 Holdout；
- Holdout 只能运行冻结 variant 一次，不得据此改模型、阈值或规则。

保存 Holdout 生成配置、seed 区间、graph/STEP manifest、质量报告和指纹审计。

## Task E：正式评价和重新冻结

对冻结 variant 运行：

- 新 `Synthetic-Holdout-v1`；
- Real-Dev：仅报告，不选择；
- Rotation-Dev：仅报告，不选择。

运行两次重复性检查。新建而不覆盖：

- `configs/paper_v1_clean_candidate_v2/`
- `reports/paper_v1_p1_a2/`
- `paper_v1_clean_candidate_v2_freeze_manifest.json`

重新生成 20 个合成样本和 10 个 Real-Dev 样本的人工审查包，必须对应最终 v2 variant。旧审查包标记为 superseded，不删除。

## 必须测试

- 全项目测试；
- 五阶段启停组合测试；
- unsupported macro preservation；
- semantic/macro compatibility；
- Val-only selection guard；
- preholdout manifest 必须早于 Holdout 输出且哈希一致；
- S3/Real-Dev 禁止训练与选择；
- 两次正式推理规范化输出完全一致。

## Agent 侧 Gate P1-A2

- 选择只使用 Synthetic-Val，不读取 Holdout 指标做决策；
- 新 Holdout 至少 1000 个有效单实体且与旧数据指纹碰撞为 0；
- 最终 variant 满足 Val 选择约束；若规则链不满足则诚实冻结 A0；
- Holdout 报告不隐藏零召回类别和失败案例；
- provenance、配置、模型、代码、数据和报告均进入新 manifest；
- 完成后停止，不进入 P2，按主计划第 14.1 节汇报。
