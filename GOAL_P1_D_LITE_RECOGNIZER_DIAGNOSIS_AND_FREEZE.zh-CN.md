# Goal P1-D Lite：受控域识别器诊断、最小修复与冻结

## 可直接用于新 Goal 的完整任务书

你正在 `W:\python\FEA\dataset_generator` 中执行 TopoCAE-Agent 第一篇论文的 P1-D Lite。你的工作是对旧版高分系统、P1-B 和 P1-C 做统一、可解释的分层诊断，完成一次 Oracle 瓶颈分解；只有证据充分时才允许做一次最小概念修复，随后冻结一个供 Agent 主线使用的受控域识别器。

本 Goal 的成功不等于把真实件识别修到很高。成功的定义是：在严格时间和范围内说明瓶颈、选出当前最佳可用流水线、冻结其入口和能力边界，并为下一阶段提供可靠接口。

## 一、开始前必须完整阅读

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\P1_B_P1_C_SUPERVISOR_AUDIT.zh-CN.md`
4. `W:\python\FEA\dataset_generator\GOAL_P1_B_GENERAL_GEOMETRY_RECOGNITION_REPAIR.zh-CN.md`
5. `W:\python\FEA\dataset_generator\GOAL_P1_C_CLEAN_GEOMETRY_FULL_REGEN_RETRAIN_COMPARISON.zh-CN.md`
6. `W:\python\FEA\dataset_generator\reports\p1_b_final_acceptance_v1\acceptance_summary.zh-CN.md`
7. `W:\python\FEA\dataset_generator\reports\p1_c_final_acceptance_v1\final_acceptance.zh-CN.md`
8. `W:\python\FEA\dataset_generator\reports\p1_b_real_dev_v1\report.zh-CN.md`
9. `W:\python\FEA\dataset_generator\reports\p1_c_real_dev_v1\report.zh-CN.md`
10. `W:\python\FEA\dataset_generator\output_real_step_s2_v5_6_a_rolefix\reports\s2_area_truth_validation_v5_6_a_mergeeval.md`
11. `W:\python\FEA\dataset_generator\output_real_step_s2_v5_6_a_rolefix\predictions\edge_aware_v5_6_a_candidate_topology_post\prediction_manifest.json`
12. `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P1_B.md`
13. `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P1_C.md`

阅读后先输出一份不超过两页的 `preflight_summary.zh-CN.md`，说明你理解的研究主线、三个待比较系统、Real-Dev 的开发集性质和本 Goal 的禁止事项，再开始运行实验。

## 二、不可违反的边界

1. 不得使用 superpower skills。
2. 不得读取、搜索、运行或评价 `test_part_s3` 及其任何派生输出。
3. 不得进入 LLM、CAE-IR、Pointer、Abaqus 编译或论文 Phase 2 以后的工作。
4. 不得重新生成数千样本，不得更换大型网络，不得扩大类别体系。
5. 不得覆盖 P1-B/P1-C checkpoint、一次性 Holdout、冻结报告和历史输出。
6. 不得重复运行 P1-B Holdout-v2 或 P1-C Holdout-v3；只读取冻结结果。
7. 不得把 `test_part`、`test_part_s`、`test_part_s1`、`test_part_s2` 再描述为未见测试集；它们统一叫 Real-Dev。
8. 不得加入 sample name、part ID、face ID、绝对面积常数、逐零件阈值或逐零件 if/else。
9. 新规则必须用旋转无关、尽量尺度无关的 B-Rep 关系表达，并具有正例、反例和旋转回归测试。
10. 正式推理不得读取 truth label、面积真值表或人工批注；truth 仅可用于离线评价和 Oracle 实验。
11. 使用 `W:\python\envs\feagpt\python.exe`。不得把依赖安装到 C 盘。
12. 所有新输出写入新目录，不修改历史报告。默认输出根目录：`reports\p1_d_lite_v1\`。
13. 最多允许一次概念级修复和一次完整复评。不得为了追分连续追加规则。
14. 若发现当前工作树已有非本 Goal 修改，保留并与之兼容，不得回退。

## 三、统一评价原则

旧 `v5_6_a_rolefix` 只有 `test_part_s2` 的完整结果，P1-B/P1-C 使用的 Real-Dev 范围更大。不得把不同样本、不同真值和不同匹配口径的数字直接放在同一列比较。

必须建立两个明确评价视图：

1. `S2-Overlap`：仅在各系统共同拥有、真值口径可统一的 `test_part_s2` 样本上比较。
2. `Real-Dev-Full`：使用 P1-B/P1-C 已冻结的 Real-Dev manifest，比较能在该 manifest 上运行的流水线。

评价器必须记录：样本清单、truth 来源、排除项、面积容差、connected same-macro merge 规则、是否具有 exact face-set truth。没有 exact face-set truth 时，不得伪造 face Macro-F1 或 exact face-set accuracy。

## 四、Task 0：预检、保护和可复现入口

1. 为以下资产计算 SHA-256 或读取现有冻结摘要：
   - P1-B/P1-C 最终 checkpoint、candidate model、配置和 label map；
   - P1-B/P1-C frozen reports；
   - 旧 `v5_6_a_rolefix` prediction/feature-object manifest；
   - Real-Dev manifest 和 truth 文件。
2. 审计 P1-B、P1-C 和旧 rolefix 的实际入口、checkpoint、candidate classifier、topology verifier/postprocess 和 decoder。
3. 在全新 Python 进程中各复现一个 synthetic-dev 样本和一个 Real-Dev 样本。
4. 检查 import-time monkeypatch/global rebind 风险，但本 Goal 只记录与复现有关的问题，不做无关架构重构。
5. 写出 `reports\p1_d_lite_v1\preflight\`：
   - `preflight_summary.zh-CN.md`
   - `protected_inventory.json`
   - `pipeline_inventory.json`
   - `environment.json`
   - `reproduction_commands.json`

## 五、Task 1：旧版高分机制与四层流水线分解

对 P1-B、P1-C、旧 `v5_6_a_rolefix` 和当前可运行的最佳流水线进行分层分析：

```text
L0 raw GNN
L1 candidate proposal/classifier
L2 topology verifier/postprocess
L3 feature-object decode
```

最低必须报告：

- 每层保留/修改/新增/删除的 face 和 candidate 数量；
- raw-to-final macro changed-face ratio；
- proposal recall@IoU 0.5/0.75/0.95，前提是存在可靠 face-set truth；
- candidate conditional macro accuracy/F1；
- membership IoU 和 exact boundary ratio；
- verifier false-veto、false-accept 和 demotion 数量；
- decoder merge/split/conflict loss；
- final object precision、recall、F1；
- hole、slot、rib/web、transition 四个家族的单独结果；
- 每层运行时间。

必须回答：

1. 旧 `v5_6_a_rolefix` 的 1.0 有多少来自 raw GNN、candidate classifier 和 topology postprocess？
2. P1-C 相比 P1-B 的真实件退化首先发生在哪一层？
3. 是 face classification、candidate coverage、candidate macro、verifier veto 还是 decoder boundary 在主导失败？

输出：

- `reports\p1_d_lite_v1\layered_comparison\metrics.json`
- `reports\p1_d_lite_v1\layered_comparison\per_class.csv`
- `reports\p1_d_lite_v1\layered_comparison\per_sample.csv`
- `reports\p1_d_lite_v1\layered_comparison\report.zh-CN.md`

## 六、Task 2：轻量 Oracle 瓶颈分解

优先复用现有人工审查、面积真值、face_count 和已经冻结的 graph，不要求用户重新标注所有 face ID。

先将 Real-Dev truth 对象自动映射到候选 face set：

- 使用 macro、face_count、面积容差、连通性、primitive 和 edge/loop 关系枚举；
- 唯一解记为 `auto_unique`；
- 多解记为 `ambiguous`，不得擅自选真值；
- 无解记为 `unresolved`。

在可靠子集上运行：

1. `O0 current_full`：当前完整流程；
2. `O1 oracle_face`：真值 face macro/role + 当前后续流水线；
3. `O2 oracle_candidate_membership`：当前 GNN + 真值 candidate face set；
4. `O3 oracle_candidate_macro`：当前 candidate membership + 真值 macro；
5. `O4 oracle_verifier_decode`：当前提议/分类 + 真值 accept/conflict decision。

必须报告每个 Oracle 相对 O0 的 object F1 增益，并据此排列首要、次要瓶颈。

### 唯一允许的人为 Gate H-Lite-1

仅当 `ambiguous + unresolved` 阻碍对主要家族作出可信 Oracle 判断时，生成：

- `human_review_ambiguous_face_sets.csv`
- `human_review_ambiguous_face_sets.html`
- `README_H_LITE_1.zh-CN.md`

然后暂停并告诉用户：只检查 CSV 中 ambiguous/unresolved 行；在 HTML 中查看高亮候选；填写 `accept_candidate_id` 或 `exclude_from_oracle`。不得要求用户重新标注全部模型，也不得要求用户填写大量 raw face ID。

若现有可靠子集已经足以判定瓶颈，则不触发人工 Gate。

输出：

- `reports\p1_d_lite_v1\oracle\truth_reconciliation.json`
- `reports\p1_d_lite_v1\oracle\metrics.json`
- `reports\p1_d_lite_v1\oracle\per_class.csv`
- `reports\p1_d_lite_v1\oracle\report.zh-CN.md`

## 七、Task 3：一次最小修复或有证据地不修

根据 Task 2 选择增益最大的瓶颈。只允许以下两种结论之一：

### 结论 A：实施一次修复

仅当满足全部条件时实施：

- Oracle 增益明确且位于一个可修改模块；
- 修复能用通用 B-Rep 关系表达；
- 不依赖零件身份、绝对面积或真值输入；
- 预计不需要重建大规模数据或更换网络；
- 能建立正例、反例、旋转、缩放和面划分回归测试。

允许修复的层只有一个：candidate proposer、candidate classifier 输入/决策、topology verifier/postprocess 或 decoder。完成后只运行一次完整复评。

### 结论 B：不修复并冻结当前最佳

出现以下任一情形必须选择 B：

- 主要瓶颈需要重做数据生成或大规模训练；
- Oracle 增益分散，没有单一高收益模块；
- 修复只能靠阈值迎合 Real-Dev；
- 一次修复没有带来明确收益；
- 修复使 frozen synthetic 指标明显下降。

输出：

- `reports\p1_d_lite_v1\minimal_fix\decision.zh-CN.md`
- 若有修改：代码、自动测试、`before_after.json` 和 `regression_results.json`
- 若无修改：给出可证伪的原因和后续 Agent 阶段如何使用 abstention/Oracle 双轨规避风险

## 八、Task 4：选择并冻结 Recognizer-Dev

候选仅限：

1. P1-B frozen；
2. P1-C frozen；
3. 旧 `v5_6_a_rolefix` 流水线；
4. P1-D Lite 唯一修复后的流水线。

选择标准：

- Real-Dev object F1；
- 四个家族的最低召回率；
- synthetic frozen 指标是否保持；
- rotation consistency；
- 是否存在 truth leakage、逐件规则或无法复现的全局状态；
- 是否能输出置信度、失败原因和 abstention。

旧 rolefix 可以作为候选，但其 `test_part_s2` 高分只能解释为开发域性能，不能作为外推证据。

### P1-D Lite 停止/冻结条件

满足以下任一条件即停止继续修识别，并冻结最佳流水线：

1. Real-Dev object F1 `>= 0.60`，且 hole/slot/rib-web/transition 四个家族均不是完全失效；
2. 相比 P1-C，Real-Dev object F1 提升 `>= 0.08`；
3. Oracle 证明主要瓶颈需要长期重做数据或 proposer，无法用本 Goal 的一次修复可靠解决。

第三种结果属于“有边界地完成”，不是失败。必须冻结当前最佳可用模型并标记 `LIMITED_CONTROLLED_DOMAIN`，后续 Agent 论文使用 Oracle/predicted 双轨评价。

冻结交付：

- `reports\p1_d_lite_v1\freeze\recognizer_dev_manifest.json`
- `reports\p1_d_lite_v1\freeze\recognizer_dev_manifest.sha256.json`
- `reports\p1_d_lite_v1\freeze\capability_contract.json`
- `reports\p1_d_lite_v1\freeze\known_limitations.zh-CN.md`
- 一个统一只读推理入口或对现有入口的明确调用契约；不得复制或覆盖历史 checkpoint

`capability_contract.json` 至少声明：输入 graph schema、输出 feature-object schema、目标类别、置信度、abstention、已知支持范围和错误码。

## 九、Task 5：最终验收报告

生成：

- `reports\p1_d_lite_v1\final_acceptance.zh-CN.md`
- `reports\p1_d_lite_v1\acceptance.json`
- `reports\p1_d_lite_v1\delivery_manifest.json`
- `reports\p1_d_lite_v1\reproduction_commands.md`
- `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P1_D_LITE.md`

最终报告必须明确回答：

1. 旧 s2 高分是如何形成的，哪些结论可用于论文，哪些不能？
2. P1-B/P1-C/旧 rolefix/P1-D Lite 中应冻结哪一个，为什么？
3. 当前几何识别对 Agent 主线是否已足够？
4. 哪些情况必须 abstain 或请求人工确认？
5. 是否需要在进入 Agent 后再补 Mixed-Context 数据？若需要，只给出有证据的类别和规模，不执行大规模生成。
6. 下一阶段是否应立即进入 CAE-IR，结论必须为 `GO` 或 `NO-GO`。

## 十、测试和工程要求

1. 所有新评价、Oracle 映射和修复逻辑必须有 unittest。
2. 测试至少覆盖：正常输入、缺 truth、歧义 truth、旋转等价、缩放等价、无候选、错误 schema 和 abstention。
3. 任何正式命令都记录到 reproduction 文件中。
4. 报告随机种子、硬件、运行时间和依赖版本。
5. 所有 JSON 必须可被标准解析器读取，CSV 使用 UTF-8 编码。
6. 完成后停止，不进入 CAE-IR 或其它阶段，等待监督者审核。

## 十一、向监督者提交时的最终摘要格式

```text
P1-D Lite 状态：PASS / BOUNDED-PASS / FAIL
冻结流水线：<name>
S2-Overlap object F1：<value>
Real-Dev-Full object F1：<value>
相对 P1-C 变化：<value>
主要瓶颈：<face/proposer/classifier/verifier/decoder>
是否实施最小修复：<yes/no>
修复收益：<value or N/A>
冻结能力等级：CONTROLLED_DOMAIN / LIMITED_CONTROLLED_DOMAIN
是否触发 H-Lite-1：<yes/no>
下一阶段建议：GO_CAE_IR / NO_GO
最终报告：<absolute path>
交付清单：<absolute path>
```
