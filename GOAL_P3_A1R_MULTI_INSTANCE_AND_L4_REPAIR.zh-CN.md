# Goal P3-A1R：多实例与 L4 Grounding 修复

## 一、Goal

在完全保留 P3-A1 `cae-query-ground/1.0.0-h3.01` 冻结资产的前提下，构造一个小而充分的同类多实例几何补充集，补齐以下真实任务：

1. L2：同一零件中多个同类 feature 的 `all` 与 `count` 选择；
2. L4：同类候选之间的 `largest`、`smallest` 和 `nearest` 精确选择；
3. 形成无泄漏、可重建、双语、可复现的 `CAE-QueryGround 1.1`；
4. 修正测试，使“level 标签出现”不再被误认为“任务语义覆盖”。

本 Goal 只修数据与评估契约，不进入 P4，不调用 LLM，不训练任何识别或语言模型。

## 二、开始前必须完整阅读

1. `W:\python\FEA\dataset_generator\reports\P3_A1_SUPERVISOR_ACCEPTANCE.zh-CN.md`
2. `W:\python\FEA\dataset_generator\GOAL_P3_A1_CAE_QUERYGROUND_DATASET.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\p3_a1_cae_queryground_v1\final_acceptance.zh-CN.md`
4. `W:\python\FEA\dataset_generator\reports\p3_a1_cae_queryground_v1\freeze\freeze_manifest.json`
5. `W:\python\FEA\dataset_generator\reports\p3_a1_cae_queryground_v1\quality_audit.json`
6. `W:\python\FEA\dataset_generator\reports\p3_a1_cae_queryground_v1\source_and_role_audit.json`
7. `W:\python\FEA\dataset_generator\output_cae_query_ground_v1\manifests\dataset_manifest.json`
8. P1-B 的 slot、rib/web、多特征非干涉生成与 graph-v2 相关实现和必要测试；只读取解决本 Goal 所需的文件。
9. P2 冻结 CAE-IR schema、role policy 和 annotation guide。

先生成 `reports\p3_a1r_multi_instance_v1\preflight_and_defect_reproduction.zh-CN.md`，逐项复现监督报告中的计数，再开始修复。

## 三、冻结与禁止范围

### 3.1 只读冻结

不得修改、覆盖或重新冻结：

- `output_cae_query_ground_v1/`
- `reports/p3_a1_cae_queryground_v1/`
- P3-A1 freeze 中的 101 项资产；
- P1/P2 checkpoint、policy、schema、review、manifest、阈值和历史报告；
- `output_generation_p1_b_corrected_balanced_5000_v3/`。

开始和结束时均核对旧 P3 101 项哈希，变化必须为 0。

### 3.2 禁止事项

1. 不使用 superpower skills。
2. 不访问、搜索、运行或评价 `test_part_s3`、真实件和一次性 Holdout。
3. 不调用本地或在线 LLM，不使用 LLM 生成语言。
4. 不训练/微调 GNN、candidate classifier、Pointer 或语言模型。
5. 不执行 Abaqus/CalculiX。
6. 不为了通过计数复制记录、只改 ID、只改数值尾数或伪造 multi-instance metadata。
7. 不把多个不同 STEP 的 feature 拼成一个虚假的多实例 graph。
8. 不把不同 feature type 放在同一 family 后跨类型比较；比较候选必须属于同一 canonical feature type。
9. 不进入 P4，不运行 P4 test split。

## 四、Task 1：缺陷回归测试先行

创建独立审计和测试，必须先在旧 P3 上正确报告：

```text
valid_feature_targets = 20700
candidate_count_ge_2 = 0
count_quantifier = 0
valid_L4 = 0
```

新增测试不能只检查 level/status 的 key 是否存在，必须检查：

- L2 executable 样本中 `candidate_count>=2`；
- `all` 的 `selected_count>=2`；
- `count` 存在且 count 与 selected_count 一致；
- L4 VALID 存在；
- largest/smallest/nearest 均有正样本；
- L4 候选数至少 2，selected_count 必须为 1；
- comparison/nearest 的 metric evidence 可以从对应 graph 重算；
- 并列或稳定间隔不足时仍返回 clarification。

旧 P3 应被该“完整任务 Gate”判定为 `SEMANTIC-COVERAGE-FAIL`，但旧文件本身不做修改。

## 五、Task 2：同类多实例几何补充集

新建：

`output_generation_p3_a1r_multi_instance_v1/`

使用现有受审计的特征生成器和 graph-v2 extractor，生成真正的单零件多实例模型。要求：

1. 每个模型包含 2-4 个同一 canonical feature type 的独立实例。
2. 实例不得重合或相互破坏；feature-set、instance_id、semantic role 和面标签完整。
3. STEP/BREP 必须是单个有效 solid；不得输出片体或多实体残片。
4. 同类实例的尺寸或空间位置必须具有稳定差异，使 largest/smallest/nearest 可唯一求解；同时保留一小部分刻意并列样本用于 clarification。
5. 不要求一个模型堆很多不同 feature；本补充集的目的就是隔离多实例与比较关系。
6. 模型坐标、尺寸和布局可变化；若生成旋转/尺度派生，必须与 base geometry 保持同 split。
7. graph 中保存可重算的 area、diameter/radius、depth、length/width/thickness、centroid 和 part reference 信息；缺少某指标时不得伪造。

规模下限：

- 17 个 canonical feature 均至少有 20 个可用于 `all/count` 的 base models；
- 至少 12 个 feature type 可用于 largest/smallest/nearest；
- accepted base models 不少于 510，目标约 600；
- train/val/test 约为 80/10/10，并按 base/signature group 划分；
- 每个 split 都必须有真实 multi-instance positive。

对 rib、fillet、chamfer 等不适合某一比较指标的类别，只使用可解释的 area 或 nearest；不得为追求指标覆盖制造不合理工程语义。

输出 dataset meta、sample manifest、split、逐图 SHA-256、类/实例/面统计、几何有效性审计和 split leakage audit。

## 六、Task 3：P3-A1R Pilot 与人工 Gate H3R-01

先生成 60-100 个补充模型 pilot，不得直接生成正式 510+：

- 覆盖全部 feature family；
- 至少包含 12 个 canonical feature；
- 每个模型 2-4 个同类实例；
- 包含 all、count、largest、smallest、nearest 和 tie/clarification。

生成人工审查包：

```text
reports/p3_a1r_multi_instance_v1/human_review_package/
  geometry_review_20.csv
  query_review_60.csv
  README_H3R_01.zh-CN.md
```

### 用户需要做的工作

`geometry_review_20.csv`：

1. 按路径在 UG 中打开 20 个 STEP；
2. 检查是否确实存在 2-4 个同类特征；
3. 检查实例是否互不破坏、零件是否单实体、尺寸差和位置关系是否清楚；
4. 填写 `PASS | MINOR | BAD`，MINOR/BAD 写一句原因。

`query_review_60.csv`：

1. 检查英文/中文 query 是否自然；
2. 检查 all/count/largest/smallest/nearest 是否与候选数量、尺寸排序和空间参考一致；
3. 检查 tie 情况是否要求澄清；
4. 填写 `approve | edit | reject`；不需要填写 face ID。

### 强制暂停

生成审查包后停止。未经用户真实回填，不得扩大正式数据、冻结 1.1 或进入 P4。

## 七、Task 4：多实例 Query Supplement

H3R-01 通过后，从正式补充几何生成 query records。最低要求：

### 7.1 L2

- executable multi-instance L2 >= 900；
- `all` 且 selected_count>=2：>= 450；
- `count` 且 count=selected_count>=2：>= 450；
- 17 个 canonical feature 均有真实 all/count 样本；
- train/val/test 均有 all 和 count。

### 7.2 L4

- executable L4 >= 900；
- `largest >= 300`、`smallest >= 300`、`nearest >= 300`；
- 每条 candidate_count>=2、selected_count=1；
- 至少 12 个 feature type 和全部主要 feature family 有 L4 正样本；
- comparison metric 至少覆盖 4 种真实几何量，并单独覆盖 nearest distance；
- train/val/test 中每种 L4 quantifier 均非零。

### 7.3 Hard negatives

- 保留旧 P3 的 200 条 L4 clarification；
- 新增稳定并列、缺 reference、缺 comparison metric 等结构化 clarification；
- 不允许用 INVALID/CLARIFICATION 数量冒充 L4 positive coverage。

### 7.4 Rendering

- 每条 canonical 仍为 2 英文 + 1 中文；
- 为 all/count/largest/smallest/nearest 提供 split-specific 的核心句法模板，不只是更换外层 wrapper；
- 模板 skeleton、family ID 和 normalized exact text 均做跨 split 审计；
- query/IR 中不得出现内部 identity。

## 八、Task 5：形成 CAE-QueryGround 1.1

新建而非覆盖：

```text
output_cae_query_ground_v1_1/
reports/p3_a1r_multi_instance_v1/
```

将旧 P3 基础数据与多实例 supplement 通过 manifest 引用或确定性重建合并。新版本建议：

`cae-query-ground/1.1.0-h3r.01`

要求：

1. 旧 21,000 条数据的语义与 target 不得被静默改变。
2. 旧 L2 中 candidate_count=1 的记录不得继续作为“multi-instance coverage”计数；可以保留为 singleton quantifier control，并在统计中单列。
3. 新增记录必须有独立 provenance，指向真实 supplement graph。
4. 数据卡必须分别报告 singleton、multi-instance、comparison、spatial、clarification 和 whole-part strata。
5. 报告所有 feature×quantifier、level×status、candidate_count、selected_count、operation、language 和 split 分布。
6. 使用一个明确的官方入口和一个只读验证入口；新 freeze manifest 只列实际依赖，不再把历史试验 wrapper 当作权威入口。

## 九、Task 6：自动测试与完整 Gate

除旧 P3 的全部检查外，必须新增：

1. 旧 P3 101 项 SHA-256 变化为 0。
2. supplement STEP/BREP/graph 100% 存在且 hash 匹配。
3. single-solid、label completeness、instance independence 通过。
4. all/count/large/small/nearest 达到 Task 4 的数量和 split 门槛。
5. multi-instance target reconstruction 100%。
6. comparison/nearest metric recomputation 100%。
7. tie/unstable comparison 不产生 VALID target。
8. geometry/base/signature/rotation/template skeleton/text leakage 为 0。
9. P2 schema 与 H2 policy 合法率 100%。
10. 全部 deterministic outputs 重跑逐字节一致。
11. P1/P2/P3-v1 冻结资产变化为 0。
12. 未调用 LLM、未训练模型、未执行 solver、未访问真实件/Holdout/test_part_s3、未进入 P4。

## 十、P3-A1R Gate

只有以下条件全部满足，才能提交监督验收：

- H3R geometry 20/20 完成，BAD=0；MINOR 已处理或有合理接受说明。
- H3R query 60/60 完成，修订后有效正确率 >=95%。
- 正式 supplement >=510 个有效 base models。
- L2 multi-instance、all、count 达到数量和 17 类覆盖门槛。
- L4 executable、largest、smallest、nearest 达到数量、split 和 feature-family 门槛。
- target/metric reconstruction、schema/policy、determinism、hash、leakage 全部通过。
- 旧 P3 101 项冻结资产变化为 0。
- 禁止阶段行为全部为 0。

完成后输出：

- `reports\p3_a1r_multi_instance_v1\final_acceptance.zh-CN.md`
- `reports\p3_a1r_multi_instance_v1\acceptance.json`
- `reports\p3_a1r_multi_instance_v1\delivery_manifest.json`
- `reports\p3_a1r_multi_instance_v1\freeze\freeze_manifest.json`
- `output_cae_query_ground_v1_1\manifests\dataset_manifest.json`
- `reports\p3_a1r_multi_instance_v1\dataset_card.zh-CN.md`
- `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P3_A1R.md`

最终状态只能是 `PASS / BOUNDED-PASS / FAIL`。完成后停止，等待监督者决定是否进入 P4，不得自行继续。

## 十一、监督摘要格式

```text
P3-A1R 状态：PASS / BOUNDED-PASS / FAIL
CAE-QueryGround version：<version>
旧 P3 frozen hash changes：<count>
multi-instance base models：<count>
L2 executable multi / all / count：<counts>
L4 executable / largest / smallest / nearest：<counts>
multi-instance target reconstruction：<passed>/<total>
metric recomputation：<passed>/<total>
H3R geometry：<reviewed>/20，BAD=<count>
H3R query：<reviewed>/60，correct rate=<rate>
geometry/template/text leakage：<counts>
P1/P2/P3-v1 frozen changes：<count>
禁止阶段访问：<none or details>
下一阶段建议：GO_P4_BASELINES / NO_GO
最终报告：<absolute path>
交付清单：<absolute path>
```

