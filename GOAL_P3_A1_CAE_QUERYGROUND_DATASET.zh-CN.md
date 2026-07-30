# Goal P3-A1：CAE-QueryGround 数据集构造与冻结

## 一、Goal

在不修改 P1 Recognizer 和 P2 CAE-IR 冻结资产的前提下，利用 P1-B 5000 件合成 STEP/B-Rep graph 的 Oracle feature-instance 与 semantic-role 真值，构造可复现、无几何泄漏、无语言模板泄漏的 `CAE-QueryGround v1` 数据集。

每个可执行 feature-targeted 样本必须形成以下可追溯链：

```text
source STEP/BREP + graph
-> Oracle feature object / semantic face role
-> exact graph-local target face set
-> canonical CAE-IR
-> split-specific natural-language rendering
-> expected CAE operation
```

本 Goal 是数据集阶段，不是 LLM、Pointer、Recognizer 修复或 Abaqus 阶段。

## 二、开始前必须完整阅读

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\P2_A1_SUPERVISOR_ACCEPTANCE.zh-CN.md`
4. `W:\python\FEA\dataset_generator\reports\p2_a1_cae_ir_v1\final_acceptance.zh-CN.md`
5. `W:\python\FEA\dataset_generator\reports\p2_a1_cae_ir_v1\freeze\freeze_manifest.json`
6. P2 冻结目录中的三个 schema、role policy、alias、annotation guide 和 approved requirements。
7. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\final_acceptance.zh-CN.md`
8. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\freeze\capability_contract.json`
9. `W:\python\FEA\dataset_generator\reports\p1_b_final_acceptance_v1\acceptance.json`
10. `W:\python\FEA\dataset_generator\output_generation_p1_b_corrected_balanced_5000_v3\dataset_meta.json`
11. `W:\python\FEA\dataset_generator\output_generation_p1_b_corrected_balanced_5000_v3\manifests\artifact_manifest.json`
12. `W:\python\FEA\dataset_generator\output_generation_p1_b_corrected_balanced_5000_v3\manifests\sample_manifest.jsonl`
13. 该数据集的 `splits\train.jsonl`、`splits\val.jsonl`、`splits\development_test.jsonl` 和少量代表性 graph；不得为了熟悉格式遍历无关历史输出。

先输出 `reports\p3_a1_cae_queryground_v1\preflight_summary.zh-CN.md`，说明实际输入、哈希、已知限制和实现路径，再开始写代码。

## 三、冻结输入与只读约束

### 3.1 P2 输入

- CAE-IR version：`1.0.0-h2.01`。
- Canonical feature：17 类。
- `web` 只能在数据入口规范化为 `rib`，不得作为第 18 个 CAE-IR feature 输出。
- Operation：`fixed_support`、`displacement`、`pressure`、`surface_traction`、`material_assignment`、`global_mesh_size`、`local_mesh_size`。
- 仅使用 H2-01 批准的 feature-role-operation 组合。
- 不得把 `tests/fixtures/cae_ir_v1/cases.jsonl` 的 `description` 字段当作语言数据。可以只把 fixtures 用作 schema 回归测试。

### 3.2 几何输入

唯一主数据源：

`W:\python\FEA\dataset_generator\output_generation_p1_b_corrected_balanced_5000_v3`

必须保持现有 `base_geometry_family_id` 分组和 split：

- `train -> train`
- `val -> val`
- `development_test -> test`

同一基础几何、旋转派生、尺度派生、exact graph signature 和 near graph signature 不得跨 split。

### 3.3 只读资产

不得修改或覆盖：

- P1-B、P1-C、P1-D Lite 的 checkpoint、阈值、代码、报告、manifest 或历史结果；
- P2 schema、policy、fixtures、review CSV、freeze 和 final acceptance；
- 上述 5000 件源数据的 graph、geometry、manifest 和 split。

所有新输出只能写到本 Goal 的新路径。

## 四、明确禁止

1. 不使用 superpower skills。
2. 不读取、搜索、运行或评价 `test_part_s3`。
3. 不读取或运行任何已冻结的一次性 Synthetic-Holdout / Real-Holdout。
4. 不访问 `test_part`、`test_part_s`、`test_part_s1` 或 `test_part_s2`；P3-A1 只构造合成 Oracle 数据。
5. 不调用在线或本地 LLM，不使用 LLM 改写，不做 prompt engineering。
6. 不训练或微调 GNN、candidate classifier、Pointer 或任何语言模型。
7. 不执行 Abaqus/CalculiX，不生成可执行求解任务。
8. 不修改 CAE-IR 1.0.0-h2.01；遇到无法表达的语义必须降级为 clarification/invalid，不能偷偷扩 schema。
9. 不让自然语言或 CAE-IR 包含 sample ID、part ID、文件名、graph-local face ID、instance ID、真值面积字段或 geometry source。
10. 不把 Oracle target 存入纯 CAE-IR；它只能位于数据记录的独立监督标签区。

## 五、Task 1：输入审计与 Role Crosswalk

实现只读输入审计，至少完成：

1. 验证 5000 条 sample manifest、graph 文件存在性和每图 SHA-256。
2. 验证源 split 计数、`base_geometry_family_id`、exact/near signature 的跨 split 碰撞均为 0。
3. 验证 graph 中 `feature_sets.node_indices`、node `instance_id`、`macro_feature` 和 `semantic_role` 一致。
4. 建立机器可读的 `semantic_role -> P2 face_role` crosswalk。
5. `feature_region` 取完整实例 face set；direct/composite role 只能由冻结 policy 的 semantic subtype candidates 组合得到。
6. `unsupported_no_distinct_subtype` 角色不得生成精确正样本。
7. `web` 的 macro 与 semantic roles 必须显式映射到 canonical `rib`，并保存 provenance；不得仅改字符串后假设角色自然一致。
8. 每个 exact target face set 必须非空、排序、去重，且是对应 feature instance 的子集；完整实例角色必须等于实例全部节点。

输出：

- `configs\paper_v1\cae_query_role_crosswalk.yaml`
- `reports\p3_a1_cae_queryground_v1\source_and_role_audit.json`
- `reports\p3_a1_cae_queryground_v1\source_and_role_audit.zh-CN.md`

如果任一 source hash 不匹配或 role crosswalk 需要发明 ontology 中不存在的标签，立即停止并报告，不得继续生成。

## 六、Task 2：数据记录 Schema

创建：

- `schemas\cae_query_ground_record.schema.json`
- `schemas\cae_query_ground_dataset_manifest.schema.json`
- `docs\cae_query_ground_annotation_guide.zh-CN.md`

记录必须严格分离四个区域：

1. `source_ref`：数据内部 provenance，可含 sample ID、graph hash、base group、split；不进入 query 或 CAE-IR。
2. `query`：自然语言、语言、template ID、rendering ID 和难度级别；不得泄漏内部 ID。
3. `intent_result`：必须严格通过冻结 `cae_intent_result.schema.json`。
4. `ground_truth`：仅监督标签，可含 graph-local feature instance ID、target face indices、semantic evidence 和 expected operation target。

必须支持三种 target contract：

- `feature_face_set`：VALID 且需要选择精确 feature object/face role；每个 operation 分别保存 target instance list 和 exact face set。
- `whole_part`：material/global mesh 等无需 face pointer 的操作；明确 `grounding_required=false`，不得伪造面选择任务。
- `non_executable`：NEEDS_CLARIFICATION/INVALID；不得携带可执行 intent 或 target face set。

`any_one` 没有唯一真值时不得作为 exact-face-set 正样本。只有可以表达完整 acceptable-target set 和相应评估契约时才能保留，否则生成 clarification。`count` 只有在匹配集合唯一且数量恰好满足时才可生成 VALID。

## 七、Task 3：Canonical Query Planner

实现主入口：

`build_cae_query_grounding_dataset.py`

建议把可测试逻辑拆到 `utils\cae_query_ground\`，主脚本只负责编排。

从每个 graph 的 Oracle feature objects 生成 4–8 个 canonical query plan，覆盖：

| Level | 语义 |
|---|---|
| L1 | 单 feature、单 role、单 operation |
| L2 | `all`、`exactly_one`、`count` 和多实例集合 |
| L3 | diameter/radius/depth/area/length/width/thickness 属性筛选 |
| L4 | largest/smallest/nearest 与明确 reference frame 的空间或比较关系 |
| L5 | 多 operation、结构化歧义、缺失信息和当前 schema 不支持的否定表达 |

生成约束：

1. 所有 VALID query 必须由 graph 真值唯一求解并通过 H2 role-operation policy。
2. 属性筛选只使用 graph/feature-set 中真实存在且单位可追溯的几何参数。
3. largest/smallest/nearest 只有在候选至少两个且排序间隔超过稳定 tolerance 时才可生成；并列必须生成 clarification。
4. 空间关系必须保存 global 或 part-local reference frame，且选择结果可重复计算。
5. `all` 可以对应多个 feature instance，ground truth 保存实例列表和面集并集。
6. 每个 operation 都有独立 target；多操作样本不得把不同 target 合成一个 face set。
7. 当前 CAE-IR 不支持的集合差、复杂布尔否定或关系链，不得伪装成 VALID。
8. 纯 whole-part 操作用于 intent parsing 对照层，不计入 face-grounding 主指标。
9. 随机性必须使用显式 seed；同一输入、配置和 seed 的 canonical JSONL SHA-256 必须一致。

## 八、Task 4：无 LLM 的双语 Rendering

实现受控模板 renderer，不调用 LLM。每个 canonical plan 默认生成：

- 2 个英文 rendering（论文主语言）；
- 1 个中文 rendering（工程辅助语言）。

要求：

1. `en-US` 与 `zh-CN` 均使用独立、人工可审查的 phrase/template bank。
2. train/val/test 使用互不重叠的 template family ID；不得只是给同一句话换编号。
3. 对生成文本做 normalized exact-text leakage 审计，跨 split 碰撞必须为 0。
4. renderer 前后的 canonical intent identity 必须不变；文本中数值、单位、数量词、方向和否定状态必须与 plan 一致。
5. 不能通过插入 sample ID、随机无意义 token 或文件名规避文本碰撞。
6. 33 条 H2 人工批准需求只能作为语言风格 seed/评估参考，不得机械复制到多个 split。
7. P2 的 8 条陈旧 fixture description 禁止作为模板来源。

输出模板配置及模板 split manifest，并提供机器检查器。

## 九、Task 5：Pilot 与人工 Gate H3-01

先生成 pilot，不得直接生成正式 20,000 条：

- 至少 800 条 canonical records；
- 至少 2400 条 bilingual renderings；
- 17 个 canonical feature、7 个 operation、L1–L5、VALID/CLARIFICATION/INVALID 均有覆盖；
- split 与来源分布必须在报告中列出。

生成人工审查包：

- `reports\p3_a1_cae_queryground_v1\human_review_package\cae_query_review_120.csv`
- `reports\p3_a1_cae_queryground_v1\human_review_package\README_H3_01.zh-CN.md`
- 可选只读 HTML 预览。

CSV 必须分层抽取 120 条，覆盖两种语言、五个 Level、全部 feature family、全部 operation 和三种状态。每行至少展示：query、简明 IR、feature/role/quantifier、operation、target instance 数、target face 数、为何唯一或为何需澄清，以及源 STEP/BREP 的只读路径。

预留人工字段：

- `review_decision`: `approve | edit | reject`
- `error_category`: `none | unnatural_language | wrong_cae_semantics | non_unique_target | wrong_target_role | wrong_value_or_unit | other`
- `corrected_query`
- `reviewer_note`

同时创建 `human_expression_additions_20.csv`，提供 20 个空白行，供用户补充 10–20 条常用自然表达；用户无需填写 face ID。

### H3-01 强制暂停

完成 pilot 和审查包后必须停止，向用户说明：

1. 打开 `cae_query_review_120.csv`；
2. 判断 query 是否自然、工程语义是否合理、IR 摘要是否等价、目标 feature/face role 是否明确；
3. 正确填 `approve`；可改文字但语义不变时填 `edit` 和 `corrected_query`；目标不唯一、工程不合理或映射错误时填 `reject` 并选 error category；
4. 不需要核对或填写 face ID；只有怀疑几何语义时才按源路径打开模型辅助判断；
5. 在 `human_expression_additions_20.csv` 中补充至少 10 条常用表达；
6. 保存后通知同一 Goal 继续。

未经用户真实回填，不得自动批准 H3-01，不得生成正式规模或冻结 P3。

## 十、Task 6：H3 后修订与正式规模

用户回复后：

1. 解析 120 条审核决定，拒绝空决定、非法枚举、edit 无 corrected query 和 reject 无原因。
2. 按错误类型修模板、planner 或 role crosswalk；不得逐 case 写 sample-specific 特判。
3. 重新生成 pilot 并做自动回归；人工发现的系统性问题必须有 regression test。
4. 将新增自然表达纳入 template/phrase bank，但仍保持 template split 隔离。
5. 生成正式数据：至少 20,000 条 canonical query-target records；每条 3 个 rendering，至少 60,000 条语言样本。
6. 输出 train/val/test JSONL、canonical-to-rendering index、dataset card、分布统计、质量过滤日志和完整 SHA-256 manifest。

推荐输出目录：

```text
output_cae_query_ground_v1/
  canonical/
    train.jsonl
    val.jsonl
    test.jsonl
  renderings/
    train.jsonl
    val.jsonl
    test.jsonl
  manifests/
  reports/
```

类别、角色、operation、level、language、status 和 target face-count 分布都必须报告。不得为达到 20,000 条而复制 payload 或仅改变内部 ID。

## 十一、Task 7：测试、审计与冻结

至少建立以下自动测试：

1. Schema strictness 和 JSON round-trip。
2. P2 intent/result schema 100% 通过。
3. H2 role-operation compatibility 100% 通过。
4. exact target face set 可由源 graph 与 crosswalk 重新计算，重建一致率 100%。
5. target faces 均存在、非重复，并满足 instance/role 边界。
6. query/intent identity leakage 为 0。
7. base/rotation/scale/exact/near signature split leakage 为 0。
8. template family 和 normalized exact text split leakage 为 0。
9. deterministic regeneration hash 一致。
10. 17 feature、7 operation、L1–L5、两种语言和三种 status 覆盖。
11. whole-part/non-executable 样本不携带伪造 face target。
12. 没有 placeholder、空 query、空 VALID target 或无法追溯的 target。

冻结版本建议为 `cae-query-ground/1.0.0-h3.01`。输出：

- `reports\p3_a1_cae_queryground_v1\dataset_card.zh-CN.md`
- `reports\p3_a1_cae_queryground_v1\quality_audit.json`
- `reports\p3_a1_cae_queryground_v1\split_leakage_audit.json`
- `reports\p3_a1_cae_queryground_v1\test_summary.json`
- `reports\p3_a1_cae_queryground_v1\freeze\freeze_manifest.json`
- `reports\p3_a1_cae_queryground_v1\final_acceptance.zh-CN.md`
- `reports\p3_a1_cae_queryground_v1\acceptance.json`
- `reports\p3_a1_cae_queryground_v1\delivery_manifest.json`
- `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P3_A1.md`

冻结 manifest 必须包含所有 generator、renderer、template bank、schema、crosswalk、tests、runner、输入引用哈希和人工审核证据；不得重复 P2 漏记 runner 的问题。

## 十二、Gate P3

只有全部满足才可提交监督验收：

1. H3-01 120/120 行已人工审核，审核后有效正确率至少 95%，系统性错误已修复并有回归测试。
2. 用户补充至少 10 条自然表达，并已记录采用或拒绝原因。
3. canonical records >= 20,000；renderings >= 60,000。
4. 目标 face set 自动追溯/重建成功率 100%。
5. Schema 与 H2 policy 合法率 100%。
6. 几何、旋转/尺度派生、graph signature、template family 和 normalized text 的跨 split leakage 均为 0。
7. 全部 17 feature、7 operation、L1–L5、两种语言和三种 status 有真实覆盖。
8. 数据内部 ID 仅存在 source/ground-truth 区，不进入 query 或 CAE-IR。
9. P1、P2 所有引用资产 SHA-256 变化为 0。
10. 未调用 LLM、未训练模型、未执行求解器、未运行 Holdout、未访问真实件或 `test_part_s3`。

任一条件失败，状态必须为 `NO_GO` 或 `BOUNDED-PASS` 并停止；不得进入 P4。

## 十三、向监督者提交的摘要格式

```text
P3-A1 状态：PASS / BOUNDED-PASS / FAIL
CAE-QueryGround version：<version>
canonical records：<count>
renderings：<count>
train / val / test：<counts>
English / Chinese：<counts>
L1 / L2 / L3 / L4 / L5：<counts>
VALID / CLARIFICATION / INVALID：<counts>
target reconstruction：<passed>/<total>
geometry/template/text leakage：<counts>
H3-01：<reviewed>/<120>，正确率 <rate>
P1/P2 frozen hash changes：<count>
禁止阶段访问：<none or details>
下一阶段建议：GO_P4_BASELINES / NO_GO
最终报告：<absolute path>
交付清单：<absolute path>
```

完成后停止，等待监督者审核，不得自行进入 LLM parser、deterministic grounder、Pointer 或 Abaqus 阶段。

