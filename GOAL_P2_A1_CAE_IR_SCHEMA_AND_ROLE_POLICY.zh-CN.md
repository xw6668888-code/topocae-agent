# Goal P2-A1：CAE-IR Schema、单位系统与 Feature-Role Policy

## 可直接用于新 Goal 的完整任务书

你正在 `W:\python\FEA\dataset_generator` 中执行 TopoCAE-Agent 第一篇论文的 Phase 2。P1-D Lite 已由监督者以 `BOUNDED-PASS` 验收，Recognizer-Dev 已冻结为 P1-B `p1_b_general_recognition_pipeline_v4`，能力等级为 `LIMITED_CONTROLLED_DOMAIN`。

本 Goal 的唯一目标是：固定自然语言结构静力需求与几何 grounding 之间的强类型 CAE-IR，建立目标特征、可选面角色与允许 CAE 操作的工程语义策略，并通过人工 Gate H2-01 冻结版本。

本 Goal 不接入任何 LLM，不生成大规模语言训练集，不训练 Pointer，不执行 Abaqus，也不修改冻结几何识别器。

## 一、开始前必须完整阅读

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\P1_D_LITE_SUPERVISOR_ACCEPTANCE.zh-CN.md`
4. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\final_acceptance.zh-CN.md`
5. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\freeze\recognizer_dev_manifest.json`
6. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\freeze\capability_contract.json`
7. `W:\python\FEA\dataset_generator\CAE_feature_ontology.yaml`
8. `W:\python\FEA\dataset_generator\face_semantic_subtype_ontology.yaml`
9. `W:\python\FEA\dataset_generator\feature_recognition_methods.md`
10. `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P1_D_LITE.md`

阅读后先输出 `reports\p2_a1_cae_ir_v1\preflight_summary.zh-CN.md`，说明：研究任务、IR 与 Recognizer 的边界、目标操作、目标特征、Oracle/predicted/abstain 三轨要求和禁止事项。

## 二、不可违反的边界

1. 不得使用 superpower skills。
2. 不得读取、搜索、运行或评价 `test_part_s3`。
3. 不得修改 P1-B、P1-C、P1-D Lite 的模型、阈值、冻结报告、manifest 或历史输出。
4. 不得重跑任何一次性 Holdout。
5. 不得调用在线或本地 LLM，不得进行 prompt engineering 或模型微调。
6. 不得训练 Pointer、GNN 或候选分类器。
7. 不得执行 Abaqus/CalculiX；本阶段只定义可编译契约。
8. CAE-IR 不得包含 raw face ID、sample ID、part ID、模型文件名或面积真值。
9. 不得让 LLM 或用户意图选择 `oracle/predicted` 模式；几何来源属于评估/运行 envelope，不属于自然语言语义。
10. 不得假设缺失的载荷、材料、方向、单位或约束值；缺失信息必须结构化输出并要求澄清。
11. 使用 `W:\python\envs\feagpt\python.exe`。新依赖只有在现有环境无法用标准库或既有库完成时才允许安装。
12. 所有新输出写入新路径，不覆盖现有文件。
13. 到 H2-01 必须暂停；未获得用户签字不得自行冻结 P2，也不得进入 P3。

## 三、固定研究范围

### 3.1 分析类型

第一篇论文只支持：

- `static_structural`
- 单零件或可作为单零件处理的 STEP/B-Rep

### 3.2 目标特征

必须覆盖当前目标类别：

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
- `fillet`
- `chamfer`

上述实际为 17 类。实现时必须对主计划、ontology 和 capability contract 自动审计并报告数量差异，不能静默删类。`web` 若仍在旧 ontology 中，只能作为 `rib` 的兼容别名，不作为论文独立目标输出。

### 3.3 首版 CAE 操作

必须通过 discriminated union 明确定义：

- `fixed_support`
- `displacement`
- `pressure`
- `surface_traction`
- `material_assignment`
- `global_mesh_size`
- `local_mesh_size`

不得把点载荷、接触、螺栓预紧、焊接、非线性材料、动力学或复杂网格优化偷偷纳入首版。

## 四、Task 0：Ontology 与现有接口审计

1. 对比主计划、两个 ontology、Recognizer capability contract 中的目标类别和 semantic subtype。
2. 输出规范化类别表和 alias 表，明确 `web -> rib` 的兼容策略。
3. 审计每个 feature 的现有 face semantic subtype；没有对应 subtype 时必须列为 policy gap，不得凭空假装已有模型输出。
4. 审计已有 `W:\python\FEA\test` 原型中可复用的需求字段和 CAE 操作概念，但不得照搬关键词解析代码或旧密钥。
5. 输出：
   - `reports\p2_a1_cae_ir_v1\ontology_audit.json`
   - `reports\p2_a1_cae_ir_v1\ontology_audit.zh-CN.md`
   - `configs\paper_v1\cae_feature_aliases.yaml`

## 五、Task 1：设计并实现 CAE-IR v1

### 5.1 Schema 文件

创建：

- `schemas\cae_intent.schema.json`
- `schemas\cae_intent_result.schema.json`
- `schemas\cae_grounding_context.schema.json`

`cae_intent.schema.json` 表示已经语义完整、允许进入 grounding 的纯意图。建议使用 JSON Schema Draft 2020-12，所有对象默认 `additionalProperties: false`。

顶层至少包含：

- `schema_version`
- `analysis_type`
- `operations`
- `source_language`
- `assumptions`，默认必须为空；未经用户确认不得自动添加

每个 operation 使用 discriminated union，必须包含：

- `operation_id`
- `operation_type`
- 与该操作匹配的 typed target 或 whole-part scope
- typed parameters

Feature target 至少包含：

- `feature_type`
- `quantifier`: `all | any_one | exactly_one | count | largest | smallest | nearest`
- `count`，仅 `quantifier=count` 时允许
- `face_role`
- `filters`
- `reference_frame`，仅存在空间条件或方向时要求

Filters 首版至少支持：diameter/radius、depth、area、length、width、thickness、spatial relation 和 nearest-to reference。

任何数值条件必须为 typed quantity，包含 value、unit 和可选 tolerance；不得使用无维度裸浮点表示有物理量的字段。

### 5.2 Intent result

`cae_intent_result.schema.json` 必须明确区分：`VALID`、`NEEDS_CLARIFICATION`、`INVALID`。

非 VALID 结果不得夹带一个可执行的半合法 intent。必须提供结构化 `error_code`、`field_path`、`missing_information`、`ambiguities` 和 `clarification_questions`。

### 5.3 Grounding context

`cae_grounding_context.schema.json` 只用于实验和运行 envelope，至少包含：

- `geometry_source`: `oracle | predicted | human_confirmed`
- `recognizer_capability_level`
- `coverage_status`
- `abstention_reasons`
- `feature_object_schema_version`

`geometry_source` 不得出现在纯 CAE intent 中，也不得由自然语言解析器决定。

## 六、Task 2：单位、方向与验证 Python API

创建 `utils\cae_intent.py`，提供稳定公共 API：

```python
parse_intent_dict(...)
validate_intent(...)
normalize_intent_units(...)
validate_intent_result(...)
load_feature_role_policy(...)
check_role_operation_compatibility(...)
build_clarification_result(...)
```

要求：

1. 单位归一覆盖长度、面积、力、压力、位移、角度，并提供可追溯转换记录。
2. Abaqus 无内置单位制，不得把单位一致误写成 Abaqus 自动换算；IR 必须显式保存项目单位上下文。
3. 方向至少支持 global Cartesian、part-local Cartesian、`surface_normal`、`opposite_surface_normal`。
4. 压力默认只允许 normal 语义，但正负号和 inward/outward 不得凭空推断。
5. normalization 必须幂等。
6. 所有错误必须具有稳定 error code，不以自由文本作为程序判断条件。
7. 禁止使用 `eval`、动态执行或宽松 JSON 修复。

## 七、Task 3：Feature-Role-Operation Policy

创建 `configs\cae_feature_role_policy.yaml`。

每个目标 feature 至少定义：`family`、`canonical_name`、`aliases`、`selectable_roles`、semantic subtype 候选、`region_kind`、`allowed_operations`、`requires_complete_instance`、`may_include_transition`、`selection_notes` 和 `unsupported_or_ambiguous_cases`。

必须至少表达：

- 通孔壁；
- 盲孔壁与孔底；
- 阶梯孔的大孔壁、台阶面、小孔壁和孔底；
- 沉孔的大孔壁、台阶面和贯通小孔壁；
- 锥沉孔锥面与贯通孔壁；
- 长圆孔完整孔壁及其可区分子角色；
- 各类槽的侧壁、底部/曲面、封闭端面和完整槽内壁；
- pocket 侧壁、底部和完整腔内壁；
- 圆柱/矩形凸台顶面、侧壁；
- rib 的侧面、顶/端面和完整 rib region；
- fillet blend face；
- chamfer bevel face。

策略必须允许“完整 feature region”和“特定 face role”两种查询，但不得制造 ontology 中不存在的精确面角色。不存在时应声明需要组合现有 subtype 或当前不支持。

不要把 support/load 写成固有几何标签。同一个面在不同工况下可以允许 fixed、displacement、pressure 或 traction；policy 只判断操作与区域类型是否相容，不替用户决定工况。

## 八、Task 4：测试语料与自动测试

创建：

- `tests\test_cae_intent_schema.py`
- `tests\test_cae_intent_semantics.py`
- `tests\fixtures\cae_ir_v1\cases.jsonl`
- `reports\p2_a1_cae_ir_v1\test_case_summary.json`

至少建立 120 个独立用例：

- 至少 50 个 VALID；
- 至少 35 个 NEEDS_CLARIFICATION；
- 至少 35 个 INVALID；
- 全部目标 feature 和 operation type 均被覆盖；
- 至少 20 个单位换算、15 个 role-operation 负例、10 个多操作、10 个数量词或几何筛选用例。

测试必须覆盖 schema strictness、additional properties、identity 字段拒绝、单位维度、normalization 幂等、quantifier/count 联动、结构化澄清、role-operation compatibility、三轨 envelope 隔离、JSON round-trip，以及错误对象不得携带可执行 operations。

不得使用 placeholder 测试充数。报告实际 case 数、断言数和通过数。

## 九、Task 5：人工 Gate H2-01 审查包

代码和自动测试通过后，生成：

- `reports\p2_a1_cae_ir_v1\human_review_package\feature_role_policy_review.csv`
- `reports\p2_a1_cae_ir_v1\human_review_package\cae_requirement_review_40.csv`
- `reports\p2_a1_cae_ir_v1\human_review_package\README_H2_01.zh-CN.md`
- 可选 HTML 只读预览

Role policy 表每行一个 feature-role-operation 组合，包含工程解释和 `review_decision/corrected/reviewer_note`。

Requirement 表由 Agent 先生成 40 条候选需求，覆盖所有 feature family 和 operation type，至少 10 条故意不完整或有歧义。用户不需要从空白开始写，也不需要填写 face ID。

### 到此必须暂停

向用户说明：

1. 打开两个 CSV；
2. 每行填写 `approve | edit | reject`；
3. `edit` 时填写 corrected 字段；
4. 至少审核 30 条 requirement；
5. role policy 所有行必须审核；
6. 保存后通知 Goal 继续。

不得在没有用户回复的情况下自动批准 H2-01。

## 十、Task 6：H2-01 后冻结 P2

用户完成审查后：

1. 解析结果，拒绝空决定、非法枚举和矛盾修改。
2. 根据批准/修改更新 schema examples、policy 和 fixtures。
3. 重跑全部测试。
4. 冻结 schema、Python API、role policy、approved requirements 和 annotation guide。
5. 计算 SHA-256，不覆盖 Recognizer-Dev 资产。

输出：

- `docs\cae_intent_annotation_guide.zh-CN.md`
- `configs\paper_v1\cae_ir_version.json`
- `reports\p2_a1_cae_ir_v1\final_acceptance.zh-CN.md`
- `reports\p2_a1_cae_ir_v1\acceptance.json`
- `reports\p2_a1_cae_ir_v1\delivery_manifest.json`
- `W:\python\FEA\dataset_generator\CODEX_CONTEXT_P2_A1.md`

## 十一、P2-A1 自动 Gate

只有全部满足才可提交监督验收：

1. H2-01 已完成；role policy 无未审行，至少 30 条 requirement 已审。
2. VALID 通过率、INVALID 拒绝率和 CLARIFICATION 结构化返回率均为 100%。
3. 至少 120 个非 placeholder 用例全部通过。
4. 全部目标 feature 至少有明确 selectable role，或显式标为当前不支持并说明原因。
5. 全部 operation 都有单位、target scope 和缺失信息契约。
6. CAE-IR 无 raw face ID、sample/part/file identity；三轨只在 context envelope。
7. 单位 normalization 幂等且有审计记录。
8. 未调用 LLM、未训练模型、未执行 Abaqus、未运行 Holdout、未读取 `test_part_s3`。
9. Recognizer-Dev 13 个引用资产 SHA-256 全部不变。
10. 完成后停止，不进入 P3，等待监督者审核。

## 十二、向监督者提交的最终摘要格式

```text
P2-A1 状态：PASS / BOUNDED-PASS / FAIL
CAE-IR schema version：<version>
目标 feature 数：<count>
selectable role 数：<count>
operation type 数：<count>
测试用例：<passed>/<total>
VALID / CLARIFICATION / INVALID：<counts>
H2-01：<completed/not completed>
人工审核 requirement：<count>
未支持的 feature-role：<list or none>
Recognizer 冻结哈希变化：<0 or count>
禁止阶段访问：<none or details>
下一阶段建议：GO_P3_QUERYGROUND / NO_GO
最终报告：<absolute path>
交付清单：<absolute path>
```
