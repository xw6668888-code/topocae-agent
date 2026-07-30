# CAE-QueryGround v1 Pilot 标注与审查指南

版本：`cae-query-ground-record/1.0.0-h3.01-pilot`  
冻结上游：CAE-IR `1.0.0-h2.01`  
当前 Gate：`H3-01` 前 pilot

## 1. 数据记录的四区隔离

每条记录严格分为四区：

1. `source_ref`：仅内部 provenance。允许保存 sample ID、graph/geometry 只读路径、graph hash、base family、源 split、派生类型和 graph signature。
2. `query`：自然语言文本及语言、template family、template、rendering ID、variant 和 L1–L5。`query.text` 禁止出现 sample/part/file 名、graph-local face/instance ID、内部路径或几何真值字段名。
3. `intent_result`：必须完整通过冻结 `cae_intent_result.schema.json` 和 H2 role-operation policy。VALID 才能携带 intent；CLARIFICATION/INVALID 不能携带半合法 operation。
4. `ground_truth`：只用于监督与重建，允许保存 graph-local instance/face indices、source macro、semantic-role 证据和每个 operation 的 target contract。Oracle target 不得写回纯 CAE-IR。

## 2. 三种 target contract

### `feature_face_set`

- 仅用于 VALID、需要精确选面的 feature-targeted operation。
- 每个 operation 独立保存 `target_instance_ids`、`target_face_indices` 和 semantic evidence；多操作不能把不同 target 合成一个面集。
- face indices 必须非空、排序、去重，且每个 selected set 是对应 instance face set 的子集。
- `feature_region` 特例取完整 instance face set，必须与 `feature_sets.node_indices` 完全相等。
- direct role 只取冻结 policy 的 subtype candidates；composite role 必须由冻结 candidates 的并集构成。
- `unsupported_no_distinct_subtype` 不得生成精确正样本。

### `whole_part`

- 仅用于 `material_assignment` 和 `global_mesh_size`。
- 固定 `grounding_required=false` 与 `scope=whole_part`，不生成伪 face target。
- whole-part 对照记录用于 intent parsing，不计入 face-grounding 主指标。

### `non_executable`

- 对应 `NEEDS_CLARIFICATION` 或 `INVALID`。
- `targets` 必须为空；只能解释缺失、歧义或当前 schema 不支持的表达。
- 不允许保留“差一点可执行”的 intent、operation 或 face set。

## 3. Role crosswalk

机器可读 crosswalk 位于 `configs/paper_v1/cae_query_role_crosswalk.yaml`，完全从冻结 H2 policy 派生。

- `web` source macro 显式规范化为 canonical `rib`，但 source macro、`web_side_plane`、`web_top_plane` provenance 必须保留。
- `slot_terminal_face` 是源 graph 中的通用 terminal membership，冻结 P2 没有独立 selectable subtype。它只能随完整 instance 进入 `feature_region`，不能作为独立 face role，也不允许发明新 ontology 标签。
- `slot_rect.closed_end_face`、`slot_v.closed_end_face`、`rib.end_face` 均为 unsupported exact role。

## 4. 唯一性规则

- `exactly_one`：匹配集合必须恰为一个 instance。
- `all`：保存所有匹配 instances 与各 face set 的并集。
- `count`：匹配集合本身唯一，且实例数恰好等于 query count。
- `any_one`：pilot 不把任意实例当作唯一真值；没有完整 acceptable-target contract 时转 clarification。
- L3 属性筛选只使用 graph/feature-set 中真实存在的数值。模型坐标长度单位约定为 mm、面积为 mm²；BREP text 自身无单位，需同时保存 P1 synthetic OCCT unit contract provenance。
- largest/smallest/nearest 至少需要两个候选；最优项与次优项的差必须大于稳定 tolerance。并列或间隔不足转 clarification。
- nearest 与空间关系必须保存 `global_cartesian` 或 `part_local_cartesian` reference frame。

## 5. L1–L5

| Level | 语义 |
|---|---|
| L1 | 单 feature、单 role、单 operation，目标由唯一实例闭合 |
| L2 | all、exactly_one、count 与多实例集合 |
| L3 | diameter/radius/depth/area/length/width/thickness 属性筛选 |
| L4 | largest/smallest/nearest 和显式 reference frame |
| L5 | 多 operation、结构化歧义、缺失信息、unsupported complex negation |

## 6. 无 LLM 双语 rendering

- 每个 canonical plan 默认生成 2 个 `en-US` 和 1 个 `zh-CN` rendering。
- en-US 与 zh-CN 使用独立人工可审查 phrase bank。
- train/val/test template family ID 与真实句式均不重叠。
- renderer 只能填充 canonical plan slots；不得调用 LLM 或使用 P2 fixture description。
- 33 条 H2 approved requirements 只用于风格参考，不机械复制。
- 跨 split normalized exact-text 碰撞必须为 0；不能靠 sample ID、文件名或随机 token 消碰撞。

## 7. H3-01 人工审查

审查者只判断语言与工程语义，不需要填写 face ID：

- `approve`：query 自然，IR 摘要等价，feature/role/operation/唯一性合理。
- `edit`：只需修改表达且语义不变；必须填写 `corrected_query`。
- `reject`：目标不唯一、工程不合理、role/operation/value/unit 映射错误；必须选 error category 并说明。

允许的 `error_category`：`none`、`unnatural_language`、`wrong_cae_semantics`、`non_unique_target`、`wrong_target_role`、`wrong_value_or_unit`、`other`。

人工字段由生成器留空。未经用户真实回填，不得自动批准 H3-01、扩大到正式 20,000 条或冻结 P3。
