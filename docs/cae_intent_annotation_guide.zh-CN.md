# CAE-IR v1 标注与工程语义指南

版本：`cae-intent/1.0.0`  
适用阶段：TopoCAE-Agent P2-A1  
分析类型：`static_structural`  
状态：H2-01 人工审核后冻结候选

## 1. 目的与边界

CAE-IR 是自然语言需求与几何 grounding 之间的强类型接口。本指南规定如何把已经明确的单零件结构静力需求标注为纯 CAE intent，以及何时必须返回结构化澄清或无效结果。

CAE-IR 只表达工程语义，不包含 raw face ID、sample ID、part ID、模型文件名、面积真值、Recognizer 置信度或几何来源。`oracle`、`predicted`、`human_confirmed` 仅存在于 grounding context envelope，不能由用户需求或 intent 选择。

本阶段不解析自然语言、不调用 LLM、不训练 Pointer、不执行 Abaqus，也不扩展到接触、螺栓预紧、焊接、非线性材料、动力学或复杂网格优化。

## 2. 三层对象

### 2.1 纯 intent

`cae_intent.schema.json` 表示语义完整、允许进入 grounding 的需求。顶层必填：

- `schema_version`: 固定为 `cae-intent/1.0.0`
- `analysis_type`: 固定为 `static_structural`
- `operations`: 至少一个 discriminated operation
- `source_language`: 如 `zh-CN`、`en-US`
- `assumptions`: 只能记录用户明确确认的假设；不得由 Agent 自动补全
- `unit_system`: 项目单位上下文
- `normalization_audit`: 单位转换审计记录；未转换时为空数组

### 2.2 intent result

`cae_intent_result.schema.json` 只有三种状态：

- `VALID`: 必须携带完整且通过验证的 `intent`
- `NEEDS_CLARIFICATION`: 不得携带 intent/operations；必须提供缺失信息、歧义和追问
- `INVALID`: 不得携带 intent/operations；必须提供稳定错误码和字段路径

非 VALID 对象不能保存“差一点可执行”的半合法操作。

### 2.3 grounding context

`cae_grounding_context.schema.json` 是实验/运行 envelope，包含：

- `geometry_source`: `oracle | predicted | human_confirmed`
- `recognizer_capability_level`: 当前固定为 `LIMITED_CONTROLLED_DOMAIN`
- `coverage_status`: `ACCEPTED | PARTIAL | ABSTAIN | ERROR`
- `abstention_reasons`: 稳定原因码
- `feature_object_schema_version`

intent 合法不等于 predicted geometry 可用。slot、rib/web 必须在 predicted 路径拒识或人工确认；`pocket_circular` 是冻结 capability contract 的覆盖缺口；transition 能力有限。

## 3. Canonical feature taxonomy

P2-A1 固定 17 类：

| family | feature types |
|---|---|
| hole | `through_hole_round`, `blind_hole_round`, `stepped_hole_round`, `counterbore_hole`, `countersink_hole`, `obround_hole` |
| slot | `slot_rect`, `slot_u`, `slot_v`, `slot_o` |
| pocket | `pocket_rect`, `pocket_circular` |
| protrusion | `cylinder_boss`, `rectangular_boss` |
| stiffener | `rib` |
| transition | `fillet`, `chamfer` |

兼容 alias 必须在进入 schema 前规范化：

- `web -> rib`
- `through_hole -> through_hole_round`
- `blind_hole -> blind_hole_round`
- `stepped_hole -> stepped_hole_round`
- `pad`、`rectangle_boss -> rectangular_boss`

CAE-IR 永远输出 canonical 名称，不能把 `web` 作为第 18 类输出。

## 4. Feature target

Feature target 必填：

- `feature_type`
- `quantifier`
- `face_role`
- `filters`，没有筛选时使用空对象

允许的 quantifier：

| quantifier | 规则 |
|---|---|
| `all` | 选择所有满足条件的实例 |
| `any_one` | 任取一个满足条件的实例；需求必须允许任意性 |
| `exactly_one` | 条件应唯一命中一个实例 |
| `count` | 必须同时给出正整数 `count`；其他 quantifier 禁止携带 `count` |
| `largest` / `smallest` | 必须给出 `comparison_metric` |
| `nearest` | 必须给出 `nearest_to` 和 `reference_frame` |

`reference_frame` 仅在空间关系或 nearest reference 存在时允许；否则必须省略。

### 4.1 Filters

首版支持：

- `diameter`, `radius`, `depth`
- `area`
- `length`, `width`, `thickness`
- `spatial_relation`
- `nearest_to`

数值 filter 使用：

```json
{
  "operator": "between",
  "value": {"dimension": "length", "value": 8, "unit": "mm"},
  "upper_value": {"dimension": "length", "value": 12, "unit": "mm"}
}
```

`between` 必须有 `upper_value`；其他 operator 禁止携带它。

## 5. Face role 与 selection status

Role policy 位于 `configs/cae_feature_role_policy.yaml`。每个角色具有：

- `direct`: 一个现有 semantic subtype 可以直接表达
- `composite`: 需要组合多个现有 subtype
- `unsupported_no_distinct_subtype`: ontology 没有足够精度单独区分该角色

`requires_complete_instance=true` 表示必须先获得完整 feature instance；缺面、错面或实例断裂时不得把局部结果当成完整 region。

H2-01 人工审核后的兼容性是工程白名单。`allowed_operations` 表示“可作为该操作的几何候选”，不自动决定用户工况。被人工 reject 的组合必须返回 `E_ROLE_OPERATION_INCOMPATIBLE`。

当前显式不可精确区分的角色：

- `slot_rect.closed_end_face`
- `slot_v.closed_end_face`
- `rib.end_face`

这些角色没有独立 subtype；不能通过位置猜测伪造精确标签。

## 6. Operation contracts

### 6.1 `fixed_support`

- target: feature target
- parameters: `constrained_dofs` 必须恰为 `U1`, `U2`, `U3`
- 不得把 support 写成固有几何标签

### 6.2 `displacement`

- target: feature target
- parameters: 至少一个唯一平移自由度 `U1/U2/U3`
- 每个分量必须是 displacement quantity
- 必须明确 global 或 part-local Cartesian reference frame

### 6.3 `pressure`

- target: feature target
- magnitude: pressure quantity，非负
- direction: 必须显式为 `surface_normal` 或 `opposite_surface_normal`
- 不得从正负号、inside/outside 或自然语言习惯推断取向

### 6.4 `surface_traction`

- target: feature target
- magnitude: pressure/stress quantity，非负
- direction: global Cartesian、part-local Cartesian、surface normal 或 opposite surface normal
- Cartesian direction 必须给出三分量向量

### 6.5 `material_assignment`

- target: `{"scope": "whole_part"}`
- 必须给出 material name、Young's modulus（pressure quantity）和 Poisson ratio
- 禁止根据“钢”“铝”等名称静默填入材料默认值

### 6.6 `global_mesh_size`

- target: whole part
- size: 正的 length quantity

### 6.7 `local_mesh_size`

- target: 人工批准的 feature role/region
- size: 正的 length quantity

## 7. 单位系统与 normalization

项目单位上下文必须同时指定：

- length
- area
- force
- pressure
- displacement
- angle

支持：

| dimension | units |
|---|---|
| length/displacement | `um`, `mm`, `cm`, `m` |
| area | `mm2`, `cm2`, `m2` |
| force | `N`, `kN`, `MN` |
| pressure | `Pa`, `kPa`, `MPa`, `GPa`, `N/mm2` |
| angle | `deg`, `rad` |

单位归一必须幂等。每次实际转换记录：field path、dimension、from/to unit、factor、原值和归一值；再次归一不能重复追加记录。

重要：Abaqus 没有内置单位制。CAE-IR 的一致性表示项目约定明确，不表示 Abaqus 会自动换算。

## 8. 缺失信息与歧义

以下内容缺失时不得猜测：

- target feature 或 face role
- quantifier/count
- load/displacement magnitude
- unit
- direction/orientation/reference frame
- material property
- global/local mesh size

常见 missing code：`M_TARGET`, `M_FACE_ROLE`, `M_MAGNITUDE`, `M_UNIT`, `M_DIRECTION`, `M_MATERIAL_PROPERTY`, `M_MESH_SIZE`。

常见 ambiguity code：`A_FEATURE_TYPE`, `A_FACE_ROLE`, `A_TARGET_INSTANCE`, `A_DIRECTION_ORIENTATION`, `A_UNIT`, `A_MATERIAL`, `A_SPATIAL_REFERENCE`。

澄清问题必须直接指向缺失或歧义字段，不能同时返回一个可执行 operation。

## 9. 稳定错误码

程序判断必须使用 error code，不得解析自由文本：

- `E_SCHEMA_VALIDATION`
- `E_IDENTITY_FIELD_FORBIDDEN`
- `E_GEOMETRY_SOURCE_IN_INTENT`
- `E_UNIT_UNSUPPORTED`
- `E_UNIT_DIMENSION_MISMATCH`
- `E_QUANTIFIER_COUNT_CONFLICT`
- `E_REFERENCE_FRAME_REQUIRED`
- `E_REFERENCE_FRAME_UNEXPECTED`
- `E_PRESSURE_DIRECTION_INVALID`
- `E_ROLE_UNKNOWN`
- `E_ROLE_OPERATION_INCOMPATIBLE`
- `E_OPERATION_ID_DUPLICATE`
- `E_DISPLACEMENT_DOF_DUPLICATE`
- `E_FILTER_RANGE_INVALID`
- `E_NONVALID_RESULT_HAS_INTENT`

`field_path` 使用 JSON Pointer 风格，以 `/` 开头。

## 10. 标注流程

1. 判断是否在单零件 `static_structural` 范围内；超范围返回 INVALID。
2. 将 feature alias 规范化为 17 类 canonical 名称。
3. 确定每个 operation type；一个需求可以包含多个 operation，但 `operation_id` 必须唯一。
4. 根据 H2-01 冻结 policy 选择允许的 face role；不允许的组合返回 `E_ROLE_OPERATION_INCOMPATIBLE`。
5. 标注 quantifier、count、comparison metric 和 filters。
6. 为所有物理量标注 dimension、value、unit 和可选 tolerance。
7. 明确方向、坐标系和法向取向。
8. 缺失或有歧义时返回 NEEDS_CLARIFICATION，不生成半合法 intent。
9. 通过 schema、单位、角色兼容性和 identity 扫描后才能标为 VALID。
10. geometry source 由实验/运行配置提供，不进入纯 intent。

## 11. H2-01 冻结依据

- role policy：302/302 行已审核
- feature-role-operation：166 个允许、134 个不兼容
- requirement：40/40 行已审核；33 条批准，7 条拒绝
- 不存在未解析的 `edit`
- 原始签字 CSV 保留在 `reports/p2_a1_cae_ir_v1/human_review_package/`

Rejected requirement 不进入 approved requirements；其 reviewer note 保留在 H2 decision summary。不得自行把 reject 解释成新的修改内容。

## 12. 质量检查

冻结前必须满足：

- 所有 schema 是 Draft 2020-12 且 object 默认 closed
- 所有 17 个 feature 至少有明确 selectable role 或显式 unsupported role
- 7 种 operation 均有 target scope、typed parameters 和缺失信息契约
- VALID acceptance、INVALID rejection、clarification structured return 均为 100%
- normalization 幂等并具有审计记录
- intent/result 不包含 identity、truth 或 geometry source
- Recognizer 13 个引用资产 SHA-256 与 P1-D Lite manifest 一致
- 未调用 LLM、未训练模型、未执行 Abaqus、未重跑 Holdout
