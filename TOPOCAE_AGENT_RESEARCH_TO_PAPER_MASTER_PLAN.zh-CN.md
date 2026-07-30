# TopoCAE-Agent 研究到论文监督总纲

> 文档性质：本课题唯一主计划、阶段验收依据和防偏航约束。
>
> 监督者：当前 Codex 主任务。
>
> 适用对象：课题负责人（人为决策与工程审查）、实施 Agent（代码、实验、报告）、后续接手本项目的新 Agent。
>
> 当前版本：v1.0，2026-07-16。
>
> 研究范围：单零件、结构静力有限元前处理、原生 STEP/B-Rep 几何、自然语言需求驱动的精确面集选择与 Abaqus 操作生成。

---

## 0. 如何使用本计划

1. 本文件是后续工作的唯一主线。历史 `docs/superpowers/plans/`、v4.x/v5.x 计划和 `test_part_s3` 诊断记录只作为技术参考，不再决定研究方向。
2. 每次只启动一个阶段内的一个 Agent 任务。实施 Agent 必须先阅读本文件对应章节，再开始修改代码。
3. 只有监督者可以把阶段状态改为“通过”。实施 Agent 不得自行宣布整个阶段完成。
4. 每次人为介入都必须由监督者用中文给出：为什么需要人工、需要检查什么、具体怎样操作、提交什么证据、什么结果算通过。
5. 每个 Agent 任务必须给出：目标、允许修改的范围、禁止事项、必须交付的文件、测试命令和验收指标。
6. 未通过当前阶段门槛时，不进入下一阶段；不得用增加模型复杂度掩盖数据、标签或评估问题。
7. 论文最终结论必须由冻结测试集和完整实验产生，不能根据个别示例的视觉效果下结论。

---

## 1. 课题一句话定义

> 给定一个 STEP 单零件和一组自然语言结构静力仿真需求，系统首先利用拓扑感知的 B-Rep 图网络识别 CAE 相关几何特征及其可选面角色，再利用开源大语言模型解析仿真意图，通过受拓扑约束的 pointer/grounding 模块选择精确面集，最后生成经过验证的 Abaqus 边界、载荷和基础前处理操作。

论文的核心研究问题不是“识别所有机械零件”，也不是“让大模型直接理解 STEP”，而是：

> 如何把自然语言 CAE 前处理意图可靠地落到原生 B-Rep 中可执行、可验证的精确面集上？

建议工作名称：

- 系统：`TopoCAE-Agent`
- 几何数据集：`CAE-FeatureGraph`
- 语言 grounding 数据集：`CAE-QueryGround`
- 几何识别器：`TopoCAE-Recognizer`
- 语言条件面集选择器：`TopoCAE-Pointer`

建议论文标题：

> **TopoCAE-Agent: Grounding Natural-Language FEA Requirements to Exact B-Rep Face Sets with Topology-Constrained Graph Reasoning**

---

## 2. 固定研究边界

### 2.1 必须包含

- 输入几何：单实体或可作为单零件处理的 STEP/STP B-Rep。
- 分析类型：单零件结构静力前处理。
- 几何识别：face-level 标签、feature instance、semantic face role、selectable region。
- 自然语言解析：仿真操作、目标特征、目标面角色、数量词、几何筛选条件、数值、方向和单位。
- 几何 grounding：输出精确 B-Rep face set，而不是只输出特征名称。
- CAE 操作：固定约束、位移约束、压力、表面力/牵引、材料、全局网格尺寸和有限的局部网格区域。
- 软件落地：至少支持 Abaqus set/surface、BC、load、material、mesh action plan 和脚本生成。
- 验证：结构化语义验证、几何/拓扑验证、单位验证、Abaqus 创建或执行反馈。

### 2.2 当前目标特征体系

孔类：

- `through_hole_round`
- `blind_hole_round`
- `stepped_hole_round`
- `counterbore_hole`
- `countersink_hole`
- `obround_hole`

槽、腔和口袋：

- `slot_rect`
- `slot_u`
- `slot_v`
- `slot_o`
- `pocket_rect`
- `pocket_circular`

凸台、薄壁和过渡：

- `cylinder_boss`
- `rectangular_boss`
- `rib`
- `web`
- `fillet`
- `chamfer`

### 2.3 明确排除

- `test_part_s3` 不作为本论文的识别目标、模型调参依据或最终评价集。
- 不继续追求任意工业复杂零件上的全特征识别。
- 不扩展法兰、导轨、凸耳、完整装配体等零件级类别。
- 不处理装配接触、螺栓预紧、焊接、运动副、复杂材料非线性、屈曲、动力学和多物理场。
- 不让 LLM 直接读取 STEP 文本、直接猜测 raw face ID 或替代底层 GNN。
- 不把“脚本能够运行”误写成“仿真在物理上一定正确”。
- 不在论文第一版中追求完整自动网格优化、求解器选择和结果解释。
- 不为迎合“大模型”而先做多 Agent、超大模型微调或无约束代码生成。

### 2.4 防偏航停止规则

出现以下情况时，必须暂停并由监督者复核：

- 有人建议重新进入 `test_part_s3` 逐件修复。
- 有人建议增加新几何类别，但无法给出明确 CAE 查询和目标面角色。
- 有人建议更换大型网络，但没有证明现有瓶颈来自模型容量。
- 有人建议用 LLM 替换 B-Rep graph encoder 或直接预测 face ID。
- 有人使用 Real-Holdout 结果反向修改规则、阈值、提示词或训练数据。
- 新任务无法说明它改善了哪个论文指标或支撑了哪个贡献。
- 单个阶段连续两次迭代没有带来可测量收益。

---

## 3. 当前工作基线

### 3.1 已完成资产

- 程序化 STEP 特征生成器及特征面标签传播。
- STEP/B-Rep 到属性邻接图的提取流程。
- 节点几何描述符、edge attributes、loop/wall role 提示。
- edge-aware multi-task GNN。
- face macro、semantic subtype、derived/idealization 标签头。
- same-instance edge 监督和实例聚合。
- candidate classifier、topology postprocess、rib boundary/proposer 等对象级修复。
- feature object 和 selectable region 输出。
- STEP 到 Abaqus topology 的几何指纹映射原型。
- action plan 和 Abaqus 脚本生成原型。
- `test_part`、`test_part_s`、`test_part_s1`、`test_part_s2` 受控真实/简化真实测试件。

### 3.2 当前证据基线

冻结前参考结果：

| 证据 | 当前结果 | 解释限制 |
|---|---:|---|
| 合成测试 macro accuracy | 0.9799 | 来自自建分布，不代表复杂工业件泛化 |
| 合成测试 macro Macro-F1 | 0.9488 | 当前几何识别主指标之一 |
| semantic Macro-F1 | 0.9324 | 需要在最终冻结模型上重跑 |
| same-instance edge F1 | 0.9971 | 需要配套 object-level 指标 |
| `test_part_s2` area-truth recall | 0.9825 | 面积匹配，不等于严格 face-set IoU |
| `test_part_s2` area-truth precision | 0.9823 | 该测试集参与过规则修改 |
| 旋转一致组 | 3/3 | 最终模型仍需重新验证 |

参考文件：

- `output_generation_v5_5_augmented_full_5800_looprole/model_runs/edge_aware_v5_5_looprole_hardconfusion_70ep/metrics.json`
- `output_real_step_s2_v5_6_a_ribfix_fresh/reports/s2_area_truth_validation_v5_6_a_ribfix_fresh_mergeeval.md`
- `output_real_step_s_v4_0_invariant_current/reports/rotation_prediction_consistency.md`

### 3.3 当前不足

- 尚未冻结一个可复现的论文版 checkpoint、配置、标签表和推理命令。
- 历史测试集多次参与规则修复，只能作为 Real-Dev。
- 缺少从自然语言到结构化 CAE 意图的正式数据集和评估协议。
- 当前中文关键词匹配器不能处理属性筛选、数量词、空间关系和组合条件。
- 旧 LLM 原型直接读取文本报告，缺少 schema、候选约束和可靠性验证。
- 缺少 learnable language-conditioned pointer/grounder。
- Abaqus 闭环尚未形成批量、可测量的执行基准。
- 缺少完整 baseline、消融、置信区间和论文级结果总表。

---

## 4. 论文研究假设

论文需要验证以下假设，而不是只展示系统：

### H1：原生 B-Rep 图比孤立面描述更适合 CAE 特征识别

边属性、邻接拓扑和 same-instance 监督应显著改善 face-level 和 object-level 识别。

### H2：学习模型与显式拓扑约束互补

GNN 提供对尺寸变化和局部几何变化的学习能力，拓扑规则消除不合法孔、槽、凸台和 rib 实例。完整方法应优于纯 GNN 和纯规则。

### H3：结构化 CAE-IR 比 LLM 直接选面更可靠

LLM 先解析语义，再由受约束 grounder 选面，应在 exact face-set accuracy、非法输出率和可解释性上优于直接预测 face ID。

### H4：language-conditioned pointer 比关键词匹配更能处理组合查询

在半径、面积、位置、数量词、最近/最大等组合需求上，learnable pointer 应优于确定性关键词规则。

### H5：验证闭环能提高可执行成功率和工程可信度

加入 schema、单位、拓扑和 Abaqus 执行反馈后，端到端成功率应高于一次性生成脚本。

---

## 5. 预期论文贡献

论文只保留四项主贡献：

1. **CAE-FeatureGraph 数据集**：面向结构静力前处理的 STEP–B-Rep graph 配对数据，提供 face label、feature instance、edge affinity 和 selectable face role。
2. **TopoCAE-Recognizer**：融合 edge-aware GNN、same-instance 学习和显式拓扑合法性约束的 CAE 几何特征对象识别器。
3. **CAE-QueryGround 与 TopoCAE-Pointer**：把自然语言仿真需求映射为结构化 CAE-IR，并以 topology-constrained pointer 选择精确 B-Rep face set。
4. **可验证的 Abaqus 闭环**：将 grounding 结果编译为 CAE 操作，并通过区域创建、脚本执行和错误反馈验证端到端可靠性。

以下内容不能单独声称为创新：

- 使用了某个开源 LLM。
- 使用了 GraphSAGE、Random Forest 或 JSON。
- 写出了 Abaqus Python 脚本。
- 生成了很多合成零件。
- 某个简单例子识别正确。

---

## 6. 最终系统架构

```text
STEP/STP
   -> B-Rep extraction
   -> attributed face-adjacency graph
   -> frozen edge-aware GNN
   -> face labels + same-instance probabilities
   -> candidate objects + topology repair
   -> feature-object memory

Natural-language FEA requirement
   -> open-weight LLM intent parser
   -> typed CAE-IR

feature-object memory + CAE-IR
   -> topology-constrained pointer grounder
   -> selected feature instances and face roles
   -> exact STEP face set
   -> semantic/topology/unit verifier
   -> STEP-to-Abaqus face grounding
   -> Abaqus action compiler
   -> set/surface/BC/load/material/mesh
   -> execution feedback / clarification / abstention
```

### 6.1 LLM 的唯一职责

- 理解用户语言。
- 输出受 JSON Schema 限制的 CAE-IR。
- 在验证失败时根据结构化反馈修正意图。
- 必要时生成澄清问题。

### 6.2 LLM 不负责

- 从 STEP 字符串恢复三维几何。
- 判断原始 B-Rep 拓扑。
- 自由编造 face ID。
- 绕过单位、操作和几何合法性检查。
- 在用户未提供信息时擅自推断载荷、材料或约束。

### 6.3 Pointer Grounder 的职责

- 将候选 feature object、selectable region 和 CAE-IR 编码为可比较表示。
- 预测一个或多个候选对象及面角色的概率。
- 使用 topology mask 排除语义或几何上不允许的目标。
- 支持 `all`、`single`、`largest`、`smallest`、`nearest`、属性范围和空间方向等约束。
- 给出选择置信度和可解释证据。

---

## 7. 数据与评估集合设计

### 7.1 CAE-FeatureGraph

现有数据集冻结后，每个样本至少包含：

- STEP 文件和 solid validity 结果。
- graph JSON。
- face primitive、几何描述符和邻域描述符。
- edge type、length、convexity、dihedral、loop/wall role。
- face macro label 和 semantic subtype。
- feature instance ID。
- same-instance edge label。
- selectable regions 和 face roles。
- 生成配置、随机种子和 feature manifest。

冻结要求：

- 删除无效 solid、空标签、标签面不存在和 instance 断裂样本。
- 训练、验证和测试按 base STEP 分割。
- 同一基础几何的旋转、尺寸扰动和派生模型不得跨 split。
- 发布统计必须包含每类模型数、实例数、面数和组合覆盖率。

### 7.2 CAE-QueryGround

每条数据是：

```text
(STEP/graph, natural-language query, CAE-IR, target feature object, target face role, exact target face set, expected CAE operation)
```

查询分五级：

| 级别 | 内容 | 示例 |
|---|---|---|
| L1 | 单特征、单角色 | 固定这个盲孔的内壁 |
| L2 | 数量词和多实例 | 固定所有通孔内壁 |
| L3 | 属性筛选 | 给直径为 10 mm 的孔施加压力 |
| L4 | 空间/比较关系 | 选择面积最大的口袋底面 |
| L5 | 多操作、歧义和否定 | 固定左侧两个孔，但排除沉孔 |

建议规模：

- 每个冻结 STEP 自动生成 4–8 条 canonical queries。
- 形成约 20,000–40,000 条结构化 query–target 对。
- 每条 canonical query 生成 2–3 个语言改写，形成约 60,000–100,000 条自然语言样本。
- 人工不逐条标注；利用已有 face/instance/role 真值自动得到目标面集。
- 人工只审查 100–200 条分层抽样语言样本和所有最终 Real-Holdout 查询。

### 7.3 数据划分

- `Synthetic-Train/Val/Test`：冻结生成数据，按基础几何划分。
- `Real-Dev`：`test_part`、`test_part_s`、`test_part_s1`、`test_part_s2`，可用于开发、错误分析和选择阈值。
- `Rotation-Dev`：Real-Dev 中已知旋转派生组。
- `Real-Holdout`：系统冻结后新建的 10 个简单零件，每个保留原始姿态并生成两种旋转，共约 30 个 STEP；只用于最终报告。
- Query split 必须同时隔离几何和语言模板，避免模型只记住句式。

### 7.4 Real-Holdout 规则

- 难度与 `test_part` 到 `test_part_s2` 相当，不要求达到 `s3`。
- 每个零件包含 2–5 个目标特征，避免严重相交和人为不可定义语义。
- 至少覆盖全部主要特征族和三类几何条件查询。
- 冻结后不得根据 Holdout 修改模型、规则、提示词或阈值。
- 若 Holdout 失败，只能记录为结果和失败分析；改进必须形成下一版并重新建立新 Holdout。

---

## 8. CAE-IR 输出契约

CAE-IR 是语言模型和几何系统之间的强制接口。建议结构：

```json
{
  "analysis_type": "static_structural",
  "operations": [
    {
      "operation_type": "pressure",
      "target": {
        "feature_type": "through_hole_round",
        "quantifier": "all",
        "face_role": "hole_cylindrical_wall",
        "filters": {
          "diameter": {"value": 10.0, "unit": "mm", "tolerance": 0.1},
          "spatial_relation": null
        }
      },
      "parameters": {
        "magnitude": 10.0,
        "unit": "MPa",
        "direction": "surface_normal"
      }
    }
  ],
  "missing_information": [],
  "ambiguities": []
}
```

CAE-IR 必须满足：

- 所有字段均为枚举或有单位的类型化值。
- 不包含 raw face ID。
- 支持明确的 `missing_information` 和 `ambiguities`。
- 解析失败时返回错误对象，不得返回半合法自由文本。
- 编译器只接受验证通过的 CAE-IR。

---

## 9. 实施阶段和验收门槛

### Phase 0：治理、冻结和安全清理

状态：`待执行`

目标：停止历史分支漂移，建立论文版唯一入口。

Agent 工作：

- 确定当前最佳 GNN checkpoint、candidate classifier、rib 模块和推理参数。
- 建立 `configs/paper_v1/`，保存数据、模型、推理和评估配置。
- 建立单一 `run_paper_recognition_pipeline.py` 或等价入口。
- 生成冻结 manifest，记录文件 SHA256、随机种子、环境和依赖。
- 将旧 LLM 原型中的硬编码凭据移除，所有密钥改为环境变量。
- 明确标记 `test_part_s3` 为 excluded research artifact。

人为工作：

- 在相关 API 服务后台撤销旧脚本中已经暴露的密钥，并新建仅供本地实验使用的密钥。
- 确认本计划第 2 节的研究范围，没有遗漏或需要删除的操作。

交付物：

- `configs/paper_v1/`
- `paper_v1_freeze_manifest.json`
- `run_paper_recognition_pipeline.py`
- `reports/paper_v1_freeze_baseline.md`
- 安全扫描结果。

Gate P0：

- 同一输入重复推理两次输出一致。
- 冻结 checkpoint 和所有辅助模型可定位、可加载。
- 仓库中不再出现明文 API key。
- 论文入口不会读取 `test_part_s3`。

### Phase 1：现有几何识别基线复现与数据审计

状态：`未开始`

目标：把“我们记得效果很好”变成可复现的论文证据。

Agent 工作：

- 重跑 Synthetic-Test、Real-Dev 和 Rotation-Dev。
- 统一 face-level、edge-level、object-level 和 rotation metrics。
- 审计 solid validity、标签一致性、split leakage 和类别分布。
- 生成 failure atlas，但只记录，不再进入复杂零件规则迭代。

人为工作：

- 抽查 Agent 提供的 20 个 STEP 可视化样本，重点确认标签面和几何实体一致。
- 用 UG/Abaqus 检查 10 个 Real-Dev 预测高亮结果。

交付物：

- `reports/paper_v1_dataset_audit.md`
- `reports/paper_v1_recognition_baseline.md`
- `reports/paper_v1_rotation_consistency.md`
- 机器可读 metrics JSON/CSV。

Gate P1：

- 关键合成指标与现有记录差异不超过 0.005，或给出可复现原因。
- 无 base geometry 或旋转派生跨 split。
- 最终训练/测试数据中的 STEP 均通过设定的 solid validity 检查。
- Real-Dev 失败有明确分类，不再以逐件修补作为默认行动。

### Phase 2：任务定义和 CAE-IR

状态：`未开始`

目标：固定自然语言任务、目标面角色和输出协议。

Agent 工作：

- 实现 `cae_intent.schema.json` 和 Python 数据模型。
- 实现单位规范化、枚举验证、缺失信息和歧义表示。
- 为所有目标特征建立 `feature_type -> selectable face roles -> allowed CAE operations` 映射。
- 编写至少 100 个 schema 和语义单元测试。

人为工作：

- 提供 30–50 条典型结构静力仿真需求，使用工程师自然表达，不需要写 face ID。
- 对每条需求注明：应该选择什么特征、什么面、施加什么操作；若需求本身不完整，注明应该追问什么。
- 审查 feature-role-operation 映射表，判断工程语义是否合理。

交付物：

- `schemas/cae_intent.schema.json`
- `utils/cae_intent.py`
- `configs/cae_feature_role_policy.yaml`
- `tests/test_cae_intent_schema.py`
- `docs/cae_intent_annotation_guide.zh-CN.md`

Gate P2：

- Schema 合法率 100%。
- 所有当前目标特征至少有一个明确 selectable role。
- 不支持或有歧义的需求能够返回结构化错误或追问。
- 用户确认 role 和 operation 映射符合结构静力工程语义。

### Phase 3：CAE-QueryGround 数据集生成

状态：`未开始`

目标：利用现有几何标签低成本生成语言 grounding 训练和评估数据。

Agent 工作：

- 实现 `build_cae_query_grounding_dataset.py`。
- 从 feature objects 和 selectable regions 自动产生 canonical CAE-IR 和 exact face-set 真值。
- 实现 L1–L5 查询生成器、模板隔离 split 和语言改写接口。
- 实现自动一致性过滤：改写前后 CAE-IR 必须一致。
- 生成数据卡、分布表和人工审查样本包。

人为工作：

- 审查 100–200 条分层抽样查询。
- 每条只需标记：`正确`、`语义不自然`、`目标不唯一`、`工程上不合理`。
- 给出 10–20 条生成器没有覆盖的自然表达。

交付物：

- `output_cae_query_ground_v1/`
- dataset card。
- 训练/验证/测试清单。
- 人工审查表和修订记录。

Gate P3：

- 至少 20,000 条 canonical query–target 对。
- 所有样本的目标面集可以由原始标签自动追溯。
- 几何、旋转派生和语言模板均不存在 split leakage。
- 人工抽查正确率达到 95%；否则修生成规则，不扩大规模。

### Phase 4：语言解析与确定性 Grounding Baseline

状态：`未开始`

目标：建立最简单、可解释、可量化的 agent baseline。

Agent 工作：

- 接入一个本地开源 instruction LLM，优先选择 8B–14B 量级。
- 用 JSON Schema constrained decoding 或严格解析器输出 CAE-IR。
- 实现关键词规则、LLM 直接选面、LLM→CAE-IR→确定性匹配三种 baseline。
- 实现统一推理日志、token/latency/失败原因记录。
- 不进行 LoRA，先完成 zero/few-shot 基线。

人为工作：

- 审查 30 条解析失败或歧义样本，判断是语言问题、Schema 问题还是工程语义问题。
- 确认追问文本符合工程沟通习惯。

交付物：

- `run_cae_intent_parser.py`
- `run_deterministic_grounding.py`
- `reports/cae_parser_baseline.md`
- `reports/deterministic_grounding_baseline.md`

Gate P4：

- JSON/schema compliance 为 100%。
- L1–L3 intent/slot F1 达到 0.95 或给出不足分析。
- 系统绝不输出不存在的 feature type、role 或单位。
- baseline 的 exact face-set、IoU、latency 和 failure rate 均已记录。

### Phase 5：TopoCAE-Pointer

状态：`未开始`

目标：实现论文的主要 language-conditioned grounding 模块。

Agent 工作：

- 将每个 feature object/selectable region 编码为候选 token。
- 融合 GNN pooled embedding、对象几何描述符、macro、face role 和 topology evidence。
- 编码自然语言或 CAE-IR，并实现 query-candidate cross-attention 或等价 pointer scorer。
- 支持多目标集合预测和 topology mask。
- 训练并比较：无 mask、无 GNN embedding、无 role、无 object-level aggregation、完整模型。
- 输出候选分数、最终 pointer 和可解释证据。

人为工作：

- 只审查 pointer 的代表性失败样本，不参与逐条训练标签制作。
- 判断被判为“语义正确但面集不同”的案例是否属于真值定义问题。

交付物：

- `models/topocae_pointer.py`
- 训练与推理脚本。
- checkpoint、配置和 metrics。
- `reports/topocae_pointer_experiment.md`

Gate P5：

- 在 L3–L5 查询上显著优于关键词和确定性 baseline。
- 相对最佳非 pointer baseline，exact face-set accuracy 或 macro query accuracy 至少提升 5 个百分点；若未达到，必须分析是否值得保留该模块。
- topology mask 能显著降低非法目标率。
- Oracle feature objects 与 predicted feature objects 两种设置均有结果，用于量化识别误差传播。

### Phase 6：验证器和 Abaqus 执行闭环

状态：`未开始`

目标：把语言 grounding 结果变成可靠的 CAE 操作，而不是展示 JSON。

Agent 工作：

- 实现 schema、单位、操作-角色、几何存在性和拓扑合法性验证器。
- 复用并重构当前 fingerprint mapping 和 Abaqus action plan。
- 建立 `STEP -> prediction -> grounding -> Abaqus set/surface -> BC/load` 单一流水线。
- 实现 dry-run、错误分类、有限次数修复和最终 abstention。
- 不自动提交耗时求解作业；先验证模型、区域和操作创建。

人为工作：

- 在 Abaqus 中检查 20 个样例：所选面、set/surface、载荷方向、约束自由度、单位是否正确。
- 对每个样例填写 `通过/不通过` 和一句原因。
- 对 5–10 个代表样例允许运行小规模静力作业，确认脚本完整性。

交付物：

- `run_topocae_agent.py`
- `utils/cae_verifier.py`
- `utils/abaqus_action_compiler.py`
- 执行日志和结果报告。

Gate P6：

- 支持范围内的 Abaqus set/surface 创建成功率不低于 95%。
- 非法或不完整需求不会静默执行。
- 验证闭环相对一次性生成显著降低执行失败。
- 人工检查的关键 BC/load 区域没有系统性错误。

### Phase 7：正式实验、消融和统计分析

状态：`未开始`

目标：形成投稿所需的完整证据链。

Agent 工作：

- 固定所有训练和推理配置，至少运行 3 个随机种子。
- 生成统一结果表、置信区间、显著性分析和 failure atlas。
- 执行数据、模型、语言 grounding、验证器和端到端消融。
- 记录硬件、时间、显存、参数量和推理延迟。
- 生成论文图表的源 CSV/JSON，不只生成截图。

人为工作：

- 审查实验矩阵是否公平。
- 判断失败分类是否符合 CAE 工程实际。
- 选择 6–10 个最能解释方法收益和局限的定性案例。

Gate P7：

- 所有主要结论至少有一个定量实验支撑。
- 所有新增模块至少有对应消融。
- 结果可由一个全新目录按文档命令复现。
- 不隐藏失败类别，不用单一 accuracy 代替类别均衡指标。

### Phase 8：Real-Holdout 最终评估

状态：`未开始`

目标：获得真正未参与开发的受控外推证据。

Agent 工作：

- 在监督者确认冻结后生成 Holdout 输入模板和检查工具。
- 对新 STEP 自动提取 graph、推理、生成 query 结果和 Abaqus 操作。
- 只生成报告，不修改模型和规则。
- 用 bootstrap 给出 query-level 和 part-level 置信区间。

人为工作：

- 在冻结之后建立或收集 10 个简单单零件 STEP。
- 每个零件给出 3–5 条真实仿真需求，不提供 face ID。
- 在 UG/Abaqus 中检查最终高亮面和操作是否正确。
- 将原始需求、正确面选择和检查结果保存为最终真值。

Gate P8：

- Holdout 全部完成，不存在回看调参。
- 报告 exact face-set、face IoU、operation accuracy、execution success 和失败类型。
- 旋转派生模型在语义和目标面角色层保持一致。
- 结果无论高低均原样进入论文讨论。

### Phase 9：论文写作、复现包和投稿

状态：`未开始`

目标：将系统工作转化为可审查、可复现的论文贡献。

Agent 工作：

- 生成数据集文档、代码使用说明、环境文件和复现实验命令。
- 汇总所有表格和图的来源。
- 检查论文中的数字与 metrics 文件一致。
- 对匿名化、许可证、开源范围和数据隐私做检查。

人为工作：

- 决定目标会议和模板。
- 审查摘要、贡献和结论是否符合实际能力。
- 确认 CAE 案例、载荷和边界条件在工程上合理。
- 最终决定数据、代码和模型的公开范围。

Gate P9：

- 论文中没有“完全自动理解任意零件”等超范围表述。
- 每项贡献均可对应到代码、数据和实验。
- 主结果可以从冻结复现包重跑。
- 附录包含 ontology、CAE-IR、数据生成、更多定性结果和局限性。

---

## 10. 正式实验矩阵

### 10.1 数据集实验

| 编号 | 实验 | 目的 |
|---|---|---|
| D1 | 类别、实例、面数和组合覆盖 | 证明数据覆盖不是只看模型数量 |
| D2 | solid/topology/label validity | 证明 STEP–graph 配对可信 |
| D3 | random split 与 composition split | 区分同分布和未见组合泛化 |
| D4 | rotation split | 验证坐标旋转鲁棒性 |
| D5 | Real-Dev 与 Real-Holdout | 验证合成到受控真实件外推 |

### 10.2 几何识别 Baseline

| 编号 | 方法 |
|---|---|
| G0 | Rule-only |
| G1 | Face-only MLP/Random Forest |
| G2 | GraphSAGE，无 edge_attr |
| G3 | Edge-aware GNN |
| G4 | G3 + same-instance head |
| G5 | G4 + candidate classifier |
| G6 | G5 + topology postprocess，完整识别器 |
| G7 | 可复现时加入 BRepNet/UV-Net/AAGNet-style 对照 |

### 10.3 语言和 Grounding Baseline

| 编号 | 方法 |
|---|---|
| L0 | 关键词规则匹配 |
| L1 | LLM 直接从候选文本预测 face ID |
| L2 | LLM -> CAE-IR -> 确定性对象匹配 |
| L3 | Text/IR encoder + 无拓扑约束 pointer |
| L4 | TopoCAE-Pointer，完整模型 |
| L5 | L4 + verifier + feedback，完整 agent |

### 10.4 关键消融

- 无 node shape descriptors。
- 无 edge attributes。
- 无 same-instance loss。
- 无 candidate-level aggregation。
- 无 topology postprocess。
- 无 face-role 输入。
- 无 topology mask。
- 无 CAE-IR，直接语言到 pointer。
- 无 verifier。
- 无 execution feedback。
- Oracle feature objects 对比 predicted feature objects。
- 原始姿态对比随机 SO(3) 旋转。

---

## 11. 指标体系

### 11.1 几何识别

- Face macro accuracy、Macro-F1、per-class precision/recall。
- Semantic subtype Macro-F1。
- Face mIoU。
- Same-instance edge precision/recall/F1。
- Object precision/recall/F1。
- Instance IoU 和 exact instance ratio。
- Rotation consistency。

### 11.2 语言解析

- JSON/schema compliance。
- Intent accuracy。
- Slot precision/recall/F1。
- Unit/value accuracy。
- Missing-information detection accuracy。
- Ambiguity detection accuracy。

### 11.3 Grounding

- Feature-instance selection accuracy。
- Exact face-set accuracy。
- Selected-face precision/recall/F1。
- Face-set IoU。
- Query-level Macro-F1。
- Invalid target rate。
- Selective accuracy / abstention coverage。

### 11.4 端到端 CAE

- Abaqus face mapping accuracy。
- Set/surface creation success。
- Operation type accuracy。
- BC/load parameter accuracy。
- Script execution success。
- End-to-end task success。
- 平均修复次数、延迟和 token 成本。
- 人工工程审查通过率。

---

## 12. 论文结构

1. **Introduction**：定义自然语言 CAE 需求到 B-Rep 精确面集 grounding 的缺口。
2. **Related Work**：B-Rep learning、CAD feature recognition、language-guided CAD entity selection、LLM/agentic CAE。
3. **Problem Definition**：STEP graph、feature object、CAE-IR、target face set 和任务边界。
4. **CAE-FeatureGraph Dataset**：生成、标签、有效性、统计和 split。
5. **Method**：TopoCAE-Recognizer、TopoCAE-Pointer、topology mask、verifier 和 Abaqus compiler。
6. **CAE-QueryGround Benchmark**：查询生成、人工审查和评估协议。
7. **Experiments**：baseline、消融、旋转、Real-Dev、Real-Holdout、端到端执行。
8. **Limitations and Safety**：复杂件、装配、物理意图缺失、LLM 幻觉和软件差异。
9. **Conclusion**：只总结已验证能力。

投稿定位：

- 若 language grounding、pointer 方法和 benchmark 贡献足够强，优先考虑 AAAI/IJCAI 类综合 AI 会议。
- 若几何表示创新更强，可考虑 CVPR/ICCV 的 CAD/B-Rep 方向。
- 若方法创新不足但工程闭环完整，转向 CAD、RCIM、Advanced Engineering Informatics 等领域期刊，不强行包装为 CCF A。

---

## 13. 预计时间表

| 周期 | 阶段 | 预期产出 |
|---|---|---|
| 第 1 周 | P0 | 冻结、入口、安全清理 |
| 第 2 周 | P1 | 数据和识别基线报告 |
| 第 3 周 | P2 | CAE-IR、role policy、测试 |
| 第 4–5 周 | P3 | CAE-QueryGround v1 |
| 第 6 周 | P4 | LLM parser 和确定性 baseline |
| 第 7–9 周 | P5 | TopoCAE-Pointer |
| 第 10–11 周 | P6 | Abaqus verifier/executor |
| 第 12–14 周 | P7 | 正式实验与消融 |
| 第 15 周 | P8 | Real-Holdout |
| 第 16–18 周 | P9 | 论文、附录、复现包 |

该时间表以单阶段通过为前提。任何阶段若未通过 Gate，不以赶时间为理由跳过。

---

## 14. 监督和任务交付协议

### 14.1 Agent 每次必须提交

1. 完成了什么，以及对应本计划的阶段和任务编号。
2. 修改/新增文件清单。
3. 实际运行的命令。
4. 测试结果和核心指标。
5. 生成报告或可视化的绝对路径。
6. 未解决问题和假设。
7. 是否触碰禁止范围；若有，为什么。
8. 建议下一步，但不得自行开始下一阶段。

### 14.2 监督者审查结论

监督者只使用三种结论：

- `通过`：所有必交付物和 Gate 满足，可进入下一任务。
- `有条件通过`：主目标满足，只允许补充指定的小问题。
- `不通过`：核心证据不足或偏离主线，需要按监督意见返工。

### 14.3 给实施 Agent 的通用任务模板

```text
你正在实施 TopoCAE-Agent 研究计划中的 [阶段/任务编号]。

首先完整阅读：
W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md

目标：
[一个可验证的具体目标]

允许修改：
[文件/目录范围]

禁止事项：
- 不处理 test_part_s3。
- 不修改已冻结 checkpoint、Holdout 或研究范围。
- 不进行未要求的架构重写。
- 不删除用户已有文件或历史输出。

必须交付：
[代码、测试、报告、配置]

必须验证：
[命令和指标]

完成后停止，不要自行进入下一阶段。按计划第 14.1 节提交结果。
```

### 14.4 给用户的人为任务模板

监督者以后必须按以下形式给出人为任务：

```text
人为任务编号：Hxx
为什么需要你：说明哪些工程判断无法由 Agent 可靠替代。
你要检查的对象：给出明确文件、模型或报告路径。
具体操作：按 1、2、3 步说明如何在 UG/Abaqus/表格中完成。
你需要返回：规定最简结果格式，例如“样本名 + 通过/不通过 + 一句原因”。
通过标准：说明什么结果允许进入下一阶段。
预计工作量：给出样本数和预计时间，不安排不可现实的大规模手工标注。
```

---

## 15. 风险清单

| 风险 | 后果 | 控制措施 |
|---|---|---|
| 合成测试过高但域外泛化有限 | 论文被认为过拟合生成器 | Real-Dev、冻结 Real-Holdout、composition split |
| Real-Dev 被用于调参 | 外推结果失去独立性 | 明确开发集身份，新建 Holdout |
| LLM 直接猜 face ID | 幻觉和不可解释错误 | CAE-IR、candidate pointer、topology mask |
| 需求本身不完整 | 系统擅自生成错误工况 | missing-information、clarification、abstention |
| 脚本可运行但物理不合理 | 工程结论不可信 | 操作白名单、单位检查、人工工程审查 |
| 模块过多，贡献不清楚 | 论文像系统集成 | 四项贡献、严格消融、单一核心问题 |
| 为复杂件不断加规则 | 再次钻牛角尖 | 明确排除 s3、两次无收益即停止 |
| 数据或 API 密钥泄露 | 安全问题 | 撤销旧密钥、环境变量、发布前扫描 |
| Abaqus face ID 导入变化 | 选面错位 | 几何指纹映射和 mapping score 验证 |
| 人工真值工作量失控 | 项目停滞 | 自动生成标签，只做分层抽查和最终 Holdout |

---

## 16. 里程碑总表

| 里程碑 | 状态 | 监督结论 |
|---|---|---|
| M0 研究主线和范围冻结 | 已定义 | 等待用户确认本文档 |
| M1 论文版识别器冻结 | 待执行 | - |
| M2 数据审计和识别证据 | 未开始 | - |
| M3 CAE-IR 和 role policy | 未开始 | - |
| M4 CAE-QueryGround v1 | 未开始 | - |
| M5 LLM parser baseline | 未开始 | - |
| M6 TopoCAE-Pointer | 未开始 | - |
| M7 Abaqus 验证闭环 | 未开始 | - |
| M8 正式实验和消融 | 未开始 | - |
| M9 Real-Holdout | 未开始 | - |
| M10 论文和复现包 | 未开始 | - |

---

## 17. 当前立即执行顺序

在用户确认本总纲后，只按以下顺序推进：

1. 启动 `P0-A1：论文版识别器冻结与安全清理`。
2. P0-A1 通过后，启动 `P1-A1：统一复现和数据泄漏审计`。
3. P1 通过后，向用户分配 `H2-01：30–50 条典型 CAE 需求编写与语义说明`。
4. 同时启动 `P2-A1：CAE-IR Schema 与 role policy 实现`。
5. 在 P2 通过前，不开始 LLM 接入；在 P3 数据集通过前，不开始 pointer 模型训练。

第一条 Agent Goal 建议如下：

```text
Goal P0-A1：冻结 TopoCAE-Agent 论文版几何识别基线。

完整阅读主计划：
W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md

请完成 Phase 0 中所有 Agent 工作：识别并冻结当前最佳 checkpoint、辅助分类器、推理参数和标签 schema；创建 configs/paper_v1、统一识别入口、freeze manifest 和基线报告；移除仓库内明文凭据并改用环境变量；确保任何论文入口都不读取 test_part_s3。

不要重新训练模型，不要修改识别规则，不要进入 CAE-IR 或 LLM 阶段。完成后运行重复推理、模型加载、安全扫描和现有相关单元测试，并按主计划第 14.1 节提交证据，然后停止。
```

---

## 18. 最终完成定义

本课题只有同时满足以下条件才算完成：

- CAE-FeatureGraph 有冻结版本、数据卡、有效性审计和可复现 split。
- TopoCAE-Recognizer 有冻结 checkpoint、完整 baseline、消融和旋转测试。
- CAE-QueryGround 有足够规模、人工抽查和无泄漏划分。
- 开源 LLM 能稳定输出合法 CAE-IR。
- TopoCAE-Pointer 能在组合查询上显著优于关键词和直接 LLM baseline。
- 系统能输出精确 face set，并在 Abaqus 中创建正确 set/surface/BC/load。
- Real-Holdout 在冻结后一次性完成，结果未被用于调参。
- 论文的每项贡献都有对应代码、数据和定量证据。
- 论文明确陈述适用范围、失败模式和不支持能力。
- 复现包能够从 STEP + query 运行到最终 action plan 和评估报告。

在这些条件之前，不以“做出了 demo”或“某几个零件识别正确”作为课题完成标准。
