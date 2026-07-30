# TopoCAE-Agent：Windows + LLM API + Abaqus 技术路线

> 日期：2026-07-28  
> 适用范围：第一篇论文，单零件结构静力前处理  
> 当前状态：P1、P2、P3 已冻结；P4 人工审核通过，等待接入真实 LLM API

## 1. 研究目标

给定一个 STEP 单零件和一条自然语言结构静力需求，系统需要：

1. 从 STEP 中识别受控范围内的 CAE 几何特征及其组成面；
2. 将自然语言需求转换为强类型 CAE-IR；
3. 将 CAE-IR 中的目标条件映射到具体特征对象和语义面角色；
4. 把 STEP/B-Rep 面稳定映射到 Abaqus 导入后的面；
5. 生成并执行可验证的 Abaqus 前处理操作；
6. 在信息不足、识别不可靠或面映射歧义时澄清或拒绝执行。

论文核心问题固定为：

> 如何把单零件结构静力自然语言需求，可靠地映射到 STEP B-Rep 的精确面集，并生成可验证的 Abaqus 前处理操作？

本路线不以任意工业 CAD 的通用特征识别为目标，也不让 LLM 直接理解 STEP、猜测 raw face ID 或自由生成可执行代码。

## 2. 总体架构

整体保留早期 `step1` 至 `step5` 的五阶段思想，但用当前冻结模块替换旧实现：

```text
STEP + natural-language CAE requirement
                    |
                    +-----------------------------+
                    |                             |
                    v                             v
        invariant attributed B-Rep graph   LLM API language parser
                    |                             |
                    v                             v
     GNN + candidate classifier + rules       typed CAE-IR
                    |                             |
                    v                             |
       feature-object memory + face roles <------+
                    |
                    v
       deterministic grounding + verifier
                    |
                    v
          exact STEP/B-Rep face set
                    |
                    v
       STEP-to-Abaqus face fingerprint mapper
                    |
                    v
        validated Abaqus action plan/compiler
                    |
                    v
      Abaqus/CAE execution + structured feedback
```

只有“自然语言到 CAE-IR”需要调用云端 API。STEP 解析、GNN 推理、对象匹配、面映射、规则校验和 Abaqus 执行均保留在 Windows 本地。

## 3. 旧五阶段与新路线的对应关系

| 旧脚本职责 | 新模块 | 处理原则 |
|---|---|---|
| `step1_topology.py`：从 Abaqus 提取简单拓扑 | Graph v2.1 B-Rep 提取器 | 使用真实节点、边及旋转/尺度适当处理后的几何拓扑属性 |
| `step2_semantic_enricher.py`：手写规则识别特征 | 冻结 P1-B recognizer | 使用 face GNN、same-instance、candidate classifier、topology verifier 和 abstention |
| `step3_topology_agent.py`：LLM 阅读报告并选 face ID | P4 Structured LLM | LLM 只把需求转换成 CAE-IR，不接收和输出 raw face ID |
| `step4_verify.py`：在 Abaqus 中高亮面 | CAE-IR grounder + verifier + 映射预检 | 验证 schema、单位、特征角色、数量、面范围、映射唯一性和能力边界 |
| `step5_script_gen.py`：拼接 Abaqus Python | 确定性 Abaqus compiler | 只接受已验证 action plan，并返回结构化执行结果 |

旧版脚本保留为历史原型和实现参考，不作为正式论文系统的默认执行入口。

## 4. 几何识别链 P1

### 4.1 输入

- 单零件 STEP/STP；
- 单实体或接近单实体的受控几何；
- 同一个 STEP 文件同时用于 B-Rep 图提取和 Abaqus 导入；
- 记录文件 SHA-256，防止识别和执行使用不同几何版本。

### 4.2 正式识别流程

```text
STEP
-> Graph v2.1
-> edge-aware face GNN
-> same-instance assembly
-> candidate proposal
-> candidate topology classifier
-> topology legality verifier
-> global decoder
-> feature objects
```

当前冻结识别器是 `p1_b_general_recognition_pipeline_v4`，能力等级为 `LIMITED_CONTROLLED_DOMAIN`。

正式目标类别包括：

- 孔：通孔、盲孔、阶梯孔、沉孔、锥沉孔、长圆孔；
- 槽：矩形槽、U 形槽、V 形槽、O 形槽；
- 口袋：矩形口袋；
- 凸台：圆柱凸台、矩形凸台；
- 加强结构：rib/web 统一为 `rib`；
- 过渡：圆角、倒角。

预测输出不是简单 face label，而是包含以下内容的 feature-object memory：

- opaque object ID；
- canonical feature type；
- instance membership；
- semantic face roles；
- geometry parameters 和 descriptors；
- confidence、coverage、conflicts 和 abstention reasons。

slot、rib/web 和部分 transition 在 predicted geometry 路径中仍属于有限能力区域。低置信、对象不完整或拓扑冲突时必须请求人工确认，不得静默纠正。

## 5. 语言理解链 P4

### 5.1 LLM 的唯一职责

LLM 只执行：

```text
natural-language CAE requirement -> CAE-IR result
```

示例：

```text
固定所有直径最小的盲孔
```

转换为描述以下内容的结构化对象：

- operation：`fixed_support`；
- feature type：`blind_hole_round`；
- quantifier：`all` 或按正式语义契约解释；
- comparison/filter：同类实例中的最小直径；
- face role：由受审核的 role policy 指定；
- constrained DOFs：`U1/U2/U3`。

LLM 不负责：

- 读取 STEP；
- 识别几何特征；
- 输出 face ID；
- 决定不在白名单中的 feature-role-operation；
- 猜测缺失的载荷、方向、单位或材料参数；
- 生成和执行任意 Abaqus Python。

### 5.2 API 方案

Windows 可以直接调用 OpenAI-compatible API，无需本地 GPU。现有 DeepSeek V4 Pro 余额可优先使用：

```text
TOPOCAE_LLM_BASE_URL=https://api.deepseek.com
TOPOCAE_LLM_MODEL=deepseek-v4-pro
TOPOCAE_LLM_API_KEY=<environment-only-secret>
```

不得把新密钥写入源码、JSON、报告、日志或 Git。旧 `step3_topology_agent.py` 中的硬编码密钥必须删除，即使密钥已经撤销。

DeepSeek 稳定 API 使用 `json_object`。当前 P4 adapter 默认发送 `json_schema`，因此需要加入 provider profile：

1. DeepSeek 使用 `response_format={"type":"json_object"}`；
2. prompt 明确要求输出 JSON；
3. 返回后由本地 Draft 2020-12 JSON Schema 验证；
4. 首次失败只允许一次显式修复重试；
5. 禁止静默 JSON 修补；
6. 记录 HTTP 状态、实际 model ID、token、延迟、重试和失败码；
7. 使用请求内容哈希缓存，避免重复计费；
8. 结构化解析默认关闭 thinking，降低延迟和波动；
9. 设置合理的 `max_tokens` 和 timeout。

开发阶段可使用 `deepseek-v4-flash` 做低成本联调；prompt、schema 和 verifier 冻结后，再使用 `deepseek-v4-pro` 运行正式实验。

### 5.3 正式主路线与基线

正式系统采用：

```text
Structured LLM -> CAE-IR -> deterministic grounding -> verifier
```

实验基线包括：

1. Keyword/rule parser；
2. Direct LLM object/role selection；
3. Structured LLM -> CAE-IR；
4. Structured LLM -> CAE-IR -> topology-constrained grounding/verifier；
5. Oracle feature-object upper bound。

Direct LLM 可以读取经过清洗的 opaque candidate memory，但不能获得 raw face IDs。它仅作为对照，不作为默认执行路线。

## 6. CAE-IR 与确定性 Grounding

CAE-IR 是语言和几何之间的强类型接口，负责表达：

- 分析类型：`static_structural`；
- operation type；
- feature target；
- quantifier/count；
- comparison metric 和 filters；
- semantic face role；
- magnitude、unit 和 direction；
- clarification 或 invalid reason。

当前支持七类操作：

- `fixed_support`；
- `displacement`；
- `pressure`；
- `surface_traction`；
- `material_assignment`；
- `global_mesh_size`；
- `local_mesh_size`。

Grounder 的职责是：

1. 根据 feature type 筛选候选对象；
2. 应用数量词、同类尺寸比较和空间条件；
3. 根据冻结 role policy 选择语义面角色；
4. 从对象和角色展开 exact STEP face set；
5. 验证 face index 范围、对象完整性、操作兼容性和覆盖状态；
6. 出现零匹配、多义匹配或能力边界时返回 clarification/abstention。

这一步必须是确定性的，同一输入、同一冻结配置应产生相同结果。

## 7. STEP-to-Abaqus 面身份映射

这是 Windows/Abaqus 闭环中优先级最高的新模块。

不能假设 OCC/STEP 图中的 `face_id` 与 Abaqus `p.faces[index]` 一致。Abaqus 导入可能重新排序、合并或拆分拓扑实体。

每个 STEP face 和 Abaqus face 应建立不依赖列表顺序的 fingerprint：

- surface type；
- area；
- centroid；
- bounding-box dimensions；
- normal、axis、radius、cone angle 等解析几何参数；
- boundary loop 数量和类型；
- adjacency degree；
- 邻接面的 surface-type/area/convexity 签名；
- 必要时加入二阶邻域签名。

映射流程：

```text
STEP face fingerprints
-> Abaqus face fingerprints
-> candidate compatibility mask
-> minimum-cost bipartite assignment
-> adjacency consistency verification
-> unique mapping / ambiguous / unmatched
```

验收要求：

- 不允许仅按面积或 face 顺序映射；
- 一对一映射必须具有唯一性和置信度证据；
- ambiguity 和 unmatched 必须阻止执行；
- 映射后重新验证目标面集合的面积、类型和邻接摘要；
- 对旋转派生件及重新导入后的同一 STEP 做回归测试。

## 8. Abaqus Action Compiler 与执行器

### 8.1 进程隔离

普通 Python/`feagpt` 与 Abaqus 自带 Python 不共享依赖。推荐使用文件协议：

```text
feagpt Python
-> validated_cae_task.json
-> abaqus cae noGUI=execute_cae_task.py -- task.json
-> abaqus_execution_result.json
```

LLM 不得写入 `execute_cae_task.py`。Abaqus compiler 只能从白名单 operation 和已验证参数生成操作。

### 8.2 第一版执行范围

- 导入 STEP；
- 创建 part/instance/static step；
- 创建 Set/Surface；
- 材料与实体截面；
- 固定、位移、压力和面力；
- 全局网格和局部网格尺寸；
- 创建 job；
- 保存 CAE；
- 返回执行清单。

第一版在提交求解前保留人工确认。论文闭环优先证明“准确选区并成功创建前处理对象”，不把复杂网格质量优化或求解收敛作为前置目标。

### 8.3 结构化反馈

执行器至少返回：

```json
{
  "status": "SUCCESS",
  "created_sets": [],
  "created_surfaces": [],
  "created_materials": [],
  "created_bcs": [],
  "created_loads": [],
  "mesh_status": "NOT_REQUESTED",
  "job_status": "CREATED",
  "failed_operations": [],
  "warnings": []
}
```

禁止用宽泛 `except: pass` 隐藏失败。每个 operation 必须有稳定 ID、结果和错误码，并支持重复执行时的幂等检查。

## 9. 关键风险和控制措施

### 9.1 几何识别错误

控制方式：

- 同时评估 Oracle feature objects 和 predicted feature objects；
- 报告误差从 recognition 到 grounding 和 Abaqus execution 的传播；
- 对 slot、rib/web、transition 保持 capability-aware abstention；
- 不根据特定 part ID、face ID 或绝对面积增加规则。

### 9.2 LLM 幻觉或结构错误

控制方式：

- 强类型 CAE-IR；
- JSON Schema 和工程白名单；
- 缺失信息必须 clarification；
- LLM 不接触 raw face ID；
- verifier 拒绝非法 object、role、unit 和 operation；
- 每条正式 test 请求只运行规定次数并保存原始响应。

### 9.3 面映射错误

控制方式：fingerprint、全局匹配、邻接复核、唯一性门禁和执行前高亮确认。

### 9.4 Abaqus 脚本误执行

控制方式：白名单 compiler、任务 JSON、独立进程、dry-run、结构化反馈、求解前人工确认。

### 9.5 API 不可复现和费用

控制方式：冻结模型 ID、调用日期、prompt、schema、参数和原始响应缓存；先小规模 smoke，再 val，最后只运行一次正式 test。API 模型升级时不得把新旧结果混在同一正式实验中。

## 10. 分阶段实施计划

### Phase P4-API：接入真实 LLM API

工作：

1. 清除旧硬编码密钥；
2. 给 P4 adapter 增加 DeepSeek provider profile；
3. 支持 `json_object`、thinking 开关、max tokens、缓存和调用审计；
4. 用 mock server 完成协议回归测试；
5. 使用少量非 test 样本做真实 API smoke；
6. 运行 C2/C3 train/val 并冻结配置；
7. 经监督者批准后，四条 baseline 各运行一次正式 test。

门禁：CAE-IR schema 合法率、语义指标、无 test leakage、费用和失败审计完整。

### Phase P5：Topology-constrained grounding/pointer

工作：

1. 完成 CAE-IR 到 feature object 的确定性筛选；
2. 完成 quantifier、尺寸比较和空间关系；
3. 完成 semantic face-role expansion；
4. 保留 Direct LLM 作为基线；
5. 输出 exact face-set、IoU、selection accuracy 和 abstention 指标。

门禁：Oracle 与 predicted 两轨均可运行，raw face ID 不进入 LLM 输入或输出。

### Phase P6-A：STEP-to-Abaqus mapper

工作：

1. 提取两侧 face fingerprint；
2. 实现候选匹配和全局 assignment；
3. 实现邻接一致性验证；
4. 对同一 STEP 重复导入、旋转件和典型特征做回归；
5. 输出 mapping report 和 ambiguity reason。

门禁：目标面映射不唯一时必须停止，不能降级为顺序或面积猜测。

### Phase P6-B：Abaqus compiler/executor

工作：

1. 定义 `validated_cae_task` schema；
2. 重构 `step5_script_gen.py` 为白名单 compiler；
3. 实现 Abaqus noGUI executor；
4. 创建 Set/Surface、BC、Load、Material 和 Mesh；
5. 返回结构化执行结果；
6. 增加 dry-run 和人工确认包。

门禁：无静默异常；每个操作都能从输入任务追踪到 Abaqus 对象和执行结果。

### Phase P7：端到端实验与论文证据

数据轨道：

- Synthetic-Test；
- Real-Dev：`test_part` 至 `test_part_s2`；
- 模型和规则冻结后创建的简化 Real-Holdout；
- Oracle geometry 与 predicted geometry 双轨。

主要指标：

- CAE-IR exact match、slot Macro-F1、clarification accuracy；
- object selection accuracy；
- face-set IoU 和 exact face-set accuracy；
- STEP-to-Abaqus mapping accuracy/coverage/ambiguity；
- Set/Surface 创建成功率；
- BC/Load/Material 创建成功率；
- script execution success；
- abstention 和 unsafe execution rate；
- 端到端成功率及各阶段误差传播。

## 11. 人与 Agent 的分工

### 用户负责

- 审核典型结构静力需求是否符合实际 CAE 语义；
- 对少量 ambiguity/abstention 案例确认目标特征和面角色；
- 在 Abaqus 中抽查高亮面、载荷方向和约束是否合理；
- 模型冻结后绘制少量简化 Real-Holdout；
- 提供 API key，但只通过环境变量配置。

用户不需要批量填写 raw face ID，也不需要手工编写大量 Abaqus 脚本。

### 监督者 Agent 负责

- 冻结研究范围、接口和实验门禁；
- 审核执行 Goal 是否偏离主线；
- 控制 test 解锁、API 正式运行和 Abaqus 执行授权；
- 审核指标、失败案例和论文结论是否超过证据范围。

### 执行 Goal/Agent 负责

- API adapter、CAE-IR parser 和评估；
- grounder/pointer 和 verifier；
- face fingerprint mapper；
- Abaqus compiler/executor；
- 自动测试、实验报告、图表和复现清单。

## 12. 当前状态与最近下一步

当前已完成：

- P1：受控域 recognizer 冻结；
- P2：CAE-IR、feature-role policy 和测试集冻结；
- P3：多实例语言查询数据集冻结；
- P4：H4 人工审核 60/60 通过；
- Windows 到 Linux 的迁移完成，但 API 路线允许 Windows 继续作为端到端主执行环境；
- P4 正式 test 尚未解锁，运行次数为 0。

最近下一步固定为：

1. 将 P4 runtime 从“必须本地部署模型”调整为“支持 DeepSeek API provider”；
2. 完成 API adapter 的协议兼容和 mock 测试；
3. 在 train/val 范围内进行小规模真实 API smoke；
4. 冻结 P4 配置并完成正式语言实验；
5. 随即进入 STEP-to-Abaqus face fingerprint mapper；
6. 最后重构 Abaqus action compiler，形成 Windows 本地端到端闭环。

## 13. 最终判断

该路线在当前受控研究范围内可实现，且能够利用已有数据集、GNN、拓扑规则、CAE-IR、语言数据和 Abaqus 原型。采用 API 后不再受本地 LLM 算力限制，但 API 只解决语言解析问题，不能替代几何识别、精确选面、面身份映射和执行验证。

论文贡献应围绕以下组合形成，而不是把“调用大模型”本身作为创新：

1. 面向 CAE 区域选择的 STEP-graph/feature-role 数据；
2. GNN 与拓扑约束耦合的受控域 feature-object recognizer；
3. 强类型 CAE-IR 与 topology-constrained grounding；
4. 不允许 LLM 猜 raw face ID 的安全 Agent 架构；
5. STEP-to-Abaqus 面身份映射和可验证执行闭环；
6. Oracle/predicted 双轨下的端到端误差传播实验。

这条路线保留了原始 `step1` 至 `step5` 原型的工程直觉，但将其升级为可审计、可拒识、可评价和适合论文论证的系统。
