# P4-API-R2A Compact CAE Query Frame Design

## 目标与边界

构建 `CAEQueryFrame v1 -> frozen P2 CAEIntentResult` 的完全离线、确定性编译链。LLM 未来只生成自然语言中可观察的 CAE 语义，不生成 canonical 样板字段、几何身份、face ID、feature instance ID、ground truth 或 source identity。本阶段不调用 API，不读取或解码 formal test，不修改 P1/P2/P3 冻结资产和 R1C/R1E/R1F 历史交付。

验证口径固定为 R1E.1 的 21,885 条：P2 fixtures 265 条、P3 train 19,180 条、P3 val 2,440 条。P2 的 159 条非法 fixture 继续按 R1E.1 规则转换为冻结 validator 首个错误对应的 `INVALID` 包络。

## 方案比较

1. **语义扁平 Frame（采用）**：根节点保留状态、语言和操作数组；每个操作直接表达 action、feature、role、selector、descriptor 与载荷/约束参数。编译器补全固定 canonical 常量、默认单位上下文、空审计数组和规则化 operation ID。优点是 CAE 语义清晰、输出紧凑、映射可逆；代价是编译器需要显式映射七类操作。
2. **短别名 Frame**：用单字母或缩写字段压缩输出。体积最小，但工程语义差、prompt 易错，不满足字段名称应体现 CAE 工程语义的要求。
3. **canonical 子树透传**：删除少量根级样板后保留 target/parameters。实现最简单，但仍要求 LLM 生成完整嵌套结构，违背阶段核心目标，且紧凑性收益不足。

## Frame 结构

根节点是关闭的普通 JSON object，版本为 `cae-query-frame/1.0.0`，状态使用 `valid`、`needs_clarification`、`invalid`。`valid` 分支包含 `language` 与一个或多个 `operations`；非 valid 分支只包含稳定错误、缺失信息、歧义和澄清问题，不允许夹带 operations。

操作对象采用语义扁平字段：`action`、`feature`、`face_role`、`selector`、`instance_count`、`compare_by`、`geometry_filters`、`selection_frame`，以及七类操作所需的 `mesh_size`、`magnitude`、`direction`、`displacements`、`material`、`constrained_dofs`。默认 operation ID 为数组位置对应的 `op1...opN`；仅当 canonical 使用其他 ID 时，Frame 才携带 `operation_key`。

数量对象省略可由字段语义唯一推断的 dimension，只保留 value、unit 和可选 tolerance。编译器按字段语义恢复 length、pressure 或 displacement。目标过滤器保留明确 operator、上下界、空间关系和语义参照，不进行字符串别名或模糊修复。

冻结数据中的 valid intent 均使用 `static_structural`、统一单位系统、空 assumptions 和空 normalization audit；这些由编译器固定补全。若未来 canonical 合法输入不满足这些 v1 前提，反向转换必须 fail closed，而不是静默丢失信息。

## 编译与错误模型

入口先验证 Python 根类型，再用交付 Schema 验证关闭字段、必填字段和枚举。错误按稳定顺序选择，映射为稳定的 `QF_E_*` 代码；未知枚举、缺字段、额外字段和冲突分支全部拒绝。编译仅做显式结构映射，最终调用冻结 `validate_intent_result`；任何 canonical 拒绝转为 `QF_E_CANONICAL_REJECTED`。

`compile_query_frame_json` 使用 UTF-8、键排序和紧凑分隔符输出 bytes，因此相同输入逐字节一致。反向 `canonical_to_query_frame` 只服务离线审计，并在遇到 v1 不可逆 canonical 状态时 fail closed。

## 数据与安全

P3 仅通过 `utils.cae_language_grounding.dataset.iter_canonical_records(train|val)` 读取。该入口只解码物理 train/val 文件，并将 supplement test 行作为 opaque bytes 跨过。审计记录 decode counters，要求 decoded test=0。Frame 构造只读取每条记录的 `intent_result`、query language/difficulty 统计字段和 split 计数；不读取 ground truth、STEP、graph 或 source identity。

重复 identity 以授权 canonical record ID 清单的 manifest 级摘要核验为 0；实现过程不把 identity 传入 Frame 或编译器。formal ledger 前后必须一致且访问计数为 0。

## Prompt 与 Repair

主 prompt 输入严格为 `{text, language}`，输出仅一个 CAEQueryFrame JSON object，不含 val/test 原句 few-shot。允许最多一次显式 repair；repair 输入仅原 Frame、稳定校验错误码、错误路径和必要 Schema 摘要。repair 必须产生可审计的第二个候选，禁止 silent repair、循环修复或 API fallback。本阶段只冻结文本和协议，不执行请求。

## 测试与验收

标准库 `unittest` 覆盖三状态、七操作、单/多操作、量词与过滤器、非法枚举/缺字段/冲突字段、无身份字段、逐字节确定性和 canonical exact。全量审计要求 21,885/21,885 schema-valid、semantic exact、deterministic reproduction；Schema UTF-8 <=10 KiB；Frame/canonical 逐记录比值的中位数 <=0.65。最后在指定解释器下运行完整 `test_p4*.py`，并依据冻结 manifest 重算 P1/P2/P3 哈希。

## 自审结论

设计不含占位项；状态与错误分支互斥；数量维度推断规则与七类操作映射一致；验证范围、正式测试隔离、紧凑性口径和停止状态均明确。Git 元数据当前不可用，因此不执行设计文档提交，以 delivery manifest SHA-256 代替提交留痕。
