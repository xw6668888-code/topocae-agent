# Goal P4-A1：CAE 语言解析与确定性 Grounding 基线

## 0. Goal 身份

你是本项目 P4-A1 的实施 Agent。你的任务是在冻结 P1/P2/P3 资产的前提下，
完成自然语言 CAE 需求解析、确定性对象/面角色 grounding、关键词基线和开源
LLM 基线，并建立严格隔离 train/val/test 的正式评估。

本 Goal 不训练任何模型，不进入 P5 Pointer，不执行 Abaqus/CalculiX，不重新
优化几何识别器。

最终状态只能是：

- `PASS`：P4-A1 全部完成且满足研究 Gate；
- `BOUNDED-PASS`：工程闭环完成，但语言指标未达到 GO-P5 门槛；
- `BLOCKED`：缺少 LLM 运行时或真实人工 H4 审查，且已按规定暂停；
- `FAIL`：存在数据泄漏、冻结资产变化、评估不可复现或关键契约失败。

完成后必须停止，等待监督者验收，不得自行进入 P5。

## 1. 必读输入

先阅读以下文件并写 preflight 摘要：

1. `W:\python\FEA\dataset_generator\ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
2. `W:\python\FEA\dataset_generator\reports\P3_A1R_SUPERVISOR_ACCEPTANCE.zh-CN.md`
3. `W:\python\FEA\dataset_generator\reports\p1_d_lite_v1\freeze\capability_contract.json`
4. `W:\python\FEA\dataset_generator\reports\p2_a1_cae_ir_v1\final_acceptance.zh-CN.md`
5. `W:\python\FEA\dataset_generator\reports\p2_a1_cae_ir_v1\freeze\freeze_manifest.json`
6. `W:\python\FEA\dataset_generator\reports\p3_a1r_multi_instance_v1\final_acceptance.zh-CN.md`
7. `W:\python\FEA\dataset_generator\reports\p3_a1r_multi_instance_v1\dataset_card.zh-CN.md`
8. `W:\python\FEA\dataset_generator\output_cae_query_ground_v1_1\manifests\dataset_manifest.json`
9. `W:\python\FEA\dataset_generator\read_cae_query_ground_v1_1.py`
10. `W:\python\FEA\dataset_generator\utils\cae_intent.py`
11. `W:\python\FEA\dataset_generator\configs\cae_feature_role_policy.yaml`
12. `W:\python\FEA\dataset_generator\schemas\cae_intent.schema.json`
13. `W:\python\FEA\dataset_generator\schemas\cae_intent_result.schema.json`

## 2. 冻结边界

以下目录和文件只读，禁止原地修改：

- P1/P1-D Lite 的全部冻结资产；
- `reports/p2_a1_cae_ir_v1/freeze/`；
- `output_cae_query_ground_v1/`；
- `output_cae_query_ground_v1_1/`；
- `output_generation_p3_a1r_multi_instance_formal_v1/`；
- `reports/p3_a1_cae_queryground_v1/`；
- `reports/p3_a1r_multi_instance_v1/`。

开始和结束时都要重新计算 P1/P2/P3 冻结清单哈希。任何变化立即 `FAIL`。

禁止访问或运行：

- `test_part_s3`；
- Real-Holdout；
- P1 Holdout；
- Abaqus/CalculiX；
- P5/Pointer 代码；
- 任何写有禁用/隔离含义的资产。

不得把 API key、token、服务器密码、私有 endpoint 写入代码、报告或日志。

## 3. 研究问题与隔离原则

P4-A1 只回答：

> 在 Oracle feature-object memory 已知时，规则方法和开源 LLM 能否把中英文
> CAE 需求解析为冻结 CAE-IR，并通过确定性、可验证的规则选择正确的对象和
> STEP face set？

必须采用 Oracle geometry 隔离语言误差：

- P4-A1 的 graph、feature instance、semantic role 和 descriptor 来自 P3
  Oracle graph；
- 不调用 P1 recognizer 生成预测标签；
- 不把 predicted geometry 误差计入 P4-A1；
- predicted geometry 的端到端误差传播留给后续独立阶段。

LLM 只做语言理解和结构化 CAE-IR，不直接读取 STEP，不自由生成 face ID。

## 4. 数据使用纪律

### 4.1 Split

- train：允许用于 prompt example、词典和规则开发；
- val：允许用于 prompt/规则选择、阈值冻结和 H4 人工审查；
- test：在 H4 通过、prompt/config/hash 全部冻结后，只允许正式运行一次。

在 test 正式运行前：

- 不得打印、浏览、抽样或人工阅读 test query/ground truth；
- 不得用 test 指标选择 prompt、模型、解析规则或重试策略；
- 只允许从 dataset manifest 读取 test 的数量和文件哈希。

实现一个 test-run ledger。正式 test 每个 baseline 只允许一次有效运行；异常中断
必须留下不可篡改的失败记录，是否重跑由监督者决定。

### 4.2 推理时不可见字段

任何 baseline 推理时都不得读取：

- `ground_truth`；
- `target_instance_ids`；
- `target_face_indices`；
- `semantic_evidence`；
- `uniqueness_contract` 中的答案字段；
- query 记录中的任何可直接还原答案的内部标识。

评估器可以在预测完成后读取 ground truth。必须用测试证明推理对象中没有上述
字段。

## 5. Task A：统一 Feature-Object Memory

从 source graph 构造只读、确定性的 candidate memory，至少包含：

- part reference 和局部参考系；
- object ID（仅本次 part 内稳定）；
- canonical feature type；
- feature family；
- semantic face roles；
- selectable face-role 到 face indices 的映射；
- `geometry_parameters`；
- `descriptor_contract`；
- centroid、area、同类排序描述符和到 part centroid 的距离；
- topology/legality/abstention 状态。

要求：

1. memory 中不得包含 query target 或 ground truth；
2. object ID 与 raw STEP face ID 分层；
3. 同一个 graph 多次构造应逐字节确定；
4. 旋转不变字段与坐标相关字段要明确标注；
5. 对 17 类和 P2 role policy 建立完整覆盖测试。

## 6. Task B：Deterministic Grounder

实现：

```text
CAE-IR + feature-object memory
-> candidate filtering
-> quantifier/descriptor/spatial selection
-> semantic face-role expansion
-> exact face set
-> verifier result
```

必须支持：

- `all`；
- `count`；
- `largest`；
- `smallest`；
- `nearest(part_centroid)`；
- singleton；
- whole-part operation；
- `VALID / NEEDS_CLARIFICATION / INVALID`；
- P2 中所有 operation、feature type、face role、unit 和 direction 契约。

确定性 grounder 不得从 query 文本解析语义，只接受已经通过 P2 schema 的
`cae-intent-result`。

先运行 Oracle-IR 上界：

- 输入 P3 冻结的正确 CAE-IR；
- 推理时不读取 ground truth target；
- 对全量 train/val 验证；
- H4 后对 test 正式运行一次。

Oracle-IR grounder 的硬门槛：

- status 一致率 100%；
- executable target reconstruction 100%；
- exact face-set accuracy 100%；
- 无越界 face ID；
- 无 silent fallback。

任何一项不足 100%，不得进入 LLM test；先修 grounder。

## 7. Task C：语言 Baselines

### C1. Keyword/Rule baseline

实现中英文关键词/规则解析器：

```text
query text -> P2 CAE-IR -> deterministic grounder
```

规则只能使用 train 开发、val 选择；不得为单个 test 句子增加特例。

### C2. Structured LLM baseline

使用一个可本地部署的开源 instruct LLM：

```text
query text -> constrained JSON CAE-IR -> P2 validator
-> deterministic grounder -> exact face set
```

运行时必须通过环境变量配置，例如：

- `TOPOCAE_LLM_BASE_URL`
- `TOPOCAE_LLM_MODEL`
- `TOPOCAE_LLM_API_KEY`（本地服务无需 key 时可省略）

要求：

- OpenAI-compatible HTTP 或明确封装的本地推理后端；
- temperature=0；
- 记录 seed、模型全名/revision、dtype、量化、backend、GPU、context length；
- 使用 JSON Schema/constrained decoding；若后端不支持必须明确记录；
- 最多一次修复重试，首轮与重试结果分别计数；
- 禁止静默修 JSON；
- prompt examples 只能来自 train；
- prompt 与配置在 H4 后冻结。

若没有可用 endpoint/model：

1. 仍完成 Task A、B、C1、评估器和 H4 包；
2. 生成 `LLM_RUNTIME_REQUEST.zh-CN.md`，写明用户需要启动的服务、环境变量和
   健康检查命令；
3. 状态标记 `BLOCKED_AT_H4_RUNTIME` 并停止，不得下载模型到 C 盘。

### C3. Direct-LLM selection baseline

使用与 C2 完全相同的 LLM、温度和 test 样本，输入 query 加去除 ground truth
的 candidate memory，让模型直接选择 object/face role。

该 baseline 只用于论文对比：

- 不允许访问 STEP；
- 不允许从 ground truth 复制 ID；
- 输出必须经过范围校验；
- 不通过 P2 CAE-IR 进行中间规划；
- 不得替代 C2 主路线。

## 8. Task D：评估协议

至少报告以下四条路线：

1. `Oracle-IR -> deterministic grounder`；
2. `Keyword -> CAE-IR -> deterministic grounder`；
3. `LLM -> CAE-IR -> deterministic grounder`；
4. `Direct LLM -> object/face-role selection`。

### 8.1 语言/CAE-IR 指标

- schema-valid rate（首轮、重试后）；
- status accuracy；
- CAE-IR canonical exact match；
- operation type accuracy；
- feature type、face role、quantifier、count、filter、parameter、unit、direction
  的 slot P/R/F1；
- clarification/invalid detection P/R/F1；
- 中英文分别报告；
- L1--L5 分层报告；
- 17 类、operation、quantifier、candidate_count 分层报告；
- paraphrase consistency；
- latency、tokens、失败和重试率。

### 8.2 Grounding 指标

- object selection accuracy；
- object P/R/F1；
- exact instance-set accuracy；
- face precision/recall/F1/IoU；
- exact face-set accuracy；
- executable success；
- clarification/abstention accuracy；
- illegal/out-of-range ID rate。

### 8.3 误差分解

必须区分：

- parser error；
- P2 schema/policy rejection；
- candidate filtering error；
- quantifier/descriptor selection error；
- face-role expansion error；
- verifier rejection；
- direct-LLM invalid ID/error。

生成 failure atlas，至少给出每类前 10 个 val 失败案例；test 只生成聚合统计和
机器可读错误类型，不得人工回看后再调参。

## 9. H4-01 人工 Gate

在读取 test 内容和正式运行 test 之前，生成：

`reports/p4_a1_language_grounding_v1/human_review_package/`

其中包含 60 条 val 审查样本，按以下维度分层：

- 中文/英文；
- 17 类；
- all/count/largest/smallest/nearest；
- VALID/NEEDS_CLARIFICATION/INVALID；
- 至少覆盖 fixed support、pressure、traction、displacement、local mesh。

每行必须用 CAE 工程人员可读形式展示：

- query；
- baseline 名称；
- 解析出的结构化操作摘要；
- feature type、face role、quantifier/filter；
- 参数、单位和方向；
- 是否请求澄清；
- Oracle 摘要；
- reviewer decision：`approve/edit/reject`；
- reviewer note。

不要要求用户逐面填写 face ID。用户只需审查自然语言理解和 CAE 语义。

生成后暂停，明确告诉用户：

1. 打开哪个 CSV；
2. 每列检查什么；
3. 如何填写 decision/note；
4. 哪些错误属于语言解析错误，哪些不属于本阶段。

收到真实人工回填之前，不得运行正式 test。

## 10. H4 后冻结与一次性 Test

H4 通过后：

1. 只使用 train/val 和人工 H4 意见修复通用 prompt/规则；
2. 冻结 prompt、模型配置、parser、grounder、verifier、评估器及哈希；
3. 写入 test-run ledger；
4. 对四条 baseline 各正式运行一次 test；
5. 生成完整对比表、分层指标、置信区间和 failure taxonomy；
6. 不根据 test 结果继续调参。

置信区间使用可复现 bootstrap，并固定随机种子。对同一样本上的 baseline 差异
使用配对统计。

## 11. GO-P5 判据

工程硬门槛：

- Oracle-IR grounder exact face-set accuracy = 100%；
- 所有输出可复现；
- P1/P2/P3 冻结资产变化 = 0；
- test run 次数符合协议；
- 推理输入 ground-truth leakage = 0；
- illegal/out-of-range face ID = 0。

主 Structured LLM 的建议 GO-P5 门槛：

- 重试后 schema-valid rate >= 99%；
- status accuracy >= 95%；
- CAE-IR slot Macro-F1 >= 95%；
- executable object selection accuracy >= 95%；
- exact face-set accuracy >= 90%；
- clarification/invalid Macro-F1 >= 85%。

若工程硬门槛通过但 LLM 指标未达到建议值，状态为 `BOUNDED-PASS`，如实报告，
由监督者决定做 P4-A1R 还是进入受限 P5。禁止为了过线读取 test 并调参。

## 12. 交付物

代码建议放在：

- `utils/cae_language_grounding/`
- `tests/test_p4_a1_*.py`
- `run_p4_a1_*.py`

数据和报告放在：

- `output_p4_a1_language_grounding_v1/`
- `reports/p4_a1_language_grounding_v1/`

至少交付：

1. `preflight.zh-CN.md`
2. feature-object memory schema 与 builder
3. deterministic grounder 与 verifier
4. keyword parser
5. structured LLM adapter
6. direct-LLM baseline
7. train/val/test 防泄漏测试
8. inference-input leakage 测试
9. H4 人工审查包及验收记录
10. baseline 对比 CSV/JSON/Markdown
11. failure atlas
12. test-run ledger
13. reproduction commands
14. dataset/model/prompt/config card
15. `acceptance.json`
16. `final_acceptance.zh-CN.md`
17. `delivery_manifest.json`
18. `freeze/freeze_manifest.json`
19. `CODEX_CONTEXT_P4_A1.md`

## 13. 实施顺序

严格按以下顺序：

1. preflight 与冻结哈希；
2. Task A feature-object memory；
3. Task B deterministic grounder；
4. Oracle-IR train/val 100% Gate；
5. Task C1 keyword baseline；
6. LLM runtime preflight；
7. Task C2/C3 仅在 train/val 开发；
8. 生成 H4-01 包并暂停；
9. 用户真实回填后冻结 prompt/config；
10. 一次性正式 test；
11. 报告、哈希和最终验收；
12. 停止，等待监督者决定 P5。

不要把 P4-A1 扩展成新的识别器优化项目，也不要提前实现 Pointer 或 Abaqus。
