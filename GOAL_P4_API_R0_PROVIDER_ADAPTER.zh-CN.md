# Goal P4-API-R0：API Provider 适配与运行可靠性准备

## 1. 身份与目标

你是 P4-API-R0 执行 Agent。你的任务是把现有 P4 Structured/Direct LLM 运行层改造成可可靠连接 DeepSeek V4 Pro 和一般 OpenAI-compatible 服务的 provider-aware adapter。

本 Goal 不调用真实收费 API，不运行 train/val，不读取或运行 formal test，不进入 P5，不执行 Abaqus。完成后必须停止在：

```text
READY_FOR_P4_API_SMOKE_REVIEW
```

## 2. 必读文件

1. `WINDOWS_API_ABAQUS_AGENT_ROUTE.zh-CN.md`
2. `TOPOCAE_AGENT_SUPERVISOR_IMPLEMENTATION_PLAN.zh-CN.md`
3. `GOAL_P4_A1_LANGUAGE_AND_GROUNDING_BASELINES.zh-CN.md`
4. `CODEX_CONTEXT_P4_A1.md`
5. `output_p4_a1_language_grounding_v1/acceptance.json`
6. `output_p4_a1_language_grounding_v1/test_run_ledger.json`
7. `utils/cae_language_grounding/llm_adapter.py`
8. `utils/cae_language_grounding/direct_llm.py`
9. `schemas/cae_intent_result.schema.json`
10. `schemas/p4_a1_direct_selection.schema.json`

## 3. 冻结和禁止事项

- P1/P2/P3、H4 和 formal-test 资产只读；
- test ledger 的所有 run 数必须保持 0；
- 不下载模型、不启动 vLLM、不访问真实 API；
- 不把 endpoint credential、API key 或 bearer header 写入交付物；
- 不改变 prompt、数据 split、指标和 Grounder 语义；
- 不修改旧五阶段原型来替代正式 P4 代码；
- 不使用 superpower 类技能。

## 4. 实施任务

### R0-A：状态与冻结预检

- 复核 H4=PASS、P4=`BLOCKED_AT_H4_RUNTIME`、formal test run=0；
- 计算上游 freeze manifest；
- 记录现有 adapter、prompt、schema、runner 的 SHA-256；
- 检查源码中是否存在硬编码 secret，只报告路径和类型，不复制 secret 内容。

### R0-B：Provider profile

实现明确的 provider/runtime 配置，至少支持：

- `openai_json_schema`：现有本地/OpenAI-compatible constrained schema；
- `deepseek_json_object`：`json_object` + 本地 schema validator；
- endpoint 可以配置为完整 chat-completions URL 或 base URL，不重复拼接 `/v1`；
- 可按 provider 开关 `seed`、`temperature`、`thinking` 和 `max_tokens`；
- DeepSeek 结构化解析默认关闭 thinking；
- 空 content、截断、无 usage、错误 response shape 均有稳定错误码；
- 返回并记录脱敏后的实际 model、system fingerprint、token 和延迟元数据。

不得使用未验证的静默 fallback。`json_schema` 不受支持时，只能根据显式 provider profile 使用 `json_object`，不能在一次正式请求中偷偷改协议。

### R0-C：可靠 runner

补齐或复用：

- 原子缓存；
- request hash，包含 provider/model/revision/prompt/schema/baseline/input；
- Structured 与 Direct 缓存隔离；
- 断点续跑；
- 可恢复 HTTP 429/5xx/timeout 有限重试；
- 不可恢复 4xx 不重试；
- 首轮 schema 失败最多一次 LLM 修复；
- deterministic limit/offset/shard；
- secret redaction；
- formal-test lock 检查。

### R0-D：离线测试

用 mock transport/server 覆盖：

- OpenAI `json_schema` 成功；
- DeepSeek `json_object` 成功；
- endpoint 组合；
- thinking disabled payload；
- 首轮非法 JSON/非法 schema 后一次修复；
- 空 content、length truncation、缺失 usage；
- 429/500/timeout 与不可恢复 4xx；
- 中断恢复、缓存命中、缓存损坏；
- Structured/Direct 隔离；
- manifest 不泄漏 secret；
- test split 在未授权状态被拒绝。

## 5. 交付物

放入 `reports/p4_api_r0_provider_adapter_v1/`：

- `preflight.json` 和 `preflight.zh-CN.md`；
- `provider_contract.json`；
- `mock_test_results.json`；
- `security_audit.json`；
- `api_smoke_procedure.zh-CN.md`；
- `acceptance.json`；
- `final_acceptance.zh-CN.md`；
- `delivery_manifest.json`；
- 最新 P4 API context。

## 6. 验收门槛

- 新旧 provider mock 全通过；
- P4 原有测试无回归；
- 上游冻结变化 0；
- formal-test run=0、内容未访问；
- secret 泄漏 0；
- 相同输入缓存幂等；
- 最终只报告 `READY_FOR_P4_API_SMOKE_REVIEW` 或 `BLOCKED_WITH_EVIDENCE`。

