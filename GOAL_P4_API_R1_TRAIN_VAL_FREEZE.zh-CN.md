# Goal P4-API-R1：真实 API Smoke、Train/Val 与配置冻结

## 1. 身份与前置条件

你是 P4-API-R1 执行 Agent。只有在监督者确认 P4-API-R0 为 PASS 后才能开始。用户通过环境变量提供 API 配置；不得要求用户把 key 写进文件。

目标是使用同一个冻结候选模型完成 Structured LLM 与 Direct LLM 的非 test 实验、费用/延迟审计和 post-H4 配置冻结。不得运行 formal test。

## 2. 运行配置

优先候选：

```text
TOPOCAE_LLM_PROVIDER=deepseek_json_object
TOPOCAE_LLM_BASE_URL=https://api.deepseek.com
TOPOCAE_LLM_MODEL=deepseek-v4-pro
TOPOCAE_LLM_API_KEY=<environment only>
```

实际 model ID 必须通过接口响应确认。禁止把 key、Authorization header 或账户余额写入日志。

## 3. 执行顺序

1. 只读验证 R0、P1/P2/P3/H4、ledger=0；
2. 调用非 test 的最小健康请求；
3. Structured/Direct 各运行 deterministic 50 条 smoke；
4. 通过后扩到 200，再扩到 1000；
5. 输出费用、延迟、token、schema failure、retry、cache 和预计全量预算；
6. 暂停并生成人工门禁 `H4-API-Budget`；
7. 用户批准后先运行完整 val，再运行完整 train；
8. 只允许根据 train/val 和既有 H4 修改通用 prompt/config；任何修改后受影响的 val 必须重新运行并记录 superseded run；
9. 选择最终模型、provider、prompt、schema、retry、grounder、verifier 和 evaluator；
10. 冻结全部 hash，生成正式 test 解锁申请，但不得自行解锁。

日常 API 推理不是训练。本 Goal 中运行 train 的目的仅是完成既有 P4 评估协议和分层统计，不得声称进行了模型微调。

## 4. 失败和止损规则

- 50 条 smoke 不能稳定返回合法 JSON：停止，不扩大调用；
- 1000 条前预计费用或时间超过用户批准预算：停止等待；
- response model 与配置不一致：停止；
- schema-valid、status 或 slot 指标明显低于 P4 门槛：只允许基于 train/val 做一次有记录的通用修订循环；
- 不允许用 formal test 调 prompt；
- API 中断使用已验证缓存续跑，禁止删除有效 ledger 后重来。

## 5. 交付物

放入 `reports/p4_api_r1_train_val_v1/`：

- 分级 smoke 结果；
- `H4-API-Budget_review.zh-CN.md` 和用户确认记录；
- Structured/Direct train/val 指标；
- latency/token/cost audit；
- failure atlas；
- frozen runtime card；
- post-H4 freeze manifest；
- formal-test unlock request；
- acceptance、final report、delivery manifest 和 context。

## 6. 门槛与停止状态

硬门槛：冻结资产变化 0、leakage 0、illegal raw face ID 0、formal test run 0、缓存审计完整。

建议指标沿用 P4-A1：retry 后 schema-valid >=99%，status accuracy >=95%，CAE-IR slot Macro-F1 >=95%，executable object selection >=95%，exact face-set >=90%，clarification/invalid Macro-F1 >=85%。

完成后停止于：

```text
READY_FOR_P4_FORMAL_TEST_REVIEW
```

