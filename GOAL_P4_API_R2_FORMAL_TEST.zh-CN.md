# Goal P4-API-R2：一次性 Formal Test 与 P4 冻结

## 1. 启动授权

本 Goal 只有在以下条件同时成立时才能启动：

- P4-API-R1 经监督者验收为 PASS/可接受 BOUNDED-PASS；
- post-H4 freeze manifest 完整；
- `supervisor_go_formal_test.json` 明确授权；
- test ledger 中四条 baseline run 均为 0。

缺少任一条件立即停止。创建本任务书本身不构成 test 授权。

## 2. 任务

1. 复核所有冻结 hash、runtime card、模型实际 ID 和 API 可用性；
2. 锁定 prompt/config/code，不再修改；
3. 按 ledger 协议分别运行：Oracle-IR、Keyword、Structured LLM、Direct LLM；
4. 每条 baseline 只运行一次有效 formal test；
5. 网络中断必须形成不可篡改失败记录，是否恢复由 ledger 协议和监督者决定；
6. 不人工阅读 test query/ground truth，不根据结果调参；
7. 生成 paired bootstrap、分层指标和 failure taxonomy；
8. 验证 P1/P2/P3/H4 和 post-H4 freeze 无变化；
9. 冻结 P4 最终资产并停止。

## 3. 交付物与状态

交付正式 baseline 表、语言/CAE-IR 指标、grounding 指标、费用/延迟、聚合 failure taxonomy、ledger、acceptance、final report、freeze manifest 和 context。

最终状态只能是：

```text
PASS_P4
BOUNDED_PASS_P4
FAIL_P4_WITH_EVIDENCE
```

不得自行进入 P5。

