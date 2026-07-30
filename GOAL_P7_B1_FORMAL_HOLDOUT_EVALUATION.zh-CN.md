# Goal P7-B1：冻结 Real-Holdout 一次性端到端评估

## 1. 前置人工工作

用户在 P7-A1 完全冻结后创建 6-10 个简化、受控、未参与开发的 STEP 零件。每个零件提供 3-5 条结构静力需求，并确认特征类型、目标语义面和预期操作。

Agent 应先生成高亮/面积/角色审查包，用户只确认目标区域，不要求批量填写 raw face ID。

## 2. 数据纪律

- Holdout 在系统冻结后才可建立；
- 审查真值与推理输入严格隔离；
- 正式运行前不得用 holdout 修改模型、prompt、规则、阈值或 mapper；
- Oracle/predicted 双轨各运行一次；
- 失败后不得调参重跑；基础设施中断按不可篡改 ledger 处理。

## 3. 评估

报告 P7-A1 中全部分层和端到端指标，并增加：

- 零件级完全成功率；
- 需求级完全成功率；
- feature family / operation / language 分层；
- rotation consistency；
- ambiguity/abstention 的安全收益；
- Oracle 上界与 predicted 实际性能；
- 95% bootstrap confidence interval；
- 失败案例只用于论文讨论，不用于回头修系统。

## 4. 最终冻结

交付 formal ledger、真值协议、聚合结果、可公开失败案例、复现命令、全树 hash、实验卡和 paper-ready CSV/JSON。

完成后状态：

```text
EXPERIMENTS_FROZEN
```

