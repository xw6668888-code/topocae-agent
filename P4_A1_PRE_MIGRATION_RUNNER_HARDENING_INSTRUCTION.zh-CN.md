# P4-A1 迁移前 Runner 加固指令

## 任务定位

这是 P4-A1 执行任务的补充工作，不是项目迁移任务。目标是在连接双 A40 Linux 推理服务前，使 Structured LLM 与 Direct LLM 的大规模 train/val 推理具备可靠、可恢复、可审计的运行能力。

## 必须完成

1. 支持断点续跑：已完成且通过 schema 校验的样本不得重复请求。
2. 原子写入：结果先写临时文件，完成校验后再替换正式文件。
3. 有界并发：并发数由参数或环境变量控制，默认值保守；不能无限创建请求。
4. 失败重试：只对可恢复错误使用指数退避和有限次数重试。
5. 请求缓存：缓存键至少包含模型、prompt 版本、推理配置和输入 payload 的稳定哈希。
6. 运行清单：记录模型标识、endpoint 的脱敏标识、prompt/config 哈希、数据 split、样本数、成功数、失败数、缓存命中数、开始/结束时间和代码版本信息。
7. Qwen3 配置：显式关闭 thinking/reasoning 输出，并保持严格 JSON 响应契约。
8. 支持受控子集运行、全量 val 运行和失败样本重放。
9. Structured 与 Direct 两条基线必须复用同一套可靠传输层，但保持各自 prompt、schema 和结果目录隔离。
10. 增加自动化测试，至少覆盖：中断恢复、损坏缓存拒绝、原子写入、并发上限、重试耗尽、重复运行幂等和敏感信息不落盘。

## 冻结约束

- 不修改 P1/P2/P3 冻结资产。
- 不修改 H4-01 已审核内容。
- 不运行任何正式 test，test ledger 必须保持 0。
- 未连接真实 endpoint 时，不伪造 Structured/Direct 的正式 train/val 结果。
- 不迁移项目，不安装 Linux 环境，不进入 P5，不执行 Abaqus/CalculiX。

## 交付

- 加固后的唯一运行入口及参数说明。
- 自动化测试与测试结果。
- 中文实现报告，说明恢复语义、并发策略、缓存键、失败策略和审计字段。
- 机器可读 acceptance JSON。
- 改动文件 SHA-256 清单。
- 最终状态只能是：
  - `READY_FOR_REMOTE_RUNTIME`；
  - `BLOCKED_WITH_EVIDENCE`。

完成后停止，等待监督者验收和 Linux endpoint 配置。
