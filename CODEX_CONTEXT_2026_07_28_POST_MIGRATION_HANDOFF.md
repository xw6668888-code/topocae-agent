# Codex Context Snapshot - 2026-07-28 Post-Migration Handoff

## 当前状态

- Linux 迁移已经完成并签署 `MIGRATION-PASS`。
- M0-M5 全部 PASS；最终 43/43 交付 SHA-256 通过。
- Linux `/home/ywx1/dataset_generator` 是主要执行环境。
- Windows 工作区与 Snapshot A/B 保留为备份和后续 Abaqus 执行节点来源。
- P1-D Lite、P2-A1、P3-A1R 已冻结。
- P4 H4 人工审查通过；Oracle/Keyword train+val 完成。
- Structured/Direct LLM 尚未正式部署或完成 train/val。
- P4 正式 test 运行 0 次、内容未访问、未解锁。
- LLM、P4 test、P5 当前仍未授权。

## 唯一下一阶段

`P4_RUNTIME_PREPARATION_SUPERVISOR_REVIEW`

监督者应先审查并授权：

- `GOAL_P4_R0_RUNTIME_PREPARATION.zh-CN.md`

该 Goal 只允许 runner 加固、mock 测试、模型部署提案和 GPU/磁盘/网络预算；不得下载模型、调用真实 LLM、运行正式 train/val 或 test。

## 新监督模式

- Linux 长期监督任务读取 `LINUX_SUPERVISOR_CONTINUATION_PROMPT.zh-CN.md`。
- 同时最多一个执行 Goal。
- Goal 完成后停止，由用户将交付摘要返回监督者。
- 监督者使用 `通过 / 有条件通过 / 不通过` 决定下一任务。

## Abaqus 工作模式

Linux 不安装 Abaqus。后续 P6 使用混合执行：

- Linux：识别、LLM、CAE-IR、grounding、pointer、verifier 和 execution package。
- Windows/Abaqus：导入 STEP、映射 face、创建 set/surface/BC/load，返回结构化执行结果。
- 第一版使用人工复制 bundle 和固定 Abaqus 命令；稳定后再决定是否实现 Windows worker。
- 无 Abaqus 不影响 P4/P5；但没有 Windows 上的真实 Abaqus 验证，就不能声称完成 P6 可执行闭环。

## 必读主计划

- `POST_MIGRATION_PROJECT_SUPERVISION_AND_HYBRID_CAE_PLAN.zh-CN.md`
- `TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
- `ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
- `reports/migration/supervisor_migration_pass.json`
- `output_p4_a1_language_grounding_v1/test_run_ledger.json`

## 禁止事项

- 不使用 `test_part_s3`。
- 不回到通用复杂 CAD 识别优化。
- 不让 LLM 直接读取 STEP 或猜 face ID。
- 不让 LLM 生成任意 Python 并在 Abaqus 执行。
- 不在正式 test 或 Real-Holdout 上调参。
- 不由执行 Goal 自行进入下一阶段。
