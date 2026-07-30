# Goal P4-R0：开源 LLM Runtime 与大规模推理可靠性准备

## 1. 角色与停止点

你是 P4-R0 执行 Agent。目标是让 Structured LLM 和 Direct LLM 在 Linux 双 A40 环境上达到“可部署、可恢复、可审计”的准备状态。

本 Goal **不下载正式模型权重、不调用真实 LLM、不运行正式 train/val、不读取或运行 test**。完成后停在：

```text
READY_FOR_MODEL_DEPLOYMENT_REVIEW
```

等待监督者决定 P4-R1。

## 2. 必读文件

1. `POST_MIGRATION_PROJECT_SUPERVISION_AND_HYBRID_CAE_PLAN.zh-CN.md`
2. `GOAL_P4_A1_LANGUAGE_AND_GROUNDING_BASELINES.zh-CN.md`
3. `P4_A1_PRE_MIGRATION_RUNNER_HARDENING_INSTRUCTION.zh-CN.md`
4. `output_p4_a1_language_grounding_v1/acceptance.json`
5. `output_p4_a1_language_grounding_v1/test_run_ledger.json`
6. `reports/p4_a1_language_grounding_v1/LLM_RUNTIME_REQUEST.zh-CN.md`
7. `migration/linux/topocae-llm.environment.yml`
8. `migration/linux/topocae-llm-pip.freeze.txt`
9. `reports/migration/supervisor_migration_pass.json`

## 3. 冻结与禁止范围

- 不修改 P1/P2/P3 冻结资产。
- 不修改 H4 人工审查结果和受保护字段。
- 不解码、打印、查看或运行 test query/ground truth。
- test ledger 的四条 baseline runs 必须保持空数组。
- 不下载模型权重，不启动 vLLM，不调用真实 endpoint。
- 不进入 P5，不执行 Abaqus/CalculiX。
- 不改变论文数据 split 和指标定义。
- 非必要不要使用 superpower 类技能。

## 4. Task R0-A：只读 Preflight

记录：

- `topocae-llm` Python、vLLM、PyTorch、CUDA；
- 两张 A40 的显存、利用率、进程和 NUMA 拓扑；
- 可用磁盘；
- Hugging Face、ModelScope 或组织批准模型源的可访问性；
- 当前 P4 candidate code/hash；
- 当前 train/val 记录数和预计总请求数；
- test ledger = 0。

不得读取 test payload 内容，只能读取 ledger 和允许的计数元数据。

## 5. Task R0-B：Runner 可靠性加固

审查并加固 Structured/Direct 公共传输层：

1. 断点续跑：只跳过缓存键一致且 schema 校验通过的结果。
2. 原子写入：临时文件 + 校验 + rename。
3. 请求缓存：键至少包含 model/revision、prompt hash、runtime config、baseline、input payload hash。
4. 有界并发：CLI/env 可配置，默认保守。
5. 有限重试：只重试可恢复网络/服务错误，指数退避，有最大次数。
6. 幂等：相同输入重复执行不增加有效请求数。
7. 失败重放：支持只重放失败样本。
8. 子集和分片：支持 deterministic limit、offset/shard，但不能改变正式样本顺序或评估集合。
9. Structured 与 Direct 的 cache/output 完全隔离。
10. Qwen3 thinking/reasoning 显式关闭，严格 JSON 契约。
11. 运行清单记录脱敏 endpoint、model、revision、dtype、quantization、seed、context、prompt/config/code hash、样本数、缓存命中、重试、失败和时间。
12. 不把 API key、代理凭证或完整敏感 endpoint 写入报告。

## 6. Task R0-C：离线测试

使用 fake/mock OpenAI-compatible endpoint，覆盖：

- 正常严格 JSON；
- 首轮 schema 失败后一次修复；
- thinking 字段/文本拒绝；
- HTTP 429/500/timeout 的有限重试；
- 不可恢复 4xx 不重试；
- 中断后恢复；
- 原子写入；
- 缓存损坏拒绝；
- 并发上限；
- 分片无重复无遗漏；
- Structured/Direct cache 隔离；
- 运行清单不泄露 secret；
- test split 调用被拒绝。

不得用 fake 输出冒充正式指标。

## 7. Task R0-D：模型部署提案

只形成提案，不下载：

- 首选 8B instruct 模型的精确名称、revision、license 和预计权重大小；
- 合规可访问来源；Hugging Face 被阻断时不得使用不明镜像；
- 权重 SHA-256/来源记录方法；
- A40 部署方式：默认每张 GPU 一个独立 replica，不以跨 NUMA tensor parallel 为首选；
- 共享 GPU 的预约窗口和冲突检测；
- context、dtype、显存预算、并发和预期吞吐；
- 50 -> 200 -> 1000 -> full train/val 的分级运行计划；
- 129,720 级别 Structured/Direct 请求的时间与存储预算；
- 正式 test 仍不在计划内。

## 8. 人工门禁 H4-R0

生成中文审查包，用户只需要确认：

1. 模型来源是否允许使用；
2. GPU 使用时段；
3. 模型缓存目录和磁盘预算；
4. 是否接受预计 train/val 时间；
5. runner 测试是否全部通过。

不得要求用户审查神经网络代码或手工修改 JSON。

## 9. 必须交付

- 加固后的唯一 runner 入口和配置说明；
- mock endpoint 与自动化测试；
- `reports/p4_r0_runtime_preparation_v1/preflight.json`；
- `runner_reliability_tests.json`；
- `model_deployment_proposal.zh-CN.md`；
- `human_review/H4-R0_review.md`；
- `acceptance.json`；
- `final_acceptance.zh-CN.md`；
- 修改前后 SHA-256 和 delivery manifest；
- 最新 P4 context。

最终状态只能是：

- `READY_FOR_MODEL_DEPLOYMENT_REVIEW`；
- `BLOCKED_WITH_EVIDENCE`。

完成后停止，不自行开始 P4-R1。
