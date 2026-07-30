# TopoCAE-Agent 全路线监督实施计划

> 监督基线：`WINDOWS_API_ABAQUS_AGENT_ROUTE.zh-CN.md`  
> 日期：2026-07-28  
> 主工作区：Windows `W:\python\FEA\dataset_generator`  
> Linux 用途：可选的大规模开源模型实验，不再是 API 路线的前置条件

## 1. 监督目标

在不重新陷入“无限修复通用几何识别”的前提下，把已经冻结的 P1-P3 资产推进为完整论文系统：

```text
STEP + CAE requirement
-> controlled-domain feature recognition
-> feature-object memory
-> API LLM -> typed CAE-IR
-> topology-constrained grounding
-> exact STEP face set
-> STEP-to-Abaqus face mapping
-> verified Abaqus preprocessing
-> dual-track evaluation and paper evidence
```

主张边界固定为单零件结构静力前处理、受控特征类别和受控操作集合。不得宣称任意工业 CAD 泛化或完全自主有限元分析。

## 2. 不可破坏的冻结资产

- P1-D Lite / P1-B recognizer 冻结资产；
- P2 CAE-IR、schema、role policy 和 302/302 审核结果；
- P3-A1R query-ground 数据、split 和 freeze manifest；
- P4 H4-01 人工审核结果；
- P4 formal-test 内容、哈希和 test-run ledger；
- `test_part_s3` 不进入第一篇论文主评估。

每个 Goal 开始和结束时必须验证相关 freeze manifest。发现未知变化立即停止，不得自行“修复”冻结资产。

## 3. 通用执行纪律

1. LLM 不读取 STEP，不输出 raw face ID，不生成任意可执行 Python。
2. API key 只通过环境变量传入，不写入源码、日志或报告。
3. train/val 可用于开发；formal test 在配置冻结后每条基线只运行一次。
4. Direct LLM 只是基线；正式路线始终是 Structured LLM -> CAE-IR -> deterministic grounding。
5. Oracle geometry 与 predicted geometry 分轨报告，禁止掩盖识别误差。
6. STEP face 与 Abaqus face 不得按列表顺序或单一面积猜测。
7. 所有不唯一、低置信、非法或不完整结果必须 clarification/abstention/fail-closed。
8. Abaqus 执行初期只到前处理创建与验证，提交求解需要人工确认。
9. 不得因 Real-Dev 某个零件失败而增加 part ID、face ID 或绝对面积特例。
10. 每个 Goal 必须交付测试、机器验收 JSON、中文报告、复现命令、哈希清单和上下文记录。

## 4. Goal 分解与依赖

| 顺序 | Goal | 目的 | 启动条件 | 停止状态 |
|---|---|---|---|---|
| 1 | P4-API-R0 | DeepSeek/OpenAI-compatible provider 适配、缓存与 mock 验证 | 当前即可开始 | `READY_FOR_P4_API_SMOKE_REVIEW` |
| 2 | P4-API-R1 | 真实 API smoke、val/train、费用审计与配置冻结 | R0 PASS + 用户提供环境变量 | `READY_FOR_P4_FORMAL_TEST_REVIEW` |
| 3 | P4-API-R2 | 四条基线一次性 formal test 与 P4 冻结 | R1 PASS + 监督者书面授权 | `PASS_P4` / `BOUNDED_PASS_P4` |
| 4 | P5-A1 | predicted feature-object memory 与 topology-constrained grounding | P4 验收 | `READY_FOR_FACE_MAPPING` |
| 5 | P6-A1 | STEP-to-Abaqus face fingerprint mapper | P5 验收 + Windows Abaqus 可用 | `READY_FOR_ABAQUS_COMPILER` |
| 6 | P6-B1 | 白名单 Abaqus action compiler/executor | P6-A1 验收 | `READY_FOR_END_TO_END_PILOT` |
| 7 | P7-A1 | Real-Dev 端到端双轨 pilot 和冻结 | P6-B1 验收 | `READY_FOR_FORMAL_HOLDOUT` |
| 8 | P7-B1 | 冻结后 Real-Holdout 一次性评估 | 用户完成 holdout 人工门禁 | `EXPERIMENTS_FROZEN` |
| 9 | P8-A1 | 论文图表、方法、实验、限制和复现包 | P7-B1 完成 | `MANUSCRIPT_PACKAGE_READY` |

不得并行执行有依赖关系的 Goal。只允许文献整理、图表模板等不写共享状态的辅助任务并行。

## 5. 监督者验收模板

每个 Goal 完成后，监督者按以下顺序验收：

1. **范围**：是否只完成任务书允许内容；
2. **冻结**：上游资产哈希变化是否为 0；
3. **测试**：自动测试是否完整通过，是否包含失败路径；
4. **数据纪律**：是否访问、泄漏或重复运行 formal test；
5. **安全**：是否泄漏密钥、执行任意代码或静默 fallback；
6. **指标**：是否满足硬门槛，指标定义是否未变化；
7. **失败案例**：是否如实记录 abstention、ambiguity 和未解决限制；
8. **复现**：命令、环境、模型 ID、prompt/config hash 是否齐全；
9. **结论**：`PASS`、`BOUNDED-PASS`、`REWORK` 或 `FAIL`；
10. **授权**：只有监督者明确写出下一 Goal 名称，才允许继续。

## 6. 用户人工介入点

### H4-API-Budget

在 P4-API-R1 的 50/200/1000 级 smoke 后，用户只需确认：

- API 账户和模型 ID 正确；
- 预计费用与时间可接受；
- 是否批准完整 val/train；
- 运行日志未包含 secret。

### H6-Mapping

在 P6-A1 中，用户通过 Abaqus/CAE 检查少量高亮结果：

- 映射后的面是否就是原 STEP 目标面；
- 是否存在漏面、错面或导入合并/拆分；
- ambiguity 是否被正确拒绝。

不要求用户填写大规模 raw face ID。

### H6-Execution

在 P6-B1 中，用户检查：

- Set/Surface；
- 约束自由度；
- 压力/面力方向；
- 材料单位；
- 网格尺寸；
- 求解前确认是否生效。

### H7-Holdout

系统完全冻结后，用户绘制 6-10 个受控、简化、未用于规则开发的零件；每个零件给出 3-5 条结构静力需求。Agent 生成高亮审查包，用户只确认目标区域和工程语义。

## 7. 当前授权

当前只授权启动：

```text
Goal P4-API-R0
```

旧 `GOAL_P4_R0_RUNTIME_PREPARATION.zh-CN.md` 的“双 A40 本地模型部署”路线被本计划替代，不再作为 P4 前置条件。它可保留为后续开源模型复现备选，但不得与 API 正式实验混用。

