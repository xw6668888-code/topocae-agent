# TopoCAE-Agent 迁移后监督、实验与混合 CAE 执行计划

更新日期：2026-07-28  
当前主要执行环境：`/home/ywx1/dataset_generator`  
迁移状态：`MIGRATION-PASS`  
当前研究门禁：`P4_RUNTIME_PREPARATION_SUPERVISOR_REVIEW`

## 1. 论文主问题

第一篇论文继续限定为：

> 给定受控类别范围内的 STEP 单零件和自然语言结构静力需求，如何将 CAE 意图可靠地映射到精确 B-Rep 面集，并生成可验证的有限元前处理操作？

系统路线保持不变：

```text
STEP
-> invariant attributed B-Rep graph
-> frozen controlled-domain recognizer
-> feature-object memory

natural-language CAE requirement
-> open-weight LLM
-> typed CAE-IR
-> deterministic grounder / topology-constrained pointer
-> exact B-Rep face set
-> verifier
-> solver-neutral action plan
-> Abaqus execution adapter
```

不得重新把论文主线改成任意工业复杂零件识别，也不得让 LLM 直接读取 STEP 或猜 raw face ID。

## 2. 当前已完成状态

### P1：几何识别器

- P1-D Lite：`BOUNDED-PASS`。
- 冻结识别器：P1-B `p1_b_general_recognition_pipeline_v4`。
- 能力边界：`LIMITED_CONTROLLED_DOMAIN`。
- 孔类可在 abstention 下使用；transition limited；slot/rib predicted path 必须允许人工确认或 Oracle 对照。

### P2：CAE-IR

- `BOUNDED-PASS`。
- CAE-IR 版本：`1.0.0-h2.01`。
- 17 个规范特征、7 类操作、role policy、单位和错误码已经冻结。

### P3：CAE-QueryGround

- P3-A1R：`PASS`。
- 数据版本：`cae-query-ground/1.1.0-h3r.01`。
- 多实例、数量词、largest/smallest/nearest、中文/英文和 exact face-set 真值已经建立。

### P4：语言与 grounding baseline

- H4 人工审查：60/60 approve。
- Oracle-IR train/val：PASS。
- Keyword train/val：完成并可确定性复现。
- Structured LLM 和 Direct LLM：代码候选已存在，但尚未正式部署模型或完成 train/val。
- 正式 test：运行 0 次、内容未访问、未解锁。

### Linux 迁移

- M0-M5：全部 PASS。
- Linux core/LLM 环境已经建立。
- Linux 可作为主要执行环境。
- Windows Snapshot A/B 和 Windows 工作区继续保留。

## 3. 后续阶段与停止门槛

### P4-R0：LLM Runtime Preparation

目标：在任何真实模型大规模调用前完成 runner 工程加固和部署方案审查。

必须完成：

- 断点续跑、原子写入、请求缓存、有界并发、有限重试；
- prompt/model/config/input 的稳定缓存键；
- Qwen thinking 显式关闭；
- Structured/Direct 输出和缓存隔离；
- mock endpoint 中断恢复和幂等测试；
- 模型来源、revision、许可、权重哈希和磁盘预算；
- 两张共享 A40 的运行时段和部署拓扑；
- 50/200/1000 样本的分级吞吐预算，但不运行正式 test。

Gate：`READY_FOR_MODEL_DEPLOYMENT_REVIEW`。

### P4-R1：模型部署与 Train/Val

仅在 P4-R0 通过后开始：

1. 部署冻结的 8B instruct 模型候选；
2. 先运行 train-only 小样本 smoke；
3. 验证 JSON/schema、thinking 关闭、缓存和恢复；
4. 冻结 model revision、dtype、context、seed、prompt 和服务配置；
5. 运行 Structured/Direct train/val；
6. 生成指标和 failure atlas；
7. 不读取 test。

Gate：达到任务书中的 schema、slot、grounding 和非法输出门槛，或给出 `BOUNDED-PASS` 证据。

### P4-R2：后 H4 冻结与一次性 Test

只有监督者确认以下内容后才解锁：

- C2/C3 train/val 完成；
- prompt/model/config/parser/grounder/verifier/evaluator 哈希冻结；
- test ledger 更新但运行次数仍为 0；
- 无 ground-truth leakage；
- 无根据 test 调参的路径。

随后四条 baseline 各正式运行一次 test。完成后不得回看 test 修改模型、规则或 prompt。

### P5：TopoCAE-Pointer

P4 通过后再执行：

- feature object/selectable region 作为候选 token；
- CAE-IR/query encoder；
- candidate cross-attention 或等价 pointer scorer；
- topology mask 与多目标集合预测；
- Oracle objects 和 predicted objects 双轨评估；
- 与 Keyword、deterministic、Direct LLM 比较。

保留条件：在 L3-L5 上相对最佳非 pointer baseline 的 exact face-set 或 macro query 指标提升至少 5 个百分点，或能显著降低非法目标率。否则降级为消融，不继续扩大模型。

### P6：混合 Abaqus 验证闭环

Linux 没有 Abaqus 不构成阻塞。P6 使用 Linux 控制面和 Windows Abaqus 执行面。

#### Linux 控制面

- STEP/graph/recognizer；
- LLM -> CAE-IR；
- deterministic/pointer grounding；
- exact face set 和 fingerprint；
- schema、单位、角色、拓扑验证；
- 生成 solver-neutral `CAEExecutionPackage`；
- 接收并分析 Abaqus 返回结果。

#### Windows Abaqus 执行面

- 接收不可变 execution package；
- 导入 STEP；
- 按 fingerprint 重新定位 Abaqus face；
- 创建 set/surface；
- 创建 BC/load/material/mesh action；
- dry-run 或提交小型静力作业；
- 返回结构化 `execution_result.json` 和日志。

#### 第一阶段采用人工文件传递

论文时间优先，不先开发常驻服务：

1. Linux 生成一个 case bundle；
2. 用户把 bundle 复制到 Windows；
3. 执行固定 Abaqus 命令；
4. 将 result bundle 复制回 Linux；
5. Linux 自动汇总。

20 个 set/surface/BC/load 创建样例和 5-10 个小型求解样例足以形成 P6 证据。人工只负责启动命令和视觉抽查，不手工修改 face ID。

#### 第二阶段可选自动化 Worker

只有手工闭环稳定后，才考虑 Windows worker：

- 轮询受控 inbox/outbox；或
- 受认证的内部 HTTP/SSH job runner。

Worker 只接受 schema-validated package，不接受任意 Python 字符串，避免 LLM 生成代码直接执行。

### P7-P9：实验、Holdout 与论文

- P7：固定配置、消融、3 个随机种子、统计分析；
- P8：冻结后建立 6-10 个简单 Real-Holdout，每件 3-5 条真实需求，只运行一次；
- P9：论文、复现包、图表和匿名化检查。

## 4. Linux/Windows 资产契约

### CAEExecutionPackage

至少包含：

- schema/version/case ID；
- STEP 文件或 STEP SHA-256；
- graph 与 recognizer manifest；
- feature objects；
- CAE-IR；
- selected face indices；
- face fingerprints 与 mapping tolerance contract；
- solver-neutral action plan；
- unit system；
- compiler/adapter version 和全部输入哈希。

### AbaqusExecutionResult

至少包含：

- package SHA-256；
- STEP import 状态；
- face fingerprint mapping score；
- selected/mapped face 数；
- set/surface 创建状态；
- BC/load/material/mesh action 创建状态；
- job submission/solve status；
- 稳定错误码；
- Abaqus 版本、worker 版本、日志哈希；
- 禁止静默 fallback。

## 5. 源码与数据权威性

- Linux 成为后续源码和实验的主要执行环境。
- Windows 不维护一套独立分叉的项目源码。
- Abaqus worker 应是一个小型、版本化的独立目录或发布包。
- P6 的 worker 改动先在 Linux 主项目形成版本和哈希，再复制到 Windows 测试。
- Windows 运行产生的 result bundle 返回 Linux 归档。
- Snapshot A/B 不删除；正式 test 和论文初稿前不得覆盖。

## 6. 监督者 + Goal 工作模式

### 监督者任务

长期保留一个任务，只负责：

- 守住论文主线和冻结边界；
- 每次只批准一个下一阶段 Goal；
- 审查 acceptance、metrics、hash 和禁止事项；
- 给用户分配中文人工检查说明；
- 使用 `通过 / 有条件通过 / 不通过` 三种结论；
- 不默认亲自大规模改代码或跑实验。

### 执行 Goal

同时最多一个，必须包含：

- 唯一阶段和指标目标；
- 必读文件；
- 允许/禁止修改范围；
- 实现和测试；
- 人工门禁；
- 交付清单和 SHA-256；
- 完成后停止，不自行进入下一阶段。

### 用户

- 把 Goal 交付摘要返回监督者；
- 按监督者说明完成少量 CAE 语义和 Abaqus 视觉检查；
- 协调共享 GPU 时段和 Windows Abaqus 许可证；
- 不需要手工修改大规模 JSON 或逐面标 face ID。

## 7. 防偏航规则

- P4 未完成前不进入 pointer 或 Abaqus。
- 不使用 `test_part_s3`。
- 不重新优化冻结识别器以追求任意复杂零件泛化。
- 不让 LLM 替代 B-Rep graph encoder。
- 不让 LLM 直接生成可执行任意代码。
- 不以“服务器没有 Abaqus”为理由更换论文主问题。
- 不在 Real-Holdout 上调参。
- 每个新模块必须对应论文假设、消融和量化指标。
