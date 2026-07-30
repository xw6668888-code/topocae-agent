# TopoCAE-Agent Linux 服务器迁移与监督交接 V2

最后更新：2026-07-24  
当前权威工作区：`W:\python\FEA\dataset_generator`  
当前 Python：`W:\python\envs\feagpt\python.exe`，Python 3.11.14  
目标：将项目完整复制到 Linux 双 A40 服务器，在保持冻结资产不变的前提下，
完成路径适配、Linux 环境重建、P4-A1 LLM 基线和后续论文路线。

本文件取代 2026-07-06 的 `SERVER_CODEX_HANDOFF.md` 作为迁移入口；旧文件仅作
历史记录。

## 1. 当前研究状态

论文主线已经冻结为：

```text
STEP
-> invariant attributed B-Rep graph
-> frozen controlled-domain recognizer
-> feature-object memory
-> natural-language CAE requirement
-> typed CAE-IR
-> deterministic grounding / later topology-constrained pointer
-> exact STEP face set
-> later Abaqus executable preprocessing
```

当前阶段为 P4-A1：

- H4-01 人工审查 60/60 approve；
- Oracle-IR deterministic grounder train/val Gate 已通过；
- Keyword train/val 已完成；
- Structured LLM 和 Direct LLM 代码已完成但未运行；
- test ledger 仍为 0；
- 当前状态为 `BLOCKED_AT_H4_RUNTIME`。

迁移期间禁止运行 test，不得进入 P5 或 Abaqus。

## 2. 当前资产盘点

- 项目工作区：44.791 GiB，386,430 个文件；
- Windows `feagpt` 环境：16.114 GiB，168,006 个文件；
- 测试文件：145 个；
- 约 30,003 个 JSON/报告文件保存了转义后的
  `W:\python\FEA\dataset_generator` 绝对路径；
- 源码中直接写死该 Windows 根路径的 Python 文件只有少量，但冻结 graph、
  manifest 和 query 中的绝对路径不能直接在 Linux 解析。

Windows 虚拟环境不能复制到 Linux 使用。迁移包只复制项目工作区，Linux 环境
必须重新创建。

## 3. 推荐迁移原则

采用双阶段、可回退迁移：

### 阶段 M1：影子副本

- Windows 继续是唯一权威副本；
- Linux 获得项目的字节级副本；
- 不修改 Windows 冻结资产；
- 在 Linux 完成环境、路径解析和只读验证；
- P4 test ledger 保持 0。

### 阶段 M2：执行接管

只有以下条件全部通过后，Linux 才成为新的执行端：

- 迁移包 SHA-256 一致；
- P1/P2/P3/P4 冻结清单无变化；
- portable path resolver 通过；
- P4 Oracle/Keyword 回归通过；
- STEP/OCC 分层抽样通过；
- GPU、vLLM 和 JSON Schema smoke test 通过；
- 新监督者签署 `MIGRATION-PASS`。

Windows 原目录至少保留到论文初稿和一次性 test 完成，不删除、不覆盖。

## 4. 绝对路径兼容方案

不得批量重写 30,003 个冻结 JSON/graph/report，这会破坏哈希和论文复现证据。

应新增运行时路径映射层，例如：

```text
TOPOCAE_ORIGINAL_WORKSPACE_ROOT=W:\python\FEA\dataset_generator
TOPOCAE_WORKSPACE_ROOT=/data/weixing/FEA/dataset_generator
```

`resolve_workspace_path()` 的行为：

1. Linux 原生绝对路径直接返回；
2. 检测 Windows 冻结根前缀；
3. 计算相对路径；
4. 映射到 `TOPOCAE_WORKSPACE_ROOT`；
5. 检查映射结果仍位于工作区内；
6. 不修改原始 JSON 字节。

优先接入：

- P4 `MemoryCache` / feature-object memory；
- P3/P4 dataset readers；
- frozen asset verifier；
- graph/STEP manifest readers；
- 正式 reproduction entrypoints。

必须增加 Windows、Linux、非法越界路径的单元测试。

## 5. Linux 环境重建

先根据服务器信息选择 micromamba/conda 或容器，不复制 Windows `feagpt`。

当前关键运行时版本：

```text
Python       3.11.14
pythonocc    7.9.x
OCCT         7.9.3
NumPy        2.4.6
SciPy        1.17.1
scikit-learn 1.8.0
PyTorch      2.10.0
NetworkX     3.6.1
PyYAML       6.0.3
jsonschema   4.26.0
VTK          9.5.2
gmsh         4.15.0
```

跨 Windows/Linux 不追求二进制环境逐包相同，而要求：

- Python 主版本一致；
- pythonocc/OCCT 主次版本一致；
- schema、graph 和规则行为一致；
- 冻结输出字节不变；
- 新生成结果的确定性和数值容差通过。

P4 LLM 推理服务与几何环境最好分离：

- `feagpt-linux`：项目、OCC、评估、训练；
- `topocae-llm` 或 vLLM 容器：Qwen3 OpenAI-compatible 服务。

## 6. 数据传输

项目包含 386,430 个小文件，不建议逐文件 `scp -r`。

推荐：

1. Windows 创建单个 tar/zstd 或 7z 迁移包；
2. 生成归档 SHA-256；
3. 使用 `scp`、SFTP 或 VSCode Remote 上传单个归档；
4. Linux 校验 SHA-256；
5. 解压到目标数据盘；
6. 再执行冻结清单验证。

不要把 `W:\python\envs\feagpt` 加入迁移包。

粗略耗时：

- 1 Gbps 局域网且服务器环境已有：4--8 小时完成全部迁移与验收；
- 100 Mbps 网络：通常 8--16 小时，其中上传约 1--4 小时；
- 服务器需重新配置驱动、CUDA、conda 或网络受限：1--2 个工作日。

单纯复制文件可能只需几十分钟，但不能称为完成迁移。

## 7. P4-A1 在迁移前的必要收尾

P4-A1 执行对话负责代码内的 LLM runner hardening，不负责搬迁整个项目：

- `--limit`；
- `--workers`；
- `--resume`；
- 确定性 `--shard`；
- 按 `example_id` 幂等 checkpoint；
- HTTP transient retry 与 schema repair retry 分离；
- `TOPOCAE_LLM_ENABLE_THINKING=false`；
- 分层 train 1,000--2,000 + 完整 val 的修订协议；
- test ledger 继续保持 0。

监督者负责：

- 迁移 Gate；
- 冻结资产和路径兼容验收；
- 是否允许 P4 正式 test；
- 是否允许进入 P5。

用户负责：

- 提供服务器非敏感配置；
- 确认目标目录和剩余空间；
- 通过 VSCode SSH/终端执行服务器命令；
- 不提供密码、私钥或 API 密钥；
- 保持 Windows 原工作区只读备份。

## 8. 迁移验收顺序

1. 收集 Linux 服务器信息；
2. 冻结当前 P4 状态和 test ledger；
3. 创建迁移包、文件清单和 SHA-256；
4. 上传并校验归档；
5. 创建 `feagpt-linux`；
6. 实现 portable path resolver；
7. 运行 P1/P2/P3/P4 冻结审计；
8. 运行 P4 和 OCC 分层回归；
9. 启动 Qwen3-8B vLLM 服务；
10. Structured/Direct 各 100 条 smoke；
11. 生成迁移验收报告；
12. `MIGRATION-PASS` 后恢复 P4-A1 val。

## 9. 当前不得执行

- 不得删除 Windows 工作区；
- 不得复制或使用 Windows `feagpt` 作为 Linux 环境；
- 不得批量改写冻结 JSON 的路径；
- 不得在迁移验收前运行正式 test；
- 不得因迁移重新训练 recognizer；
- 不得进入 `test_part_s3`、P5 或 Abaqus。

## 10. 关键入口

- `ONE_MONTH_PAPER_ROUTE_DECISION.zh-CN.md`
- `reports/P3_A1R_SUPERVISOR_ACCEPTANCE.zh-CN.md`
- `GOAL_P4_A1_LANGUAGE_AND_GROUNDING_BASELINES.zh-CN.md`
- `reports/p4_a1_language_grounding_v1/final_acceptance.zh-CN.md`
- `reports/p4_a1_language_grounding_v1/LLM_RUNTIME_REQUEST.zh-CN.md`
- `CODEX_CONTEXT_P4_A1.md`
- `CODEX_CONTEXT.md`

