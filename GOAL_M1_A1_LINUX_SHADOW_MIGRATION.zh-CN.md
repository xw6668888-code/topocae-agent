# Goal M1-A1：Linux 双 A40 影子迁移与跨平台复现

## 角色

你是迁移执行 Agent，不是论文路线监督者。你的任务是把 Windows 权威工作区建立为 Linux 上可校验、可运行的影子副本。完成后停止并交由监督者验收，不得自行解锁 P4 正式 test。

## 当前输入

- Windows 权威工作区：`W:\python\FEA\dataset_generator`
- Linux SSH Host：`node6`
- Linux 用户：`ywx1`
- Linux 目标目录：`/home/ywx1/dataset_generator`
- Linux：Ubuntu 24.04.4 LTS
- GPU：2 x NVIDIA A40 46 GiB
- Linux 可用磁盘：约 375 GiB，用户配额未暴露
- PyPI、conda-forge、GitHub 可访问；Hugging Face 不可访问
- 当前监督结论：`GO-M1`，但 `NO-GO-P4-RUNTIME`

## 必读文件

1. `NEW_SUPERVISOR_BOOTSTRAP_PROMPT.zh-CN.md`
2. `SERVER_MIGRATION_AND_SUPERVISOR_HANDOFF_V2.zh-CN.md`
3. `reports/SERVER_MIGRATION_PREFLIGHT_2026_07_26.zh-CN.md`
4. `CODEX_CONTEXT_2026_07_26_SERVER_PREFLIGHT.md`
5. `P4_A1_PRE_MIGRATION_RUNNER_HARDENING_INSTRUCTION.zh-CN.md`
6. P1/P2/P3/P4 的冻结清单和最终验收报告

## 总目标

建立满足以下定义的 Linux 影子副本：

- 冻结项目资产逐字节不变；
- Windows 绝对路径通过运行时解析层映射，不修改冻结 JSON/manifest；
- Linux 环境可重复创建；
- STEP/OCC、graph、CAE-IR、Query-Ground 和 P4 非 LLM 回归通过；
- Windows 继续保留权威副本，直到监督者签署 `MIGRATION-PASS`；
- 正式 test ledger 保持 0。

## Task 1：冻结与依赖盘点

1. 记录当前工作区总文件数、总字节数和冻结清单状态。
2. 导出 Windows `feagpt` 的只读依赖清单，但不得把 Windows 环境复制到 Linux。
3. 根据项目实际 import、入口脚本和冻结报告，建立：
   - `environment.core.linux.yml`：STEP/OCC、图、训练和评估；
   - `environment.llm.linux.yml`：vLLM/OpenAI-compatible 服务；
   - 两份可机器读取的依赖说明。
4. 优先使用 Python 3.11；核心环境和 LLM 服务环境分离，避免 pythonocc 与 vLLM 依赖互相污染。

## Task 2：跨平台路径层

新增统一的 `resolve_workspace_path()` 或等价模块，读取：

```text
TOPOCAE_ORIGINAL_WORKSPACE_ROOT=W:\python\FEA\dataset_generator
TOPOCAE_WORKSPACE_ROOT=/home/ywx1/dataset_generator
```

要求：

1. Windows 原生路径在 Windows 正常工作。
2. Linux 读取冻结 Windows 路径时映射到 Linux 项目根目录。
3. 不修改原始 JSON、graph、manifest 或 freeze 文件。
4. 拒绝目录穿越和映射到工作区外部的路径。
5. 接入 P3/P4 dataset reader、feature-object memory、manifest verifier 和正式 reproduction entrypoint。
6. 添加 Windows/Linux/非法路径单元测试。

## Task 3：可靠归档

1. 在项目内建立 `.migration/`，归档输出必须被归档输入排除，避免自包含。
2. 生成稳定文件清单，至少包含相对路径、字节数和 SHA-256。
3. 将整个工作区打包为单个 `tar.zst`；仅允许排除：
   - 归档输出本身；
   - 明确可再生的临时缓存，并在报告中逐项列出。
4. 不允许排除数据集、模型、报告、冻结文件、测试件和源代码。
5. 记录归档 SHA-256、压缩前后字节数、文件数和生成命令。

## Task 4：传输与 Linux 解包

由于当前 Codex 沙箱不能读取用户 SSH 配置，传输命令由用户在本地 Windows 终端执行。执行 Agent 必须生成准确命令，不索要密码或私钥。

1. 通过 `scp` 或可恢复的传输方式上传单个归档和校验文件。
2. Linux 端先校验归档 SHA-256，再解压到临时目录。
3. 对解压结果运行文件树清单校验。
4. 校验通过后才命名为 `/home/ywx1/dataset_generator`。
5. 不覆盖或删除 Windows 工作区。

## Task 5：Linux 环境重建

1. 不需要 sudo；在 `/home/ywx1` 下安装 Miniforge/Micromamba。
2. 创建独立的 `topocae-core` Python 3.11 环境。
3. 从 conda-forge 安装与冻结环境兼容的 `pythonocc-core/OCCT`，再安装其余依赖。
4. 创建独立的 `topocae-llm` 环境，但在模型来源确认前只安装服务代码和执行 smoke test，不下载正式模型。
5. 不假设系统存在 `nvcc`；优先使用带 CUDA runtime 的 PyTorch/vLLM 二进制包。
6. 记录全部安装命令、channel、版本和导出的显式环境清单。

## Task 6：影子副本回归

至少执行：

1. 冻结资产哈希校验。
2. Python import smoke test。
3. OCC 读取简单 STEP 并输出 solid/face/edge 数量。
4. 固定小样本的 graph 提取与关键不变量对比。
5. P1/P2/P3 只读回归。
6. P4 Oracle-IR 与 Keyword 的只读 train/val 回归。
7. 路径映射测试。
8. 严格确认四条 P4 正式 test 运行次数仍为 0。

## Task 7：LLM 运行时预备检查

1. 探测 ModelScope 或经管理员批准的内部模型源是否可用。
2. 不绕过组织网络策略，不使用不明镜像。
3. 检查 GPU 空闲时段；当前两张 A40 均有其他 Python 进程，各约占 5 + 5 GiB。
4. Qwen3-8B 优先每张 GPU 一个独立 replica，不使用跨 NUMA、无 NVLink 的 tensor parallel 作为默认方案。
5. 未获得模型权重和 GPU 使用确认时，状态保持 `NO-GO-P4-RUNTIME`。

## 禁止事项

- 禁止复制 Windows 虚拟环境。
- 禁止修改冻结资产以适配 Linux。
- 禁止删除、覆盖 Windows 原工作区。
- 禁止运行 P4 正式 test。
- 禁止进入 P5、Abaqus/CalculiX 或 `test_part_s3`。
- 禁止记录密码、私钥、token、代理凭证。
- 非必要不要使用 superpower 类技能。

## 人工检查点

### H-M1-01：归档前

向用户展示：包含/排除目录、原始文件数、预计归档大小和本地剩余空间。用户批准后才创建大型归档。

### H-M1-02：上传前

向用户展示：归档路径、大小、SHA-256、目标服务器路径和预计上传时间。用户批准后才上传。

### H-M1-03：Linux 验收

向用户展示：冻结哈希、环境版本、OCC smoke、graph 回归和 test ledger。用户确认后提交监督验收。

## 最终交付

- 中文迁移报告。
- 机器可读 `acceptance.json`。
- Windows 与 Linux 文件树/冻结资产校验结果。
- 两份 Linux 环境文件和显式版本清单。
- 路径解析实现及测试。
- 归档与复现命令。
- 交付清单与 SHA-256。
- 最终状态只能是：
  - `READY_FOR_MIGRATION_ACCEPTANCE`；
  - `BLOCKED_WITH_EVIDENCE`。

完成后停止，等待监督者签署 `MIGRATION-PASS`，不得自行进入 P4 正式推理。
