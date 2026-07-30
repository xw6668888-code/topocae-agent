# Goal M1-A1 V2：TopoCAE 完整备份、Linux 影子迁移与 Codex 接管

## 0. 任务性质

你是迁移执行 Agent。Windows 工作区在监督者签署 `MIGRATION-PASS` 前始终是唯一权威副本。你的任务包含：不可变备份、跨平台路径改造、完整归档、用户辅助上传、Linux 环境重建、只读回归和新 Codex 接管材料。

不得自行运行 P4 正式 test，不得进入 P5。

## 1. 已确认的现状

- Windows 权威根目录：`W:\python\FEA\dataset_generator`
- 当前盘点基线：386,444 个文件，48,094,527,104 bytes，约 44.792 GiB
- 工作区不是 Git 仓库
- 无 reparse point / symbolic link
- Windows 环境：`W:\python\envs\feagpt`，不得复制到 Linux
- 根目录现有 `environment.yml` 是旧 Python 3.10 + CPU 配置，不能作为 Linux 正式环境规格
- Linux SSH Host：`node6`
- Linux 用户/目标：`ywx1` / `/home/ywx1/dataset_generator`
- Linux：Ubuntu 24.04.4，双 A40 46 GiB，约 375 GiB 文件系统余量
- PyPI、conda-forge、GitHub 可访问；Hugging Face 不可访问
- 服务器当前两张 A40 均有其他进程
- 当前迁移门禁：`GO-M1`；P4 runtime：`NO-GO`；P4 test：`LOCKED`

## 2. 必读文件

1. `SERVER_MIGRATION_AND_SUPERVISOR_HANDOFF_V2.zh-CN.md`
2. `reports/SERVER_MIGRATION_PREFLIGHT_2026_07_26.zh-CN.md`
3. `reports/SERVER_MIGRATION_GATE_M1_2026_07_26.zh-CN.md`
4. `M1_A1_PATH_PORTABILITY_FINDINGS.zh-CN.md`
5. `CODEX_CONTEXT_2026_07_26_SERVER_PREFLIGHT.md`
6. `P4_A1_PRE_MIGRATION_RUNNER_HARDENING_INSTRUCTION.zh-CN.md`
7. P1-D Lite、P2-A1、P3-A1R、P4-A1 acceptance/freeze/delivery 文件

## 3. 总体策略

采用两份归档，不创建未经校验的手工精简副本：

### Snapshot A：迁移前权威备份

在任何跨平台源码修改前，对当前完整工作区建立不可变本地备份。

### Snapshot B：迁移就绪版本

完成 Windows 端路径兼容改造并通过回归后，再建立第二个完整归档，上传 Linux。

第一轮迁移不裁剪历史数据。原因是项目无 Git、冻结清单跨多个阶段、存在大量路径和报告引用；44.8 GiB 对当前服务器仍可接受。精简执行包只能在完整迁移验收后另行构建。

## 4. 阶段 M0：写入静默与外部存储确认

1. 检查是否有其他 Codex Goal、训练、生成、报告脚本正在写入工作区。
2. 要求用户暂停所有写入任务；只读查看可以继续。
3. 重新统计文件数和总 bytes，记录与上述基线的差异。
4. 让用户选择**工作区外部**的备份目录，例如：

```text
E:\TopoCAE_Migration_20260726
```

5. 外部备份盘至少需要：
   - 120 GiB 可用空间：允许 `.tar` + `.tar.zst` + manifest 共存；
   - 推荐 160 GiB。
6. 不允许把备份归档写入 `dataset_generator` 内部。

### H-M0-01

向用户展示：当前写入任务、文件数、总 bytes、备份目标、可用空间。获得用户批准后才能建立 Snapshot A。

## 5. 阶段 M0-A：Snapshot A 不可变备份

执行 Agent 必须实现可重复的快照工具，要求：

1. 生成按 UTF-8 相对路径排序的树清单：
   - relative_path
   - bytes
   - sha256
2. 清单覆盖工作区所有普通文件，不跟随工作区外路径。
3. 创建完整 `tar`，再单独使用 `zstd` 压缩；优先可靠性，不依赖 Windows PowerShell 的二进制管道。
4. 计算 `.tar.zst` SHA-256。
5. 输出机器可读 snapshot metadata。
6. 随机抽检不少于 100 个文件，并完整校验 P1/P2/P3 的冻结资产。
7. Snapshot A 命名必须含 UTC 时间和内容清单哈希。
8. Snapshot A 完成后设为只读或由用户放入不再修改的备份目录。

不得把 Windows 虚拟环境打入项目快照；只记录其依赖清单和 Python/关键包版本。

## 6. 阶段 M1：在 Windows 权威工作区进行源码适配

跨平台代码优先在 Windows 修改，原因：

- Windows 是当前权威工作区；
- 可以立即验证旧流程没有回归；
- 修改后只产生一个迁移就绪版本，避免 Windows/Linux 双分支漂移；
- Snapshot A 可以随时恢复迁移前状态。

### 6.1 变更边界

允许：

- 新增跨平台路径模块和测试；
- 修改尚未正式冻结的 P4 运行代码；
- 新增 Linux 环境与迁移脚本；
- 新增迁移报告。

禁止：

- 修改 P1/P2/P3 冻结资产；
- 全局替换 JSON/JSONL/graph/manifest/report 中的 `W:\...`；
- 修改 H4 人工审查表及受保护字段；
- 重新生成 P1/P2/P3 数据。

P4 当前 freeze manifest 状态为 `NOT_FROZEN_H4_AND_RUNTIME_PENDING`。路径兼容和 runner 加固可以修改 P4 候选代码，但必须重新生成候选哈希并证明 H4 审查内容未变。

### 6.2 路径兼容层

严格执行 `M1_A1_PATH_PORTABILITY_FINDINGS.zh-CN.md`：

- 新增唯一的 `resolve_workspace_path()`；
- 支持 `PureWindowsPath`；
- 使用 `TOPOCAE_ORIGINAL_WORKSPACE_ROOT` 和 `TOPOCAE_WORKSPACE_ROOT`；
- 相对路径固定相对于项目根目录；
- 防止路径穿越和符号链接越界；
- 不回写冻结文件。

必须先接入：

1. `utils/cae_language_grounding/evaluation.py::MemoryCache.get()`；
2. P4 graph/geometry/manifest reader；
3. Linux 上需要运行的 frozen asset verifier；
4. reproduction entrypoint 中实际打开文件的代码。

以下旧字符串默认仅作为历史证据，不修改：P1-D Lite 旧 Windows 命令、P2 报告 command、旧原型目录 metadata、测试字面 fixture。

### 6.3 Runner 协调

路径适配与 P4 runner 加固不得由两个 Agent 同时修改同一个文件。

- 如果 P4 runner 加固 Goal 正在运行，先等待其停在 `READY_FOR_REMOTE_RUNTIME`，再合并路径层。
- 如果尚未开始，则 M1 只实现路径层；runner 加固仍由 P4-A1 Goal 完成。
- 每次编辑前后记录受影响文件 SHA-256。

### 6.4 Windows 验收

至少验证：

- P1/P2/P3 冻结清单零变化；
- H4 review 内容零变化；
- 路径解析单元测试；
- P3-A1R 一个真实 Windows `graph_path` 通过新 resolver 加载；
- P4 Oracle/Keyword train/val 只读回归；
- test ledger 仍为 0；
- 原始 JSON/JSONL SHA-256 不变。

### H-M1-01

提交：修改文件清单、前后哈希、测试结果、冻结资产结果。用户和监督者批准后才能建立 Snapshot B。

## 7. 阶段 M2：Snapshot B 迁移归档

1. 停止工作区写入。
2. 生成新的完整文件树 SHA-256 清单。
3. 创建完整 `tar` + `tar.zst`。
4. 记录 Snapshot A 与 B 的差异，差异必须仅限批准的迁移/P4 可变文件。
5. 计算归档哈希。
6. 输出准确的 Windows 上传命令，例如：

```powershell
scp "<snapshot-b.tar.zst>" node6:/home/ywx1/
scp "<snapshot-b.sha256>" node6:/home/ywx1/
```

### H-M2-01

向用户展示归档路径、大小、SHA-256、差异清单、上传目标和按 3 MiB/s 估算时间。批准后用户执行上传。

## 8. 阶段 M3：Linux 校验与环境

1. Linux 先执行 `sha256sum -c`，失败则不得解压。
2. 解压到临时目录，例如 `/home/ywx1/dataset_generator.migration_tmp`。
3. 重新生成树清单并与 Snapshot B 比对。
4. 校验通过后才命名为 `/home/ywx1/dataset_generator`。
5. 在用户目录安装 Miniforge/Micromamba，不需要 sudo。
6. 分离两个环境：
   - `topocae-core`：Python 3.11、pythonocc/OCCT、PyTorch、图与评估依赖；
   - `topocae-llm`：vLLM/OpenAI-compatible 服务。
7. 不使用根目录旧 `environment.yml` 作为正式规格；生成新的 Linux 环境文件和显式锁定清单。
8. Hugging Face 不可访问时，不使用未经批准的镜像；先完成 core 环境，模型权重作为独立门禁。

## 9. 阶段 M4：Linux 回归

必须运行：

1. 全部冻结资产哈希检查；
2. Python 关键 import；
3. OCC 读取固定 STEP，验证 solid/face/edge 数；
4. 固定样本 graph 提取和不变量；
5. P1-D Lite 只读识别 smoke；
6. P2-A1 测试；
7. P3-A1R 只读验证；
8. P4 Oracle/Keyword train/val 回归；
9. Windows path -> Linux path 的真实 P3/P4 graph 加载；
10. test ledger = 0，LLM 未调用，P5 未进入。

Linux 发现源码问题时，不允许只在服务器改完就继续。必须：

1. 记录补丁；
2. 将同一改动同步回 Windows 权威工作区；
3. Windows 回归；
4. 生成增量补丁清单或新的 Snapshot B revision；
5. 两端文件哈希重新一致。

## 10. 阶段 M5：新 Codex 接管

确保 Linux 项目根目录包含：

- `LINUX_CODEX_RECOVERY_PROMPT.zh-CN.md`
- 最新 migration acceptance/report/manifest
- 最新 `CODEX_CONTEXT_*.md`
- P1/P2/P3/P4 acceptance/freeze/delivery 文件

用户在 Linux VSCode/Codex 新任务中发送：

```text
工作目录是 /home/ywx1/dataset_generator。
请完整读取 LINUX_CODEX_RECOVERY_PROMPT.zh-CN.md，先执行其中的只读恢复检查。
在报告当前阶段、冻结边界、迁移验收状态和下一门禁之前，不得修改文件、运行 P4 正式 test 或进入 P5。
```

新 Codex 必须先给出恢复摘要，监督者确认后才能接管。

## 11. 最终交付

- Snapshot A/B 归档、树清单和归档 SHA-256；
- A/B 差异清单；
- 路径解析层及测试；
- Linux core/LLM 环境规格与显式包清单；
- Windows/Linux 回归结果；
- `migration_acceptance.json`；
- 中文迁移验收报告；
- 新 Codex 恢复报告；
- 全部交付哈希清单。

最终状态只能是：

- `READY_FOR_MIGRATION_ACCEPTANCE`；
- `BLOCKED_WITH_EVIDENCE`。

完成后停止，等待监督者签署 `MIGRATION-PASS`。
