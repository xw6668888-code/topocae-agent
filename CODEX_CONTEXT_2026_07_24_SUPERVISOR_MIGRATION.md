# Codex Context Snapshot - 2026-07-24

## 当前决策

用户决定尝试把 TopoCAE 项目迁移到一台通过 VSCode SSH 管理的 Linux 双 A40 服务器，并希望创建新的 Codex 监督者任务接管后续工作。

迁移采用两阶段策略：

1. 先建立 Linux 影子副本，并让服务器承担无状态 LLM 推理。
2. 只有在冻结资产、依赖环境、跨平台路径和关键回归全部通过后，才考虑把 Linux 变成权威执行环境。

Windows 工作区在迁移验收完成前仍是唯一权威副本。

## 当前研究阶段

- `P1-D Lite`：`BOUNDED-PASS`，识别器冻结。
- `P2-A1`：`BOUNDED-PASS`，CAE-IR 冻结。
- `P3-A1R`：`PASS`，Query-Ground 数据冻结。
- `P4-A1`：
  - `H4-01` 60/60 approve；
  - Oracle-IR 与 Keyword train/val 已完成；
  - Structured LLM 与 Direct LLM 尚未运行；
  - 正式 test 未解锁，四条基线 test 运行次数均为 0；
  - 当前状态为 `BLOCKED_AT_H4_RUNTIME`。

## 迁移审计发现

- 项目目录约 `44.791 GiB`，约 `386,430` 个文件。
- Windows `feagpt` 环境约 `16.114 GiB`，不可复制到 Linux 使用，必须重建。
- 工作区不是 Git 仓库，因此迁移需要归档、文件清单和 SHA-256 校验。
- 约 `30,003` 个文件含有序列化的 Windows 根路径；冻结文件不得批量重写。
- Linux 适配应提供运行时路径映射：
  - `TOPOCAE_ORIGINAL_WORKSPACE_ROOT=W:\python\FEA\dataset_generator`
  - `TOPOCAE_WORKSPACE_ROOT=<Linux 项目绝对路径>`
- 源码中仅少量文件硬编码旧根路径，可由兼容层和针对性修复处理。
- P4 LLM runner 当前还需要断点续跑、原子写入、有界并发、失败重试、缓存、运行清单和显式关闭 Qwen3 thinking。

## 已创建交付物

- `SERVER_MIGRATION_AND_SUPERVISOR_HANDOFF_V2.zh-CN.md`
- `scripts/collect_linux_migration_info.sh`
- `NEW_SUPERVISOR_BOOTSTRAP_PROMPT.zh-CN.md`
- `C:\tmp\TopoCAE_SUPERVISOR_HANDOFF_2026-07-24.md`

## 分工

- 当前/新监督者：迁移门禁、冻结资产保护、执行 Agent 验收、test 解锁与论文路线决策。
- P4-A1 执行 Goal：只负责 P4 runner 加固和后续经过批准的 LLM 基线执行，不负责整个项目搬迁。
- 用户：提供脱敏服务器信息、确认目标路径和权限、执行或授权远程命令、保留 Windows 原始副本。

## 禁止事项

- 不把 Windows 虚拟环境复制到 Linux。
- 不重写冻结 JSON 中的 Windows 路径。
- 不在验证前删除或覆盖 Windows 项目。
- 不提前运行正式 test。
- 不进入 P5，不执行求解器，不扩大到 `test_part_s3`。
