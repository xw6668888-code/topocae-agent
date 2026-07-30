# Linux 迁移后监督交接增量包 V2

更新日期：2026-07-28

## 1. 性质

本增量包是在 43/43 迁移交付完成并签署 `MIGRATION-PASS` 后新增的监督资料。它不修改或追溯替换原迁移包，也不改变任何 P1-P4 冻结资产。

## 2. 需要同步到 Linux 项目根目录的文件

```text
POST_MIGRATION_PROJECT_SUPERVISION_AND_HYBRID_CAE_PLAN.zh-CN.md
LINUX_SUPERVISOR_CONTINUATION_PROMPT.zh-CN.md
LINUX_SUPERVISOR_CONTINUATION_PROMPT_V2.zh-CN.md
GOAL_P4_R0_RUNTIME_PREPARATION.zh-CN.md
CODEX_CONTEXT_2026_07_28_POST_MIGRATION_HANDOFF.md
NON_GIT_WORKSPACE_DECLARATION.zh-CN.md
POST_MIGRATION_HANDOFF_DELIVERY_V2.zh-CN.md
POST_MIGRATION_HANDOFF_CHECKSUMS_V2.sha256
```

目标目录：

```text
/home/ywx1/dataset_generator
```

## 3. Linux 校验

进入项目根目录后运行：

```bash
sha256sum -c POST_MIGRATION_HANDOFF_CHECKSUMS_V2.sha256
```

七个被校验文件必须全部显示 `OK`。校验文件本身不做自哈希。原版监督提示词作为迁移后原始记录保留，实际接管使用 V2。

## 4. 范围边界

同步与校验本包不构成以下授权：

- 不授权下载或调用 LLM；
- 不授权 P4 正式 test；
- 不授权进入 P5；
- 不授权在 Linux 或 Windows 上执行 `git init`；
- 不授权修改 P1/P2/P3/P4 冻结资产。

完成校验后，仅重新触发 Linux 监督者的只读门禁审查。

