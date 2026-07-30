# TopoCAE Linux 影子迁移最终验收

迁移候选状态：`READY_FOR_MIGRATION_ACCEPTANCE`  
监督决策：`MIGRATION-PASS`  
签署时间：`2026-07-27T19:49:26.8024220+08:00`

Linux 影子迁移已由监督者正式接受。`/home/ywx1/dataset_generator` 可作为主要执行环境；Windows Snapshot A/B 和当前 Windows 工作区必须按监督决策继续保留。

## 阶段结果

- M0-M4：全部 `PASS`。
- M5：新 Codex 只读恢复报告已提交并确认，状态 `PASS`。
- 新 Codex 恢复报告：`LINUX_CODEX_RECOVERY_REPORT.20260727T100554Z.json`，SHA-256 `a9f7c7e0a0b5434fb9eca82f4abedcae0287bfb7ca2e9f1f85d4531264012611`。
- 监督决策：`reports/migration/supervisor_migration_pass.json`，SHA-256 `965b72cd63f8bc7dad64da8615440113c7aaecaf2d5d43bd6d17bf9d8445c800`。

## 验收证据

- Snapshot B archive SHA-256：`92bf5b961e0fc8cd74e2d845b0d3b10f472edfc20d6291305259701e6e6ce084`。
- Snapshot B tree manifest SHA-256：`05ecd6f9e92137a00bedc561599ad8a5749f6562dc4b7796d12062921ee18a6f`。
- Linux 重建：386455 文件、48094569238 字节，与 Snapshot B 完全一致。
- 签署候选交付清单：41/41 SHA-256 通过。
- 冻结资产：P1 13/13、P2 13/13、P3-A1R 1885/1885，变化 0。
- P4 Oracle/Keyword 仅运行 train/val；正式 test ledger 保持 0。
- 未调用 LLM，未下载模型权重，未进入 P5。

## 签署后归档

主管签署文件已与候选交付一并纳入最终 `migration_delivery_checksums.sha256`，最终归档 43/43 SHA-256 通过。签署前恢复报告保持原始字节，不回写其 `supervisor_confirmed` 字段；最终确认以本报告、`migration_acceptance.json` 和主管签署文件为准。

## 授权边界

`MIGRATION-PASS` 只接受 Linux 迁移结果并允许 Linux 成为主要执行环境。它不授权：

- 下载或调用 Qwen/其他 LLM；
- 启动 P4 Structured/Direct 正式 train/val；
- 解锁或运行 P4 正式 test；
- 进入 P5 或执行 Abaqus/CalculiX。

下一门禁：`P4_RUNTIME_PREPARATION_SUPERVISOR_REVIEW`。