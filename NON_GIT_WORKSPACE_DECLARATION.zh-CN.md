# TopoCAE 非 Git 工作区声明

更新日期：2026-07-28

## 1. 结论

`W:/python/FEA/dataset_generator` 在迁移前就不是一个有效的 Git 仓库。其 `.git` 为空目录，不含 `HEAD`、`config`、objects 或 refs。Linux 端 `/home/ywx1/dataset_generator/.git` 同样为空，属于迁移前状态的忠实保留，不是迁移缺失或恢复失败。

因此：

- Linux 监督者不需要恢复 Git 状态；
- 空 `.git` 不阻塞 P4-R0；
- 当前阶段不得为了消除该提示而执行 `git init`；
- 是否正式建立版本库应作为独立的工程治理任务，在不改变冻结资产和论文实验证据链的前提下另行审批。

## 2. 当前权威完整性机制

项目迁移、冻结和阶段验收不依赖 Git，而依赖以下可审计证据：

1. Windows Snapshot A/B 及对应 SHA-256；
2. Snapshot B 全树 manifest；
3. M3、M4、M5 增量 patch manifest；
4. P1、P2、P3 阶段 freeze manifest 与 delivery manifest；
5. P4 `test_run_ledger.json`；
6. `migration_delivery_checksums.sha256`；
7. `migration_acceptance.json`；
8. `reports/migration/supervisor_migration_pass.json`。

已签署的迁移状态为 `MIGRATION-PASS`。M0-M5 全部通过，正式 P4 test 仍为 0 次且未解锁。

## 3. 对后续 Goal 的要求

在正式建立版本控制之前，每个执行 Goal 必须继续提供：

- 输入冻结资产及其哈希；
- 新增、修改、删除文件列表；
- delivery manifest 与 acceptance；
- 测试结果和复现命令；
- 范围审计，特别是 P4 test、P5 和 Abaqus 的授权边界。

监督验收应依据上述材料，不以 `git status` 是否可用作为门禁条件。

