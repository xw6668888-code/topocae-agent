# Linux Codex 恢复提示

工作目录必须是 `/home/ywx1/dataset_generator`。

在修改任何文件前，完整读取：

1. `GOAL_M1_A1_LINUX_SHADOW_MIGRATION_V2.zh-CN.md`
2. `CODEX_CONTEXT_2026_07_27_LINUX_MIGRATION.md`
3. `migration_acceptance.json`
4. `MIGRATION_ACCEPTANCE_2026_07_27.zh-CN.md`
5. `reports/migration/m3_linux_acceptance.json`
6. `reports/migration/m4_linux_acceptance.json`
7. `reports/migration/m3_incremental_patch_manifest.json`
8. `reports/migration/m4_incremental_patch_manifest.json`
9. `reports/migration/m5_incremental_patch_manifest.json`

然后只执行以下恢复检查：

```bash
pwd
sha256sum -c migration_delivery_checksums.sha256
/home/ywx1/miniforge3/bin/conda env list
/home/ywx1/miniforge3/bin/conda run -n topocae-core python -c 'import OCC, torch; print(OCC.VERSION, torch.__version__, torch.version.cuda, torch.cuda.device_count())'
python -m json.tool migration_acceptance.json >/dev/null
python -m json.tool reports/migration/m4_linux_acceptance.json >/dev/null
```

如需重查冻结资产，只能把新输出写到项目外：

```bash
mkdir -p /home/ywx1/recovery_evidence
PYTHONDONTWRITEBYTECODE=1 \
TOPOCAE_ORIGINAL_WORKSPACE_ROOT='W:\python\FEA\dataset_generator' \
TOPOCAE_WORKSPACE_ROOT='/home/ywx1/dataset_generator' \
/home/ywx1/miniforge3/envs/topocae-core/bin/python \
  verify_p4_a1_frozen_assets.py \
  --output /home/ywx1/recovery_evidence/frozen_assets.json
```

首次回复必须报告：

- 当前阶段为 M5，状态为 `READY_FOR_MIGRATION_ACCEPTANCE`；
- Snapshot B 和增量清单链；
- P1/P2/P3/P4 冻结边界与 M4 结果；
- `topocae-core` / `topocae-llm` 状态；
- P4 test ledger 是否仍为 0；
- 已知旧 P3 冻结路径限制；
- 下一门禁只能是监督者签署 `MIGRATION-PASS`。

把首次恢复摘要填写到 `LINUX_CODEX_RECOVERY_REPORT.template.json` 的副本或新的带时间戳报告中，提交监督者确认。

在监督者确认恢复摘要前：不得修改项目文件，不得运行 P4 正式 test，不得读取受保护 test 内容，不得调用 LLM，不得下载模型权重，不得进入 P5，不得自行宣布 `MIGRATION-PASS`。
