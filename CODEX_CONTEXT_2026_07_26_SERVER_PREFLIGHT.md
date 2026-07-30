# Codex Context Snapshot - 2026-07-26 Server Preflight

## 状态

- 已读取 `scripts/linux_server_info.txt`。
- 服务器为 Ubuntu 24.04.4、双 A40 46 GiB、125 GiB RAM。
- `/home/ywx1` 可写，所在文件系统剩余约 375 GiB。
- 硬件条件通过；当前迁移监督结论为 `CONDITIONAL-GO`。

## 阻塞项

- Hugging Face HTTPS 连接超时。
- 无 Conda/Mamba/Micromamba/uv。
- Docker 无当前用户权限。
- 用户磁盘配额未知。
- PyPI、conda-forge、GitHub 可访问性未知。
- 两张 A40 当前占用未知。
- Windows SSH Host 别名尚未确认；`ywx1@node6:~$` 只是远端 shell 提示符。

## 新交付物

- `scripts/collect_linux_migration_blockers.sh`
- `reports/SERVER_MIGRATION_PREFLIGHT_2026_07_26.zh-CN.md`

## 下一门禁

用户在服务器运行二次探测脚本，并把输出保存为 `scripts/linux_migration_blockers.txt`。监督者据此选择在线 Micromamba、管理员环境或离线依赖迁移路线；通过后状态升级为 `GO-M1`，才开始项目归档与上传。
