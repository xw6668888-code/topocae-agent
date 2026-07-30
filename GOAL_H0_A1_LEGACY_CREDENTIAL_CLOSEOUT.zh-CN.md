# Goal H0-A1：Legacy Credential Closeout

## 1. 目标

关闭 `W:\python\FEA\cae\step3_topology_agent.py` 中已撤销但仍硬编码保存的旧 credential，使真实 API smoke 前达到全范围 secret gate。

本 Goal 不调用任何 API，不运行 train/val，不访问 formal test，不进入 P5，不执行 Abaqus。

## 2. 安全要求

- 不在消息、终端、diff、报告或日志中打印、复制或引用旧 key 值；
- 不创建包含旧 key 的备份文件；
- 不把新 key 写入源码；
- 只修改 credential 获取和缺失时 fail-closed 逻辑，不改变旧原型的几何/CAE 行为；
- 新 API 正式路线仍使用 `dataset_generator` 的 provider adapter，旧原型不能成为正式入口。

## 3. 修改要求

1. 删除硬编码 credential 字面值；
2. 只从进程环境变量读取，例如 `TOPOCAE_LLM_API_KEY`；
3. 环境变量缺失时，在创建 client 或发请求前抛出清晰错误并停止；
4. 不提供默认 key、fallback key 或示例 key；
5. 扫描 `W:\python\FEA\cae` 和 `W:\python\FEA\dataset_generator` 的生产源码；
6. 测试环境变量缺失路径，不得进行网络连接；
7. 确认 hardcoded credential=0、environment read=true、missing credential fail-closed=true。

## 4. 交付

在 `W:\python\FEA\dataset_generator\reports\p4_api_r0_1_legacy_secret_closeout_v1\` 输出：

- `security_audit.json`；
- `legacy_closeout_test.json`；
- `acceptance.json`；
- `final_acceptance.zh-CN.md`；
- `delivery_manifest.json`。

不得在交付物中包含任何 secret、Authorization header 或旧 key hash；这里只需要证明字面凭据已经不存在。

最终状态：

```text
READY_FOR_P4_API_R1_AUTHORIZATION_REVIEW
BLOCKED_WITH_EVIDENCE
```

