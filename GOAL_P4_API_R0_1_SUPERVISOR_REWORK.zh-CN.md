# Goal P4-API-R0.1：监督验收后的最小修复

## 1. 目标与停止点

你是 P4-API-R0.1 执行 Agent。只修复监督验收发现的审计和 provenance 问题，不改变 P4 数据、prompt 语义、CAE-IR、Grounder、指标或正式测试协议。

禁止真实 API、train/val、formal test、P5、Abaqus 和模型下载。完成后停止于：

```text
READY_FOR_P4_API_SMOKE_REVIEW_R0_1
```

## 2. 必读

1. `GOAL_P4_API_R0_PROVIDER_ADAPTER.zh-CN.md`
2. `reports/p4_api_r0_provider_adapter_v1/final_acceptance.zh-CN.md`
3. `reports/p4_api_r0_supervisor_review/supervisor_acceptance.zh-CN.md`
4. `reports/p4_api_r0_supervisor_review/supervisor_acceptance.json`
5. `verify_p4_a1_frozen_assets.py`

## 3. 必须修复

### A. 全量冻结 SHA-256

- R0 build/verify 必须逐项重算 P1/P2/P3 manifest 中全部资产的 bytes + SHA-256；
- 不得只复用历史 PASS 或比较文件大小；
- 不打开 formal-test payload；
- 输出 matching/total/change_count 和稳定错误码；
- 当前预期为 P1 13/13、P2 13/13、P3 1885/1885。

### B. Cache provenance

- request hash 加入规范化 backend/provider identity 和 endpoint identity；
- endpoint 若包含 userinfo、query credential 或 secret，hash 前可使用完整值但不得写入日志，公开元数据只保存 SHA-256/脱敏 origin；
- 保留 model、revision、prompt、schema、baseline、input 和全部实际 payload switch；
- 新增测试证明：同 profile/model/revision、不同 endpoint 或 backend 不得 cache hit。

### C. Immutable run identity

- 输出目录使用稳定 `run_id`；
- run_id 至少绑定 endpoint hash、backend、profile、model、revision、prompt/schema/config、baseline、split、offset/limit/shard；
- 同一 run_id 的 manifest 不一致时 fail closed，不得覆盖；
- 相同配置恢复运行仍能命中缓存；
- 增加模型/配置切换不会覆盖旧输出的测试。

### D. Canonical adapter import

- 消除 `llm_adapter.py` 与 `llm_adapter/` 的同名遮蔽；
- 保留一个唯一权威 `utils.cae_language_grounding.llm_adapter` import path；
- 如需保留旧实现，显式命名为 legacy 模块，并写真正独立加载的回归测试；
- 更新 manifest 和文档，不得声称未实际执行的 legacy comparison。

### E. Legacy secret closeout

- 检查 `W:/python/FEA/cae/step3_topology_agent.py`；
- 不记录或复制旧 key 值；
- 将硬编码 credential 攓为环境变量读取，缺失时 fail closed；
- 不写入新 key；
- 若执行环境无权修改该文件，生成明确的人工操作门禁，不得报告全范围 secret=0。

## 4. 测试与交付

- 重跑全部 `test_p4*.py`；
- 新增 endpoint cache isolation、run identity、全量 freeze SHA 和 canonical import 测试；
- 重新验证 ledger run=0、test content 未访问、API call=0；
- 新交付放在 `reports/p4_api_r0_1_supervisor_rework_v1/`；
- 提供 acceptance、中文报告、delivery manifest、复现命令和 context；
- 原 R0 报告保留，不得覆盖或改写监督者审查文件。

最终状态只能是：

```text
READY_FOR_P4_API_SMOKE_REVIEW_R0_1
BLOCKED_WITH_EVIDENCE
```

