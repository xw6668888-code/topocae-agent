# M1-A1 路径可移植性扫描结论

日期：2026-07-26

## 结论

迁移前必须增加运行时路径解析层；不得对冻结数据执行全局字符串替换。

## 已确认的关键故障链

P3-A1R 冻结 canonical query 的 `source_ref.graph_path` 保存为：

```text
W:\python\FEA\dataset_generator\...\graphs\sample.json
```

P4 当前调用链：

```text
run_p4_a1_llm_baselines.py
-> example.source_ref["graph_path"]
-> utils.cae_language_grounding.evaluation.MemoryCache.get()
-> Path(graph_path).read_bytes()
```

在 Linux 上，`pathlib.Path` 不会把 Windows drive path 映射到项目目录，因此会读取失败。该链路必须在迁移归档前修复。

## 源码扫描分类

### A. 必须修复的运行时路径

- `utils/cae_language_grounding/evaluation.py` 的 `MemoryCache.get()`。
- 任何从 graph/query/manifest 读取 `graph_path`、`geometry_path`、`source_step_path`、`source_graph_path` 后直接 `Path(value)`、`open(value)` 或 `read_*()` 的代码。
- Linux 上需要执行的 frozen asset verifier 和 reproduction entrypoint。

### B. 本身可移植，无需修改

大量当前脚本通过以下方式定位根目录：

```python
ROOT = Path(__file__).resolve().parent
```

或：

```python
ROOT = Path(__file__).resolve().parents[n]
```

这些代码移动到 `/home/ywx1/dataset_generator` 后会自动得到 Linux 路径。

### C. 历史证据或测试字面量，不应为迁移而改写

- `p1_d_lite/finalize.py`、`p1_d_lite/preflight.py` 中生成旧 Windows 复现命令的文本。
- `p1_d_lite/freeze.py` 中冻结的 Windows CLI 说明。
- `utils/run_p2_a1_cae_ir_tests.py` 中写入报告的历史 command 字段。
- `utils/cae_ontology_audit.py` 中描述旧原型目录的 metadata。
- 部分测试中故意构造的 Windows path fixture。
- 冻结 JSON、JSONL、graph、manifest、report 中的 Windows 路径字段。

这些内容可能是冻结哈希的一部分。除非监督者证明它是当前运行入口且不在冻结范围内，否则不得直接修改。

## 解析器契约

新增唯一公共函数，例如：

```python
resolve_workspace_path(value: str | Path) -> Path
```

使用环境变量：

```text
TOPOCAE_ORIGINAL_WORKSPACE_ROOT=W:\python\FEA\dataset_generator
TOPOCAE_WORKSPACE_ROOT=/home/ywx1/dataset_generator
```

行为要求：

1. 当前平台上已经有效的绝对路径直接解析。
2. 用 `PureWindowsPath` 识别 Windows drive path，不能依赖 Linux `Path.is_absolute()`。
3. Windows 根路径匹配应忽略 drive letter 大小写，但不能模糊匹配其他目录。
4. 将根路径之后的相对部分拼接到 `TOPOCAE_WORKSPACE_ROOT`。
5. 解析后必须仍位于目标工作区，拒绝 `..`、不同 drive、UNC 越界和符号链接越界。
6. 相对路径按项目根目录解析，而不是按当前 shell 工作目录解析。
7. 返回解析后的路径，但保留原始字符串用于审计和哈希，不回写源 JSON。

## 最低测试集

1. Windows 当前根路径在 Windows 正常解析。
2. Windows 冻结路径在模拟 Linux root 下映射到相同相对文件。
3. `/home/ywx1/dataset_generator/...` 在 Linux 直接解析。
4. 项目相对路径正常解析。
5. `..` 越界、其他盘符、非法 UNC 路径被拒绝。
6. P3-A1R train 中一个真实 `source_ref.graph_path` 可由 `MemoryCache.get()` 成功加载。
7. 路径解析前后源 JSON 字节和 SHA-256 不变。
8. P4 Oracle/Keyword 回归结果不变。

## 迁移门禁

只有路径解析实现、上述测试和冻结哈希检查全部通过后，才允许进入 H-M1-01 和创建大型迁移归档。
