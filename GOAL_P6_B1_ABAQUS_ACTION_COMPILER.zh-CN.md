# Goal P6-B1：可验证 Abaqus Action Compiler 与 Executor

## 1. 目标

把通过 CAE-IR、grounder、verifier 和 face mapper 的结果编译成白名单 Abaqus 前处理操作，并通过 noGUI 执行后返回结构化结果。

LLM 不能生成或修改 Abaqus Python；执行器只解释版本化任务 JSON。

## 2. 实施范围

定义并实现：

```text
validated_cae_task.json
-> deterministic compiler/executor
-> Abaqus model objects
-> abaqus_execution_result.json
```

第一版支持：STEP 导入、part/instance、StaticStep、Set/Surface、材料/实体截面、fixed support、displacement、pressure、surface traction、global/local mesh size、job creation 和 CAE 保存。

默认不提交 job。提交求解必须是独立显式开关并经过人工确认。

## 3. 工程要求

- JSON Schema closed-world；
- operation ID 唯一且可追踪；
- 任务引用 STEP SHA-256、mapping manifest 和单位上下文；
- 字符串、名称和数值严格校验，禁止代码注入；
- 不允许 `except: pass`；
- 每个操作返回 success/failure/error code；
- 支持 dry-run、幂等重放和已存在对象冲突策略；
- 压力方向、Surface side、位移自由度和材料单位显式；
- Abaqus 无内置单位制，任务必须声明并验证一致单位约定；
- 普通 Python 与 Abaqus Python 使用文件协议隔离。

## 4. 测试

- schema 和非法任务拒绝；
- 非法/歧义 mapping 拒绝；
- Set/Surface 创建；
- 七类 P2 operation；
- 重复执行；
- 部分操作失败时的事务/清理策略；
- 路径带空格和中文时的处理；
- noGUI 退出码、stdout/stderr 和 result JSON 一致；
- 不提交求解的默认安全行为。

## 5. H6-Execution

用户在 Abaqus 中抽查至少 10 个任务，确认区域、约束、载荷方向、材料、网格和 job 状态。Agent 提供截图/高亮索引和简洁审查表，不要求用户阅读代码。

## 6. 验收与停止

- 自动 fixture operation creation success=100%；
- 人工审查任务无错误区域或方向；
- silent failure=0；
- arbitrary-code path=0；
- 所有冻结资产变化=0。

完成后停止于：

```text
READY_FOR_END_TO_END_PILOT
```

