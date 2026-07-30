# Goal P5-A1：Predicted Geometry Memory 与拓扑约束 Grounding

## 1. 目标

在 P4 已冻结的语言解析和 Oracle grounder 基础上，把 P1-B predicted feature objects 接入同一 CAE-IR grounding 接口，形成 Oracle/predicted 双轨，并量化识别误差如何传播到对象和精确面集选择。

本 Goal 不训练新的通用识别器，不读取 `test_part_s3`，不执行 Abaqus，不修改 P4 formal-test 结果。

## 2. 必读与冻结

- `WINDOWS_API_ABAQUS_AGENT_ROUTE.zh-CN.md`；
- `TOPOCAE_AGENT_SUPERVISOR_IMPLEMENTATION_PLAN.zh-CN.md`；
- P1 capability contract 和冻结 pipeline；
- P2 CAE-IR/role policy；
- P4 final freeze、Structured LLM 输出契约和 deterministic grounder。

P1-P4 冻结资产只读。

## 3. 实施任务

1. 定义 predicted feature-object memory adapter，与 Oracle memory 使用同一公开 schema；
2. 保留 `geometry_source=predicted|oracle|human_confirmed` 和 recognizer coverage；
3. 将 object confidence、conflicts、abstention 和 semantic face roles正确传入 grounder；
4. predicted 路径不得补用 truth、面积真值或人工 face ID；
5. 复用 CAE-IR 的 feature type、quantifier、descriptor、spatial selector 和 role policy；
6. 对 slot、rib/web、transition 和不完整实例执行 capability-aware abstention；
7. 输出 exact STEP face set、对象来源和每一步 selection trace；
8. 在 Synthetic-Test 与 Real-Dev (`test_part` 至 `test_part_s2`) 上评估 Oracle/predicted 双轨；
9. 分解 parser、recognizer、candidate、role expansion、verifier 五类误差；
10. 评估是否真的需要 learned pointer。只有确定性 selector 在候选完整且描述符充分时仍存在系统性歧义，才提出后续模型，不得为“高级模块”强行训练。

## 4. 指标

- object selection P/R/F1；
- exact instance-set accuracy；
- face P/R/F1/IoU；
- exact face-set accuracy；
- coverage-risk 和 abstention accuracy；
- Oracle-to-predicted performance gap；
- unsafe silent fallback rate，必须为 0；
- rotation consistency。

## 5. 人工门禁

仅对 predicted 路径中的少量 abstention/ambiguity 生成审查包。用户只判断“候选对象和面角色是否适合该 CAE 操作”，不填写大规模 raw face ID。

## 6. 交付与停止

交付 adapter、selection trace、双轨评估、failure atlas、learned-pointer necessity decision、测试、acceptance、freeze/delivery manifest 和 context。

完成后停止于：

```text
READY_FOR_FACE_MAPPING
```

