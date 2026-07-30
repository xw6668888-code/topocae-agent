# Goal P7-A1：Real-Dev 端到端双轨 Pilot 与系统冻结

## 1. 目标

在不创建新 holdout、不继续调识别器的前提下，用 `test_part` 至 `test_part_s2` 完成 Oracle/predicted 两条端到端链路，发现接口错误并冻结最终系统。

## 2. 流程

```text
STEP + requirement
-> feature objects (oracle / predicted)
-> API LLM -> CAE-IR
-> grounding/verifier
-> STEP face set
-> Abaqus face mapping
-> action compiler/executor
-> structured result
```

## 3. Pilot 设计

- 选择覆盖 17 类、七类 operation、中英文、单/多实例和 clarification 的代表性任务；
- 每个任务在 Oracle 和 predicted geometry 下运行；
- API 使用 P4 冻结配置和缓存，不重调 prompt；
- 识别器使用 P1 冻结配置，不按 pilot 错误调阈值；
- 允许修复通用接口 bug，但必须证明不改变模型/规则语义；
- 每次修复都运行完整回归并记录变更原因。

## 4. 指标和误差传播

- CAE-IR exact match；
- object/instance selection；
- exact STEP face set；
- STEP-Abaqus mapping；
- Set/Surface、BC/Load/Material/Mesh 创建；
- overall end-to-end success；
- clarification/abstention；
- unsafe execution rate；
- Oracle 与 predicted 差值；
- recognition/parser/grounder/mapper/compiler 分层 failure attribution。

## 5. 冻结门禁

Pilot 完成后冻结：recognizer、API model/prompt/schema、grounder、role policy、mapper、compiler、evaluator 和所有指标定义。生成正式 Holdout 协议和空白人工审查模板。

完成后停止于：

```text
READY_FOR_FORMAL_HOLDOUT
```

不得自行查看或创建正式 holdout 结果。

