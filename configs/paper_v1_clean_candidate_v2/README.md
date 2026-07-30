# Paper v1 clean candidate v2（P1-A2）

该目录只服务 P1-A2。方案选择仅使用原正式 `Synthetic-Val` 及其确定性旋转派生集。原 `Synthetic-Test` 在本阶段逻辑重命名为 `Synthetic-Development-Test`，只保留历史诊断身份，不参与任何选择。`Real-Dev`、`Rotation-Dev` 和冻结后新建的 `Synthetic-Holdout-v1` 均禁止训练、选模型、选阈值或新增规则。

执行顺序必须为：Val 阶段消融 → 写入并校验 `preholdout_selection.json` → 生成 Holdout → 冻结方案一次性评价 → 重复性检查（非 Holdout 输入）→ v2 manifest。
