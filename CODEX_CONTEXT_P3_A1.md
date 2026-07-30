# CODEX Context P3-A1

Goal P3-A1 已完成正式数据构造并冻结 `cae-query-ground/1.0.0-h3.01`。正式数据为 21000 canonical / 63000 renderings，每条 canonical 固定 2 英文 + 1 中文。H3-01 为 120/120，有效正确率 100.0%；20 条人工表达均已记录采用或拒绝理由。输出级确定性复现、P2 校验、目标重建和跨 split 泄漏审计均通过。P1/P2 冻结资产变化为 0。

本 Goal 未调用 LLM、未训练模型、未执行 Abaqus/CalculiX、未运行 Holdout、未访问真实件或 `test_part_s3`，未进入 P4。下一步仅可将验收报告交监督者审核；不得在本 Goal 内进入 P4。
