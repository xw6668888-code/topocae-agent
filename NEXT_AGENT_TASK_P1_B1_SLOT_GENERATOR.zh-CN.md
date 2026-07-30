# 下一 Agent 任务：P1-B1 槽生成器统一与连续 guide 修复

## 必读

1. `W:\python\FEA\dataset_generator\TOPOCAE_AGENT_RESEARCH_TO_PAPER_MASTER_PLAN.zh-CN.md`
2. `W:\python\FEA\dataset_generator\reports\p1_a2r_human_review_supervisor_gate.zh-CN.md`
3. `W:\python\FEA\dataset_generator\features\swept_channel_slot.py`
4. `W:\python\FEA\dataset_generator\features\swept_slot.py`
5. `W:\python\FEA\dataset_generator\features\slot_rect.py`
6. `W:\python\FEA\dataset_generator\tests\test_swept_slot_v2.py`

## 唯一目标

把 `slot_rect/slot_u/slot_v/slot_o` 统一为“指定闭合截面沿连续 guide 扫掠后与现有实体求差”的一个参数化生成框架，并生成可人工检查的小样本。只修生成与审计，不训练模型。

## 禁止事项

- 不修改 GNN、候选分类器、后处理阈值或真实件规则。
- 不运行正式训练。
- 不覆盖现有 5400 数据、Synthetic-Holdout-v1、checkpoint、freeze manifest 或 P1-A2 报告。
- 不读取或使用 `test_part_s3`。
- 不把多段直线拟合称为连续曲线。
- 不新增槽 macro 类别；四个 macro 仍由截面类型决定，而不是由 guide 类型决定。

## 1. 统一参数模型

每个槽实例必须显式记录以下正交参数：

- `profile_shape`: `rect | u | v | o`
- `guide_family`: `straight | curved_open | curved_closed`
- `end_condition`: `open_open | open_closed | closed_loop`
- `guide_curve_kind`: `line | circular_arc | bezier_or_bspline | circle_or_periodic_bspline`
- `guide_continuity`: 至少 `C1`
- `host_face_kind`: 本阶段只允许现有实体的平面宿主面
- `host_local_frame`: 原点、法向和两个面内主方向
- `path_length/profile_width`
- `path_start_on_boundary`、`path_end_on_boundary`

`slot_o` 表示 O 形截面，不表示闭合环形路径。闭合环槽由 `end_condition=closed_loop` 表示，四种截面均可配闭合 guide。

## 2. 必须覆盖的 20 个组合

四种 profile 均覆盖：

1. straight + open_open
2. straight + open_closed
3. curved_open + open_open
4. curved_open + open_closed
5. curved_closed + closed_loop

正式 pilot 每个组合生成 4 个有效样本，共 80 个；另导出每个组合 1 个固定种子 canonical STEP，共 20 个供用户检查。

## 3. 直线路径规则

- guide 必须沿宿主平面的两个局部主方向之一，禁止连续随机斜角。
- 对轴对齐块/板/梁，guide 在相应平面内与 stock 的局部 X/Y/Z 主轴平行。
- `open_open`：可见 guide 两端到达宿主面边界，工具 guide 在两端继续延伸到实体外，保证没有残余端壁。
- `open_closed`：一端到达边界并向实体外延伸，另一端位于宿主面内部并保留一个完整终止壁。
- 延伸量由截面在端点切向上的支撑尺寸和布尔容差计算，不使用固定“路径长度 10%”作为唯一规则。

## 4. 曲线路径规则

- open curve 必须由单条解析圆弧、Bezier 或 B-spline edge 表示，至少 C1 连续。
- closed curve 必须由圆、椭圆或 periodic B-spline 的单条闭合 edge 表示，首尾至少 C1 连续。
- 禁止先采样点、再为相邻点逐段建立直线 edge。
- `open_open` 的两个可见端点与宿主边界相交，工具曲线沿原曲线切向/参数域延伸到实体外。
- `open_closed` 仅开放端越过边界；闭合端保留终止壁。
- `closed_loop` 完全位于宿主面内部，不得与边界相交。

可以先只实现解析 line、circular arc 和 circle；如果 Bezier/B-spline 不能在本任务中稳定形成单实体，不得用折线替代，应记录为后续扩展。

## 5. 截面规则

- rect：四条直线闭合。
- U：一个精确半圆弧加三条直线闭合。
- V：三条直线组成闭合三角切削截面。
- O：一条完整圆或解析圆弧闭合 wire。
- 截面宽度/深度与 guide 长度解耦，但 `path_length/profile_width >= 4`。
- 截面与宿主面相交，使布尔求差后形成开放槽，不得生成完全埋在实体内部的隧道。

## 6. 标签和语义

- 所有由槽切削形成的可见面必须属于同一 instance，并标为对应 slot macro。
- 开放端不存在 feature terminal face。
- 半通槽的终止壁必须被标为槽的一部分。若现有 semantic subtype 无法准确表示，新增一个共享的 `slot_terminal_face`，并同步 ontology、selectable region 和导出 schema；不得将其标为 stock。
- 曲线槽不得因 OCC 的面类型差异把底面漏标为 stock。
- `raw_parameters` 必须保存真实的 guide 类型和连续性，不得把实际 polyline 记为 circular arc。

## 7. 生成后自动审计

每个样本至少验证：

- STEP 可读、单 solid、正体积、B-Rep valid、无 issues。
- guide wire 的 edge 数：line/arc/closed analytic guide 应为 1。
- guide 连续性至少 C1；closed loop 首尾闭合。
- end condition 与边界关系一致。
- open_open 没有残余端壁；open_closed 恰有一个终止壁；closed_loop 无端壁。
- 所有槽面都有 macro、instance 和 semantic 标签。
- 曲线槽不再出现由固定采样段数导致的 16/32/48 面峰值。
- 20 个组合均有 4 个有效样本。

## 8. 测试

新增或重写测试，至少覆盖：

- 四种 profile wire 的几何组成。
- 解析 line、arc、closed curve guide 的单 edge 和连续性。
- 五种 guide/end 条件的边界关系。
- 四种 profile 与五种条件的 20 组合参数覆盖。
- boolean 后单实体有效性。
- open/half-open/closed-loop 的端壁数量与标签。
- 曲线槽特征面数量不随内部采样点数量线性增加。
- 旧测试中“每个折线段一条 edge”的断言必须删除或改成拒绝正式 polyline guide。

运行全部项目测试，并报告新增测试与全量测试结果。

## 9. 交付物

- 统一后的槽生成代码。
- `tests/test_swept_slot_continuous_guides.py` 或等价测试。
- `output_generation_p1_b1_slot_pilot_80/`
- `output_generation_p1_b1_slot_pilot_80/reports/slot_generation_audit.json`
- `output_generation_p1_b1_slot_pilot_80/reports/slot_generation_audit.zh-CN.md`
- `output_generation_p1_b1_slot_pilot_80/human_review_canonical_20.csv`
- `output_generation_p1_b1_slot_pilot_80/human_review_canonical_20/` 中的 20 个 STEP。

## 10. Agent Gate

- 80/80 样本通过单实体和 B-Rep 有效性检查。
- 20 个组合各 4 个样本，无缺失。
- 所有正式 curved guide 都是单条连续 curve edge，不存在 polyline spine。
- 端部状态自动审计 100% 通过。
- 槽面标签覆盖率 100%，无槽面被标为 stock。
- 全量测试通过。
- 未修改任何冻结数据、模型和报告。

完成后按主计划汇报并停止。不得开始 P1-B2 或训练。
