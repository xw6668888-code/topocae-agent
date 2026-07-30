# 下一 Agent 任务：P1-A2R

## 任务名称

`P1-A2R：人工审查包可执行性修复`

## 必读

1. `W:\python\FEA\dataset_generator\reports\p1_a2_supervisor_gate_review.zh-CN.md`
2. `W:\python\FEA\dataset_generator\reports\paper_v1_p1_a2\p1_a2_report.zh-CN.md`
3. `W:\python\FEA\dataset_generator\paper_v1_clean_candidate_v2_freeze_manifest.json`

## 唯一目标

生成用户可以在 UG 中实际执行的 20 件 Synthetic-Holdout + 10 件 Real-Dev 人工审查包。该任务只修审查材料，不重新推理、不重算指标、不改变冻结模型、数据、配置、阈值、preholdout selection 或 manifest。

## 硬性禁止

- 不修改或覆盖任何 P0、P1-A1、P1-A2 冻结文件。
- 不运行 Holdout 第二次推理。
- 不根据 Holdout/Real-Dev 结果修改模型、规则或阈值。
- 不读取 S3。
- 不进入 P2。

## 必须实现

新建 `build_paper_v1_p1_a2_human_review_package.py`，输出到：

`reports/paper_v1_p1_a2/human_review_package_v2/`

### Synthetic 20 件

- 使用已经冻结的 Holdout 预测，不重新推理。
- 覆盖全部 18 个目标类别；stepped hole 和 through hole 可各增加 1 件，合计 20 件。
- 每件必须包含存在且可打开的 `step_path`、graph、prediction 和 feature objects 路径。
- 从 graph 真值提取：对象 macro、实例 ID、face IDs、每个 face 的面积、semantic role、对象总面积。
- 从冻结预测提取：预测 macro、face IDs、face 面积、semantic role、置信度、对象总面积。
- 明确列出 truth/prediction 对象数和可能的漏检、误检、碎裂、合并提示，但不得代替人工结论。
- `holdout_rib_000` 必须进入审查包，并增加专门问题：该 rib 是否连接 L 形两板或位于腔内；若只是普通块/板边界上的孤立薄板，标记 `INVALID_TEST_GEOMETRY`。

### Real-Dev 10 件

- 只选择当前真实存在的 `.stp` 文件。
- 从 `test_part`、`test_part_s`、`test_part_s1`、`test_part_s2` 中至少覆盖 3 个目录；不得使用 s3。
- 旧包中 source STEP 不存在的 `test_part/part1、part2、part3、part5` 必须替换，不能保留死路径。
- 每件包含真实 `step_path`、预测对象 macro、face IDs、面积、semantic role、置信度和风险提示。
- 优先保留已有真值或用户此前检查过的简单件，同时覆盖孔、槽、boss、transition、rib/web 中尽可能多的类别。

### 用户填写字段

两个 CSV 均必须有：

- `review_result`：`PASS` / `FAIL` / `INVALID_TEST_GEOMETRY`
- `error_type`：`NONE` / `MISS` / `FALSE_POSITIVE` / `FRAGMENT` / `MERGE` / `ROLE_ERROR` / `GEOMETRY_LABEL_ERROR`
- `review_note`

不得要求用户批量重新标注 face ID。

### 审查说明

生成 `human_review_guide.zh-CN.md`，逐步说明：

1. 如何按 CSV 的 step_path 在 UG 打开模型；
2. 如何用 face area、macro 和 semantic role 核对 graph/预测；
3. synthetic 与 real 两种表分别判断什么；
4. rib 样本的专门判断标准；
5. 用户只需填写哪三列；
6. 完成后应返回哪两个 CSV。

同时生成一个只读 HTML 汇总，方便用户浏览对象摘要；CSV 仍是唯一回填文件。

## 验证

- 30 个 step_path 全部存在；
- synthetic 覆盖 18 类且包含 `holdout_rib_000`；
- real 覆盖至少 3 个允许目录；
- 每个对象摘要至少含 macro、face IDs、面积和 semantic role；预测摘要还必须有置信度；
- 输出中不存在任何 S3 路径；
- 新增单元测试；
- 全项目测试通过；
- 生成独立 `review_package_manifest.json`，记录新文件 SHA256，但不得改写 v2 freeze manifest。

完成后按主计划第 14.1 节汇报并停止，不进入 P2。
