# Codex Context：Paper-v1 P1-B

Last updated: 2026-07-18 20:30 Asia/Shanghai  
Workspace: `W:\python\FEA\dataset_generator`  
Python: `W:\python\envs\feagpt\python.exe`

## Current Snapshot

P1-B 通用几何识别修复已完成实现、训练、冻结 Real-Dev 诊断与 Synthetic-Holdout-v2 一次性评估。当前不进入 P2；最终论文真实件泛化 claim 仍需监督者根据 Real-Dev F1=0.6000 决定是否收缩。

## Active Goal

严格执行 `GOAL_P1_B_GENERAL_GEOMETRY_RECOGNITION_REPAIR.zh-CN.md`。工程完成条件由 `reports/p1_b_final_acceptance_v1/acceptance.json` 逐项审计；禁止读取或使用 `test_part_s3`、真实件专用阈值、LLM 直接 STEP 识别和用户禁用的 skills。

## User Preferences And Constraints

- 仅 Human Gate H-B1/H-B2 暂停请求人工；两项均已由用户回填 PASS。
- Real-Dev 只做最终诊断，不参与训练、阈值、early stopping 或模型选择。
- Holdout-v2 结果不得反向修改模型、规则或阈值。
- 未满足全部 Goal 条件不得进入 P2。

## Key Decisions

- 槽使用连续闭合截面 sweep-cut；20 个 profile × guide/end canonical 组合全部通过。
- rib/web 使用具备至少两个真实支撑接触的薄壁增强体；12 个工程场景全部通过。
- Graph v2 使用相对/无量纲节点边特征、identity-safe 双向邻接和非破坏 canonical patch。
- same-instance 双向概率用算术均值对称化；阈值只由 Synthetic-Val 选择。
- 候选级模型、verifier-v4 与 global-decode-v2 接入 `p1_b_general_recognition_pipeline_v4.py`。
- 最终主模型固定为 seed 20260718；preholdout selection SHA-256 为 `8DBEAF19C77F5AEF930772B9B9AD7CF60C98168B7BDE97A8EA120E6D8BC4E051`。
- 两轮允许的通用迭代已用完；Real-Dev 不达 0.75 后未继续调参。

## Important Files And Paths

- 单一入口：`run_paper_v1_p1_b_pipeline.py`
- 最终推理：`p1_b_general_recognition_pipeline_v4.py`
- 修正数据集：`output_generation_p1_b_corrected_balanced_5000_v3/`
- 三种子与 checkpoints：`reports/p1_b_training_runs_v1/`
- 训练消融：`reports/p1_b_training_ablation_summary_v1/`
- Real-Dev：`reports/p1_b_real_dev_v1/`
- preholdout freeze：`reports/p1_b_freeze_v1/`
- Holdout-v2 数据：`output_generation_p1_b_synthetic_holdout_v2/`
- Holdout-v2 正式结果：`reports/p1_b_holdout_v2_once/`
- 最终验收：`reports/p1_b_final_acceptance_v1/`

## Completed Work

- H-B1 20/20、H-B2 12/12 PASS。
- 修正数据集 5000 件、8050 实例、18 类每类至少 300；STEP/BREP 100% 有效，split leakage=0。
- Synthetic-Val 三种子均通过：face Macro-F1 均值 0.977427，object Macro-F1 0.964711，exact 0.952443，edge F1 0.999627。
- Real-Dev 26 个允许 STEP 已冻结评估；统一 area+macro P/R/F1=0.672897/0.541353/0.600000，并生成 7 类 failure atlas。
- Holdout-v2 1080 件（18 类各 60），相对 5000 件训练/验证数据 base/exact/near 碰撞均为 0；仅正式推理一次。
- Holdout-v2：face 0.934394，semantic 0.934036，edge 0.988558，proposal recall 1.0，object 0.919950，exact 0.937877，boundary 0.999212。
- 20 件分层审查 STEP 覆盖全部 18 类以及槽、rib、web；该包仅留档，不触发返工。

## Verification

- P1-B 必要专项 21/21 通过。
- Graph v2 表示专项 6/6 通过。
- 既有全仓 discovery 的两个 P1-A2 失败属于用户人工回填后旧测试仍要求空字段/旧哈希；不是本次新增回归。
- Holdout state 与 generation manifest 的正式推理次数均为 1，1080 个预测文件和 20 个审查 STEP 均已写入冻结 manifest。

## Open Questions Or Blockers

- Real-Dev F1=0.6000 高于 A0 0.5340，但未达到优先开发目标 0.75；slot_rect 与两个 rib face-set 未命中。
- failure atlas 以 abstention 与 invalid_topology 为主。若继续研究，必须使用下一论文版本和全新开发/Holdout，不得污染 Holdout-v2。
- canonical patch 消融未显示正收益；不应在论文中声称其贡献。

## Next Steps

1. 运行 `python run_paper_v1_p1_b_pipeline.py --verify --run-tests` 做只读验收。
2. 监督者决定是否收缩真实件泛化 claim。
3. 不进入 P2，除非用户另行明确授权。

## Suggested Skills

无。继续工作时先读取本文件和最终验收 JSON；不要启用用户禁用的 skills。

## Update Log

### 2026-07-18 20:30 Asia/Shanghai

- 完成 P1-B 所有工程阶段、一次性 Holdout-v2、Real-Dev 失败图谱与最终冻结证据。
- 新建独立 P1-B 上下文，以免旧全局 `CODEX_CONTEXT.md` 中历史实验与本次严格禁用范围混淆。
