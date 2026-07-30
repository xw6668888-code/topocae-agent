# Goal：P1-C 干净基体、全量特征重生成、受控组合重训与冻结对比

## 一、执行指令

严格执行本文件全部要求，持续完成18类训练几何从零重生成、干净基体修复、单/多特征受控组合、Graph v2 重提取、同架构公平重训、三随机种子、消融、Real-Dev 冻结对比和全新 Synthetic-Holdout-v3 一次性评估。

除本文规定的 Human Gate 外，自主完成全部代码、测试、数据生成、训练、评估和报告。不得因任务耗时而缩减类别、样本量、随机种子、消融或质量审计。未满足本文全部 Goal 完成条件，不得将 Goal 标记为 complete，不得进入 P2。

建议启动命令：

```text
/goal 严格执行 W:\python\FEA\dataset_generator\GOAL_P1_C_CLEAN_GEOMETRY_FULL_REGEN_RETRAIN_COMPARISON.zh-CN.md 中的全部要求。除文档规定的 Human Gate 外自主完成所有代码、测试、数据生成、训练和实验。不得使用 test_part_s3、真实件专用阈值、LLM 直接识别 STEP、旧 Holdout 结果调参或 superpower skills。未满足全部完成条件不得标记 complete，也不得进入 P2。
```

## 二、背景和已确认问题

当前 P1-B 结果只作为冻结基线，不作为本次数据源：

- P1-B 修正数据集：5000 样本、8050 实例。
- Synthetic-Val 三种子均值：face Macro-F1 `0.977427`，object Macro-F1 `0.964711`，exact instance `0.952443`，same-instance edge F1 `0.999627`。
- Synthetic-Holdout-v2：face `0.934394`，object `0.919950`，exact `0.937877`，boundary `0.999212`。
- Real-Dev：area+macro `P=0.672897 / R=0.541353 / F1=0.600000`。
- Real-Dev 中 rib 为 `0/10`，四类槽合计 `0/12`；fillet、blind hole、chamfer、cylinder boss 也存在明显漏检。

已确认 P1-B rib 基体存在可清理冗余面：

- `new_rib_0000_identity`：17 faces，经 `ShapeUpgrade_UnifySameDomain` 后 11 faces，体积不变。
- `new_rib_0002_identity`：19 faces，清理后 13 faces，体积不变。
- `new_rib_0011_identity`：19 faces，清理后 13 faces，体积不变。
- 原因是底板、竖板或支撑体采用重叠 Box 后直接 Fuse，且未完成同域面清理。
- Graph v2 canonical patch 只能聚合部分冗余面；face GNN 和 edge head 仍使用原始 face/edge，因此冗余面会进入训练。

P1-B train split 的上下文分布也不平衡：

- 全局单特征 2392/4046，多特征 1654/4046。
- rib：258 个训练样本中仅 63 个单特征，195 个多特征；与 pocket_rect、pocket_circular、through_hole_round 高度共现。
- 四种槽在 train 中全部为单特征，缺少真实多特征上下文。

本 Goal 的核心假设是：清除生成器特有冗余拓扑，并对每类单/多特征上下文做显式平衡，可以减少模型对生成器痕迹和共现关系的依赖，改善真实件 rib/slot 泛化。

## 三、硬性禁令

以下要求在整个 Goal 中不可违反：

1. 不得读取、枚举、复制、评估或引用 `test_part_s3` 中的任何文件。
2. 不得使用 Real-Dev 的样本名、face ID、面积、零件类别或错误模式设计阈值、规则、候选来源或训练特征。
3. 不得增加零件专用、目录专用、样本专用或真实件专用阈值。
4. 不得让 LLM 查看 STEP、截图或文件名后直接输出几何识别结果；识别必须来自 B-Rep、Graph、模型和通用拓扑。
5. 不得使用任何 superpower skills。允许使用普通文件、终端、测试和必要的非 superpower 工作流，但不得借助被禁用 skills 规划或执行。
6. 不得将 P1-B 的旧 STEP/BREP、旧修正数据集或 Synthetic-Holdout-v2 几何复制进新训练集。
7. 不得用 Synthetic-Holdout-v2 指标、逐样本预测或错误记录选择模型、阈值、规则、数据比例或生成器参数。
8. 不得根据新 Synthetic-Holdout-v3 结果回头修改模型、生成器、规则或阈值。
9. 不得删除困难样本、隐藏困难类别、合并类别或通过降低真值标准使指标通过。
10. 不得修改、覆盖或删除以下冻结基线：
    - `output_generation_p1_b_corrected_balanced_5000_v3/`
    - `reports/p1_b_training_runs_v1/`
    - `reports/p1_b_training_ablation_summary_v1/`
    - `reports/p1_b_real_dev_v1/`
    - `output_generation_p1_b_synthetic_holdout_v2/`
    - `reports/p1_b_holdout_v2_once/`
    - `reports/p1_b_final_acceptance_v1/`
11. 不得进入自然语言 CAE Agent、P2 或其他扩大范围的工作。

## 四、阶段 0：preflight 与冻结基线

开始后首先生成不超过 250 行的中文 preflight 报告，至少包括：

- Python、OpenCASCADE、PyTorch、CUDA、GPU、可用显存和关键包版本。
- 当前 P1-B 数据、三种子 checkpoints、threshold selection、Real-Dev、Holdout-v2 和最终验收文件 SHA-256。
- 受保护目录的文件数和字节数清单。
- 本 Goal 将创建的新目录、配置和单一入口。
- 明确声明 `test_part_s3` 未使用、Real-Dev 不参与开发、旧 Holdout 不参与开发、未使用 superpower skills。
- 记录 P1-B 的冻结数值，后续对比不得重写历史数字。

输出：

- `reports/p1_c_preflight/preflight.zh-CN.md`
- `reports/p1_c_preflight/preflight.json`
- `reports/p1_c_preflight/protected_inventory.json`

完成 preflight 后直接实施，不等待用户确认。

## 五、阶段 1：18 类生成器全部从零重建

新训练集必须完全由本阶段的新生成器和新 seed 产生。18 类全部重新生成：

1. `through_hole_round`
2. `blind_hole_round`
3. `stepped_hole_round`
4. `counterbore_hole`
5. `countersink_hole`
6. `obround_hole`
7. `slot_u`
8. `slot_rect`
9. `pocket_rect`
10. `pocket_circular`
11. `cylinder_boss`
12. `rib`
13. `web`
14. `rectangular_boss`
15. `fillet`
16. `chamfer`
17. `slot_v`
18. `slot_o`

### 5.1 通用干净几何原则

- host/base 优先由单个闭合截面一次拉伸、旋转或一次 Boolean Cut 构造，不得用深度重叠的 Box 堆叠模拟一个简单基体。
- 必须发生 Fuse 的位置只允许使用最小、可配置且记录在 manifest 中的建模容差；最终 shape 必须清理同域面。
- Boolean 操作完成后调用通用 cleanup，例如 `SimplifyResult(True, True)` 和/或 `ShapeUpgrade_UnifySameDomain`。
- cleanup 必须发生在最终 face labeling 之前。
- face 标签必须根据最终 shape 的 Boolean history、最终几何身份或可靠的几何匹配重新传播；不得继续使用 cleanup 前的整数 hash 作为最终真值。
- cleanup 不得吞掉合法 fillet/chamfer、槽终止面、孔阶梯面、rib/web 过渡面或不同实例边界。
- 所有生成器输出必须是一个有效 solid、正体积、无自交、无内部 shell、无悬空面。
- 全部训练几何必须写出可直接在 UG/NX 打开的 STEP；BREP 可作为附加工件，但不得成为唯一几何格式。
- 文件 provenance 使用 manifest 中的 `generated_at_utc`、seed、generator revision 和 SHA-256，不使用文件修改时间作为数据身份。

### 5.2 冗余面自动定义

对每个最终 shape 执行只读同域面探针：

```text
cleaned_probe = ShapeUpgrade_UnifySameDomain(final_shape)
```

必须满足：

- `face_count(final_shape) == face_count(cleaned_probe)`；
- `edge_count(final_shape) == edge_count(cleaned_probe)`，允许仅删除经严格证明不影响拓扑语义的退化 edge；
- cleanup 前后 solid count 均为 1；
- 相对体积误差 `< 1e-9`；
- `removable_same_domain_face_count == 0`；
- `redundant_coplanar_stock_partition_count == 0`；
- 不存在同实例、同曲面、同法向、切向连续且可无损合并的相邻 raw faces。

合法的建模历史 face partition 只能作为单独的鲁棒性派生样本，不得出现在 identity clean geometry 中，并必须与 base family 固定在同一 split。

### 5.3 rib/web 特殊要求

- L 型 rib host 必须由单个 L 截面拉伸形成，不得由底板与竖板重叠 Fuse 形成。
- cavity rib host 必须由干净 block 一次 Cut 得到，Cut 后进行同域面清理。
- two-support web host 必须由单个阶梯/桥接截面拉伸，或使用经过 cleanup 且探针幂等的支撑 host。
- rib/web 必须真实连接至少两个支撑区域，但支撑接触不得通过遗留内部重叠体或冗余分割面证明。
- rib/web feature faces、host faces、pocket faces 和 hole faces 必须互斥；每个辅助特征有独立 instance ID。
- rib/web 薄壁的 span/thickness、height/thickness、支撑接触、过渡、孔变体和 host family 参数必须进入 manifest。
- identity rib/web 样本的 canonical patch 不得被用来掩盖 raw face 污染；raw shape 本身必须干净。

### 5.4 其他类别概念要求

- hole family 必须由材料去除产生，区分贯通、盲孔底、阶梯、沉孔、沉头和 obround 的真实 B-Rep 结构。
- 四类槽继续使用闭合截面连续 sweep-cut，增加干净的多特征 host 和邻近干扰，但不得让不同特征相交或共享真值 face。
- pocket 与 boss 必须分别是材料去除和材料添加；圆/矩形概念、侧壁和底/顶面明确。
- fillet/chamfer 必须来自真实 edge transition 操作，不能把任意小圆柱或斜面直接标成过渡特征。
- 所有类别至少提供 4 种 host family、旋转、尺度和相对位置变化；任何单一 host family 不得超过该类样本的 35%。

## 六、阶段 2：pilot、自动门槛与 Human Gate

在大规模生成前必须完成两个 Human Gate。

### 6.1 H-C1：干净 rib/web 基体 Gate

生成：

- rib 9 个场景：L-corner、rect-cavity、circular-cavity × plain/hole/transition。
- web 3 个场景：two-support × plain/hole/transition。
- 每个场景先生成 4 个 pilot，共 48 个。
- 从固定 seed 中生成 12 个 canonical STEP，每个场景 1 个。

自动门槛：48/48 必须满足单实体、有效 B-Rep、冗余面探针幂等、标签互斥、支撑接触、STEP 重读一致和 finite 审计。

输出：

- `output_generation_p1_c1_clean_rib_web_pilot_48/`
- `output_generation_p1_c1_clean_rib_web_pilot_48/human_review_canonical_12/`
- `output_generation_p1_c1_clean_rib_web_pilot_48/human_review_canonical_12.csv`

### Human Gate H-C1：必须暂停

自动审计通过后必须停止，并用中文明确告诉用户：

- 打开上述 12 个 STEP。
- 在 UG/NX 中确认一个实体、基体没有重叠体、共面分割线或多余小面。
- 确认 rib/web 是受支撑薄壁增强体，未吞掉基体，孔和 transition 变体成立。
- 不要求用户检查 raw face ID。
- 填写 `human_review_canonical_12.csv` 的 `single_solid`、`clean_host`、`feature_concept_valid`、`decision` 和 `notes`。
- 12/12 必须 PASS；任何 FAIL 都要修复、重生成并重新 Gate。

### 6.2 H-C2：18 类单/多特征概念 Gate

H-C1 通过后，为18类各生成：

- 1 个 target-only canonical STEP；
- 1 个 target + compatible secondary feature canonical STEP。

共36个 STEP，固定 seed，覆盖全部18类及单/多特征上下文。

输出：

- `output_generation_p1_c2_all_class_canonical_36/`
- `output_generation_p1_c2_all_class_canonical_36/human_review_canonical_36/`
- `output_generation_p1_c2_all_class_canonical_36/human_review_canonical_36.csv`

### Human Gate H-C2：必须暂停

自动审计通过后停止，并用中文要求用户：

- 打开36个 STEP。
- 检查 target 特征概念、secondary 特征概念、两者不相交且没有不必要基体面。
- 检查多特征样本中不同实例边界清楚，不能互相吞并。
- 填写 CSV 的 `target_valid`、`secondary_valid`、`clean_host`、`composition_valid`、`decision` 和 `notes`。
- 36/36 必须 PASS 后才能生成正式数据集。

## 七、阶段 3：全新 Clean-Context-7200 数据集

正式数据集输出目录：

```text
output_generation_p1_c_clean_context_7200
```

### 7.1 固定规模和组成

总样本数必须为 7200，每类恰好 400 个 primary target 样本：

- target-only：240/类，共 4320。
- target + 1 secondary：120/类，共 2160。
- target + 2 secondary：40/类，共 720。

多特征样本必须使用预先写入配置的兼容性矩阵。secondary 分配必须近似 Latin-square 平衡：

- 全数据集中每类作为 secondary 的出现次数目标为 200，允许范围 190–210。
- 任何 directed primary-secondary pair 不得占该 primary 多特征样本的 10% 以上。
- 每个 primary 至少与 10 个不同 compatible secondary 类组合。
- pair/triple 特征不得几何相交，不得共享真值 face，不得通过难以解释的 Boolean 残片连接。

### 7.2 split

- 按 base geometry family、host family、组合 parent 和所有 rotation/scale/partition 派生分组。
- 目标比例 train/val/development-test 为 80/10/10；允许因 group 约束产生不超过 1%的偏差。
- base family、exact graph signature、near graph signature、source STEP hash 和 parametric family 跨 split 碰撞均为 0。
- 组合样本的所有成分、派生和 parent 必须固定在同一 split。

### 7.3 等规模公平子集

从7200中在任何训练结果产生前冻结一个 group-safe、类别和上下文分层的 5000 样本子集：

```text
output_generation_p1_c_clean_context_7200/splits_matched_5000/
```

该子集只用于与 P1-B 5000 样本规模做公平对比，不能根据模型结果调整。

### 7.4 每个样本的交付物

- STEP；
- Graph v2 JSON；
- 可选 BREP；
- face-level macro、semantic role、instance ID；
- feature_sets/object truth；
- primary/secondary 角色；
- host family、generator revision、seed、参数；
- cleanup 审计；
- exact/near/base signatures；
- SHA-256；
- generated_at_utc。

### 7.5 数据自动验收

- 7200/7200 STEP 可读、单实体、有效、正体积。
- 7200/7200 cleanup 幂等，冗余可合并面为 0。
- face label coverage 100%，face 多重归属为 0。
- 18 类 primary 各400，single/pair/triple 配额精确满足。
- secondary 分布和 host family 分布通过。
- 无 NaN/Inf，Graph v2 输入绝对值上限通过。
- split leakage 为 0。
- H-C1/H-C2 CSV 哈希写入 dataset card 和 manifest。
- P1-B 旧数据和旧 Holdout 的受保护清单不变。

必须生成 dataset card、sample manifest、split manifests、compatibility matrix、co-occurrence matrix、cleanup audit、label audit 和 protected inventory diff。

## 八、阶段 4：Graph v2 与训练表示

- 继续使用 P1-B 的 Graph v2 相对表示作为固定起点，不因 Real-Dev 修改特征。
- identity clean geometry 必须直接通过 raw face audit；canonical patch 不得替代 clean geometry。
- 训练输入分位数只在新 train split 拟合。
- 新增 `primary_target`、`single/pair/triple context` 只能作为采样/评估元数据，不得作为推理输入特征。
- 不得把 sample ID、generator name、host template name、seed 或 source path 输入模型。
- same-instance 真值由最终 clean shape 的 instance ID 产生。
- 候选生成仍使用通用拓扑来源；hard negatives 覆盖缺面、超集、邻近实例合并、跨 primary/secondary 错误合并和同轴异功能混淆。

增加以下鲁棒性测试：

- 任意轴旋转 consistency >=0.98。
- uniform scale consistency >=0.98。
- 合法 face partition consistency >=0.98。
- cleanup 前探针与最终 clean identity 的 canonical signature 一致。
- single 与 multi context 下同类候选特征分布可比较，禁止读取 context 标签。

## 九、阶段 5：同架构训练与公平消融

### 9.1 固定公平原则

- P1-B checkpoints、config 和 threshold 作为只读历史基线。
- P1-C 首轮必须使用与 P1-B 相同的 GNN、same-instance head、candidate classifier、verifier-v4 和 global-decode-v2 架构及默认超参数。
- 不得在看到 Real-Dev 后更换模型、阈值或数据比例。
- early stopping、阈值、全局 decode 参数只使用各自 Synthetic-Val。
- face loss、edge loss和candidate loss必须按 sample/instance/class 归一，避免 face-rich multi-feature 样本因节点多而支配训练。

### 9.2 必须训练的正式变体

1. `P1-B-frozen-baseline`：不训练，只读取冻结报告。
2. `P1-C-clean-5000`：新干净等规模子集，seed `20260718/19/20`。
3. `P1-C-clean-7200`：新完整数据集，seed `20260718/19/20`。
4. `P1-C-single-only`：4320 target-only 样本，主 seed，用于量化多特征上下文收益。
5. `P1-C-no-instance-balanced-loss`：7200，主 seed，用于量化 sample/instance loss 归一收益。
6. `P1-C-no-canonical-patch`：7200，主 seed，用于重新验证 canonical patch 是否有收益。

每个正式三种子变体必须保存：

- checkpoint；
- config；
- epoch history；
- Val predictions；
- threshold selections 及 selection SHA；
- face/semantic/edge/candidate/object/boundary 指标；
- 单/多特征分层指标；
- 每类和每 host family 指标；
- 训练时间、推理时间、参数量、VRAM、环境记录。

### 9.3 比较问题

报告必须明确回答：

- Clean-5000 相对 P1-B-5000 的变化来自几何清理还是训练随机波动？
- Clean-7200 相对 Clean-5000 的变化是否来自数据量？
- multi-feature context 是否改善 multi-context Val，而不显著损害 single-context？
- instance-balanced loss 是否降低 face-rich composite 的支配效应？
- canonical patch 在 clean raw geometry 上是否仍有收益？
- rib/slot 在不同 host/context 上的最差类指标是否改善？

## 十、阶段 6：Synthetic-Val 自动门槛和有限迭代

正式冻结前至少满足：

- face Macro-F1 >=0.92。
- semantic-role Macro-F1 >=0.92。
- object Macro-F1 >=0.90。
- exact instance ratio >=0.88。
- same-instance edge F1 >=0.96。
- candidate proposal oracle recall (IoU>=0.5) >=0.95。
- rotation/scale/partition consistency >=0.98。
- single-context 与 multi-context object Macro-F1 差值绝对值 <=0.05。
- 每类最差 object F1 >=0.80；若某类未达到，必须单独报告。
- 三种子均单独通过绝对门槛，报告 mean/std。
- Clean-5000 与 P1-B 的 matched-size face/object 指标不得显著退化；允许预先定义不超过0.01 face、0.02 object 的非劣界限。

如果未达到，可在 preholdout 之前最多进行两轮通用迭代。每轮必须预先写明：

- 单一可证伪假设；
- 只使用 Synthetic-Train/Val 的证据；
- 改动的数据、表示或候选层；
- 预期改善指标；
- 结果和是否保留。

不得使用 Real-Dev、Holdout-v2 或未来 Holdout-v3 选择迭代。两轮后仍未达到，停止调参，提交瓶颈报告并请求监督者决定是否继续；不得绕过门槛。

## 十一、阶段 7：模型选择和 preholdout 冻结

只根据 Synthetic-Val 和预先声明的选择函数选择最终 P1-C 模型。默认候选为 `P1-C-clean-7200` 三种子中的预声明主 seed；若采用聚合或其他 seed，规则必须在读取结果前写入配置。

冻结文件必须包括：

- 最终 GNN checkpoint/config；
- candidate checkpoint/config；
- same-instance、candidate、global-decode 阈值；
- Graph/export/pipeline/verifier/global decode 源码；
- 新 dataset/split/quantile hashes；
- H-C1/H-C2 Gate hashes；
- 训练环境和选择函数；
- 比较用 P1-B 冻结 baseline hashes；
- 明确列出新 Holdout 正式作业中允许比较的模型变体。

写入：

- `reports/p1_c_freeze_v1/preholdout_selection.json`
- `reports/p1_c_freeze_v1/preholdout_selection.sha256.json`

preholdout selection 必须严格早于 Holdout-v3 生成。

## 十二、阶段 8：Real-Dev 冻结对比

模型选择和阈值冻结后，使用与 P1-B 完全相同的允许 Real-Dev 协议：

- 只允许 `test_part_s`、`test_part_s1`、`test_part_s2`。
- 继续排除 `test_part_s2/part10`。
- 不得访问 `test_part_s3`。
- 使用同一面积真值、同一 `5 mm² 或 1%` area+macro 一对一匹配。
- 使用同一可恢复 face-set 真值和 failure taxonomy。
- 不根据结果修改任何模型、数据、规则或阈值。

只正式评估一个由 Synthetic-Val 选定的 P1-C 模型；P1-B 直接引用冻结结果，不重新开发。报告至少比较：

- P1-B A0：F1 `0.534`。
- P1-B final：F1 `0.600`。
- P1-C clean selected：新结果。

必须按类报告 rib、四类 slot、fillet、blind hole、chamfer、cylinder boss、hole family 和其他类，并输出逐样本 failure atlas。

Real-Dev 比较目标但不是调参门槛：

- area+macro F1 高于 `0.600`；
- rib recall 从 `0` 提升；
- slot family recall 从 `0` 提升；
- 不以显著降低 through-hole 等强类为代价。

无论是否达到，都必须如实保存。若未达到，只能报告假设未被真实件支持，不能回头开发。

## 十三、阶段 9：全新 Synthetic-Holdout-v3

preholdout selection 写入后：

1. 生成 `Synthetic-Holdout-v3`，不得复用任何训练、验证、development-test 或 Holdout-v2 几何。
2. 规模为 1440：18 类各80个 primary target，其中40个 target-only、30个 pair、10个 triple。
3. 使用全新 seed namespace、未用于训练的 host templates 和参数区间。
4. 检查与 P1-C train/val/development-test 以及 Holdout-v2 的 base、STEP hash、exact 和 near graph signature 碰撞，全部必须为0。读取 Holdout-v2 只允许使用 fingerprint manifest，不得读取预测或错误结果。
5. 正式推理作业只能运行一次。preholdout manifest 中预先声明 P1-B frozen 和 P1-C selected 两个比较变体；它们必须在同一个正式作业中一次性完成。
6. 启动作业前写入 execution lock、selection hash、script hash；中断时只允许同一脚本 hash 续跑缺失样本，不得重算已完成样本。
7. 完成后将 inference pass count 固定为1并拒绝第二次执行。
8. 不根据结果修改任何内容。

Holdout-v3 报告必须包含：

- P1-B 与 P1-C 的 face、semantic、edge、proposal、object、exact、IoU、boundary；
- 每类、single/pair/triple、host family 分层；
- P1-C 相对 P1-B 的差值和 bootstrap confidence interval；
- 生成 36 个分层人工审查 STEP：18类各1个 single 和1个 multi；
- 审查包只留档，不是返工 Gate，不得污染 Holdout。

## 十四、测试和防回归

至少新增并运行：

- clean host construction tests；
- unify idempotence tests；
- face/edge count invariance tests；
- final-shape label propagation tests；
- rib/web support and independent-instance tests；
- all-class single/multi generator tests；
- compatibility matrix and co-occurrence quota tests；
- split base/exact/near leakage tests；
- STEP round-trip tests；
- sample/instance-balanced loss tests；
- inference truth-stripping tests；
- Real/holdout forbidden-input tests；
- exactly-once Holdout execution tests；
- protected inventory tests；
- single reproducible entry tests。

任何 P1-C 新增测试必须全部通过。全仓既有失败必须区分：若与本次改动相关则修复；若为已确认的历史人工回填测试冲突，单独记录但不得掩盖新增失败。

## 十五、必须交付的对比报告

至少交付：

- clean geometry generator 设计和实现说明；
- H-C1/H-C2 审查包及用户回填结果；
- 7200 数据集 card、manifest、splits、co-occurrence matrix、cleanup audit；
- matched-5000 子集 manifest；
- 三种子 Clean-5000 与 Clean-7200 checkpoints/config/metrics；
- single-only、no-balanced-loss、no-canonical-patch 消融；
- P1-B vs P1-C Synthetic-Val 公平对比；
- P1-B vs P1-C Real-Dev 对比和 failure atlas；
- P1-B vs P1-C Holdout-v3 一次性正式报告；
- 36 件 Holdout-v3 审查包；
- 环境、时间、参数量和 VRAM 记录；
- 冻结 manifest 和所有 SHA-256；
- 单一复现入口，例如 `run_paper_v1_p1_c_pipeline.py`；
- 更新后的 `CODEX_CONTEXT_P1_C.md`；
- 中文最终验收摘要。

所有数字必须可追溯到 JSON/CSV，不能只出现在 Markdown 或截图中。

## 十六、进度、Human Gate 和长任务规则

- 每完成一个阶段更新计划状态、关键文件、测试、指标和下一步。
- 大约每30–60秒提供一次简短实质进展。
- 长时间数据生成和训练不得留下未结束 shell；允许使用可验证的 resume state，但不得重算已完成正式 Holdout 样本。
- 仅 H-C1/H-C2 或真正无法从资料判断的工程语义可以暂停请求用户。
- Human Gate 必须用中文明确说明打开哪些 STEP、看什么、填哪个 CSV、什么算 PASS。
- 不要求用户检查 raw face ID。
- 可编码、可测试、可运行的工作不得转交用户。

## 十七、自动验收汇总

最终自动验收至少要求：

- H-C1 12/12 PASS。
- H-C2 36/36 PASS。
- 18类全部由新生成器从零生成，旧训练几何复用数为0。
- 7200/7200 STEP 单实体、有效、cleanup 幂等。
- removable same-domain faces 为0。
- face label overlap 为0，标签覆盖100%。
- primary、single/pair/triple、secondary、host family 配额通过。
- split base/exact/near/hash leakage 为0。
- Graph v2 finite、rotation/scale/partition 通过。
- Clean-5000 和 Clean-7200 三种子完整。
- face/object/exact/edge/proposal 绝对门槛通过。
- single/multi gap 通过或有诚实瓶颈报告和监督者决定。
- Real-Dev 只做冻结比较并生成 failure atlas。
- Holdout-v3 只正式推理一次。
- 受保护 P1-B 数据和报告未修改。
- 禁用项全部未使用。

## 十八、Goal 完成条件

只有以下全部成立才能将 Goal 标记为 complete：

1. H-C1 和 H-C2 已完成且全部 PASS。
2. 18类生成器全部从零重建，rib/web host 冗余面问题已由自动测试和人工检查关闭。
3. 新7200数据集、matched-5000、STEP、Graph、labels、splits、card 和 manifests 全部完成。
4. 单/多特征组成和 secondary 共现达到预设配额，旧训练几何复用为0。
5. Clean-5000 与 Clean-7200 三种子训练、全部消融和环境记录完成。
6. Synthetic-Val 自动门槛通过；若两轮通用迭代后仍有未通过项，必须已有监督者对 claim/范围的明确决定。
7. Real-Dev 已冻结比较且没有据其修改模型、数据、规则或阈值。
8. Synthetic-Holdout-v3 已按协议正式推理恰好一次，并生成36件审查包。
9. 必要测试、复现入口、selection hash、freeze manifest 和最终交付清单完整。
10. 未使用 `test_part_s3`、真实件专用阈值、LLM STEP 识别、旧 Holdout 调参或 superpower skills。
11. P1-B 受保护资产未修改。
12. 未进入 P2。

完成后向监督者提交中文验收摘要，逐项对应生成器修复、数据组成、Human Gate、三种子、消融、Synthetic-Val、Real-Dev、Holdout-v3、禁用项和未解决限制。即使 P1-C 优于 P1-B，也不得自动进入 P2。
