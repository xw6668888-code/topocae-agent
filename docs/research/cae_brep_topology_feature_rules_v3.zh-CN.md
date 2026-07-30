# CAE STEP Recognition v3.0: B-Rep 拓扑特征表示规则报告

更新时间：2026-06-05  
工作目录：`W:\python\FEA\dataset_generator`

## 1. 结论摘要

当前 v2.6 模型在合成数据上表现较好，但在真实 STEP 上失败，根本原因不是单纯“样本不够”或“模型太小”，而是任务定义和数据结构仍停留在逐面分类层面。CAE 前处理中的区域选择，本质上需要识别由若干面和共享边组成的拓扑实例。仅用 face node 的面积、曲面类型、包围盒、邻居均值，不能稳定区分“孔壁”和“法兰外圆面”、“槽壁”和“导轨外侧面”、“圆角面”和“小通孔壁”。

v3.0 应改为：

1. 使用 attributed AAG：面是节点，面之间共享的真实 B-Rep 边是图边属性。
2. 模型同时学习 face label、same-instance edge label 和可选的 face role。
3. 后处理必须加入拓扑合法性检查，过滤“不可能特征”，例如两个面组成矩形腔、单独一个圆柱面组成盲孔。
4. 数据集应先回到干净、少干涉、少特征数量的概念学习，再加入工程背景负样本。

## 2. 参考依据

本报告采用的核心依据：

- AAG 是特征识别中常用的辅助数据结构，节点表示面邻接关系，边属性可存储 concave/convex/smooth 等 vexity 信息；这些边属性是特征识别的重要几何启发。来源：Analysis Situs AAG overview，https://analysissitus.org/features/features_aag.html
- AAGNet 使用 geometric Attributed Adjacency Graph，强调 B-Rep 中的拓扑、几何和扩展面/边属性对于实例识别是必要的。来源：Wu 等，AAGNet: A graph neural network towards multi-task machining feature recognition，https://doi.org/10.1016/j.rcim.2023.102661
- 面向 FEA 的 rib recognition 文献强调：FEA 不是只要识别几何实体，还要分解区域数据；rib 往往是互相连接的复杂结构，需要识别所有属于 rib structure 的面，并区分 rib segment 与 transition region。来源：Lai 等，Recognition and decomposition of rib features in thin-shell plastic parts for finite element analysis，https://doi.org/10.1080/16864360.2017.1375678
- Rib 在工程/CAD 语义上通常是从主体伸出的薄壁状支撑结构，基本参数包括 length、height、thickness、draft angle。来源：CAD Exchanger MTK Molding_Rib class reference，https://docs.cadexsoft.com/mtk/classcadex_1_1_molding___rib.html

## 3. B-Rep 规则词表

为避免“凸边/凹边”含义混乱，v3.0 中建议统一采用以下计算词表：

| 概念 | B-Rep/AAG 表示 | 说明 |
|---|---|---|
| face | graph node | STEP B-Rep 中的 TopoDS_Face |
| adjacency edge | graph edge / arc | 两个面共享至少一条 B-Rep edge |
| shared edge curve type | edge_attr | line、circle、ellipse、bspline、other |
| shared edge length | edge_attr | 用 BRepGProp 计算共享边长度 |
| edge vexity | edge_attr | convex、concave、smooth、unknown |
| dihedral angle | edge_attr | 相邻两面沿共享边的夹角 |
| loop role | face/edge derived attr | inner loop、outer loop、hole loop、slot loop |
| face role | supervised/derived label | wall、bottom、top、shoulder、side、transition、stock/context |
| same-instance | edge label | 相邻两个面是否属于同一特征实例 |

注意：用户描述中的“凸边/凹边”应在代码中对应为统一的 AAG vexity，而不是肉眼方向。孔口、凸台根部、槽底转角等边的凹凸性要以 OpenCascade face orientation 和 solid orientation 计算结果为准，并在小样本上人工核对一次。

## 4. 各特征规则评估

### 4.1 通孔 through_hole_round

用户描述：

`入口面 - 圆形边 - 圆柱面 - 圆形边`

可计算性：高。  
稳定性：中高。  
建议作为硬约束。

最小 B-Rep 规则：

- 至少包含 1 个圆柱面作为 hole wall。
- 圆柱面应存在两个开口侧边界，通常对应两个圆形 edge loop。
- 圆柱面不应连接 bottom face。
- 两端邻接的上下文面可以是平面，也可以是曲面。
- 两端圆形边的 vexity 应一致，但具体 convex/concave 名称要以实现核对为准。

容易误判对象：

- 法兰外圆柱面。
- 凸耳外圆侧面。
- 圆柱毛坯外侧面。

必须加入的负样本：

- 大圆柱外轮廓面不能被识别为通孔。
- 只有一个圆柱面但没有内环/穿透关系时不能直接判定为通孔。

### 4.2 盲孔 blind_hole_round

用户描述：

`入口面 - 圆形边 - 圆柱面 - 圆形边 - 底部圆面`

可计算性：高。  
稳定性：高。  
建议作为硬约束。

最小 B-Rep 规则：

- 至少包含 1 个圆柱壁面和 1 个底部圆形平面。
- 圆柱壁面与底面直接邻接，共享圆形边。
- 圆柱壁面另一端连接入口上下文面。
- 底面法向应近似平行于圆柱轴。
- 深度/半径比应大于圆形腔阈值。

建议阈值：

- `depth / radius >= 1.0` 可作为盲孔候选。
- `depth / radius < 0.6` 更倾向圆形腔。
- 中间区间设为不确定或依赖语义需求。

### 4.3 圆形腔 circular_pocket

用户描述：

与盲孔一样，但深度尺寸远小于半径尺寸。

可计算性：高。  
稳定性：中高。  
建议作为硬约束 + 尺度软阈值。

最小 B-Rep 规则：

- 拓扑组成与盲孔相同：圆柱侧壁 + 底面。
- 主要差别是尺度比例：浅而宽。
- 圆形腔更常作为加载/压力/接触区域，盲孔更常作为固定/定位/装配孔。

训练注意：

- 必须把 `depth/radius` 或等价尺度描述符放入节点/实例特征。
- 合成数据中二者比例区间必须明显分开，否则真实件上会不可解释地混淆。

### 4.4 阶梯孔 stepped_hole_round

用户描述：

`大孔入口面 - 圆形边 - 大圆柱面 - 圆形边 - 台阶平面 - 圆形边 - 小圆柱面 - 圆形边 - 小孔底面`

可计算性：高。  
稳定性：中高。  
建议作为硬约束，但要区分盲阶梯孔和通阶梯孔。

最小 B-Rep 规则：

- 至少 2 个同轴圆柱面。
- 大圆柱半径大于小圆柱半径。
- 两个圆柱之间存在环形台阶面，台阶面法向近似平行于孔轴。
- 若是盲阶梯孔，末端有小孔底面。
- 若是通阶梯孔，末端没有底面，而是穿透到另一侧上下文面。

当前需要明确：

- 第一篇论文中建议只保留 `stepped_hole_round`，但在 semantic subtype 中区分 `stepped_hole_large_wall`、`stepped_hole_shoulder`、`stepped_hole_small_wall`、`stepped_hole_bottom_or_exit`。

### 4.5 沉孔 counterbore_hole

用户描述：

`入口面 - 圆形边 - 大圆柱面 - 圆形边 - 平面 - 圆形边 - 小圆柱面 - 圆形边 - 出口面`

可计算性：高。  
稳定性：高。  
建议作为硬约束。

最小 B-Rep 规则：

- 大圆柱面 + 台阶平面 + 小圆柱面。
- 大小圆柱同轴。
- 大孔半径大于小孔半径。
- 通常小圆柱面贯穿实体或连接另一侧上下文面。
- 台阶平面是 CAE 中非常重要的 selectable region。

与阶梯孔区别：

- 两者 B-Rep 很接近；工程语义上 counterbore 更强调安装/螺栓沉头，通常入口大孔较浅，小孔贯穿。
- 在第一阶段可把它们都视作“同轴多半径孔”，再用比例和出口关系细分。

### 4.6 锥沉孔 countersink_hole

用户描述：

`入口面 - 圆形边 - 锥面 - 圆形边 - 平面`

可计算性：高。  
稳定性：高。  
建议作为硬约束。

最小 B-Rep 规则：

- 至少一个 cone face。
- 锥面轴线与孔轴一致。
- 锥面邻接入口面和小孔壁/底面/出口面。
- 如果只识别到孤立小圆柱面而没有锥面，不应判为 countersink。

### 4.7 长圆孔 obround_hole

用户描述：

至少有一个封闭的“两个半圆弧 + 两条直线”组成的跑道状边。

可计算性：中高。  
稳定性：中高。  
建议作为硬约束 + loop 识别。

最小 B-Rep 规则：

- 存在 obround/slot-like inner loop。
- loop 由两段圆弧和两段直线组成，或由等价的圆柱端壁 + 平面侧壁组成。
- 若为穿透长圆孔：没有底面。
- 若为长圆盲腔：存在底面，但应从宏类上归为 slot/pocket，而不是 through obround hole。

实现重点：

- 当前只靠 face 形状容易把单个长条面误判；必须识别 loop 或聚合端部半圆壁与两侧平面壁。

### 4.8 槽类 slot

用户描述：

截面尺度远小于扫描方向尺度，可抽象成一个以上长条状面，通过长边拼接。

可计算性：中。  
稳定性：中。  
建议作为软硬结合规则。

最小 B-Rep 规则：

- 主方向长度明显大于截面宽度/深度。
- 至少包含长条状侧壁或底部区域。
- 组成面之间多通过长直边或沿扫掠方向的边相连。
- U/O/V 槽由截面类型决定，但在真实 B-Rep 中可能被分裂为多个面。

建议拆分：

- `slot_rect`：底面 + 两个长侧面 + 端面，或至少两个长侧面。
- `slot_u`：包含圆弧底/端部过渡面，截面近似 U 形。
- `slot_v`：包含两个倾斜侧面，截面夹角形成 V 形。
- `slot_o`：更接近圆管状/圆弧槽，必须小心不要与圆孔混淆。

### 4.9 矩形腔 rect_pocket

用户描述：

最起码由四个侧面 + 一个底面组成，且以凹边相连。

可计算性：高。  
稳定性：高。  
建议作为硬约束。

最小 B-Rep 规则：

- 至少 5 个面：1 个底面 + 至少 4 个侧面。
- 底面与侧面通过 concave edges 相连。
- 相邻侧面也应形成腔体边界。
- 侧面法向通常成两组近似平行/反平行方向。
- 只有 2 个或 3 个面组成的结果必须判为 invalid pocket。

这条规则应直接解决 v2.6 中“两个面/三个面组成矩形腔”的明显错误。

### 4.10 圆柱凸台 cylindrical_boss

用户描述：

`基准面 - 圆形边 - 圆柱面 - 圆形边 - 顶部圆面`

可计算性：高。  
稳定性：高。  
建议作为硬约束。

最小 B-Rep 规则：

- 圆柱侧壁 + 顶面。
- 圆柱侧壁与基准上下文面相连。
- 顶面法向近似平行于圆柱轴。
- 与盲孔的主要区别是凸/凹方向相反：凸台是 protrusion，盲孔是 depression。

易混淆对象：

- 法兰外圆。
- 圆柱毛坯外侧。
- 凸耳外侧曲面。

必须加入负样本：

- 一个大型圆柱体的外表面不能被当作圆柱凸台。

### 4.11 矩形凸台 rectangular_boss

用户描述：

最起码由四个侧面 + 一个顶面组成，且以凸边相连。

可计算性：高。  
稳定性：高。  
建议作为硬约束。

修正：

- 用户描述中的“底面”在凸台语义中应改为“顶面/端面”。基准面是上下文面，不属于凸台实例本身。

最小 B-Rep 规则：

- 至少 5 个面：4 个侧面 + 1 个顶面。
- 侧面与顶面、侧面与上下文基准面之间形成 protrusion 关系。
- 与矩形腔区别是 edge vexity 相反，且实例位于主体外部而不是主体内部凹陷。

### 4.12 肋 / 腹板 rib / web

用户要求：查找可靠材料，并判断如何用 B-Rep 信息表示。

文献与工程定义：

- Rib 通常是从主体伸出的薄壁状支撑结构，用于增加结构强度/刚度而不显著增加材料量。
- 面向 FEA 的 rib recognition 不能只识别几何实体，还要识别 rib segment、transition region，并为网格分解提供区域数据。
- Rib 往往组成网络，而不是孤立简单体。

可计算性：中。  
稳定性：中。  
建议第一阶段不要做过细分类，而是作为 `thin_wall_support_candidate` / `rib_web_candidate`。

最小 B-Rep 规则：

- 存在一对面积较大、近似平行、间距较小的主侧面。
- 两主侧面之间的距离为 thickness，显著小于 height 和 length。
- `thickness / length` 和 `thickness / height` 小于阈值。
- 该薄壁体通过边或过渡面连接到一个或多个主体面。
- 常伴随根部圆角/倒角，但根部过渡不应作为判断必要条件。

建议尺度规则：

- `thickness < 0.2 * min(length, height)` 作为候选。
- 若 `length / thickness > 4` 且 `height / thickness > 2`，可认为是强候选。
- 两侧面法向夹角接近 180 度。

肋与腹板的暂定区别：

- `rib`：局部加强筋，通常从一个主体面伸出并连接到另一个面/壁，长度相对有限。
- `web`：更大面积的板状连接区，常用于连接两个主体结构或分隔腔体。

第一篇论文建议：

- 不强行区分 rib 和 web 的细粒度工程语义。
- 统一作为 `rib_web_candidate` 或保留两个宏类但共享一套薄壁识别规则。
- CAE downstream 中把它们映射为 `shell_candidate` / `stiffener_candidate`。

### 4.13 圆角 / 倒角 fillet / chamfer

用户描述：说不清楚。

可计算性：中高。  
稳定性：中。  
建议作为局部 transition feature，但不要让其破坏主特征。

Fillet 最小规则：

- 面类型通常为圆柱面、圆环面、surface of revolution 或近似圆弧扫掠面。
- 相邻两个或多个主面。
- 面积和半径相对于邻接主面较小。
- 连接边通常为 smooth 或特定 vexity。

Chamfer 最小规则：

- 通常为窄长平面。
- 法向不与主要基准轴或邻接主面平行。
- 连接两个主面，面积相对较小。

关键约束：

- transition 不能单独抢占小孔壁。
- transition 应作为主特征的附属区域或独立可选区域，取决于 CAE 需求。
- 训练中应增加 `transition_context_negative`：小通孔壁、薄侧面、法兰连接孔壁不能误判为 fillet/chamfer。

## 5. v3.0 规则稳定性评级

| 特征 | 推荐规则类型 | 稳定性 | 是否适合硬约束 |
|---|---|---:|---|
| 通孔 | 圆柱壁 + 双开口/双环 | 中高 | 是 |
| 盲孔 | 圆柱壁 + 底面 | 高 | 是 |
| 圆形腔 | 盲孔拓扑 + 浅宽比例 | 中高 | 是，尺度用软阈值 |
| 阶梯孔 | 同轴多圆柱 + 台阶面 | 中高 | 是 |
| 沉孔 | 同轴大小孔 + 台阶面 + 通孔出口 | 高 | 是 |
| 锥沉孔 | 锥面 + 同轴小孔/出口 | 高 | 是 |
| 长圆孔 | 跑道 loop 或端圆弧 + 侧壁 | 中高 | 是 |
| 槽 | 长宽尺度 + 长边拼接 + 截面形态 | 中 | 部分硬约束 |
| 矩形腔 | 底面 + 至少四侧面 + 凹边 | 高 | 是 |
| 圆柱凸台 | 圆柱壁 + 顶面 + protrusion vexity | 高 | 是 |
| 矩形凸台 | 顶面 + 至少四侧面 + 凸边 | 高 | 是 |
| 肋/腹板 | 薄壁尺度 + 成对大面 + 主体连接 | 中 | 软硬结合 |
| 圆角/倒角 | 小尺度 transition + 邻接主面 | 中 | 软硬结合 |

## 6. v3.0 数据集建议

### 6.1 v3.0_clean_concept

目标：让模型先学会拓扑语法。

生成规则：

- 每个样本 1-3 个主特征。
- 避免主特征互相破坏。
- 不追求复杂组合。
- 每个目标特征必须满足拓扑合法性规则。
- stock/context 面明确标注为背景。

### 6.2 v3.1_context_negative

目标：解决真实件误判。

新增背景结构：

- 简化法兰盘：作为背景，不作为目标宏类。
- 凸耳：作为背景，凸耳上的孔才是目标。
- 导轨：作为背景，导轨上的槽/孔才是目标。
- 圆柱毛坯和圆柱外壁：作为背景负样本。
- 分隔腔体的大板/腹板：根据需要标为 rib/web 或 context。

### 6.3 v3.2_engineering_composite

目标：接近简化真实工程件。

生成规则：

- 每个样本 3-6 个特征。
- 允许特征位于凸耳、法兰、导轨等背景结构上。
- 允许腔体被腹板分隔，但必须保证标注仍可解释。
- 禁止布尔操作严重破坏已有目标特征的核心拓扑。

## 7. v3.0 模型建议

### 7.1 输入图

每个样本包含：

- `node_features`
- `edge_index`
- `edge_features`
- `macro_feature_ids`
- `semantic_subtype_ids`
- `instance_ids`
- `edge_same_instance_labels`
- 可选：`face_role_ids`

### 7.2 模型结构

建议从轻量可实现版本开始：

- Node encoder：MLP + primitive embedding。
- Edge encoder：MLP 编码 edge_attr。
- Message passing：edge-aware MPNN / GINE-style layer。
- Heads：
  - face macro head；
  - face semantic head；
  - same-instance edge head；
  - optional face role head。

### 7.3 损失函数

总损失：

`L = L_face_macro + L_face_semantic + L_edge_same_instance + L_role + L_topology_penalty`

其中：

- `L_edge_same_instance` 是关键，用于学习哪些邻接面属于同一特征。
- `L_topology_penalty` 第一阶段可先不反向传播，只作为后处理 invalid rate；第二阶段再转为可微或近似可微惩罚。

### 7.4 后处理

不再使用“同类节点连通分量”。

新流程：

1. 根据 face macro logits 得到候选类别。
2. 根据 same-instance edge logits 裁剪邻接图。
3. 在裁剪图中做连通分量，得到候选实例。
4. 对每个候选实例执行拓扑合法性规则。
5. 合法实例输出；非法实例降级为 `context/unknown` 或拆分。

## 8. v3.0 验证指标

除了原来的 face-level 指标，必须增加：

- same-instance edge F1。
- instance IoU。
- instance precision/recall/F1。
- topology invalid rate。
- per-class invalid reason count。
- CAE selectable-region accuracy。
- real simplified STEP compact review accuracy。

其中 `topology invalid rate` 应作为核心论文指标之一，因为它能直接体现“拓扑先验”对明显错误的抑制效果。

## 9. 当前需要用户确认的设计决策

1. 法兰、凸耳、导轨是否只作为 context/background，不作为第一篇论文的目标识别类别？
2. 肋/腹板是否先合并为 `rib_web_candidate`，后续再细分？
3. 阶梯孔与沉孔是否先共享“同轴多半径孔”规则，再通过深度/出口/用途细分？
4. 圆角/倒角是否允许作为独立候选区域，但同时被主特征规则吸收为附属 transition？

## 10. 推荐执行顺序

1. 增加 edge_attr 提取与训练导出。
2. 增加拓扑规则库和 invalid feature checker。
3. 增加 same-instance edge labels。
4. 实现 edge-aware GNN。
5. 生成 `v3.0_clean_concept` 小样本并训练。
6. 加入 `v3.1_context_negative`，专门压制法兰外圆、凸耳外圆、导轨外侧等误判。
7. 用用户简化真实 STEP 做 compact review。

