# CAE STEP 特征识别实验对比设计建议

最后更新：2026-07-02  
工作目录：`W:/python/FEA/dataset_generator`  
用途：用于论文实验设计、组会汇报和后续代码实现排期。

## 1. 结论先行

对于本项目，最合理的实验对比不是简单地做：

1. “我的模型跑别人的数据集”；或
2. “别人的模型跑我的数据集”。

这两件事都可以做一部分，但都不适合作为论文主实验的唯一依据。

更科学、也更容易落地的实验结构应当是：

```text
主实验：在我的 CAE-STEP 数据集上，比较不同方法对 CAE 特征对象和可选区域的识别能力。
辅助实验 1：在公开 machining / B-Rep 数据集上验证我的 B-Rep 图学习模块不是只对自建数据有效。
辅助实验 2：将公开模型思想复现成可运行 baseline，在我的数据集上做同任务公平比较。
辅助实验 3：在真实/简化真实 STEP 上做泛化测试和案例分析。
```

一句话：

> 第一篇论文的主线应该是“面向 CAE 前处理的 STEP 特征对象识别”，所以主评价必须落在自己的 CAE 数据集和真实件测试集上；公开数据集和公开方法只作为可信 baseline 与泛化补充。

## 2. 已阅读和参考的核心文献

### 2.1 B-Rep 特征/分割类

| 文献 | 权威性 | 任务 | 实验设计重点 | 对本项目可借鉴点 |
|---|---|---|---|---|
| BRepNet: A topological message passing system for solid models, CVPR 2021 Oral | 权威经典 | Fusion 360 B-Rep face segmentation | 与 mesh/point cloud/ECC 对比；指标为 per-face accuracy、IoU；做 kernel 和 input feature 消融 | 应作为 B-Rep 图网络 baseline 的概念来源；说明 B-Rep 比 mesh/point cloud 更适合 CAD |
| UV-Net: Learning from Boundary Representations, CVPR 2021 | 权威经典 | B-Rep classification、segmentation、retrieval | MFCAD face segmentation；指标为 accuracy、per-class accuracy、IoU；与 PointNet++、MeshCNN、BRepNet 等对比 | 可借鉴 UV-grid + adjacency graph；也可作为“更强面几何编码”的 future baseline |
| AAGNet: A graph neural network towards multi-task machining feature recognition, RCIM 2024 | 与当前项目最接近 | MFCAD、MFCAD++、MFInstSeg machining feature recognition | 语义分割、实例分割、bottom face 多任务；指标包含 face-level accuracy/mIoU、link prediction F1、feature-level F1 和 localization IoU | 是当前项目最重要的直接参考；但其目标是 CAPP/加工特征，不是 CAE 区域选择 |
| FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting, CVPR 2026 | 最新且高度相关 | B-Rep classification/segmentation 的旋转鲁棒性 | 原始测试集 vs SO(3) 随机旋转测试集；accuracy、IoU；数据量消融 | 本项目必须加入 rotation consistency / rotated test set；可借鉴旋转不变局部描述符 |
| FeatureFox: Sample-Efficient Panoptic Graph Segmentation for Machining Feature Recognition in B-Rep 3D-CAD Models, 2026 | 最新且最适合复现 | MFInstSeg panoptic machining feature recognition | edge boundary classifier + connected components + instance classifier；指标为 Panoptic Quality、feature-level recognition/localization | 最推荐作为本项目的先进 baseline：对象级、样本高效、和我们的候选对象路线高度一致 |

### 2.2 大模型 / LLM-CAD 类

| 文献 | 任务 | 实验设计重点 | 对本项目的启发 |
|---|---|---|---|
| Pointer-CAD, CVPR 2026 | text-to-CAD generation with B-Rep face/edge pointer | DeepCAD/Recap-DeepCAD/OmniCAD+；命令 F1、Chamfer Distance、SegE、DangEL、Invalidity Ratio；GNN component ablation | 大模型没有直接替代 B-Rep；它通过 pointer 显式选择 face/edge。对本项目启发是 LLM 应用于 candidate reasoning / entity selection |
| CAD-Llama, CVPR 2025 | LLM 生成参数化 CAD command/code | DeepCAD；Text-to-CAD、caption、addition/deletion；ACC、Chamfer、BLEU/Rouge 等 | 证明 LLM 需要结构化 CAD code 和专项训练；不适合作为裸 STEP 面识别器 |
| CAD-MLLM, 2025 | 多模态条件 CAD generation | Omni-CAD；text/image/point 输入；Chamfer、F-score、Normal Consistency、SegE、DangEL、SIR、FluxEE | 评价的是生成质量和拓扑质量，不是已有 STEP 特征面识别；可借鉴 topology metrics 和 multimodal/LLM 设计 |
| BrepCoder, 2026 | B-Rep 到 CAD code 的多任务大模型推理 | reverse engineering、completion、error correction、CAD-QA；ACCcmd、ACCparam、CAD-QA accuracy | 最新大模型路线仍然先编码 B-Rep，再与 code/LLM 对齐；适合作为未来 CAE agent 参考 |
| GeoCAD / Seek-CAD, 2025 | LLM-based parametric CAD generation / self-refinement | DeepCAD 或自建工业风格 CAD code；Chamfer、Hausdorff、IoU/user study/validity | 适合说明 LLM 在 CAD 生成中有价值，但不直接解决 STEP 面级识别 |

## 3. 回答你的两个想法

### 3.1 想法 1：用我的模型跑其他人的数据集

可以做，但只能作为辅助实验。

原因：

- AAGNet/MFCAD/MFInstSeg 是加工特征识别数据集，目标是 CAPP，不是 CAE 前处理。
- BRepNet/Fusion360 segmentation 是建模操作分割，不是孔、槽、肋、CAE 语义区域。
- Pointer-CAD、CAD-Llama、CAD-MLLM 多数是 CAD 生成任务，不提供与你当前任务完全一致的 face-level CAE feature labels。
- 公开数据集是否开源、是否有 STEP、是否有 face label、是否许可证允许使用，都需要逐项确认。

适合做的版本：

| 公开数据集 | 是否建议 | 怎么做 |
|---|---|---|
| MFCAD | 建议 | 将我的 graph extractor + GNN backbone 跑在 MFCAD face segmentation 上，只评价可对应的加工特征类别。 |
| MFCAD++ | 建议，但注意数据质量 | 同上；适合测试更复杂的加工特征语义分割。 |
| MFInstSeg | 很建议 | 如果可获取，最适合作为外部实例级验证；评价 face-level + instance-level。 |
| Fusion360 Gallery Segmentation | 可选 | 任务是建模操作面分割，不是 CAE 特征；适合证明 B-Rep graph backbone 的通用性。 |
| DeepCAD / Text2CAD / Omni-CAD | 不建议作为主实验 | 它们是 command sequence / text-to-CAD generation，和已有 STEP 面选择任务不一致。 |

结论：

> 如果要做“我的模型跑别人数据集”，优先跑 MFCAD/MFCAD++/MFInstSeg，而不是 Pointer-CAD 或 CAD-Llama 的生成数据集。

### 3.2 想法 2：拿其它模型跑我的数据集

这是更适合作为主实验的方向，但要注意“任务公平性”。

不能直接拿 Pointer-CAD/CAD-Llama 这种生成模型跑你的数据集，因为它们的输出是 CAD command sequence，不是已有 STEP 中的 face ID。真正可比的是：

| baseline | 是否能跑我的数据集 | 对比方式 |
|---|---|---|
| Rule-only | 可以 | 用手写拓扑规则识别孔、槽、凸台、transition；作为传统可解释基线。 |
| Face MLP / RandomForest | 可以 | 只用单面 descriptor，不用图邻接；证明拓扑信息必要。 |
| Vanilla GCN/GAT/GraphSAGE | 可以 | 使用同样 node_attr/edge_attr，作为基础图网络 baseline。 |
| BRepNet-style | 可以复现简化版 | 用 coedge/winged-edge 邻域或近似邻接 kernel；比较 B-Rep 专用消息传递是否更强。 |
| UV-Net-style | 可以复现简化版 | 增加 face UV-grid/edge-grid 几何编码；比较连续曲面采样是否提升。 |
| AAGNet-style | 可以复现 | 多任务：face semantic + same-instance edge/link + subregion/bottom/role head。 |
| FeatureFox-style | 强烈建议 | edge boundary classifier -> connected components -> instance classifier；指标可用 PQ、object-level F1。 |
| LLM-only | 不建议作为正式强 baseline | 可作为失败/上限讨论：输入候选摘要或图摘要，而不是裸 STEP。 |
| LLM candidate reasoner | 建议作为高级模块 | 在候选对象层做 rerank/merge/split/classify，和非 LLM candidate classifier 比较。 |

结论：

> 论文主实验应该是“在我的 CAE 数据集上复现/适配几个代表性 baseline”，而不是强行拿生成模型输出面 ID。

## 4. 推荐实验矩阵

### 4.1 主实验：自建 CAE 数据集上的方法对比

数据集划分：

- `Synthetic-Clean`：单一特征、非交叉、多方向、多毛坯。
- `Synthetic-Multi`：多个不交叉或轻微邻接特征。
- `Synthetic-Hard`：专门针对混淆构造：
  - through hole vs outer cylinder；
  - blind hole vs cylinder boss；
  - slot_u vs obround_hole；
  - slot_o vs fillet；
  - rib vs stock/pocket/web；
  - countersink/counterbore/stepped/chamfer。
- `Simplified-Real STEP`：用户手工画的 `test_part_s/s1/s2/s3`。

对比方法：

| 编号 | 方法 | 目的 |
|---|---|---|
| M0 | Rule-only | 传统规则上限/下限，展示可解释但覆盖不足。 |
| M1 | Face-only MLP/RF | 证明单面属性不够。 |
| M2 | Vanilla GNN | 基础图结构 baseline。 |
| M3 | Edge-aware GNN | 当前主模型的核心。 |
| M4 | Edge-aware GNN + topology postprocess | 证明显式几何规则能修复明显不合法预测。 |
| M5 | M4 + candidate classifier | 证明对象级分类优于面级分类。 |
| M6 | FeatureFox-style | 最推荐的先进 baseline：edge boundary + connected component + instance classifier。 |
| M7 | M5/M6 + LLM candidate reasoner | 大模型增强模块：只处理候选对象，不裸读 STEP。 |

### 4.2 外部数据集辅助实验

目的不是证明你的 CAE 特征体系在加工数据集上最好，而是证明：

- B-Rep 图提取和网络 backbone 有通用性；
- 与 AAGNet/BRepNet/UV-Net 类方法的差异不是数据集偶然造成的。

建议安排：

| 数据集 | 任务 | 我们应该跑什么 |
|---|---|---|
| MFCAD | face semantic segmentation | 跑 M1/M2/M3，和文献报告值比较，作为 sanity check。 |
| MFCAD++ | face semantic segmentation | 同上，注意类别映射和损坏 STEP。 |
| MFInstSeg | face semantic + instance | 跑 M3/M6，比较 face-level 和 object-level。 |
| Fusion360 Gallery | modeling-operation face segmentation | 可选，验证 B-Rep backbone 不是只能识别孔槽。 |

不建议优先跑：

- Pointer-CAD 的 DeepCAD/Recap-DeepCAD；
- CAD-Llama 的 text-to-CAD；
- CAD-MLLM 的 Omni-CAD。

这些任务是生成，不是既有 STEP 面识别。

## 5. 指标设计

### 5.1 face-level 指标

用于衡量每个面分类是否正确：

- Overall Accuracy；
- Macro-F1；
- Per-class Precision / Recall / F1；
- mIoU；
- Confusion Matrix。

适合对比：M1/M2/M3/M4。

### 5.2 object-level / instance-level 指标

用于衡量一个特征对象是否被正确识别：

- Object Precision / Recall / F1；
- Instance IoU：预测面集和真值面集的 IoU；
- Localization F1：IoU 超过阈值才算 TP；
- PQ / SQ / RQ：参考 FeatureFox 的 panoptic evaluation。

推荐阈值：

```text
IoU >= 0.5: 可认为识别到对象
IoU >= 0.75: 高质量定位
IoU >= 0.9: 面集几乎完全正确
```

### 5.3 CAE-region 指标

这是你的论文应该区别于加工特征识别的指标：

- Query Success Rate：给定仿真需求，是否能找到正确区域。
- Region IoU：查询输出的区域与真值区域的面集 IoU。
- Critical Region Recall：对孔壁、支承面、肋、槽内壁这类关键区域的召回率。
- False Selection Rate：错误选择 stock 大面、外圆柱、非目标面等比例。
- Topology Validity Rate：输出对象是否满足基本拓扑合法性。

示例查询：

```text
固定所有盲孔
在通孔内壁施加压力
选择 U 形槽内壁
选择 rib/web 作为薄壁候选
忽略主特征附属倒角，仅输出主载荷区域
```

### 5.4 rotation consistency 指标

参考 FoV-Net，必须做。

方法：

- 对同一个 STEP 生成 3-5 个随机旋转版本；
- 由于 face ID 可能变化，用面积、拓扑签名、面类型、邻接摘要做近似匹配；
- 比较旋转前后 feature objects 是否一致。

指标：

- Rotated Face Macro-F1；
- Object Recall under Rotation；
- Prediction Consistency Rate；
- Feature Count Consistency；
- Query Success Consistency。

## 6. 大模型实验怎么设计

### 6.1 不建议做 LLM-only

不建议把整张 STEP graph 文本化后直接丢给 LLM 输出面 ID，原因：

- face 数量不固定；
- face ID 不稳定；
- 图结构 token 很长；
- LLM 缺少几何内核约束；
- 很难保证输出面存在且拓扑合法；
- 不是现有 LLM-CAD 文献的主流做法。

可以把 LLM-only 作为一个小规模负例实验：

```text
Input: graph summary + face table
Output: predicted feature objects
Metric: object-level recall / invalid output rate
Expected: 可解释性较好，但准确性和稳定性不足
```

### 6.2 推荐做 LLM candidate reasoner

流程：

```text
STEP
 -> B-Rep graph
 -> GNN / FeatureFox-style candidate proposal
 -> candidate object summary JSON
 -> LLM reasoner
 -> keep/delete/merge/split/classify
 -> topology postprocess
 -> CAE selectable regions
```

LLM 输入：

- 候选面集合；
- 面类型和面积；
- 邻接边类型、凹凸性、长度、loop role；
- 候选对象尺寸比例；
- GNN/candidate classifier 概率；
- 拓扑规则证据；
- CAE 查询语义。

LLM 输出：

- 是否保留；
- 类别修正；
- 是否合并/拆分；
- CAE 子区域选择；
- 简短理由。

对比：

| 方法 | 是否有 LLM | 评价 |
|---|---|---|
| M5 candidate classifier | 无 | 传统机器学习对象分类器。 |
| M5 + GPT/DeepSeek prompt rerank | 有，零样本/少样本 | 测试大模型推理是否能修复 hard cases。 |
| M5 + LoRA-tuned CAD/CAE LLM | 有，微调 | 如果 A40 资源足够，作为增强实验。 |

指标：

- hard-case object F1；
- invalid candidate removal rate；
- false merge/split rate；
- query success rate；
- inference cost/time；
- explanation usefulness。

## 7. 论文实验章节建议结构

### 7.1 Dataset

介绍：

- Synthetic CAE-STEP；
- hard cases；
- simplified real STEP；
- 类别体系；
- 标注方式；
- 数据划分；
- 旋转增强和非旋转测试。

### 7.2 Baselines

分三类：

1. Rule-based；
2. B-Rep learning；
3. Candidate-level / LLM-enhanced。

### 7.3 Metrics

分四层：

1. Face-level；
2. Object-level；
3. CAE query-level；
4. Rotation robustness。

### 7.4 Main Results

表格建议：

```text
Method | Face Macro-F1 | Object F1 | PQ | Query Success | Rotation Consistency
```

### 7.5 Ablation

建议做：

- w/o edge_attr；
- w/o loop-role；
- w/o topology rules；
- w/o candidate classifier；
- w/o rib boundary/proposer；
- w/o LLM reasoner；
- UV/FoV descriptor 替换或增强。

### 7.6 Generalization

包括：

- synthetic -> simplified real；
- clean -> hard；
- unrotated -> rotated；
- 自建数据 -> MFCAD/MFInstSeg 可选。

## 8. 最推荐的落地顺序

### Step 1：先整理当前已有最佳流程结果

把当前 pipeline 固化为：

```text
Ours-v5.x = edge-aware GNN + candidate classifier + topology postprocess + rib repair/proposer
```

输出统一报告：

- 自建测试集；
- `test_part_s/s1/s2/s3`；
- face-level；
- object-level；
- query-level。

### Step 2：补一个最简单但有说服力的 baseline

实现：

- Rule-only；
- Face-only MLP/RF；
- Vanilla GNN。

这些容易实现，能证明图和拓扑规则有效。

### Step 3：实现 FeatureFox-style baseline

这是最重要的先进 baseline：

```text
same-instance edge classifier
 -> prune adjacency graph
 -> connected components
 -> instance classifier
```

它与当前项目非常接近，公平、可复现、指标也高级。

### Step 4：做 rotation robustness 实验

参考 FoV-Net：

- 原始测试集；
- 随机 SO(3) 旋转测试集；
- 对比有无旋转增强、有无旋转不变 descriptor。

### Step 5：做 LLM candidate reasoner 小实验

不一开始微调。

先做：

- 选 30-50 个 hard cases；
- 给 LLM 输入 candidate summary；
- 输出 candidate 修正；
- 和 candidate classifier 比较。

如果有效，再考虑：

- LoRA 微调；
- 用 A40 训练本地小模型；
- 接入智能体流程。

## 9. 最终建议

对于你的两个想法，我的建议是：

```text
主线采用想法 2：让适配后的公开模型/公开思想跑我的 CAE 数据集。
辅助采用想法 1：让我的 backbone 跑 MFCAD/MFCAD++/MFInstSeg，证明方法不是只对自建数据有效。
不要把 Pointer-CAD/CAD-Llama/CAD-MLLM 作为直接对比模型；它们是 CAD 生成任务，不是 STEP 面识别任务。
把它们作为“大模型模块设计依据”，即 LLM + B-Rep encoder / pointer / candidate reasoner。
```

这样实验既公平，也更容易讲清楚创新点：

1. 面向 CAE 前处理，而不是传统 CAPP 加工特征；
2. 输出可选面集，而不是只分类整个零件；
3. 结合 B-Rep 图学习、拓扑规则和对象级修复；
4. 引入 LLM 作为候选对象推理和 CAE 查询理解模块，而不是不可靠地裸读 STEP。

