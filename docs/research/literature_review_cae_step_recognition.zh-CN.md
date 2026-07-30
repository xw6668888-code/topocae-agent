# 面向 CAE 的 STEP/B-Rep 特征识别文献审查与工作重规划建议

## 0. 当前阅读范围

本次重点阅读了 `W:\python\FEA\reference` 下的两篇 PDF：

1. Wu 等，2024，`AAGNet: A graph neural network towards multi-task machining feature recognition`
2. Qi 等，2026，`Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection`

同时结合 Connected Papers 线索和网络检索，补查了以下对当前课题最相关的工作：

- BRepNet: A topological message passing system for solid models
- UV-Net: Learning from Boundary Representations
- Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition
- MFTReNet: Machining feature and topological relationship recognition based on a multi-task graph neural network
- BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation
- Self-Supervised Representation Learning for CAD
- DeepCAD / Text2CAD / CADmium 等命令序列式 CAD 生成工作

Connected Papers 页面本身未能直接抓取，但通过论文题名、引用链和官方源码仓库补齐了核心相关工作。

---

## 1. AAGNet 论文总结

### 1.1 它解决什么问题

AAGNet 面向的是加工特征识别，也就是从 CAD/B-Rep 模型中识别孔、槽、口袋、凸台等加工语义，服务于 CAPP/CAM。它指出传统规则方法难以覆盖复杂相交特征，点云、体素、网格、多视图等深度学习表示又会损失 B-Rep 中精确的几何和拓扑信息。

### 1.2 它的核心做法

它提出 `gAAG`，即 Geometric Attributed Adjacency Graph。这个图的节点是 B-Rep face，边是相邻 face 共享的 B-Rep edge。与普通 AAG 不同，它不仅保留 face adjacency，还加入：

- face 属性：面类型、面积、质心、loop 数量等
- edge 属性：边类型、长度、凹凸性、二面角等
- UV grid：对曲面和曲线做规则采样，保留连续几何信息

模型 AAGNet 是一个多任务图网络，主要任务包括：

- semantic segmentation：每个 face 属于哪类加工特征
- instance segmentation：哪些 face 组成同一个特征实例
- bottom face segmentation：识别加工特征底面

实例分割被建模为 link prediction，即预测两个 face 是否属于同一个特征实例，而不是后处理式地按类别和连通性硬分组。

### 1.3 它的数据集贡献

它基于 MFCAD++ 生成了 MFInstSeg，包含约 62,495 个 STEP 模型，24 类加工特征，并提供：

- face-level semantic labels
- feature instance labels
- bottom face labels

论文还强调 topological sanity check，即生成数据时检查模型是否有效、闭合、流形、coedge 唯一。这一点对当前工作很重要。

### 1.4 它的实验怎么做

实验分三层：

- 在 MFCAD 上做 face-level semantic segmentation，对比 UV-Net、MeshCNN、PointNet、DGCNN 等。
- 在 MFCAD++ 上验证复杂相交特征。
- 在 MFInstSeg 上验证 semantic segmentation、instance segmentation、bottom face segmentation，并报告 face-level 和 feature-level 指标。

关键指标包括：

- accuracy
- mIoU
- instance F1
- feature-level recognition F1
- feature-level localization F1
- 参数量
- 推理时间
- ablation：去掉 face attributes、edge attributes、UV grid，比较性能下降

### 1.5 源码对你的启发

AAGNet 源码结构很清楚：

- `utils/AAGExtractor.py`：从 STEP/B-Rep 转成 gAAG
- `models/segmentors.py`：face-level segmentation
- `models/inst_segmentors.py`：semantic + instance + bottom 多任务模型
- `models/encoders.py`：GCN、GraphSAGE、GIN、GAT、DeeperGCN、AAGNetGraphEncoder 等 encoder

对你最有价值的不是完全照搬模型，而是三点：

1. 你的 AAG 需要补 `edge_attr`，至少包含边长、凹凸性/二面角、共享边类型。
2. 你的实例识别不应长期停留在“同类连通分量后处理”，应加入 edge/pair affinity 监督。
3. 你的数据集需要有拓扑合法性检查和标签审计，否则合成数据可信度不足。

---

## 2. Pointer-CAD 论文总结

### 2.1 它解决什么问题

Pointer-CAD 面向 text-to-CAD / CAD generation。已有命令序列方法可以生成 sketch、extrude 等操作，但在复杂编辑中有两个问题：

- 无法精确选择已有 face 或 edge，因此难以支持 chamfer、fillet 等需要选择实体的操作。
- 连续参数离散化会带来量化误差，导致草图平面错位、边不对齐、拓扑断裂。

### 2.2 它的核心做法

Pointer-CAD 引入 pointer-based command sequence。命令序列中不再只包含 label token 和 value token，还包含 pointer token，用于指向已有 B-Rep 中的 face 或 edge。

它的流程是多步生成：

```text
自然语言描述
  -> 当前已生成的 B-Rep
  -> B-Rep encoder / GNN 编码 face 和 edge
  -> LLM 预测下一步命令
  -> 若命令需要选择实体，则输出 pointer embedding
  -> 用 cosine similarity 在候选 face/edge embedding 中选择目标
  -> 执行 CAD 操作并更新 B-Rep
```

也就是说，它把“选择几何实体”从坐标回归问题变成了“在有限候选 face/edge 中检索”的问题。

### 2.3 它的表示和模型

它定义三类 token：

- Label Token：操作类型，如 sketch、extrude、chamfer、fillet
- Value Token：数值参数，如坐标、角度、半径、深度
- Pointer：指向 B-Rep face 或 edge

B-Rep encoder 采样方式：

- face：`32 x 32` UV grid，每个点包含坐标、法向、高斯曲率、visibility mask
- edge：32 个采样点，每个点包含坐标、切向、反向切向、一阶导数
- 之后用 CNN + GNN 得到 128 维 face/edge embedding

LLM 使用 Qwen2.5，并通过 LoRA 微调。输出包括 value head 和 pointer head。pointer head 输出 128 维向量，与所有候选 face/edge embedding 做相似度匹配。

### 2.4 它的数据和实验

论文构建了约 575K CAD 模型数据，扩展了 chamfer 和 fillet 操作，并用视觉模型和大语言模型生成较高质量的自然语言描述。

实验指标包括：

- operation F1：line、arc、circle、extrusion、chamfer、fillet 的命令级准确率
- Chamfer Distance：几何相似度
- Segment Error：拓扑片段错误
- Flux Enclosure Error：封闭性/水密性质量
- Invalidity Ratio：生成失败率
- ablation：去掉 GNN，对比性能下降

### 2.5 对你的启发

Pointer-CAD 和你的任务不一样：它是“生成 CAD”，你是“识别并映射 CAE 区域”。但它有一个非常适合借鉴的思想：

> 下游任务不要靠坐标硬猜，而要通过 B-Rep entity selection / pointer / fingerprint 对齐到真实 face 或 edge。

你的 Abaqus mapping 本质上就是一个工程版 pointer：

```text
识别到的 graph face / feature object
  -> 生成几何指纹
  -> 在 Abaqus topology face 集合中选择最匹配的 face
```

因此你可以把自己论文中的映射模块表述为：

> task-conditioned B-Rep entity grounding for CAE preprocessing

这比“简单后处理匹配”更有论文味道，也更贴合 Pointer-CAD 的思想。

---

## 3. BRepNet 对你的启发

BRepNet 是 B-Rep 深度学习的基础工作之一。它不是把 B-Rep 简化成普通图，而是利用 coedge 的有向拓扑结构做 message passing。B-Rep 中的 coedge 有 next、previous、mate、parent face、parent edge 等关系，这些拓扑关系比普通 face adjacency 更丰富。

它的启发是：

- 普通 AAG 只知道两个面相邻，但不知道沿哪个 coedge、loop 顺序如何、相邻关系方向如何。
- 对于 CAE，孔边界、倒角链、圆角链、薄壁边界都依赖更细的拓扑关系。
- 你的近期版本不必完全实现 BRepNet，但至少应补 edge attributes；中期可以考虑引入 oriented edge / coedge-like 信息。

最可实现的借鉴路径：

```text
当前 AAG:
face node + adjacency edge

下一步:
face node + edge_attr

再下一步:
face-edge bipartite graph 或 face-edge-face 三元结构

长期:
coedge/topological-walk style encoder
```

---

## 4. UV-Net 对你的启发

UV-Net 的贡献是把 B-Rep 的连续曲面和曲线通过 UV 参数域规则采样，变成神经网络容易处理的 grid，同时保留 face adjacency。AAGNet 和 Pointer-CAD 都明显吸收了这个思路。

你的当前节点特征主要是面积、质心、法向、轴向、半径等手工属性。这对通用性和 Abaqus 对齐很好，但对识别复杂曲面、倒角、圆角、自由曲面、边界形状可能不够。

建议不要一开始就完整做 32x32 UV grid，成本较高。可以分两步：

1. 先补轻量几何统计：bbox、面积归一化、法向方向、二面角、边长分布、loop 数量。
2. 再做小尺寸 UV grid，例如 `8 x 8` 或 `16 x 16`，作为 ablation，而不是一开始作为主方法。

---

## 5. Hierarchical CADNet 对你的启发

Hierarchical CADNet 关注 B-Rep 的层次表示，用面邻接图和离散化网格之间的层次关系来学习加工特征，尤其针对相交特征。它的价值在于提醒你：真实 CAD/B-Rep 不是单一尺度结构。

对你而言，层次可以不是 mesh hierarchy，而是 CAE 任务层次：

```text
face
  -> feature instance
  -> CAE region
  -> action plan operation
```

这比单纯照搬它的 mesh hierarchy 更贴合你的工作。

---

## 6. MFTReNet 对你的启发

MFTReNet 在 AAGNet 基础上进一步做“特征之间拓扑关系识别”。它不仅识别每个特征，还预测特征之间的拓扑关系，并构建 MFTRCAD 数据集。

这对你的工作非常关键。因为 CAE 前处理不只需要知道“这里是孔/槽/肋”，还需要知道：

- 哪些孔属于同一孔阵列
- 哪个凸台与哪个孔关联
- 哪个肋支撑哪个壁/板
- 哪些圆角/倒角是应力集中或网格敏感区域
- 哪个载荷区域与哪个支撑区域形成分析工况

但你不必做完整的“加工特征拓扑关系识别”。你可以定义更小、更 CAE-oriented 的关系：

- `feature_adjacent_to_support`
- `feature_on_load_path`
- `hole_pattern_member`
- `mesh_sensitive_adjacent`
- `thin_wall_connected_to_pad`

这会成为你与 AAGNet 的差异点。

---

## 7. BrepMFR 和域适应工作的启发

BrepMFR 的关键词是 domain adaptation。它指出合成数据训练的模型迁移到真实 CAD 时会有性能下降，因此需要跨域适应。

这正好对应你的真实问题：你现在大量依赖程序化合成 STEP，但论文最终要说服读者它对真实 CAE 有用。

可实现的做法不是马上做复杂域适应，而是：

1. 先建立 20 个真实/近真实 STEP + Abaqus topology 小验证集。
2. 报告 synthetic test 和 real test 的性能落差。
3. 做最简单的增强：尺寸扰动、坐标旋转、特征组合扰动、STEP 导入导出扰动。
4. 如果时间允许，再做 feature normalization 或少量真实样本 fine-tuning。

这已经足够支撑“泛化性讨论”。

---

## 8. 这些文献对你当前工作的综合启发

### 8.1 你的工作不应追求“大而全”

这些顶会/顶刊论文并不是把 CAD/CAE 全自动化都做完了，而是抓住一个很窄但清楚的问题：

- UV-Net：如何从 B-Rep 学表示
- BRepNet：如何利用 coedge 拓扑做卷积
- AAGNet：如何基于 gAAG 做加工特征多任务识别
- Pointer-CAD：如何让命令序列能选择 B-Rep face/edge

所以你的论文也应该简化。不要声称“从 STEP 自动生成完整仿真”。更合适的窄问题是：

> 面向 CAE 前处理的 STEP/B-Rep 特征识别与仿真区域映射。

再具体一点：

> 从 STEP 模型中识别 face-level CAE-relevant features 和 feature objects，并将这些区域通过几何指纹映射到 Abaqus face regions，用于边界、载荷和网格操作的候选区域生成。

### 8.2 你的创新点应该放在哪里

不建议把创新点主要放在“又提出一个 GNN encoder”。这条线竞争太强，BRepNet、UV-Net、AAGNet、MFTReNet 都已经很成熟。

更适合你的创新点是：

1. CAE-oriented ontology：把传统加工特征重新组织成 CAE 前处理可用标签。
2. CAE region proposal：从 face/feature 识别进一步产生支撑、载荷、网格加密、壳化候选区域。
3. Software-agnostic fingerprint mapping：识别结果不是停留在 graph node，而是对齐到 Abaqus face id。
4. STEP-to-Abaqus minimal closed loop：形成真实仿真前处理闭环，而不是只做离线分类指标。

### 8.3 你的技术路线应该简化成三层

建议论文只保留三层主线：

```text
Layer 1: B-Rep graph recognition
识别 face label、feature instance、CAE tag

Layer 2: Task-conditioned region proposal
根据仿真需求选择 support/load/mesh/shell candidate

Layer 3: CAE software grounding
通过 fingerprint 将 graph feature object 映射到 Abaqus topology face id
```

不要把 LLM SFT、完整 Abaqus 自动仿真、复杂真实工况推理作为主贡献。它们可以放在扩展或未来工作。

---

## 9. 你还需要补哪些工作才像一篇合格论文

### 必须补

1. `edge_attr`
   - shared edge length
   - edge type
   - dihedral angle / convexity
   - adjacent face normal angle

2. instance affinity
   - 用 `instance_id` 生成 edge/pair label
   - 报告 edge-affinity F1 或 feature-instance IoU

3. topology/data validity audit
   - BRepCheck validity
   - closed/manifold 检查
   - 标签一致性检查
   - 至少导出可视化审计样例

4. CAE-oriented labels
   - 不要直接说真实 load/support 是几何推断
   - 应定义为 `candidate region`
   - 由 synthetic analysis case 或用户需求驱动

5. Abaqus real validation
   - 至少 20 个真实/近真实 STEP
   - 导出 Abaqus topology
   - 验证 mapping score、face IoU、region 是否为空、face id 是否复用

### 可以补但不宜作为第一优先级

1. UV grid
2. coedge/topological walk
3. LLM-based demand parsing
4. 完整 Abaqus 求解结果闭环
5. 复杂域适应模型

---

## 10. 建议实验设计

### 10.1 数据集实验

| 实验 | 目的 | 指标 |
| --- | --- | --- |
| Dataset coverage | 证明合成数据覆盖 CAE 特征类别 | 每类 instance count、face count、组合覆盖率 |
| Validity audit | 证明 STEP/B-Rep 合法 | valid rate、closed rate、manifold rate、label consistency |
| Random split | 标准识别能力 | accuracy、macro-F1、mIoU |
| Composition split | 未见组合泛化 | macro-F1、feature IoU、性能下降比例 |
| Real small set | 合成到真实迁移 | mapping score、人工通过率、失败类型统计 |

### 10.2 模型实验

| 模型 | 目的 |
| --- | --- |
| Node-MLP | 不用拓扑的下界 |
| GraphSAGE no edge_attr | 当前 baseline |
| GraphSAGE + edge_attr | 验证边属性贡献 |
| Multi-task + instance affinity | 验证实例分组贡献 |
| Rule-only region selection | 验证下游规则是否足够 |

### 10.3 指标

face-level：

- primitive surface accuracy
- macro feature accuracy / macro-F1
- semantic subtype accuracy / macro-F1
- CAE tag micro-F1 / macro-F1

instance-level：

- edge-affinity precision / recall / F1
- feature instance IoU
- feature-level recognition F1
- feature-level localization F1

downstream-level：

- graph feature object -> Abaqus face mapping mean score
- mapped face reuse count
- feature region IoU
- BC/load candidate IoU
- action plan success rate
- empty-region failure rate

ablation：

- no edge_attr
- no instance head
- no CAE tags
- no fingerprint matching, using index matching only
- no task condition, selecting regions only by largest face

---

## 11. 是否需要重新规划

需要小幅重新规划，但不是推翻已有工作。

原路线偏工程闭环：

```text
STEP -> graph -> GNN -> feature objects -> Abaqus mapping -> action plan
```

建议调整为论文路线：

```text
STEP -> CAE-AAG representation
     -> multi-task CAE feature recognition
     -> task-conditioned region proposal
     -> fingerprint-based Abaqus grounding
     -> evaluation on synthetic + real-small validation
```

保留已有代码，但把下一阶段优先级调整为：

1. 补 `edge_attr`
2. 补 `instance_affinity`
3. 补 `analysis_role / region proposal`
4. 做真实 Abaqus 小验证集
5. 做正式实验表格

暂缓：

- 大规模 LLM SFT
- 完整自然语言仿真需求解析
- 全自动真实载荷意图识别
- 复杂 coedge 网络

---

## 12. Grill-me 式关键问题

### 问题 1：你的论文到底是“特征识别论文”还是“CAE 自动前处理论文”？

推荐答案：是“以特征识别为基础的 CAE 自动前处理区域映射论文”。

也就是说，模型识别不是终点，Abaqus 区域映射才是你的差异点。但论文不能承诺完整仿真自动化，只承诺 support/load/mesh 等候选区域生成与软件拓扑对齐。

### 问题 2：你要不要像 AAGNet 一样追求 60k 数据集规模？

推荐答案：不需要。

你的目标不是和 AAGNet 在加工特征 benchmark 上拼 SOTA。你需要的是一个覆盖合理、标签可信、能支撑 CAE 映射的中等规模数据集。建议目标：

- 3000-10000 个合成样本
- 每类 CAE-relevant feature 至少 200-500 个实例
- 20-50 个真实/近真实 Abaqus 验证样本

### 问题 3：是否要简化功能？

推荐答案：要简化，而且要主动简化。

第一篇论文只做：

- 单零件
- 结构静力分析前处理
- face/feature recognition
- support/load/mesh/shell candidate region proposal
- Abaqus grounding

不要做：

- 装配接触
- 螺栓预紧
- 热边界
- 完整工况自动推理
- 自然语言到仿真脚本全自动生成

### 问题 4：仿真需求怎么进入系统？

推荐答案：先用结构化需求，不要先做自然语言。

例如：

```json
{
  "analysis_type": "static_structural",
  "support_policy": "largest_stock_planar_face",
  "load_policy": "feature_region",
  "target_feature_types": ["hole", "boss", "pad"],
  "mesh_refinement_policy": ["fillet", "chamfer", "thickness_transition"]
}
```

自然语言可以后续映射成这个 JSON，但不作为第一篇论文主线。

### 问题 5：你的最小创新闭环是什么？

推荐答案：

```text
CAE-AAG + feature instance recognition + task-conditioned CAE region proposal + Abaqus face grounding
```

这四件事连起来，就是你的论文工作量和创新点。
