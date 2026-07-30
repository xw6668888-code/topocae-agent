# 单零件结构静力 CAE 特征范围设计

## 目标

本阶段项目聚焦于“单零件结构静力分析前处理”。数据集和识别流程应从 STEP/B-Rep 模型中识别局部、可查询的面特征，并进一步支持有限元前处理中的边界条件、载荷、网格准备和几何理想化区域选择，最终将选中的面区域映射为 Abaqus 可执行的 set、surface 或 action plan。

本阶段的重点不是识别所有 CAD/机械特征，而是识别那些在结构静力工况中能够被明确描述、明确选面、明确落到 Abaqus 操作上的局部几何特征。

## 研究范围

V1 特征范围刻意保持局部化和可查询性。当前默认数据集包括：

- 圆孔类特征
- 长圆孔
- 槽
- 口袋/凹槽
- 简单凸台
- 肋/腹板
- 圆角
- 倒角
- 由毛坯面派生出的外部面角色

V1 不包含零件级或装配级语义，例如：

- 法兰
- 孔组/螺栓孔组作为训练类别
- 轴承装配语义
- 接触
- 螺栓预紧
- 完整自然语言仿真意图理解

长圆孔保留在孔类中，因为它是工程中常见的孔型特征。孔组不作为宏特征训练；它应当在单个孔实例识别完成后，根据孔径、轴线、中心距、共面关系和排列规律进行后处理聚类得到。

## V1 核心宏特征

V1 默认生成器应产生以下规范化宏特征：

- `through_hole_round`：圆形通孔
- `blind_hole_round`：圆形盲孔
- `stepped_hole_round`：圆形阶梯孔
- `counterbore_hole`：沉孔/平底扩孔
- `countersink_hole`：锥沉孔
- `obround_hole`：长圆孔
- `slot_u`：U 形槽
- `slot_rect`：矩形槽
- `pocket_rect`：矩形口袋/矩形凹槽
- `pocket_circular`：圆形口袋/圆形凹槽
- `cylinder_boss`：圆柱凸台
- `rectangular_boss`：矩形凸台
- `rib`：肋
- `web`：腹板
- `fillet`：圆角
- `chamfer`：倒角

以下 CAE 前处理标签与宏几何特征分离，作为派生标签或理想化标签存在：

- `thin_wall_region`
- `thickness_transition`
- `midsurface_candidate`
- `mesh_sensitive_region`
- `shell_candidate`

这样可以避免把“几何类别”和“分析工况角色”混在同一个标签头中。

## 排除或降级的特征

以下标签可保留在 ontology 中以兼容旧代码和历史数据，但不启用于 V1 论文默认数据集：

- `bolt_hole_pattern`：孔组应由多个单孔对象聚类得到，不作为训练宏特征。
- `flange`：过于接近零件级或连接结构语义，不适合作为当前局部特征识别的核心类别。
- `support_pad`：合并到 `rectangular_boss`；支撑或加载含义由具体分析工况决定。
- `pad`：规范化为 `rectangular_boss`，保留为代码别名。
- `cone_boss`、多边形/棱台凸台、三角/矩形切口、`notch_relief`、`bearing_seat`、`gusset`：作为 V2 可选内容，仅在具有明确 CAE 查询场景和稳定生成器时再考虑。

## 可选面区域

模型输出不应只停留在“这是某类宏特征”。每个 feature object 还应暴露可供结构化仿真需求选择的面角色，即 selectable face regions。

典型例子：

- 盲孔：圆柱孔壁、孔底面。
- 沉孔：大圆柱面、台阶平面、小圆柱面。
- 长圆孔：两侧平面、两端圆柱面。
- 矩形凸台：顶面、侧面。
- 口袋/凹槽：底面、侧面。
- 肋/腹板：侧面、顶面、薄壁候选区域。
- 毛坯：顶部外平面、底部外平面、侧向外平面、最大外部平面。

这样可以支持类似以下查询：

```text
固定直径为 5 的盲孔圆柱面
在圆柱凸台顶面施加压力
固定最大底部平面
对圆角区域进行网格加密
将肋/腹板区域标记为 shell candidate
```

## 实施策略

保留现有生成器模块，避免丢失已有工作。V1 的收敛方式不是删除旧代码，而是通过默认配置、coverage target、本体 enabled 标志和别名映射来限制默认数据集范围。

第一轮实施应完成：

1. 将 `pad` 输出规范化为 `rectangular_boss`。
2. 保持 `obround_hole` 启用，并归入孔类。
3. 在默认 V1 coverage 中禁用零件级和非核心宏标签。
4. 将默认生成器限制为 V1 核心集合。
5. 让 `BossFeatureGenerator` 默认只生成圆柱凸台，保证 V1 数据稳定。
6. 更新派生 CAE 标签逻辑，使薄壁逻辑只依赖 `rib/web`，厚度过渡逻辑支持 `rectangular_boss`。
7. 添加轻量回归测试，验证 ontology 和默认配置没有偏离 V1 范围。

## 成功标准

- 默认数据集不再包含 `flange`、`bolt_hole_pattern`、`support_pad`、`bearing_seat`、`gusset`、`notch_relief`、多边形切口或非核心多边形/棱台凸台。
- `pad` 生成器输出能够解析为 `rectangular_boss`。
- `obround_hole` 仍保持启用，并属于孔类。
- coverage targets 只包含 V1 核心启用特征。
- `rib/web` 和 `rectangular_boss` 相关区域仍能生成薄壁、厚度过渡、网格敏感或壳候选等派生标签。
- 在 `feagpt` 环境中能正常导入 OCC 并完成最小数据集生成与校验。
