# Goal P6-A1：STEP-to-Abaqus Face Fingerprint Mapper

## 1. 目标

实现不依赖 face 列表顺序的 STEP/OCC face 到 Abaqus `Part.faces` 的稳定映射，使经过 grounding 的 exact STEP face set 可以安全转换为 Abaqus Set/Surface 所需面集合。

本 Goal 只做映射和高亮验证，不创建载荷/约束，不提交求解。

## 2. 环境

- Windows 为唯一权威执行环境；
- 普通 Python 负责 STEP/OCC fingerprint 和全局匹配；
- Abaqus noGUI 脚本负责导入 STEP、提取 Abaqus-side fingerprint 和生成高亮审查对象；
- 两侧通过版本化 JSON 交换，不共享第三方 Python 依赖。

## 3. 禁止事项

- 不假设 OCC face ID 等于 Abaqus face index；
- 不使用单一面积、数组顺序、part 名或人工特例映射；
- 不在歧义时选择“最接近的一个”继续执行；
- 不修改 P1-P5 冻结结果；
- 不访问 `test_part_s3`。

## 4. 实施任务

### A. Fingerprint schema

两侧尽可能提取：surface type、area、centroid、bbox、normal/axis、radius、cone angle、loop/edge 摘要、adjacency degree、邻接 surface/area/convexity 签名和二阶邻域摘要。所有容差相对 part scale 定义，不使用针对单个零件的绝对阈值。

### B. Matching

1. 建立 surface/type/topology compatibility mask；
2. 计算归一化多属性 cost；
3. 使用一对一全局 assignment；
4. 用邻接一致性和映射后的局部子图复核；
5. 输出 `UNIQUE`、`AMBIGUOUS`、`UNMATCHED`、`TOPOLOGY_CHANGED`；
6. 只允许 `UNIQUE` 进入后续流程。

### C. 回归验证

- 解析几何简单体；
- 当前特征类别的典型单特征和多特征件；
- `test_part` 至 `test_part_s2` 受控样本；
- 同一 STEP 重复导入；
- 旋转派生件；
- 人为制造等面积对称面歧义，验证 fail-closed；
- 导入后可能发生面合并/拆分的案例。

### D. H6-Mapping 人工包

为 10-20 个覆盖主要 surface/feature family 的样本创建 Abaqus 高亮 Set 和审查表。用户检查映射是否正确、歧义是否被拒绝。

## 5. 验收门槛

- 自动 fixture 的已知映射准确率 100%；
- 人工批准样本的目标面映射错误为 0；
- unsafe ambiguity acceptance 为 0；
- 重复导入 deterministic；
- 不允许用降低门槛换 coverage；
- 所有失败有稳定 reason code。

## 6. 交付与停止

交付 fingerprint schemas、双侧 extractor、matcher、Abaqus highlighter、测试集、H6 审查包、mapping metrics、acceptance、manifest 和 context。

完成后停止于：

```text
READY_FOR_ABAQUS_COMPILER
```

