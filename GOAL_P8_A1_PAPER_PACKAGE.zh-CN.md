# Goal P8-A1：论文初稿、图表与复现交付包

## 1. 目标

在实验完全冻结后，整理形成一篇范围诚实、证据闭合、可复现的论文初稿和补充材料。不得为美化结果重新训练、重跑 formal test 或改指标。

## 2. 论文结构

1. Introduction：CAE 区域 grounding 问题，而非泛化 CAD 识别口号；
2. Related Work：B-Rep recognition、CAD/LLM grounding、CAE agents；
3. Dataset：CAE-FeatureGraph、feature instances、face roles、query ground；
4. Method：recognizer、CAE-IR、Structured LLM、topology-constrained grounding、face mapper、Abaqus verifier；
5. Experiments：P1、P4、P5、P6、P7 分层和端到端；
6. Ablations：edge/topology、Direct vs Structured、verifier、Oracle vs predicted；
7. Failure/Limitations：controlled domain、slot/rib/transition、CAD import mapping、API reproducibility；
8. Conclusion。

## 3. 必须生成

- 中英文题目、摘要和贡献点；
- 方法总图、数据图、误差传播图；
- 主结果表、消融表、类别分层表、效率/费用表；
- 典型成功/失败案例；
- dataset/model/API/runtime cards；
- reproducibility checklist；
- 伦理、安全和 API 数据说明；
- 中文完整初稿和英文投稿初稿骨架；
- supplement 与 artifact README；
- 所有数字到机器结果文件的 provenance map。

## 4. 结论纪律

- 不保证或宣称 CCF-B/CCF-A 录用；
- 不把 Real-Dev 当未见外推；
- 不把 API 调用本身当创新；
- 不隐藏 predicted geometry 的性能下降；
- 不把 Abaqus job creation 等同于求解结果正确；
- 所有主张必须能指向冻结实验或人工审核证据。

完成后状态：

```text
MANUSCRIPT_PACKAGE_READY
```

