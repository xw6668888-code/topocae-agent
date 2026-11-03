# TopoCAE-Agent 论文最小复现包

本项目对应 STEP-FO18 数据集和 BiRel-FO 识别方法，保留论文主线所需的代码、冻结模型、逐例预测、标注、真实语言模型响应及 CAD/CAE 工件。研究开发目录和废弃实验队列不进入当前发布版本。

## 最小闭环

```bash
conda env create -f environment-inference.yml
conda activate topocae-inference
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

流程为：STEP → 新执行的 BiRel-FO 识别 → 解析真实存档 Qwen 响应 → 特征对象与操作区域绑定 → CAD 编辑或 Abaqus 脚本生成。不是直接复制已有编辑结果，也不以人工参考计划代替模型响应。无需再次支付 API 费用。

Bearing Housing 会实际修改盲孔半径并回读 STEP。Angle Bracket 默认生成完整 Abaqus 脚本；提供 `--abaqus "实际的abaqus启动路径"` 后会提交求解。可以对 `data/engineering/cases.json` 内其余案例使用同一入口。求解需自行安装并获得 Abaqus 许可。

## 论文表格与图像

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python plot_training.py
```

识别表格从 1,200 件测试零件的逐例预测重新计算，包含三种子及 I/P/S 子集；语言模型表格从真实原始响应重新解析、定位评价并汇总耗时和 token。训练曲线由实际记录重绘，保留各次运行的真实轮数。

全部大文件通过 `python fetch_assets.py` 下载，总计约 1.61 GB；`python verify.py` 校验完整性。使用 `--profile demo` 仅下载最小演示需要的约 2.84 MB 权重。完整数据含 8,370 件训练/验证 STEP、1,200 件测试 STEP、冻结权重及工程求解工件。

## 结果使用边界

- 历史对象绑定流程：Qwen 辅助组为 13/13 项定位正确，并有 6 项 CAD、7 项 CAE 的核验工件；其中包含等价命令的既有工件复用。
- 统一面组输出实验：Qwen3.6-Plus、Qwen3.6-Plus＋BiRel-FO、Qwen3.8-Max-0902 分别为 3/13、12/13、10/13 定位正确，任务通过为 3/13、11/13、10/13。
- 两种协议的结果分别保存，不能将历史 13/13 填入统一比较表。均值、中位数、总费用、平均费用也分别输出。
- 本包验证冻结模型推理、实际 CAD 编辑、CAE 脚本生成、已存 ODB 读取及表格复算；不把这些检查表述为重新完成全部训练或全部求解。

详细文件对应及初稿数值核对见 [PAPER_RESULTS.md](docs/PAPER_RESULTS.md)。仓库仍为私有；上游许可声明已保留，作者代码尚未指定面向公众的统一许可证。原研究目录和 Git 历史未删除。
