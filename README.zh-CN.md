# TopoCAE-Agent：论文最小复现包

按2026年11月11日英文稿收束：仅保留论文中的四组识别消融、两种文献方法对比、13个工程案例、三组Qwen实验及其必要运行代码与证据。

## 复算论文表格

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python export_training_data.py
```

结果写入 `results/reproduced/`，已核对副本在 `results/paper/`。识别部分完整保留6种配置×3种子×1200件，即21,600份预测，不删除失败样本。表1、表2的数值与稿件一致；BiRel-FO的精确实例macro F1为92.79%，零件识别准确率为77.08%。

表4按照统一的工程验收标准复核原始39项任务：CAD检查正确对象上的编辑结果；CAE检查实际物理模型，求解完成状态单独保留。结果为 **4/13、13/13、11/13**。这三项计数已与更新稿一致；纯Plus在Bearing Housing中只选择孔壁也确实完成了四个目标盲孔的半径修改。这个规则同样适用于全部比较组。原先的完整面集合评分、求解超时以及全部42次API尝试均保留，未新增API调用或改写模型输出。

## 最小闭环

创建 `environment-inference.yml` 环境后执行：

```bash
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

从STEP重新识别，读取真实归档Qwen回答，再绑定操作对象，执行CAD编辑或生成Abaqus脚本。`replay_plan.py`可按三组对照的统一面选择接口重放原始回答，例如：

```bash
python replay_plan.py QP-B bearing_housing --output results/runs/plus_bearing
python replay_plan.py QP-S clevis --output results/runs/assisted_clevis
```

需要实际求解时提供 `--abaqus`；默认仅生成CAE脚本。`python fetch_assets.py`下载全部大文件，`python verify.py`校验完整包。Abaqus需自行安装并具备许可证。

## 图表与数据

`figures/`放作者最新8张TIFF，Fig. 3仅保留作者版本。`data/training_curves/fig3_data.csv`包含287行真实训练记录，另有6份原始JSON和种子对应表。导出脚本不会插值、补齐轮数或重新绘制另一张Fig. 3。

图与稿件仍有需要作者修改的标注，详见 [稿件核对清单](docs/MANUSCRIPT_ALIGNMENT.md)。原始实验数据不会为了匹配图中错误而修改。未写入当前稿件的补充试验和旧图已从当前发布树中移除；未重写Git历史。

该包验证的是冻结模型推理、工程操作及表格复算；不等同于已验证全部模型从零训练。仓库目前仍为私有，公开代码/数据许可证尚未由作者确定。参考 [README](README.md) 中的具体命令与证据范围。

最新固定版本为 `paper-v1.3-20261111`。Fig. 3(d)已对应33轮Q分支记录，Fig. 6(a)已使用R4/R3。仓库公开后可运行 `python fetch_assets.py --anonymous` 下载，无需GitHub密钥；当前私有访问仍需已获授权的凭据。
