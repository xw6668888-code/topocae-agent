# BiRel-FO：面向CAD编辑与CAE建模的工程几何特征对象识别

[English](README.md) | [数据与模型](docs/DATA.md) | [复现指南](docs/REPRODUCTION.md) | [论文结果](docs/PAPER_RESULTS.md)

本仓库提供论文 *BiRel-FO: Engineering geometric feature object recognition for CAD editing and CAE modeling* 对应的BiRel-FO实现、STEP-FO18数据集、预训练模型及实验材料。

BiRel-FO从STEP B-Rep几何中提取属性图，结合稀疏与稠密同实例关系、对象评分及冲突选择，识别特征类别和完整成员面。识别对象进一步与操作面角色和实测几何参数关联，用于自然语言驱动的CAD编辑与CAE模型准备。

## 主要结果

- 在1,200件测试零件上，三个随机种子的平均精确实例macro F1为92.79%，零件识别准确率为77.08%。
- 在13件工程零件上识别182个特征实例，完成6项CAD编辑和7项CAE建模与分析案例。
- 在统一工程任务评价下，Qwen3.6-Plus、Qwen3.6-Plus + BiRel-FO、Qwen3.8-Max-0902分别完成4/13、13/13和11/13项任务。评价口径见[工程任务与API统计](docs/ENGINEERING_ACCEPTANCE.md)。

仓库包含对应的模型实现、数据划分与标注、权重、训练日志、预测记录、语言模型响应及工程输出。

## 安装

```bash
git clone https://github.com/xw6668888-code/topocae-agent.git
cd topocae-agent
conda env create -f environment-inference.yml
conda activate topocae-inference
```

该环境用于特征识别及CAD操作，识别入口支持CPU。CAE求解和ODB读取使用单独安装并已授权的Abaqus。

## 快速运行

```bash
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

程序从STEP执行识别，将已记录的Qwen计划绑定到识别对象，随后执行CAD编辑或生成Abaqus脚本。示例复用原始语言模型响应，因此无需API密钥。部署权重总计约2.84 MB。

识别其他STEP零件：

```bash
python infer.py part.step --output results/runs/prediction.json
```

输出包括特征类别、成员面、操作角色、几何参数及STEP标识。CAE提交计算和三组语言模型计划重放的用法见[复现指南](docs/REPRODUCTION.md)。

## 复现论文表格与曲线数据

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python export_training_data.py
```

识别评价覆盖6种配置、3个种子和1,200件零件，共21,600份预测。结果写入 `results/reproduced/`，论文表格位于 `results/paper/`。Fig. 3使用六份训练日志，CSV保留287行实际训练与验证记录。

## 下载完整数据

```bash
python fetch_assets.py
python verify.py
```

大文件从GitHub Releases下载，`assets.json`记录其来源、大小及SHA-256。可使用 `--group training`、`--group test`、`--group engineering` 或 `--group models` 选择内容；`--anonymous`用于不读取任何凭据的公开下载。

固定论文版本：[paper-v1.4-20261111](https://github.com/xw6668888-code/topocae-agent/releases/tag/paper-v1.4-20261111)。数据划分、目录结构、评价方法及第三方软件信息见[英文说明](README.md)和 `docs/`。
