# Linux 服务器迁移与新 Codex 接手说明

最后更新：2026-07-06
当前 Windows 工作目录：`W:\python\FEA\dataset_generator`
目标场景：将本项目迁移到 Linux 服务器，通过 Win10 VSCode SSH + Codex 继续开发、训练和评估。

---

## 1. 这个项目一句话是什么

本项目研究面向结构静力 CAE 前处理的 STEP/STP 单零件几何特征识别与区域选择。

核心目标不是做通用 CAD 理解，而是让系统能够从 STEP/STP 的 B-Rep 模型中识别对有限元前处理有用的可选区域，例如孔、槽、口袋、凸台、肋/腹板、圆角、倒角、沉孔、阶梯孔等，并最终服务于约束、载荷、理想化和 Abaqus 区域映射。

当前技术路线是：

```text
STEP/STP
  -> B-Rep 面邻接图
  -> face/node attribute + edge_attr
  -> edge-aware GNN 面级预测
  -> 候选对象提议 candidate proposal
  -> 候选级分类器 candidate classifier
  -> topology postprocess / role repair / semantic policy
  -> feature objects / selectable CAE regions
  -> 真实 STEP 面积真值表评估
```

---

## 2. 迁移到服务器时必须保留的文件

### 2.1 必须保留：代码与配置

这些是项目可以继续运行的最小主体：

```text
README.md
environment.yml
CODEX_CONTEXT.md
CODEX_CONTEXT_SUMMARY.md
PROJECT_AGENT_HANDOFF.md
PROJECT_AGENT_HANDOFF.zh-CN.md
PROJECT_SCRIPT_GUIDE.md
CAE_feature_ontology.yaml
face_semantic_subtype_ontology.yaml
engineering_semantics.txt

configs/
core/
features/
models/
schemas/
tests/
utils/
docs/

*.py              # 工作目录根目录下所有 Python 脚本
*.md              # 根目录下已有研究说明、计划、结果摘要
*.yaml
*.yml
*.txt
```

特别重要的计划文件：

```text
docs/superpowers/plans/2026-07-05-v5-21-reviewed-truth-and-semantic-policy-recall.md
docs/superpowers/plans/2026-07-05-v5-22-review-aware-residual-failure-atlas.md
docs/superpowers/plans/2026-07-06-v5-23-targeted-proposer-repair.md
docs/superpowers/plans/2026-07-06-v5-24-reason-aware-candidate-classifier-repair.md
```

其中 `2026-07-06-v5-24-reason-aware-candidate-classifier-repair.md` 是当前下一步最直接的执行入口。

### 2.2 必须保留：真实件与人工真值

这些是真实 STEP 泛化评估所需数据：

```text
test_part/
test_part_s/
test_part_s1/
test_part_s2/
test_part_s3/
test_part_s3_rib_colored/
```

其中最重要的是：

```text
test_part_s3/
test_part_s3/Feature_Comparison_Table_s3_mm2.xlsx
test_part_s3_rib_colored/
```

`test_part_s3` 是当前主要真实件评估集，`test_part_s3_rib_colored` 是用颜色标注 rib face truth 的补充数据。

### 2.3 建议保留：当前阶段关键输出

不要把所有 `output_*` 都无脑搬过去，历史输出非常多。为了延续当前 v5.21-v5.24 工作，建议至少保留这些：

```text
output_generation_v5_5_v51dist_looprole_5400/
output_generation_v5_10_instance_candidate_dataset_v55/
output_generation_v5_24_reason_aware_instance_candidate_dataset/

output_real_step_s3_v5_20_user_semantic_review/
output_real_step_s3_v5_21_reviewed_truth_semantic_policy/
output_real_step_s3_v5_22_review_aware_residual_atlas/
output_real_step_s3_v5_23_targeted_proposer_repair/
output_real_step_s3_v5_24_reason_aware_candidate_classifier/
```

原因：

- `output_generation_v5_5_v51dist_looprole_5400/`：v5.24 构建候选分类器数据集时需要用到其中的 synthetic graphs。
- `output_generation_v5_10_instance_candidate_dataset_v55/`：候选级分类器历史数据、模型和 raw predictions 的重要来源。
- `output_real_step_s3_v5_20...v5_24...`：当前真实件评估链路、用户语义审查、review-aware residual atlas 和阶段对比报告。

如果磁盘空间紧张，优先保留这些目录下的：

```text
graphs/
reports/
model_runs/
instance_candidate_dataset/
predictions_raw/
*.json
*.md
*.joblib
```

### 2.4 可以暂时不搬：历史大输出

这些目录主要是早期 v1-v4 或中间试验结果。除非要复现实验历史，一般不需要第一批迁移：

```text
output_experiment_v1_*
output_generation_v2_*
output_generation_v3_*
output_generation_v4_*
output_ablation_*
output_real_step_s*_v3_*
output_real_step_s*_v4_*
output_train_smoke_v1
output_smoke_*
__pycache__/
*.log
```

注意：不要搬 Windows 虚拟环境：

```text
W:\python\envs\feagpt
```

Linux 服务器上应重新建环境。

---

## 3. 推荐打包方式

### 3.1 最稳妥方式：先复制一个精简包，再传服务器

建议在 Windows 上先建立一个迁移目录，例如：

```powershell
$src = "W:\python\FEA\dataset_generator"
$dst = "W:\python\FEA\dataset_generator_server_pack"
```

复制时保留：

```text
configs
core
features
models
schemas
tests
utils
docs
test_part
test_part_s
test_part_s1
test_part_s2
test_part_s3
test_part_s3_rib_colored
output_generation_v5_5_v51dist_looprole_5400
output_generation_v5_10_instance_candidate_dataset_v55
output_generation_v5_24_reason_aware_instance_candidate_dataset
output_real_step_s3_v5_20_user_semantic_review
output_real_step_s3_v5_21_reviewed_truth_semantic_policy
output_real_step_s3_v5_22_review_aware_residual_atlas
output_real_step_s3_v5_23_targeted_proposer_repair
output_real_step_s3_v5_24_reason_aware_candidate_classifier
```

再复制根目录文件：

```text
*.py
*.md
*.yaml
*.yml
*.txt
*.code-workspace
```

压缩后上传到服务器目录，例如：

```text
/home/<user>/projects/FEA/dataset_generator
```

### 3.2 如果使用 rsync / scp

可以从 WSL、Git Bash 或支持 rsync 的终端上传。建议排除历史大目录和缓存：

```bash
rsync -av --progress /mnt/w/python/FEA/dataset_generator/ <user>@<server>:/home/<user>/projects/FEA/dataset_generator/ \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.log' \
  --exclude 'output_experiment_v1_*/' \
  --exclude 'output_generation_v2_*/' \
  --exclude 'output_generation_v3_*/' \
  --exclude 'output_generation_v4_*/' \
  --exclude 'output_ablation_*/'
```

如果不确定排除是否过多，优先采用 3.1 的手动精简包方式。

---

## 4. Linux 服务器环境建议

### 4.1 不要直接复制 Windows venv

Windows 的：

```text
W:\python\envs\feagpt
```

不能直接用于 Linux。服务器上应重新创建环境。

### 4.2 基础环境

当前 `environment.yml` 是早期 CPU 环境：

```yaml
name: aagnet
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pythonocc-core=7.9.3
  - cpuonly
  - pytorch
  - pyyaml
  - pip
```

服务器有 NVIDIA A40 时，建议新建 GPU 环境，不要保留 `cpuonly`。示例：

```bash
conda create -n feagpt python=3.10 -y
conda activate feagpt
conda install -c conda-forge pythonocc-core=7.9.3 pyyaml scikit-learn joblib numpy pandas openpyxl tqdm networkx -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果服务器 CUDA 版本不同，需要让新 Codex 先执行：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

然后再确定 PyTorch CUDA 版本。

### 4.3 迁移后的路径调整

Windows 上历史命令使用：

```text
W:\python\envs\feagpt\python.exe
W:\python\FEA\dataset_generator
```

Linux 上应改为：

```text
python
/home/<user>/projects/FEA/dataset_generator
```

多数脚本应尽量用相对路径运行。旧 JSON/MD 里可能包含 Windows 绝对路径，这些通常只是历史元数据；如果脚本真正依赖绝对路径，新 Codex 需要改为参数传入或相对路径。

---

## 5. 当前研究进度

### 5.1 已形成的共识

第一篇论文目标：

```text
单零件结构静力 CAE 前处理中的几何特征识别与区域选择
```

当前目标类别包括：

```text
through_hole_round
blind_hole_round
stepped_hole_round
counterbore_hole
countersink_hole
obround_hole
pocket_rect
pocket_circular
slot_rect
slot_u
slot_v
slot_o
cylinder_boss
rectangular_boss
rib / rib_web_candidate
fillet
chamfer
```

重要语义边界：

- `counterbore_hole`：小孔贯通，没有底面。
- `stepped_hole_round`：同轴多半径盲孔，通常有小孔底面。
- `countersink_hole`：通孔加圆周锥面；其锥面不等同于普通狭长倒角。
- `fillet/chamfer` 在第一篇论文中作为独立目标输出。
- rib 带倒角时，倒角可作为 rib 的附属 transition，不一定独立输出。
- pocket 内长 rib 时，pocket 仍保留独立目标。
- 法兰、导轨、安装板等不作为第一篇论文目标类别，它们只是毛坯/上下文变化。

### 5.2 当前瓶颈

生成数据内部指标较高，但真实 STEP 泛化仍受限。主要问题不是单一模型分数，而是：

- synthetic-to-real gap；
- 候选对象边界不稳；
- rib / stock / pocket 混淆；
- fillet / slot_o / slot_u 等拓扑相近类别混淆；
- through hole / outer cylinder / inner cylinder / boss 的功能角色混淆；
- proposal recall 提升后，新候选没有稳定通过 candidate classifier。

### 5.3 最近版本进展

v5.20 用户语义审查：

- 用户填写了 `output_real_step_s3_v5_20_user_semantic_review/reports/semantic_review_template.csv`。
- 审查决策包括 `keep_strict`、`truth_label_should_change`、`accept_family_match`。
- strict macro metric 仍为主指标；hole-family match 只作为辅助语义指标。

v5.21：

- 更新真值到 `configs/test_part_s3_area_truth_v5_21_reviewed.json`。
- strict recall：约 `106/204 = 0.5196`。
- semantic-policy recall：约 `108/204 = 0.5294`。

v5.22：

- 引入 review-aware residual failure atlas。
- residual 重点集中在 rib、fillet、through_hole_round、chamfer、counterbore_hole。

v5.23：

- 增加 rib/fillet/chamfer targeted topology proposer。
- proposal recall 明显提升，但 final-survival recall 基本没变。
- 结论：瓶颈从“候选没提出”转移到“候选没被分类器接受或没进入最终对象”。

v5.24 当前计划：

- 做 reason-aware candidate classifier repair。
- 让候选分类器看到 candidate 的 assembly reason / source reason。
- 目标是让 v5.23 新提出的 topology-rule 候选能通过分类器和最终后处理。

---

## 6. 新服务器上的第一步检查

进入项目目录后，先不要改代码。先做：

```bash
pwd
find . -maxdepth 2 -type f | sed 's#^\./##' | sort | head -200
python --version
python - <<'PY'
import sys
print(sys.executable)
for name in ["OCC", "torch", "sklearn", "joblib", "yaml", "numpy"]:
    try:
        __import__(name)
        print(name, "OK")
    except Exception as e:
        print(name, "FAIL", repr(e))
PY
```

再运行最小测试：

```bash
python -m unittest \
  tests.test_topology_postprocess_v5_23 \
  tests.test_object_candidate_proposer_v5_23 \
  tests.test_compare_review_aware_residual_atlases \
  tests.test_object_candidate_proposer \
  tests.test_topology_postprocess_v5_8 \
  tests.test_review_aware_residual_failure_atlas -v
```

如果这些测试因为 Windows/Linux 路径分隔符失败，优先修路径兼容性，不要先改算法。

---

## 7. 新 Codex 应优先阅读的文件

让新 Codex 按下面顺序读：

```text
SERVER_CODEX_HANDOFF.md
README.md
CODEX_CONTEXT_SUMMARY.md
PROJECT_SCRIPT_GUIDE.md
docs/superpowers/plans/2026-07-06-v5-24-reason-aware-candidate-classifier-repair.md
utils/object_candidate_proposer.py
utils/topology_postprocess.py
utils/instance_candidate_features.py
build_instance_candidate_dataset.py
train_candidate_topology_classifier.py
object_proposal_recall_report.py
review_aware_residual_failure_atlas.py
```

如果需要更完整历史，再读：

```text
CODEX_CONTEXT.md
PROJECT_AGENT_HANDOFF.md
docs/superpowers/plans/2026-07-06-v5-23-targeted-proposer-repair.md
docs/superpowers/plans/2026-07-05-v5-22-review-aware-residual-failure-atlas.md
```

---

## 8. 给新 Codex 的启动提示词

可以直接复制下面这段给服务器上的新 Codex：

```text
你现在接手的是一个面向结构静力 CAE 前处理的 STEP/STP B-Rep 几何特征识别项目。请先阅读当前目录下的 SERVER_CODEX_HANDOFF.md、README.md、CODEX_CONTEXT_SUMMARY.md、PROJECT_SCRIPT_GUIDE.md，以及 docs/superpowers/plans/2026-07-06-v5-24-reason-aware-candidate-classifier-repair.md。

请不要一上来重构项目。先完成三件事：
1. 检查 Linux 环境依赖是否齐全，尤其是 pythonocc-core/OCC、torch、scikit-learn、joblib、pyyaml、numpy。
2. 运行 v5.23/v5.24 相关的最小回归测试，确认迁移后路径和基础逻辑没有坏。
3. 按 v5.24 计划执行 reason-aware candidate classifier repair：添加 assembly-reason/source-reason 特征，重建候选级数据集，训练候选分类器，在 test_part_s3 上生成和 v5.23 对比的 recall-stage/report/failure-atlas。

工作方式：
- 修改前先读相关代码，不要凭文件名猜测。
- 所有新增计划、执行结果和报告都写到 docs/superpowers/plans 或对应 output_* / reports 目录。
- 运行命令用 Linux 相对路径，不要继续使用 W:\ Windows 绝对路径。
- 不要删除历史输出。
- 如果指标没有改善，要说明瓶颈在 proposal、classifier、postprocess、truth policy 还是真实件数据分布，不要只继续调阈值。

最后请给出：
- 修改文件清单；
- 运行过的测试命令和结果；
- 新报告路径；
- v5.24 相比 v5.23 是否真正改善；
- 下一步建议。
```

---

## 9. 延续当前协作模式的提示词

如果用户希望新 Codex 继续采用“一个 agent 执行、当前 Codex 审查和写下一步计划”的模式，可以这样提示：

```text
接下来请延续原项目的工作模式：你不是只写代码，而是作为研究协作 agent。

每一轮请按以下流程工作：
1. 先审查上一轮 agent 的修改、报告、测试结果和建议，判断是否真的满足上一份计划。
2. 判断该轮是否带来实质性进展，尤其要区分：
   - 训练集内部指标提升；
   - 真实 STEP 泛化提升；
   - proposal recall 提升；
   - candidate classifier 通过率提升；
   - topology postprocess 后最终对象提升。
3. 如果需要继续实现，请写一份很详细的 Markdown 计划，放到 docs/superpowers/plans/，计划要让另一个 agent 能照着执行。
4. 计划中必须包含：
   - 目标；
   - 不做什么；
   - 需要修改的文件；
   - 需要新建的文件；
   - 具体实现步骤；
   - 需要运行的命令；
   - 验收标准；
   - 失败时如何判断下一步；
   - 给执行 agent 的提示词。
5. 如果需要用户介入，请明确告诉用户要做什么，例如：
   - 是否需要新增真实 STEP；
   - 是否需要补面积真值表；
   - 是否需要用颜色标注某类 face truth；
   - 是否需要抽查生成 STEP 几何合理性。

注意：不要绕着一个小阈值无限调参。每轮都要从论文目标、数据真实性、拓扑表达、候选边界、真实泛化这几个层面判断是否还值得继续修。
```

---

## 10. 用户在服务器阶段需要做什么

短期内用户不需要立即手工画大量新零件。优先让 Codex 完成：

1. Linux 环境迁移检查；
2. v5.23 回归测试；
3. v5.24 reason-aware candidate classifier；
4. test_part_s3 对比报告；
5. 判断当前错误是否仍集中在 rib/transition/hole-role。

只有在以下情况才需要用户介入：

- 真实件剩余错误无法从面积真值表判断，需要用户确认语义边界；
- 某一类真实 STEP 明显缺少训练覆盖，需要用户补 3-10 个代表性简化真实件；
- rib、slot、transition 等对象边界必须用颜色或面积表补 face-level truth；
- 需要决定论文中某类特征是否作为独立目标，还是作为附属 transition/context。

---

## 11. 服务器上推荐的下一步任务

当前最直接任务是执行 v5.24：

```text
docs/superpowers/plans/2026-07-06-v5-24-reason-aware-candidate-classifier-repair.md
```

如果 v5.24 后：

- final-survival recall 提升明显：继续 candidate-level topology/classifier 路线；
- proposal recall 高但 final 仍不动：检查 postprocess/final feature object assembly；
- classifier 指标提升但真实 s3 不提升：停止分类器微调，转向真实件 hard cases 和 domain adaptation；
- rib 仍是最大残差：继续基于 colored rib truth 做 rib candidate / boundary 的上游修复；
- through-hole / outer-cylinder / boss 仍混淆：继续 loop-role edge_attr 和 cylinder role repair。

---

## 12. 论文角度的当前结论

当前项目已经有一个可以写成论文的主线：

```text
面向 CAE 前处理的 B-Rep 图特征识别：
从合成 STEP 数据生成，到方向不敏感图表示，到 edge-aware GNN，
再到候选对象级拓扑修复和真实 STEP 泛化评估。
```

但论文不能只报告生成数据集上的高分。必须重点展示：

- synthetic-to-real gap；
- 为什么单纯 face-level GNN 不够；
- 为什么需要 edge_attr、loop role、candidate object assembly 和 topology-aware repair；
- 用户面积真值表和颜色真值如何作为少量真实件监督；
- 最终系统对 CAE selectable region 的帮助，而不仅是传统加工特征识别。

---

## 13. 最后提醒

迁移到服务器后，最容易踩的坑是：

- 把 Windows venv 复制过去；
- 让脚本继续依赖 `W:\...`；
- 把所有旧输出都搬过去导致服务器目录巨大；
- 没有先跑回归测试就开始改算法；
- 只看训练集指标，不看真实 STEP；
- 只调阈值，不分析 proposal/classifier/postprocess 哪个阶段失败。

正确的接手方式是：

```text
先复现当前 v5.23/v5.24 基线
-> 再做 v5.24 reason-aware candidate classifier
-> 再用 test_part_s3 的 review-aware reports 判断瓶颈
-> 再决定是否继续模型、数据或规则路线
```
