# Paper results and source files

| Paper item | Source files | Command |
| --- | --- | --- |
| STEP-FO18 and dataset splits | `data/d0/split_and_labels.json`, training/validation STEP archive, `data/t10/` | `python fetch_assets.py --group training`; `python fetch_assets.py --group test` |
| BiRel-FO recognition | `comparison/topocae/` and deployment checkpoints | `python infer.py part.step --output prediction.json` |
| Fig. 3 | `figures/fig3.tif`, six JSON histories, `data/training_curves/fig3_data.csv` | `python export_training_data.py` |
| Table 1: component ablations | R-only, Q-only, rule-scored and BiRel-FO predictions | `python reproduce.py --only recognition` → `TABLE_1.csv` |
| Table 2: method comparison | BiRel-FO, AAGNet18-SI and MFTReNet-SI-D0 predictions on I/P/S subsets | `python reproduce.py --only recognition` → `TABLE_2.csv` |
| Table 3: engineering applications | `data/engineering/`, object-based plans in `data/language/` | `python reproduce.py --only historical`; `run_case.py`; `read_odb.py` |
| Table 4: language-model comparison | `data/qwen_unified/`, original responses, timing, usage and engineering outputs | `python reproduce.py --only unified` → `TABLE_4.csv` |
| Figs. 1–2 and 4–8 | TIFF files in `figures/` | Dataset, method and application illustrations |

Recognition tables report the mean and sample standard deviation across three seeds. All methods use 1,200 test parts: 400 same-strategy (I), 400 parameter-variation (P) and 400 structural-combination (S) parts. Exact-instance matching evaluates both category and full member-face membership.

Table 3 evaluates the 13 object-based engineering applications. Table 4 compares three language-model configurations through a shared direct-face plan interface, with 13 tasks per configuration. The common engineering criterion yields 4/13, 13/13 and 11/13 successful tasks. These two experiments have their own plans and records; their settings and metrics are described in [ENGINEERING_ACCEPTANCE.md](ENGINEERING_ACCEPTANCE.md).

The ablation table evaluates the contribution of the R/Q branches and the combined learned object-decision mechanism relative to rule-based scoring. Individual seed results and per-part predictions are provided alongside the aggregate tables.
