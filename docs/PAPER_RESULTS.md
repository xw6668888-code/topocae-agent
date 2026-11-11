# Paper-to-artifact mapping

| Paper item | Sources | Reproduction |
| --- | --- | --- |
| STEP-FO18 data and splits | `data/d0/split_and_labels.json`, train/validation STEP archive, `data/t10/` | `fetch_assets.py --group training`, `--group test`; `verify.py` |
| BiRel-FO method | `comparison/topocae/` and frozen weights | `infer.py` |
| Fig. 3 | `figures/fig3.tif`, six histories, `data/training_curves/fig3_data.csv` | `export_training_data.py` |
| Table 1 | R, Q, rule and BiRel-FO predictions for all three seeds | `reproduce.py --only recognition`; `TABLE_1.csv` |
| Table 2 | BiRel-FO, AAGNet18-SI, MFTReNet-SI-D0 on all I/P/S parts | `reproduce.py --only recognition`; `TABLE_2.csv` |
| Table 3 / 13 application cases | `data/engineering/`, assisted responses in `data/language/` | `run_case.py`, `reproduce.py --only historical`, `read_odb.py` |
| Table 4 | `data/qwen_unified/`, raw responses, original timing/usage, actual CAD results and model readbacks | `reproduce.py --only unified`; `TABLE_4.csv` |
| Figs. 1–2, 4–8 | author TIFF exports in `figures/` | Design illustrations and case visualization; see figure notes |

Recognition tables report percentages, with sample standard deviation over three seed runs. No best-seed selection is used for those tables. The 1,200 test parts and all relevant failed predictions remain in every method's denominator.

Six model configurations remain in the paper package: R, Q, rule-scored, BiRel-FO, AAGNet18-SI and MFTReNet-SI-D0. An unpublished learned-score-only supplementary comparison and geometric-perturbation study are outside this manuscript's release scope. This packaging choice does not establish a separate causal benefit for empirical calibration; the current four-row ablation supports the combined learned object decision system against rule scoring.

The historical Table 3 demonstration and the Table 4 common direct-face experiment have different interfaces and are not merged. Table 4 is now independently evaluated from its own original responses under the author's common functional engineering criterion. See `ENGINEERING_ACCEPTANCE.md` for the 4/13–13/13–11/13 counts and the matching counts in the 11 November manuscript.

Internal IDs such as E3, R and legacy module names identify the frozen executable system. Active imported files are retained for reproduction; they are not experiment queue logs.
