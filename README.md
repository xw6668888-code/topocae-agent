# TopoCAE-Agent: BiRel-FO and STEP-FO18

Paper artifact for engineering feature-object recognition from STEP B-Rep, CAD editing and CAE modeling. This checkout contains the experiments in the author's 10 November 2026 manuscript: four recognition ablations, two literature baselines, 13 engineering cases, and three Qwen configurations.

## Reproduce the tables

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python export_training_data.py
```

The recount checks **21,600 individual predictions** (six configurations × three seeds × 1,200 parts), all 18 classes, I/P/S subsets, and the paired bootstrap comparisons. Tables 1–2 reproduce the manuscript's 92.79% exact-instance macro F1 and 77.08% whole-part accuracy. Table 3 is supported by the 13 object-bound engineering demonstrations. Outputs are written to `results/reproduced/`; checked copies are in `results/paper/`.

For Table 4, the same engineering acceptance criterion is applied to all 39 archived tasks: correct edited geometry for CAD, and a correctly constructed physical model for CAE. The resulting counts are **4/13, 13/13 and 11/13** for Qwen3.6-Plus, Plus + BiRel-FO and Qwen3.8-Max-0902. The manuscript's first count (3/13) needs correction: Plus correctly edits all four Bearing Housing holes using their cylindrical walls. The original complete-member-face and solver-completion scores remain in the per-task records. See [acceptance and costs](docs/ENGINEERING_ACCEPTANCE.md) and [manuscript alignment](docs/MANUSCRIPT_ALIGNMENT.md).

## Run the smallest closed loop

```bash
git clone https://github.com/xw6668888-code/topocae-agent.git
cd topocae-agent
conda env create -f environment-inference.yml
conda activate topocae-inference
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

`run_case.py` performs fresh STEP recognition, reads an actual archived assisted Qwen response, binds its targets to recognized objects, and executes CAD editing or generates an Abaqus script. It makes no new API call. The demo needs about 2.84 MB of deployment weights. For a new STEP file, use `python infer.py part.step --output results/runs/prediction.json`.

The three-condition comparison uses a separate, common direct-face plan interface. Replay any original model response with:

```bash
python replay_plan.py QP-B bearing_housing --output results/runs/plus_bearing
python replay_plan.py QP-S clevis --output results/runs/assisted_clevis
```

Conditions are `QP-B` (Plus), `QP-S` (Plus + BiRel-FO), and `QM-B` (Max). Non-executable responses remain non-executable; the replay does not repair model plans. To submit a generated CAE model, add `--abaqus /path/to/abaqus` and choose available CPU/memory resources. A licensed Abaqus installation is required. Solver completion is reported separately from modeling correctness.

## Data, weights and solver outputs

```bash
python fetch_assets.py
python verify.py
python replay_cad.py --output results/runs/reference_cad
abaqus python read_odb.py --output results/runs/odb_readback.json
```

Large immutable files are downloaded from the releases identified in `assets.json`, with size and SHA-256 verification. Private-repository access uses a local Git credential manager or `GH_TOKEN`/`GITHUB_TOKEN`; credentials are never stored here. Existing data assets are reused without re-encoding them.

| Location | Paper content |
| --- | --- |
| `comparison/topocae/` | Frozen R/Q networks, feature recovery, scorer, calibration and object selection |
| `comparison/ablations/` | R-only, Q-only, rule-scored and complete BiRel-FO configurations |
| `comparison/aagnet/`, `comparison/mft/` | AAGNet18-SI and MFTReNet-SI-D0 adaptations and checkpoints |
| `data/d0/`, `data/t10/` | 7,440 training, 930 validation and 1,200 test STEP parts and labels |
| `data/engineering/` | 13 engineering inputs, annotations, predictions, six edited STEPs and seven archived solver results |
| `data/language/` | Assisted object-binding responses supporting Table 3 and the minimum replay |
| `data/qwen_unified/` | All three Table 4 groups, 39 tasks, 42 API attempts, execution and readback evidence |
| `direct_execution/`, `replay_plan.py` | Common face-based CAD/CAE execution for the three-condition comparison |
| `figures/` | The author's eight current TIFF figures; only one Fig. 3 image |
| `data/training_curves/` | Six raw histories, 287-row CSV, seed/panel mapping and author Origin project |
| `results/paper/` | Recomputed paper tables and per-task evidence |

The figures are the author's supplied exports. Their drawing/label discrepancies are documented in [manuscript alignment](docs/MANUSCRIPT_ALIGNMENT.md); raw observations have not been changed to match a drawing. No old alternate Fig. 3 image, unpublished robustness dataset, experimental queue, paper draft or API key is included in the current paper tree. Historical module names that remain in the inference dependency closure are required executable code.

## Scope and availability

This minimum artifact supports frozen inference, fresh CAD operations, CAE script generation, reading archived ODBs, and independent table reconstruction. Exact data, seeds, weights and training histories are provided; a portable, fully certified from-scratch training/regeneration workflow is not claimed. The 13 engineering cases are development/application examples. Baseline results apply to the paper's dataset, 18-class ontology and task-compatible adaptations.

The repository remains **private** under the author's earlier instruction. It is prepared as a paper artifact, not yet a publicly licensed open-source release. Public availability and the authors' code/data license must be finalized before a public-access claim is made. Third-party licenses are preserved in `licenses/` and [THIRD_PARTY.md](THIRD_PARTY.md).

See [reproduction scope](docs/REPRODUCTION.md), [paper-to-file mapping](docs/PAPER_RESULTS.md), [data layout](docs/DATA.md) and [中文说明](README.zh-CN.md).
