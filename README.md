# TopoCAE-Agent: BiRel-FO and STEP-FO18

Reproducibility artifact for engineering feature-object recognition from STEP B-Rep and its use in CAD editing and CAE preparation.

The package contains the frozen **BiRel-FO** recognition system, the **STEP-FO18** data used by the paper, per-part predictions for the reported comparisons, actual Qwen requests/responses, and the 13 engineering examples. The deployment system is identified as `E3` in the original machine-readable records.

## Start with the smallest closed loop

```bash
git clone https://github.com/xw6668888-code/topocae-agent.git
cd topocae-agent
conda env create -f environment-inference.yml
conda activate topocae-inference
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

The demo downloads approximately 2.84 MB of deployment weights. Engineering STEP inputs and archived language responses are already in Git.

- **Bearing Housing:** fresh recognition of the input STEP; parse the actual archived Qwen response; locate the blind holes of radius 4 mm; edit their radius to 3 mm; read back the edited STEP.
- **Angle Bracket:** fresh recognition; bind the archived material, mesh, support and load plan to recognized objects; generate a complete Abaqus script. To also solve, add `--abaqus /path/to/abaqus` (Windows example: `--abaqus "F:/ABAQUS2020/commands/abaqus.bat"`). A licensed Abaqus 2020 installation is required for this step.

These commands replay an actual language-model response and execute fresh geometry recognition and engineering operations. They make **no new API calls** and do not substitute an author reference plan. Generated files are written into a new output directory. The frozen model is a closed-set recognizer for the paper's 18 feature classes.

For a new STEP file:

```bash
python infer.py /path/to/part.step --output results/runs/part_prediction.json
```

## Reproduce the paper's numbers

The lightweight table reproduction uses the per-part observations in Git and does not require an API key, GPU, CAD kernel or solver:

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python plot_training.py
```

Outputs appear in `results/reproduced/` and `results/reproduced_figures/`. `reproduce.py` independently recounts 25,200 predictions (seven reported/supplementary configurations, three seeds, 1,200 parts), the registered 5,000-resample paired comparisons, the engineering face/instance records, and the archived language responses. The training plot uses six recorded R/Q histories; Times New Roman must be installed to recreate the paper styling.

The principal recognition result is 92.7866% exact-instance macro F1, 94.2530% exact-instance micro F1, 77.0833% whole-part accuracy and 99.6832% face accuracy, averaged over the three frozen seeds. Full per-seed and I/P/S results remain available.

## Full data and solver results

```bash
python fetch_assets.py                  # all 37 immutable assets, about 1.61 GB
python verify.py
python replay_cad.py --output results/runs/reference_cad
abaqus python read_odb.py --output results/runs/odb_readback.json
```

Downloads are also selectable with `--group training`, `test`, `models` or `engineering`. Private-release downloads use `GH_TOKEN`, `GITHUB_TOKEN`, or the local Git credential manager. Credentials are never stored in the artifact. Every downloaded file is verified against its SHA-256 digest. See [the release](https://github.com/xw6668888-code/topocae-agent/releases/tag/paper-v1.0-20261103).

`replay_cad.py` replays the six **author-reference** CAD plans. `run_case.py` instead consumes the actual archived **Qwen-assisted** plan. `read_odb.py` reopens seven archived reference-case ODBs and verifies their fields and equilibrium; it does not rerun those solvers.

## What is included

| Directory / entry | Purpose |
| --- | --- |
| `infer.py`, `run_case.py` | Fresh STEP recognition and reproducible engineering examples |
| `comparison/topocae/` | R/Q encoders, task heads, candidate recovery, object scorer, calibration and constrained selection |
| `comparison/aagnet/`, `comparison/mft/` | Paper-specific baseline implementations, configurations and frozen checkpoints |
| `data/d0/` | 7,440 training and 930 validation STEP parts, split and annotations |
| `data/t10/` | 1,200 final STEP parts; I/P/S subsets of 400 each; targets and complete per-seed predictions |
| `data/engineering/` | 13 STEP parts, recognition outputs, operation plans, six edited STEPs and seven solver input/output archives |
| `data/language/` | Historical Qwen object-binding requests/responses and operation-equivalence evidence |
| `data/qwen_unified/` | Three Qwen conditions on the same 13 tasks; 42 API attempts for 39 slots, including quota failures |
| `data/training_curves/`, `data/robustness/` | Recorded training curves and supplementary geometric perturbation observations |
| `reproduce.py`, `src/` | Independent metric recalculation |
| `assets.json`, `MANIFEST.json`, `VALIDATION.json` | Download identities, source integrity and measured package validation |

## Interpretation of the experiments

The historical object-binding demonstration and the unified face-output comparison are distinct experiments. In the historical Qwen-assisted workflow, 13/13 tasks have correct operation-face sets and verified engineering artifacts, including reuse of equivalent previously executed commands. In the unified comparison, Qwen3.6-Plus alone, Qwen3.6-Plus with BiRel-FO, and Qwen3.8-Max-0902 locate 3/13, 12/13 and 10/13 tasks correctly; complete task success is 3/13, 11/13 and 10/13. Both records are retained. They must not be combined into a single supposedly uniform 3/13–13/13–10/13 comparison.

See [paper/result mapping](docs/PAPER_RESULTS.md), [reproduction scope and methods](docs/REPRODUCTION.md), [data layout](docs/DATA.md) and [中文说明](README.zh-CN.md).

This release reproduces frozen-model inference, CAD execution, CAE script generation and archived-result evaluation. It does not claim a new end-to-end retraining run or renewed provider/solver experiments. The 13 engineering cases were used during development and application validation; they are not a blind estimate of general industrial performance. The adapted AAGNet18-SI and MFTReNet-SI-D0 scores apply to this dataset and task definition.

## Access and attribution

This repository remains **private**, as requested by the authors. It is prepared for a future paper code release; repository visibility and a repository-wide public license have not been changed. Third-party notices are in [THIRD_PARTY.md](THIRD_PARTY.md) and `licenses/`. Abaqus, Open CASCADE and other dependencies must be installed separately. No API key, paper draft, downloaded publication, experimental queue or abandoned run archive is included in the current paper tree. Earlier Git history is retained.
