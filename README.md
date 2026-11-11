# BiRel-FO: Engineering Geometric Feature Object Recognition for CAD Editing and CAE Modeling

[中文](README.zh-CN.md) | [Dataset and checkpoints](docs/DATA.md) | [Reproduction guide](docs/REPRODUCTION.md) | [Paper results](docs/PAPER_RESULTS.md)

This repository provides the implementation of **BiRel-FO**, the **STEP-FO18** dataset, pretrained models, and the experiments accompanying *BiRel-FO: Engineering geometric feature object recognition for CAD editing and CAE modeling*.

BiRel-FO recognizes engineering feature objects directly from STEP B-Rep geometry. It combines sparse and dense same-instance relations with object-level scoring and conflict selection to recover feature categories and complete member-face sets. The resulting feature objects, face roles and measured geometric parameters support language-guided CAD editing and CAE model preparation.

## Main results

- **Feature recognition:** 92.79% exact-instance macro F1 and 77.08% whole-part accuracy on 1,200 test parts, averaged over three random seeds.
- **Engineering applications:** 182 recognized feature instances across 13 parts, with six CAD edits and seven CAE modeling and analysis examples.
- **Language-model assistance:** 4/13, 13/13 and 11/13 successful engineering tasks for Qwen3.6-Plus, Qwen3.6-Plus + BiRel-FO and Qwen3.8-Max-0902, respectively, under the common task criterion described in [evaluation and API accounting](docs/ENGINEERING_ACCEPTANCE.md).

The release includes the model implementations, dataset splits and annotations, checkpoints, training histories, predictions, language-model responses, and engineering outputs used to obtain these results.

## Installation

```bash
git clone https://github.com/xw6668888-code/topocae-agent.git
cd topocae-agent
conda env create -f environment-inference.yml
conda activate topocae-inference
```

The environment includes PyTorch and pythonocc-core for recognition and CAD operations. The supplied inference entry point runs on CPU. CAE solving and ODB reading use a separately installed, licensed Abaqus installation.

## Quick start

Download the deployment checkpoints and run the CAD and CAE examples:

```bash
python fetch_assets.py --profile demo
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

`run_case.py` recognizes the input STEP, binds the recorded Qwen operation plan to the recognized feature objects, and executes the CAD edit or generates an Abaqus script. Recorded language-model responses make the examples reproducible without an API key. The deployment checkpoints total approximately 2.84 MB.

To recognize a new part:

```bash
python infer.py part.step --output results/runs/prediction.json
```

The prediction contains feature classes, member-face sets, operation roles, geometric parameters and the STEP identity. See the [reproduction guide](docs/REPRODUCTION.md) for submitting CAE jobs and replaying the three language-model configurations.

## Reproduce the paper tables

The table-reproduction workflow requires Python, NumPy and JSON Schema:

```bash
python -m pip install -r requirements-recount.txt
python verify.py --code-only
python reproduce.py
python export_training_data.py
```

`reproduce.py` reconstructs the recognition, engineering and language-model tables from the supplied per-part and per-task records. The recognition evaluation includes **21,600 predictions** from six configurations, three seeds and 1,200 test parts. Generated tables are written to `results/reproduced/`; the paper tables are provided in `results/paper/`.

`export_training_data.py` exports the six training and validation histories used in Fig. 3. The CSV contains 287 recorded epochs across the R/Q branches and three seeds.

## Download data and checkpoints

```bash
python fetch_assets.py
python verify.py
```

Large files are distributed through GitHub Releases. `assets.json` specifies each file's release, download name, size and SHA-256 checksum. The downloader places files in the expected directories and verifies their integrity. Use `--group training`, `--group test`, `--group engineering` or `--group models` to download a subset; `--anonymous` explicitly downloads public assets without credentials.

The fixed paper package is [paper-v1.4-20261111](https://github.com/xw6668888-code/topocae-agent/releases/tag/paper-v1.4-20261111).

## Repository structure

| Location | Contents |
| --- | --- |
| `comparison/topocae/` | BiRel-FO graph networks, candidate recovery, object scorer, calibration and selection |
| `comparison/ablations/` | R-only, Q-only, rule-scored and complete BiRel-FO configurations |
| `comparison/aagnet/`, `comparison/mft/` | AAGNet18-SI and MFTReNet-SI-D0 implementations, settings and checkpoints |
| `data/d0/`, `data/t10/` | 7,440 training, 930 validation and 1,200 test parts with annotations |
| `data/engineering/` | 13 engineering parts, annotations, predictions, edited STEP files and Abaqus results |
| `data/language/` | Object-based operation plans and the engineering demonstration records |
| `data/qwen_unified/` | Three language-model configurations, 39 tasks, API records and engineering outputs |
| `direct_execution/` | Shared CAD/CAE executor for the language-model comparison |
| `figures/`, `data/training_curves/` | Paper figures, training histories and Fig. 3 source data |
| `results/paper/` | Paper tables and per-task results |

## Documentation

- [Reproduction guide](docs/REPRODUCTION.md)
- [Dataset, annotations and model weights](docs/DATA.md)
- [Paper-to-file mapping](docs/PAPER_RESULTS.md)
- [Engineering evaluation and API accounting](docs/ENGINEERING_ACCEPTANCE.md)
- [Data and code availability](docs/AVAILABILITY.md)

## Acknowledgments

The baseline comparisons build on [AAGNet](https://github.com/whjdark/AAGNet) and MFTReNet. Their license notices are included in `licenses/`; dependency information is provided in [THIRD_PARTY.md](THIRD_PARTY.md).
