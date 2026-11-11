# Reproduction guide

## 1. Reproduce the reported tables

Install `requirements-recount.txt`, then run:

```bash
python verify.py --code-only
python reproduce.py
```

The generated CSV and JSON files are written to `results/reproduced/`. The corresponding paper tables are in `results/paper/`. To run one evaluation, use `--only recognition`, `--only engineering`, `--only historical` or `--only unified`.

The recognition evaluation aggregates 21,600 predictions across six configurations, three seeds and 1,200 parts. Exact-instance matching requires the predicted class and complete member-face set to match the reference. Results include whole-part accuracy, face accuracy, exact-instance macro F1 and micro F1, together with I/P/S subset results and paired bootstrap comparisons. Each method uses the same test parts and 18-class definition.

The ablations compare R-only, Q-only, rule-based scoring and BiRel-FO. AAGNet18-SI and MFTReNet-SI-D0 are the literature-method adaptations evaluated on STEP-FO18. The supplied settings document their supervision, capacity and decoding configurations.

## 2. Run STEP recognition

Create and activate `environment-inference.yml`, then download the deployment weights:

```bash
python fetch_assets.py --profile demo
python infer.py data/engineering/bearing_housing/input.step --output results/runs/bearing_prediction.json
```

The recognition entry point extracts a B-Rep graph from the STEP file and runs the pretrained BiRel-FO system on CPU. Output objects contain the feature class, complete member-face set, supported operation roles, measured parameters and source STEP hash. Use a new output filename for each run.

The R and Q encoders have independent weights and four message-passing layers. Node input combines 40 numerical attributes with a surface-type embedding; edge observations contain 20 attributes. Both encoders propagate over physical adjacency. R scores adjacent face pairs, while Q additionally scores non-adjacent pairs. Recovered candidates are evaluated by a 66→64→32→1 object scorer, calibrated with temperature 3 and smoothing strength 20, and selected under feature constraints. Conflict components of at most 32 candidates use exact search; larger components use beam search with width 128.

The executable implementation is under `comparison/topocae/`. `runtime_loader.py` prepares the required modules in a temporary directory and verifies their identities. All entry points resolve inputs relative to the repository.

## 3. Run the CAD/CAE applications

```bash
python run_case.py bearing_housing --output results/runs/bearing
python run_case.py angle_bracket --output results/runs/angle
```

Each run performs recognition, parses the recorded Qwen response, binds its semantic targets to feature objects, and executes the corresponding CAD operation or generates an Abaqus script. The saved language-model responses provide a deterministic replay of the reported application workflow without a live API call. A fresh prediction can also be supplied with `--prediction`.

CAD execution writes an edited STEP and performs geometric readback. CAE scripts cover import, material and section assignment, meshing, boundary conditions, loads, job submission and result checks. The engineering cases use the mm–N–MPa–tonne–s unit system. Material and mesh settings are included in each task plan.

To submit the Angle Bracket analysis, provide the Abaqus command and use a new output directory:

```bash
python run_case.py angle_bracket --output results/runs/angle_solved --abaqus abaqus
```

The package's CAE environment was validated with Abaqus 2020. A licensed installation is required for job submission and ODB access.

## 4. Replay the language-model comparison

The three conditions are `QP-B` (Qwen3.6-Plus), `QP-S` (Qwen3.6-Plus + BiRel-FO), and `QM-B` (Qwen3.8-Max-0902). They share the direct-face operation interface implemented in `direct_execution/`.

```bash
python replay_plan.py QP-B bearing_housing --output results/runs/plus_bearing
python replay_plan.py QP-S clevis --output results/runs/assisted_clevis
```

`replay_plan.py` parses the selected recorded response and executes its face selections, actions and parameters. Clarification responses are recorded without execution. For CAE solving, add `--abaqus abaqus`; `--cpus` and `--memory-mb` configure solver resources. Model correctness and solver completion are recorded separately. [ENGINEERING_ACCEPTANCE.md](ENGINEERING_ACCEPTANCE.md) defines the task metrics and API accounting.

## 5. Read the supplied engineering outputs

```bash
python fetch_assets.py --group engineering
python replay_cad.py --output results/runs/reference_cad
abaqus python read_odb.py --output odb_readback.json
```

The engineering data include six edited STEP files, seven solver-input archives and seven primary ODB archives. The readback script checks the supplied ODBs against their recorded fields and engineering checks. Per-case plans and result summaries are included alongside these files.

## 6. Training records and checkpoints

`comparison/` contains model implementations, settings and checkpoints for the three recorded seed runs. The corresponding training and validation inputs, labels and fixed split are provided under `data/d0/`. Model-specific baseline dependencies are documented with the supplied source configurations; the minimal inference environment above is for BiRel-FO.

```bash
python fetch_assets.py --group training
python fetch_assets.py --group models
python export_training_data.py
```

The last command exports the six R/Q training histories for Fig. 3. The release's main reproduction commands evaluate saved predictions, run pretrained recognition and replay CAD/CAE applications. File checks and completed inference, CAD and ODB validation records are provided in `VALIDATION.json`.
