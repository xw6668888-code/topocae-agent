# STEP-FO18, engineering cases and model weights

## STEP-FO18

STEP-FO18 covers 18 local engineering feature classes and provides face-level labels, complete instance membership and same-instance relations associated with the final STEP geometry.

| Split | Parts | Geometry | Labels and membership |
| --- | ---: | --- | --- |
| Training | 7,440 | `data/d0/steps.zip` | `data/d0/split_and_labels.json` |
| Validation | 930 | `data/d0/steps.zip` | `data/d0/split_and_labels.json` |
| Test | 1,200 | `data/t10/steps.zip` | Per-part targets and indices in `data/t10/` |

The test set contains 400 same-strategy (I), 400 parameter-variation (P) and 400 structural-combination (S) parts. Base-body faces use the background category for face classification. Each feature instance specifies its class and complete set of final STEP member faces.

Use `python fetch_assets.py --group training` and `python fetch_assets.py --group test` to download the geometry archives. Labels, indices and prediction records are available directly in the repository for table reproduction. `infer.py` constructs the model input from STEP geometry.

## Engineering parts

`data/engineering/` contains the 13 application parts and their independent face annotations. Each case includes an input STEP, recognized objects, its operation plan and a result summary. CAD cases include an edited STEP; CAE cases provide `solver_input.zip` with `model.inp` and `solver_output.zip` with `model.odb`.

The `.step` and `.stp` extensions denote the same STEP exchange-format family. The engineering parts were also used during application development and are reported as engineering case studies.

## Language-model experiments

`data/language/` provides the recorded Qwen responses and object-based operation plans for the application workflow. Additional engineering artifacts are in `data/language/physical/`, with identities recorded in `recorded_equivalence.json`.

`data/qwen_unified/` contains the three-condition comparison: input geometry descriptions, task instructions, request bodies, responses, token usage, timings and execution records. The dataset covers 39 tasks and 42 API attempts. It includes all 15 generated CAD STEP files and the saved-model/INP evidence used for CAE evaluation. The primary case-study ODBs are supplied in `data/engineering/`.

## Model checkpoints

- `comparison/topocae/engine/weights/`: the deployed R encoder, Q encoder and object scorer used by `infer.py` and `run_case.py`.
- `comparison/topocae/weights/`: recognition-model checkpoints for the reported seed runs.
- `comparison/aagnet/weights/` and `comparison/mft/weights/`: the adapted baseline checkpoints.

`python fetch_assets.py --profile demo` downloads the deployment models. `python fetch_assets.py --group models` downloads the model asset group.

## File integrity

`assets.json` records the size, SHA-256 and release location of large files. `MANIFEST.json` records the remaining package files. `python verify.py` checks the downloaded package. Historical source-path fields provide provenance; execution uses repository-relative paths.
