# Reproduction scope and implementation

## Frozen recognition

`infer.py` reads STEP geometry through pythonocc-core 7.9.0, extracts the B-Rep attributes, runs the sparse R and dense Q networks, recovers candidates, scores their 66-dimensional evidence, applies the frozen class/source calibration and selects compatible feature objects. The output includes category, complete member faces, supported operation roles, measured parameters and the input geometry hash.

The two four-layer graph encoders have independent weights. Node input combines 40 numeric attributes with the surface-type embedding; edge observations contain 20 attributes. Both encoders propagate on physical adjacency. R predicts same-instance relations for adjacent face pairs; Q also predicts non-adjacent face pairs. The object scorer is a 66→64→32→1 MLP. The frozen calibration uses temperature 3 and smoothing strength 20. Conflict selection uses exact search for components of at most 32 candidates and beam width 128 for larger components. Precise feature construction, recovery and selection code is included under `comparison/topocae/`.

`runtime_loader.py` materializes the 99 hash-checked dependencies into a temporary namespace. Some internal module identifiers retain their historical names because the frozen code imports them. These files are the active inference/execution dependency closure, not copies of the original run directories. No original drive path is needed to run the public entry points.

## Evaluation

Exact-instance matching requires both the category and the complete member-face set to agree. Missing and extra predictions remain false negatives and false positives. The 1,200-part denominator is fixed for every method and seed. The result package contains all seeds and all cases, including failures. Archived predictions are used for offline table reproduction; a fresh STEP inference is a separate computation.

Face accuracy measures predicted face classes and does not establish correct grouping into complete objects. R-derived face outputs can therefore be identical across several object-decoding variants while object scores differ. The supplied AAGNet and MFT implementations are the task-compatible adaptations named in the paper, not claims of reproducing their original benchmark scores. Dataset supervision, model capacity and object-level training differ across systems and are described in their configuration files.

The retained learned-score-only control scores higher than the full calibrated system on the generated test set. It remains in the supplementary prediction data so the package supports the paper's stated calibration limitation.

The original independent recount returns `status: PASS` for the numeric checks. Its legacy `reproducibility_pass: false` and `paper_MAIN_RESULTS_csv_verified: false` fields describe the scope of that one original function: it neither executes models/solvers nor checks the publication CSV. The public `reproduce.py` wrapper additionally checks the CSV values, and `VALIDATION.json` separately records the completed fresh inference, CAD and ODB checks. These legacy fields are preserved rather than relabeled as broader experimental acceptance.

## Engineering replay

`run_case.py` first runs or accepts a fresh `infer.py` output and verifies the STEP SHA-256 identity. It parses the original archived Qwen response, converts the selector schema without changing its action or numeric values, and binds the semantic selector to the newly recognized objects. A hash-scoped object binding determines the affected face sets. Reference labels are used only by evaluation scripts.

CAD operations create a new STEP and run geometric readback. CAE operations generate a script covering import, material, section, mesh, boundary/load application, job submission and output checks. The compiler maintains the actual plan-to-geometry binding. Abaqus face indices are derived geometrically from the source STEP, rather than copied as permanent IDs across exports.

The archived reference cases use a consistent mm–N–MPa–tonne–s unit system. Material values and mesh settings are recorded in the individual plans. The Qwen replay retains its historical material choice; subsequent edits to author instructions do not overwrite these original observations.

## Environment and validation limits

- Lightweight recount: Python, NumPy, JSON Schema; Matplotlib for figures.
- Frozen BiRel-FO inference and CAD: `environment-inference.yml`, CPU supported. CUDA is not required for the minimum replay.
- Abaqus result reading and new solving: a licensed Abaqus 2020 installation. ODB reading uses the Abaqus Python interpreter.
- The baseline code/checkpoints are supplied for inspection and further use; their author-library dependencies are separate from the minimal BiRel-FO environment. The portable package independently rechecks their complete archived prediction tables.
- Complete retraining and random regeneration of the entire dataset are outside this minimum replay. The exact training/validation data, labels, split, frozen settings, weights and observed curves are supplied. The frozen dependency source contains the relevant graph and scorer training components, but this release does not certify a portable from-scratch training driver.
- Validation on 2026-11-03 reruns the packaged inference and CAD paths, compiles CAE scripts, and reads the archived ODBs. It does not start competing solver jobs while other Abaqus jobs are running on the host. See `VALIDATION.json` for the actual completed checks.

The 13 engineering parts are development/application examples. No result in this package establishes open-set recognition of unsupported feature classes or universal CAD editing.
