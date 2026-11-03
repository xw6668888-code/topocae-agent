# Data, labels and identities

`data/d0/steps.zip` contains the exact 8,370 training/validation STEP parts; `split_and_labels.json` defines 7,440 training and 930 validation members and their labels. `data/t10/steps.zip` contains a separate 1,200-part test set, with 400 same-strategy (I), 400 parameter-variation (P) and 400 structural-combination (S) parts. Per-part targets and the STEP index are stored outside the ZIP so table reproduction does not require downloading large geometry files.

Target records define the 18 local feature classes, face classes and full instance membership. Stock/base faces belong to the background category for face classification. These labels are evaluation/training data; the `infer.py` entry point constructs its numeric input from the STEP file.

The engineering folder contains 13 author-supplied or author-created STEP parts and independent face annotations. Each case's `prediction.json` is an archived frozen-model result. `input.step` is the original input; `output.step` is the historical edited result for CAD cases. Each CAE `solver_input.zip` contains `model.inp`, and `solver_output.zip` contains `model.odb`. The `.stp` and `.step` suffixes refer to the same exchange-format family.

`data/language/physical/` additionally includes the original Qwen-driven engineering artifacts that differed in plan expression from the reference cases. A supplementary baseline failure is retained alongside the successful assisted cases. Archive member and original SHA-256 identities are recorded in `recorded_equivalence.json`.

`data/qwen_unified/requests/` stores the exact provider request bodies (without authentication headers), and `responses/` stores the raw streaming content. Per-attempt metadata retains failed quota calls, timing, model IDs and reported usage. Physical execution readbacks are retained as small records; the unified comparison's complete additional solver binaries are not duplicated in this minimum package. The selected 13-case closed-loop solver binaries are supplied separately.

Stored historical provenance may include source-workspace path strings. They are provenance only; the public entry points use paths relative to this repository and temporary runtime directories. Hashes identify the actual files and prevent silent substitution. All large release attachments are enumerated in `assets.json`.
