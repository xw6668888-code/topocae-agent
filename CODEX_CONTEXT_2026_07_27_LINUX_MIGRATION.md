# TopoCAE Linux Shadow Migration Context

## Current state

- Project root: `/home/ywx1/dataset_generator`.
- Windows authority retained at `W:\python\FEA\dataset_generator` until supervisor signs `MIGRATION-PASS`.
- Final migration state: `READY_FOR_MIGRATION_ACCEPTANCE`.
- Next and only gate: supervisor signature `MIGRATION-PASS`.
- P4 formal test remains locked. P5 has not started.

## Snapshot chain

- Snapshot A manifest SHA-256: `fafcf0e82ea78d09ab2d6bff99a13e179df614cffad020ecc1f3c3b41ff1d561`.
- Snapshot B archive SHA-256: `92bf5b961e0fc8cd74e2d845b0d3b10f472edfc20d6291305259701e6e6ce084`.
- Snapshot B tree manifest SHA-256: `05ecd6f9e92137a00bedc561599ad8a5749f6562dc4b7796d12062921ee18a6f`.
- Linux extraction reproduced 386455 files and 48094569238 bytes before post-snapshot migration reports were added.
- Post-Snapshot B additions and modifications are recorded in `reports/migration/m3_incremental_patch_manifest.json` and `reports/migration/m4_incremental_patch_manifest.json`.

## Environments

- Miniforge prefix: `/home/ywx1/miniforge3`.
- `topocae-core`: Python 3.11.15, pythonocc-core 7.9.3, PyTorch 2.10.0, CUDA 13.0, 2 x NVIDIA A40.
- `topocae-llm`: Python 3.12.13, vLLM 0.26.0, OpenAI 2.48.0, PyTorch 2.11.0+cu130, 2 x NVIDIA A40.
- Specs and explicit package lists: `migration/linux/`.
- No model weights were downloaded. No vLLM service or LLM call was made.

## M4 evidence

- Frozen assets: P1 13/13, P2 13/13, P3-A1R 1885/1885.
- OCC/graph: 13/13.
- P1-D Lite smoke: deterministic fields match the historical result.
- P2-A1: 27/27 tests and 1088/1088 fixture assertions.
- P3-A1R: 13/13 through an external compatibility harness while frozen files remain byte-identical.
- Path portability: 15/15 and real P3/P4 Windows-path graph loading succeeded.
- P4 Oracle train/val: PASS, failure count 0, decoded test lines 0.
- P4 Keyword train 57540 and val 7320: deterministic metrics match the Windows baseline.
- P4 test ledger: content not accessed, run count 0, unlock not authorized.

## Retained patches

- `tests/test_graph_v2_representation.py`: imports the already-used `Counter`.
- `tests/test_p4_a1_path_portability.py`: resolves the frozen graph path before creating a fake Windows path.
- `tests/test_m1_a1_path_integrations.py`: resolves frozen graph and geometry paths before reading source bytes.

The attempted direct P3 frozen-file resolver edits were fully reverted. Their final SHA-256 values match Snapshot B and the P3 frozen verifier passes 1885/1885.

## Known frozen limitation

The legacy pre-P3-A1R formal suite is 6/7 on Linux because frozen `utils/cae_query_ground/constants.py` treats a Windows path as a relative Linux path. M1 already documented this incompatibility. Do not modify that frozen file to satisfy the historical assertion. P3-A1R and real P3/P4 resolver loading are the migration acceptance boundaries.

## Mandatory guards

- Do not modify files before the recovery summary is reviewed.
- Do not run P4 formal test or read its protected content.
- Do not call LLMs or download model weights without a new explicit gate.
- Do not enter P5.
- Do not declare `MIGRATION-PASS`; only the supervisor can sign it.
