# Codex Context — P4-A1

Last updated: 2026-07-24 Asia/Shanghai  
Workspace: `W:\python\FEA\dataset_generator`

## Current Snapshot

P4-A1 remains `BLOCKED_AT_H4_RUNTIME`. Task A, Task B, Oracle-IR train/val
Gate, Keyword train/val, the evaluator, C2/C3 adapters, leakage tests, and H4
are complete. The human acceptance record now passes its contract: 60 approve,
0 edit, 0 reject; reviewer identity/role/time and all three attestations are
present, while personal fields remain redacted from this journal. No test query/GT
has been decoded, printed, browsed, sampled, or manually read.

## Active Goal

Implement natural-language CAE-IR parsing and deterministic Oracle-geometry
grounding under the frozen P1/P2/P3 contracts. Do not train a model, run solvers,
enter P5 Pointer, or tune from test.

## User Preferences And Hard Constraints

- Follow `goal-objective.md` from the Codex attachment exactly.
- Frozen P1/P2/P3 assets must not change.
- H4 is accepted, but test remains locked until the LLM train/val baselines and
  post-H4 prompt/model/config freeze are complete.
- Structured LLM may output only CAE-IR. Direct LLM sees sanitized candidates and
  may output opaque object IDs/semantic roles, never raw face IDs.
- No model download to C drive; no use of the prohibited superpower skill.

## Key Decisions

- `utils/cae_language_grounding/dataset.py` directly addresses physical
  train/val records and treats scheduled supplement test lines as opaque bytes.
- Feature objects use opaque `obj-####` IDs and P3 Oracle descriptors/roles.
- The deterministic grounder receives validated P2 intent only, never query text.
- The keyword parser was developed on train/val only.
- LLM adapter uses OpenAI-compatible HTTP, temperature 0, seed 20260401, strict
  JSON Schema when supported, at most one explicit repair retry, and no silent
  JSON repair.
- Direct candidate input omits source identity and raw face indices; local code
  expands roles and rejects unknown/out-of-range selection.

## Important Files And Paths

- Preflight: `reports/p4_a1_language_grounding_v1/preflight.zh-CN.md`
- Oracle Gate: `output_p4_a1_language_grounding_v1/task_b/oracle_ir_train_val_gate.json`
- Keyword metrics: `output_p4_a1_language_grounding_v1/baselines/keyword/`
- H4 CSV: `reports/p4_a1_language_grounding_v1/human_review_package/H4-01_val_language_review_60.csv`
- H4 instructions: `reports/p4_a1_language_grounding_v1/human_review_package/README.zh-CN.md`
- H4 machine validation: `reports/p4_a1_language_grounding_v1/human_review_package/H4-01_review_validation.json`
- Runtime request: `reports/p4_a1_language_grounding_v1/LLM_RUNTIME_REQUEST.zh-CN.md`
- Current acceptance: `output_p4_a1_language_grounding_v1/acceptance.json`
- Test ledger: `output_p4_a1_language_grounding_v1/test_run_ledger.json`

## Completed Work

- Initial and H4-pre-pause P1/P2/P3 frozen-hash audits: PASS.
- Feature-object memory audit: 21,620 train/val canonical records, 3,244 graphs,
  5,887 objects, 17 classes, no audit failures.
- Oracle-IR Gate: 21,620 records; status, reconstruction, instance and face-set
  metrics all 100%; illegal IDs and silent fallbacks both 0.
- Keyword train: 57,540 renderings; schema 100%; slot Macro-F1 0.954461; exact
  face set 0.977295; executable 0.974939.
- Keyword val: 7,320 renderings; schema 100%; slot Macro-F1 0.949605; exact face
  set 0.978778; executable 0.973770.
- H4 package: 60 val rows, 30 zh-CN/30 en-US, all 17 classes, required
  quantifiers/statuses/operations.
- H4 row review: 60 approve, 0 edit, 0 reject; all non-review fields match a
  deterministic regeneration of the original package.
- H4 acceptance: PASS; the non-template record satisfies reviewer metadata,
  decision-count, Gate-decision, and three-attestation contracts.

## Verification

- `tests.test_p4_a1_memory`: 6/6 passed.
- `tests.test_p4_a1_grounder`: 6/6 passed.
- `tests.test_p4_a1_keyword_parser`: 4/4 passed.
- `tests.test_p4_a1_llm_and_leakage`: 5/5 passed.
- `tests.test_p4_a1_h4_package`: 4/4 passed after H4 acceptance.
- H4-pre-pause frozen assets: P1 13/13, P2 13/13, P3 1885/1885; no changes.
- LLM runner with missing configuration exits fail-closed with code 2 and
  `BLOCKED_AT_H4_RUNTIME`.
- `validate_p4_a1_h4_review.py` reports `PASS`; this satisfies the H4
  prerequisite but does not itself unlock the global test ledger.

## Open Questions Or Blockers

- `TOPOCAE_LLM_BASE_URL` and `TOPOCAE_LLM_MODEL` are not configured.
- Structured/Direct LLM train/val runs are therefore still not executed.
- Candidate prompt/config hashes are recorded but intentionally marked
  `NOT_FROZEN_H4_AND_RUNTIME_PENDING`.
- Formal test ledger is `LOCKED_RUNTIME_PENDING`; H4 is recorded as accepted,
  test unlock authorization remains false, and every test run count remains 0.

## Next Steps

1. Start an approved local or controlled remote open-source instruct LLM endpoint
   and set the metadata in `LLM_RUNTIME_REQUEST.zh-CN.md`.
2. Verify health, model revision, dtype/quantization, structured JSON support,
   context length, and request-log policy.
3. Run Structured and Direct LLM on val, then train; never use test for model or
   prompt selection.
4. Freeze prompt/model/config/parser/grounder/verifier/evaluator hashes, update
   the test ledger, and only then run each formal test baseline once.
5. Never tune from test; generate final reports, recheck frozen hashes, and stop
   at P4-A1.

## Suggested Skills

- Use `context-journal` again only to update this file after runtime/H4 progress.
- No other skill is required for the next runtime-validation step.

## Update Log

### 2026-07-23 Asia/Shanghai

- Created the durable H4 pause-point snapshot.

### 2026-07-24 Asia/Shanghai

- Detected 60/60 `approve` decisions in the H4 CSV.
- Regenerated the original val package in a temporary directory and verified
  that all protected columns are unchanged.
- Kept test locked because the real-human acceptance/attestation record and LLM
  runtime are still missing.

- H4 acceptance record normalized with machine-only schema metadata; human fields
  were not altered.
- H4 validator reached `PASS`; remaining blocker is the missing LLM runtime.
- Corrected the ledger and H4 validator semantics: H4 acceptance satisfies only
  the human prerequisite and does not directly authorize formal test.
- Re-ran all 25 P4-A1 tests successfully, reverified frozen assets with zero
  changes, and independently rehashed all 56 delivery-manifest assets with zero
  mismatches.
