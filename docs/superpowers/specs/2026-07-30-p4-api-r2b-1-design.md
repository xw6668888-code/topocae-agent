# P4-API-R2B.1 Prompt Revision Design

## Scope

Perform one general runtime-prompt revision based only on aggregate R2B v2 failures, then run one paired re-smoke on the frozen ordered 20-item val manifest. QueryFrame/P2 schemas, compiler, validator, grounder, repair policy, samples, metrics, thresholds, and historical reports remain unchanged.

## Architecture

The sole semantic behavior change is in `utils/cae_query_frame_runtime.py`. It loads the frozen P2 feature-role policy and frozen CAE-IR schema, deterministically emits a compact compatibility table and fail-closed decision rules, and adds five handcrafted general examples. It contains no sample IDs, smoke queries, part-specific rules, or ground-truth payloads.

A new offline R2B.1 module replays v2 cached responses through the unchanged Frame validator/compiler/P2 validator. It produces field-level differences for status, action, feature, face role, quantifier/count, parameters, operation completeness/order, protocol, and grounding while preserving canonical semantic exact as the semantic gate.

A new execution module reuses the frozen 20-item manifest and v2 transport limits with a new cache namespace and output/work directories. It publishes operation-parameter exact, target-contract exact, v2-to-R2B.1 paired changes, and singleton wrong-quantifier/same-face-set counts. Passing maps to `READY_FOR_P4_API_R2C_DISJOINT_100_VAL_REVIEW`; a completed non-passing model run maps to `DEEPSEEK_COMPACT_FRAME_INSUFFICIENT_WITH_EVIDENCE`; external/preflight failure maps to `BLOCKED_WITH_EVIDENCE`.

## Isolation And Safety

Formal test stays fail-closed and undecoded. Leakage audit decodes only train/val and records opaque formal-test lines without reading their text. R2B v1/v2 and R2A/R2A.1 report trees are hashed before and after. P1, STEP/BRep, Direct baseline, P5, and Abaqus are not invoked.

## Verification

Tests first cover deterministic prompt bytes, exact policy-table derivation, schema-derived status rules, quantifier and operation rules, five example categories, cache-only diagnosis, paired metrics, namespace isolation, stop-state logic, and frozen hashes. The offline gate then runs mock Frame/compiler/P2 validation and complete `test_p4*.py` before any API request.
