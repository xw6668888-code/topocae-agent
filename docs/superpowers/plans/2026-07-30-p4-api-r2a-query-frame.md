# P4-API-R2A Query Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver and verify a compact CAEQueryFrame v1 plus deterministic compiler to frozen P2 CAEIntentResult.

**Architecture:** A closed JSON Schema describes an LLM-facing semantic frame. A standalone Python module validates, compiles, serializes, and reverses canonical results for offline audit; a separate offline runner reads only authorized P2/train/val sources and writes evidence.

**Tech Stack:** Python 3 standard library, `jsonschema`, frozen `utils.cae_intent`, standard-library `unittest`.

---

### Task 1: Contract tests and Schema

**Files:** Create `tests/test_p4_api_r2a_query_frame.py`; create `reports/p4_api_r2a_compact_query_frame_v1/query_frame.schema.json`.

- [x] Write failing tests that load the absent module/schema and assert root object, no external refs/recursion, <=10 KiB, closed objects, Chinese descriptions, three statuses, seven actions, identity-field exclusion and single/multi-operation examples.
- [x] Run `W:\python\envs\feagpt\python.exe -B -m unittest tests.test_p4_api_r2a_query_frame -v`; verify failure is caused by the missing delivery.
- [x] Add the minimal Schema and module constants required to pass structural tests.
- [x] Re-run the focused tests and retain green output in the later regression audit.

### Task 2: Compiler red-green cycle

**Files:** Create `utils/cae_query_frame.py`; modify `tests/test_p4_api_r2a_query_frame.py`.

- [x] Add failing valid-path tests for all seven actions, quantity dimension restoration, selectors/filters and multi-operation ID generation.
- [x] Implement explicit action-to-canonical mappings and final frozen P2 validation.
- [x] Add failing negative tests for wrong root type, invalid enum, missing field, unexpected field, conflicts, duplicate IDs and forbidden identity keys.
- [x] Implement stable `QueryFrameCompileError` classification without fuzzy repair.
- [x] Add failing byte determinism and canonical-to-frame exact roundtrip tests, then implement sorted compact UTF-8 serialization and strict inverse mapping.
- [x] Run the focused unittest file after each red/green slice.

### Task 3: Offline audit runner

**Files:** Create `run_p4_api_r2a_compact_query_frame.py`; create audit JSON files under `reports/p4_api_r2a_compact_query_frame_v1/`.

- [x] Add tests for authorized source counts, decode audits, no formal access, coverage dimensions and deterministic manifest generation.
- [x] Implement source iteration using P2 fixtures plus split-safe P3 train/val readers only.
- [x] Compute schema-valid, frozen-validator-valid, semantic exact, deterministic bytes, record-level compactness ratios, and language/difficulty/status/action/quantifier coverage.
- [x] Compute leakage/duplicate/formal-ledger evidence and frozen P1/P2/P3 manifest hash evidence.
- [x] Run the full 21,885-record audit twice where required by determinism checks.

### Task 4: Prompt, protocol and contract documentation

**Files:** Create `query_frame_contract.zh-CN.md`, `query_frame_prompt.zh-CN.md`, `repair_protocol.zh-CN.md`, `CODEX_CONTEXT_P4_API_R2A.md`.

- [x] Document every field and every enum value in Chinese, default reconstruction rules, stable errors, and examples without val/test sentences.
- [x] Freeze a generic `{text, language}` prompt whose only output is one Frame object.
- [x] Freeze a maximum-one-attempt explicit repair protocol accepting only error code/path/schema summary plus the rejected Frame.
- [x] Add tests that reject few-shot dataset strings, API execution code and silent/multiple repair semantics.

### Task 5: Acceptance and verification

**Files:** Create `acceptance.json`, `final_acceptance.zh-CN.md`, `regression_results.json`, `freeze_audit.json`, `delivery_manifest.json`.

- [x] Run `W:\python\envs\feagpt\python.exe -B -m unittest discover -s tests -p test_p4*.py` and record interpreter path/version, test count and return code.
- [x] Recompute all in-scope freeze hashes from authoritative manifests and assert zero changes.
- [x] Assert every numerical gate and every required artifact before assigning the only successful stop status `READY_FOR_P4_API_R2B_20_VAL_SMOKE_REVIEW`.
- [x] Generate the delivery manifest last, excluding itself from its asset hash list, then run the delivery verifier/focused tests again.

## Plan Self-Review

The plan covers every named artifact, numerical gate, source-isolation rule, prompt/repair constraint, required interpreter, full P4 regression and freeze audit. It contains no deferred implementation placeholders. Function names and status/error conventions match the design.
