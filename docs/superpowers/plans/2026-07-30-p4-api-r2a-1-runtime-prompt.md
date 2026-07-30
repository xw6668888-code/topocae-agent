# P4-API-R2A.1 Runtime Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the offline QueryFrame runtime/repair prompt contract, mock DeepSeek JSON-object request, and deterministic 20-sample val smoke manifest.

**Architecture:** A new sidecar module reads frozen R2A assets and owns deterministic prompt/request logic. A separate offline runner performs val-only set-cover, cost/freeze/leakage audits, and report generation without changing frozen code.

**Tech Stack:** Python 3.11, standard-library `unittest`, frozen QueryFrame/P2 APIs, existing split-safe P3 reader.

---

### Task 1: Runtime and repair contract

**Files:** Create `tests/test_p4_api_r2a_1_runtime.py`; create `utils/cae_query_frame_runtime.py`.

- [ ] Write failing tests for deterministic prompt bytes, Schema SHA, compact embedded Schema, generated inventory, four validated examples, banned evidence terms, and no enum drift.
- [ ] Run the focused unittest and confirm failure is the absent module.
- [ ] Implement Schema-derived prompt assembly and re-run to green.
- [ ] Add failing tests for one-shot repair, exact input keys, stable error codes, bounded summary, and second-attempt rejection.
- [ ] Implement `RuntimeContractError` and repair assembly, then re-run to green.

### Task 2: Mock JSON-object request

**Files:** Modify `tests/test_p4_api_r2a_1_runtime.py`; modify `utils/cae_query_frame_runtime.py`.

- [ ] Add failing tests for normal endpoint, JSON-object profile/payload, two-field user message, absence of strict/json-schema fields, and zero network calls.
- [ ] Implement deterministic request construction and mock-only execution.
- [ ] Add failing tests proving raw mock output passes Frame validation, compilation, and frozen P2 validation; implement minimal parsing pipeline and re-run.

### Task 3: Smoke set-cover and audits

**Files:** Create `tests/test_p4_api_r2a_1_smoke.py`; create `run_p4_api_r2a_1_runtime_prompt.py`.

- [ ] Write failing tests for 20 rows, 10/10 languages, all difficulty/status/scope targets, requested actions/quantifiers, unique sample IDs, unique canonical IDs where feasible, hash-only payload evidence, and deterministic reproduction.
- [ ] Implement val-only candidate extraction and weighted deterministic set-cover with hard quotas.
- [ ] Compute runtime, repair, mock, smoke coverage, cost, leakage, and freeze audit dictionaries; run focused tests to green.

### Task 4: Delivery and regression

**Files:** Create all JSON/Markdown artifacts under `reports/p4_api_r2a_1_runtime_prompt_v1/`.

- [ ] Run the offline delivery runner and persist audits without API/formal/STEP/graph access.
- [ ] Run full `W:\python\envs\feagpt\python.exe -B -m unittest discover -s tests -p test_p4*.py` and record exact count/version/result.
- [ ] Verify R2A/P1/P2/P3/history hashes and formal ledger are unchanged.
- [ ] Generate acceptance and delivery manifest last; verify every report and implementation SHA-256.

## Plan Self-Review

All requested artifacts, gates, quotas, prompt/repair constraints, mock transport assertions, cost bounds, freeze checks, and the required interpreter are covered. Function names and error semantics match the design. No implementation placeholder remains.
