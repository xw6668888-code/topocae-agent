# P4-R3A Hybrid Evidence Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an offline bilingual text-evidence extractor and frozen-policy Hybrid Resolver over cached R2B.1 QueryFrame proposals.

**Architecture:** Separate deterministic evidence extraction, policy-aware resolution, and final replay/report publication. Reuse the frozen QueryFrame compiler, P2 validator, role policy, R2B.1 cache, and existing evaluation utilities without modifying them.

**Tech Stack:** Python 3, `unittest`, regular expressions, JSON, PyYAML, existing QueryFrame/compiler/P2 modules.

---

### Task 1: Evidence Contract And Unit Normalization

**Files:**
- Create: `tests/test_p4_r3a_text_evidence.py`
- Create: `utils/cae_text_evidence.py`

- [ ] Write handcrafted en-US and zh-CN tests asserting half-open spans, clause indices, stable rule IDs, selector preservation, quantities, direction/frame/DOF evidence, exclusions, nearest references, and missing-parameter observations.
- [ ] Run `W:\python\envs\feagpt\python.exe -m unittest tests.test_p4_r3a_text_evidence -v` and confirm failure because the module is absent.
- [ ] Implement strict `{text, language}` validation, clause segmentation, general lexicons, dimension-aware unit normalization, evidence ordering, and missing-parameter derivation.
- [ ] Re-run the focused test and require zero failures.

### Task 2: Frozen-Policy Hybrid Resolver

**Files:**
- Create: `tests/test_p4_r3a_hybrid_resolver.py`
- Create: `utils/cae_hybrid_resolver.py`

- [ ] Write tests for compatible proposal acceptance, explicit selector/quantity override, policy role repair, ambiguous role clarification, set-difference rejection, nearest clarification, missing parameters, multi-operation clause isolation, illegal identities, and complete decision traces.
- [ ] Run `W:\python\envs\feagpt\python.exe -m unittest tests.test_p4_r3a_hybrid_resolver -v` and confirm failure because the module is absent.
- [ ] Implement clause alignment, frozen policy lookup with review IDs, explicit field overrides, fail-closed status frames, trace creation, QueryFrame validation, compilation, and P2 validation.
- [ ] Re-run both focused modules and require zero failures.

### Task 3: Offline Replay And Requested Metrics

**Files:**
- Create: `tests/test_p4_r3a_offline_replay.py`
- Create: `run_p4_r3a_hybrid_evidence_resolver.py`

- [ ] Write tests using synthetic cached outcomes for zero transport dependencies, paired R2B-v2/R2B.1/Hybrid rows, requested metric denominators, provenance coverage, stop-state thresholds, and immutable report publication.
- [ ] Run `W:\python\envs\feagpt\python.exe -m unittest tests.test_p4_r3a_offline_replay -v` and confirm failure because the runner is absent.
- [ ] Implement cache-only proposal loading, existing-evaluator reuse, metric aggregation, paired comparison, and JSON/Markdown publication to the isolated delivery directory.
- [ ] Re-run all P4-R3A tests and require zero failures.

### Task 4: Leakage, Train Coverage, And Freeze Audits

**Files:**
- Modify: `tests/test_p4_r3a_offline_replay.py`
- Modify: `run_p4_r3a_hybrid_evidence_resolver.py`

- [ ] Add failing tests for forbidden identifier tokens in production rule sources, illegal identity keys, P3-train-only coverage counting, historical tree hashes, and an explicit list of prohibited systems not accessed.
- [ ] Implement static scans, train coverage counters without text export, SHA-256 tree snapshots, delivery manifest hashing, and audit payloads.
- [ ] Re-run all P4-R3A tests and require zero failures.

### Task 5: Final Offline Acceptance

**Files:**
- Create: `reports/p4_r3a_hybrid_evidence_resolver_v1/` artifacts named in the Goal.

- [ ] Snapshot P1/P2/P3/R2A/R2B historical hashes before replay.
- [ ] Run the offline replay once; assert backend requests are zero and publish all required metrics and paired rows.
- [ ] Run `W:\python\envs\feagpt\python.exe -m unittest discover -s tests -p test_p4*.py -v` and capture return code and output summary.
- [ ] Recompute historical hashes, verify no changes, verify every manifest asset hash, and run the forbidden-token and illegal-identity scans.
- [ ] Compare every acceptance metric with the Goal thresholds and publish exactly one allowed stop state.

