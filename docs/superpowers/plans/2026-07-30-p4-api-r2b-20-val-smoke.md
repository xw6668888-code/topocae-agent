# P4-API-R2B 20-Val Smoke Execution Plan

> **For agentic workers:** Execute this plan inline; the experiment must stop before any request when preflight identity differs from the authorized profile.

**Goal:** Execute the frozen 20-sample Compact QueryFrame smoke or produce a complete request-free blocked evidence package.

**Architecture:** Reuse the frozen R2A.1 manifest and runtime without modification. Gate all backend access behind identity, sample-hash, freeze, formal-test-lock, request-cap, and cost-cap checks; on a failed identity gate, emit only audit artifacts with metrics marked not executed.

**Tech Stack:** Python 3.11, frozen TopoCAE P2/P3 assets, DeepSeek OpenAI-compatible JSON-object API.

---

### Task 1: Preflight

- [x] Freeze provider/profile, base URL, endpoint, model, response format, temperature, seed, max tokens, HTTP retry, request caps, and cost cap.
- [x] Compare environment identity before constructing or sending a backend request.
- [x] Stop on beta/base-URL mismatch.

### Task 2: Frozen Sample Verification

- [x] Load exactly the 20 IDs in `reports/p4_api_r2a_1_runtime_prompt_v1/smoke_sample_manifest.json` in manifest order.
- [x] Recompute every val query and ground-truth SHA-256 with the frozen canonical JSON procedure.
- [x] Persist only IDs, hashes/match flags, and coverage metadata.

### Task 3: Blocked Delivery

- [x] Record zero first-round, repair, retry, and backend requests.
- [x] Record zero cost, unchanged formal-test lock, and unchanged frozen hashes.
- [x] Mark every unavailable protocol, semantic, grounding, and stratified metric as `not_executed_preflight_block` rather than zero.
- [ ] Verify all required artifacts and generate `delivery_manifest.json` last.

