# P4-API-R2A.1 Runtime Prompt and Smoke Manifest Design

## Scope

R2A.1 adds an offline runtime contract around the frozen R2A QueryFrame assets. It does not modify the QueryFrame Schema, compiler, P2/P3 assets, provider adapter, or historical P4 reports. No network, formal-test, STEP, graph, P5, or Abaqus access is allowed.

## Chosen architecture

Create `utils/cae_query_frame_runtime.py` as a sidecar module. It reads the frozen R2A Schema and base prompt, verifies their hashes, emits a byte-deterministic runtime system prompt, builds a one-shot repair message, constructs a normal DeepSeek JSON-object request, and validates mock output through the frozen QueryFrame validator/compiler/P2 validator. The old provider adapter remains untouched.

The runtime prompt embeds the compact, sorted Schema JSON verbatim and a generated field inventory. Enum values come only from the Schema. Operation mutual-exclusion rules are declared once in the runtime module because the compact Schema intentionally leaves them to the compiler; tests verify every referenced action/field exists in the frozen Schema and every example compiles exactly.

Four handcrafted examples cover valid, needs-clarification, invalid, and multi-operation valid branches. Their provenance is a code constant, not a dataset record. Examples use generic CAE requests and are checked against authorized train/val texts; formal test remains inaccessible.

## Repair contract

`build_query_frame_repair_prompt(inference_input, error_code, schema_summary, attempt)` accepts only the original two-field input, one stable `QF_E_*` error code, a bounded minimal Schema summary, and `attempt=1`. Any second attempt, extra input field, unknown error code, oversized summary, or forbidden evidence fails closed. It never accepts the rejected output, canonical target, ground truth, or corrected answer.

## Mock request contract

The request builder fixes profile `deepseek_json_object`, endpoint `https://api.deepseek.com/chat/completions`, and `response_format={"type":"json_object"}`. The system message contains the assembled runtime prompt; the user message is sorted compact JSON with only `text` and `language`. No strict tool, `/beta`, `json_schema`, or provider-side Schema field is emitted. A mock-only execution function parses duplicate-safe JSON, validates the Frame, compiles it, and revalidates the frozen P2 result.

## Smoke set-cover

Read only `iter_rendering_records("val")`. Candidate labels come from `query.language`, `query.difficulty_level`, and frozen `intent_result`; query and ground-truth contents are used only to compute canonical SHA-256. The manifest stores rendering ID as `sample_id`, allowed metadata, coverage tags, and the two hashes.

Selection uses deterministic weighted greedy set-cover with exact 10/10 language quotas and unique canonical IDs as hard constraints. Uncovered labels receive inverse-frequency weights, so rare invalid, clarification, multi-operation, action, and quantifier labels are preferred. Ties use the lexicographic rendering ID. Selection never uses model output and is run twice for byte equality.

## Cost model

Use the previously frozen R1C planning prices, not a live price claim: cache-miss input USD 0.435/M tokens, output USD 0.87/M tokens, conservative USD/CNY 8.0. Input tokens use UTF-8 payload bytes as an upper bound; output is bounded from the largest allowed QueryFrame plus margin. Report 20 first attempts, at most 20 repairs, and hard request cap 40.

## Verification

New standard-library `unittest` tests cover prompt determinism/hash/schema completeness/examples, repair fail-closed behavior, request payload isolation, mock compilation, deterministic set-cover and redacted manifest. The delivery runner writes all required audits, checks R2A/P1/P2/P3/history hashes, verifies the formal ledger, and runs complete `test_p4*.py` under `W:\python\envs\feagpt\python.exe`.

## Self-review

The design has no placeholders. Runtime and repair responsibilities are isolated, enum ownership remains with the frozen Schema, dataset access is val-only, and the manifest cannot expose query or ground-truth payloads. Git is unavailable, so no commit operation is attempted.
