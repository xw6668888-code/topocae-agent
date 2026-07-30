# P4-R3A Hybrid Evidence Resolver Design

## Scope

Build an offline semantic layer that combines an R2B.1 QueryFrame proposal with explicit evidence extracted only from `{text, language}` and the frozen P2 feature-role policy. The implementation must not modify prompts, schemas, thresholds, historical reports, or cached responses. The 20 cached development-smoke items are used only by the final replay, never to author or tune rules.

## Architecture

`utils/cae_text_evidence.py` owns a deterministic bilingual evidence contract. It validates the two-field input, splits text into operation clauses, applies general lexicons and regular expressions, normalizes units through one dimension-aware unit dictionary, and emits evidence items with source text, half-open character span, normalized value, rule ID, and clause index. Missing parameters are observations only; they do not decide the QueryFrame status.

`utils/cae_hybrid_resolver.py` owns policy-aware resolution. It accepts a proposal plus extracted evidence, anchors proposal operations to text clauses by action cues and order, preserves LLM action/feature/role when policy-compatible, and treats explicit quantifiers, quantities, units, directions, frames, DOFs, and material parameters as authoritative. It never transfers parameters between clauses. An explicit unique conflict creates an `override_by_explicit_evidence` trace; unresolved conflicts create a valid `needs_clarification` frame; contract-exceeding set difference or complex exclusion creates a valid `invalid` frame. Nearest without a reference creates clarification. Every resulting frame is validated, compiled, and checked by the frozen P2 validator before return.

`run_p4_r3a_hybrid_evidence_resolver.py` owns final offline replay and report publication. It reads the frozen R2B.1 sample results and existing development-smoke evaluation inputs, performs no transport calls, calculates the requested paired metrics, audits decision provenance and illegal identities, snapshots historical asset hashes, and writes only `reports/p4_r3a_hybrid_evidence_resolver_v1/` plus an isolated work directory if needed.

## Evidence Contract

Evidence categories cover operation cues, features, face roles, selectors, numbers and units, pressure/displacement/mesh/material parameters, direction, coordinate frame, DOFs, clause boundaries, exclusions, spatial references, and missing required parameters. Each item has `category`, `span`, `text`, `normalized`, `rule_id`, and `clause_index`. Rule IDs are stable semantic identifiers, not data identifiers. The production lexicon contains no smoke text or sample/rendering/template/canonical identifiers.

## Resolution Rules

1. Reject identity-bearing proposal fields through the existing QueryFrame validator.
2. Reject explicit set difference and complex exclusion as `invalid`.
3. Align operations by action cue and text order; ambiguity fails closed.
4. Preserve LLM action, feature, and role only when the frozen policy allows their combination.
5. Use a unique explicit local role to repair an incompatible proposal role; otherwise clarify.
6. Override explicit selector and parameter fields only within the aligned clause, preserving `exactly_one` and `count` regardless of downstream candidate cardinality.
7. Clarify nearest without an explicit spatial reference and any required parameter that remains missing.
8. Emit a trace for every considered field with original value, final value, evidence span, rule ID, policy reference, and one of `accept`, `override_by_explicit_evidence`, `clarify`, or `reject`.

## Isolation And Safety

Development tests use only handcrafted en-US/zh-CN sentences and frozen contracts. A train coverage audit may count rules over P3 train without writing train text into code or reports. The final replay is a separate command and is the first point where the 20 development-smoke inputs are decoded. Formal test, P5, STEP/BRep, P1 recognizer, Abaqus, and network adapters are outside the import graph.

## Verification

New unittest cases first fail for the absent extractor, resolver, replay, and audits. Focused tests then cover spans, normalization, bilingual clauses, exclusions, nearest, missing parameters, policy repair, conflict decisions, operation isolation, compiler/P2 validity, and provenance completeness. Final verification runs the full `test_p4*.py` suite with `W:\python\envs\feagpt\python.exe`, static leakage and identity scans, frozen-tree hash comparison, and the offline 20-item replay. The stop state is selected only from the three states named in the Goal.

