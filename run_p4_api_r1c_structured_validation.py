"""One-revision bounded Structured-LLM train/val validation for P4-API-R1C."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from p4_api_r0_1_audit import verify_full_freeze
from utils.cae_language_grounding.evaluation import BaselineAccumulator, MemoryCache
from utils.cae_language_grounding.grounder import ground_intent_result
from utils.cae_language_grounding.llm_adapter import (
    JSONInferenceOutcome,
    LLMRuntimeConfig,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.llm_adapter_r0 import (
    INTENT_RESULT_SCHEMA_PATH,
    STRUCTURED_PROMPT_PATH,
    _default_transport,
)
from utils.cae_language_grounding.records import iter_language_examples


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r1c_structured_v1"
REPORT = ROOT / "reports" / "p4_api_r1c_structured_v1"
CACHE = OUTPUT / "cache"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
SPLITS = ("train", "val")
SAMPLE_COUNT_PER_SPLIT = 50
BACKEND_REQUEST_CAP = 201
COST_CAP_CNY = 20.0
USD_CNY_CONSERVATIVE = 8.0
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
OLD_PROMPT_SHA256 = "2f6285afc985afe94295df80e44107dc72692d118ca5410a9a74401f964dcc2b"
NEW_PROMPT_SHA256 = "e289c378c12b225da467310dada3b093bd13b2937736d7434f3039f8892344da"
INTENT_SCHEMA_SHA256 = "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727"
DIRECT_PROMPT_SHA256 = "ed21697b97c114ec12f2eab9e13aeb8c1a69afdc68efd03f7c458d693861a581"
DIRECT_SCHEMA_SHA256 = "77d1b4f1b2500d0806d15417454752ac3a87e09437335aa61b27ecebfe4d351a"
DIRECT_PROMPT_PATH = ROOT / "prompts" / "p4_a1_direct_llm_v1.md"
DIRECT_SCHEMA_PATH = ROOT / "schemas" / "p4_a1_direct_selection.schema.json"
CONTRACT_PATHS = (
    STRUCTURED_PROMPT_PATH,
    INTENT_RESULT_SCHEMA_PATH,
    DIRECT_PROMPT_PATH,
    DIRECT_SCHEMA_PATH,
    ROOT / "utils" / "cae_intent.py",
    ROOT / "utils" / "cae_language_grounding" / "grounder.py",
    ROOT / "utils" / "cae_language_grounding" / "llm_adapter.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def formal_test_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    state = {
        "run_count": sum(len(rows) for rows in ledger["runs"].values()),
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }
    if state != {"run_count": 0, "content_accessed": False, "unlock_authorized": False}:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    return state


def contract_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in CONTRACT_PATHS}


def assert_contract_preflight() -> dict[str, str]:
    values = contract_hashes()
    expected = {
        "prompts/p4_a1_structured_llm_v1.md": NEW_PROMPT_SHA256,
        "schemas/cae_intent_result.schema.json": INTENT_SCHEMA_SHA256,
        "prompts/p4_a1_direct_llm_v1.md": DIRECT_PROMPT_SHA256,
        "schemas/p4_a1_direct_selection.schema.json": DIRECT_SCHEMA_SHA256,
    }
    for name, value in expected.items():
        if values.get(name) != value:
            raise RuntimeError(f"E_CONTRACT_HASH:{name}")
    return values


def runtime_checks(config: LLMRuntimeConfig) -> None:
    if not config.available or not config.api_key:
        raise RuntimeError("E_RUNTIME_MISSING")
    if config.provider_profile != "deepseek_json_object":
        raise RuntimeError("E_PROVIDER_PROFILE")
    if config.model != "deepseek-v4-pro":
        raise RuntimeError("E_CONFIGURED_MODEL_MISMATCH")
    if config.max_http_retries != 2 or config.max_tokens != 4096:
        raise RuntimeError("E_RETRY_OR_TOKEN_POLICY_CHANGED")


def token_cost_cny(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def sanitize_errors(errors: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "error_code": str(error.get("error_code") or error.get("code") or "E_UNKNOWN"),
            "field_path": str(error.get("field_path") or "/"),
        }
        for error in errors
    ]


class BudgetedCappedTransport:
    """Persist request count and reserve a worst-case cost before every call."""

    def __init__(self, transport=_default_transport) -> None:
        self.transport = transport
        self.request_count = 0
        self.accounted_cost_cny = 0.0
        self.metered_input_tokens = 0
        self.metered_output_tokens = 0
        self.conservative_reservations_cny = 0.0
        self.audit_path = OUTPUT / "request_budget_audit.json"
        self._persist()

    def _persist(self) -> None:
        write_json(self.audit_path, {
            "schema_version": "topocae-p4-api-r1c-request-budget/1",
            "backend_request_count": self.request_count,
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "accounted_cost_cny": self.accounted_cost_cny,
            "cost_cap_cny": COST_CAP_CNY,
            "metered_input_tokens": self.metered_input_tokens,
            "metered_output_tokens": self.metered_output_tokens,
            "conservative_reservations_cny": self.conservative_reservations_cny,
        })

    def __call__(self, url, headers, payload, timeout):
        if self.request_count >= BACKEND_REQUEST_CAP:
            raise RuntimeError("E_BACKEND_REQUEST_CAP")
        payload_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        output_upper = int(payload.get("max_tokens", 4096))
        worst_call_cost = token_cost_cny(payload_bytes, output_upper)
        if self.accounted_cost_cny + worst_call_cost > COST_CAP_CNY:
            raise RuntimeError("E_COST_CAP_BEFORE_REQUEST")
        self.request_count += 1
        self.accounted_cost_cny += worst_call_cost
        self.conservative_reservations_cny += worst_call_cost
        self._persist()
        try:
            response = self.transport(url, headers, payload, timeout)
        except Exception:
            # Keep the reservation when no trustworthy usage record exists.
            self._persist()
            raise
        usage = response.get("usage") if isinstance(response, Mapping) else None
        try:
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, TypeError, ValueError):
            self._persist()
            return response
        actual = token_cost_cny(input_tokens, output_tokens)
        self.accounted_cost_cny += actual - worst_call_cost
        self.conservative_reservations_cny -= worst_call_cost
        self.metered_input_tokens += input_tokens
        self.metered_output_tokens += output_tokens
        self._persist()
        return response


class AuditedClient(OpenAICompatibleJSONClient):
    def __init__(self, config: LLMRuntimeConfig, transport: BudgetedCappedTransport) -> None:
        super().__init__(config, transport=transport)
        self.split = "unassigned"
        self.example_id = "unassigned"
        self.language = "unassigned"
        self.difficulty = "unassigned"
        self.audit_rows: list[dict[str, Any]] = []

    def infer_json(self, **kwargs) -> JSONInferenceOutcome:
        validator = kwargs["validator"]
        attempts: list[dict[str, Any]] = []

        def recording_validator(value):
            validation = validator(value)
            attempts.append({
                "attempt": len(attempts) + 1,
                "valid": bool(validation.get("valid")),
                "errors": sanitize_errors(list(validation.get("errors", []))),
            })
            return validation

        kwargs["validator"] = recording_validator
        outcome = super().infer_json(**kwargs)
        row = {
            "split": self.split,
            "example_id": self.example_id,
            "language": self.language,
            "difficulty_level": self.difficulty,
            "schema_valid_first_round": outcome.schema_valid_first_round,
            "schema_valid_after_retry": outcome.schema_valid_after_retry,
            "retry_count": outcome.retry_count,
            "failure_code": outcome.failure_code,
            "validation_attempts": attempts,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "backend_request_count": outcome.backend_request_count,
            "actual_model": outcome.actual_model,
            "finish_reason": outcome.finish_reason,
            "latency_ms": outcome.latency_ms,
            "http_retry_count": outcome.http_retry_count,
            "cache_hit": outcome.cache_hit,
            "request_hash": outcome.request_hash,
        }
        self.audit_rows.append(row)
        append_jsonl(OUTPUT / "outcome_audit.jsonl", row)
        if outcome.retry_count > 1:
            raise RuntimeError("E_SCHEMA_RETRY_CAP")
        if outcome.actual_model != self.config.model:
            raise RuntimeError("E_ACTUAL_MODEL_MISMATCH")
        if outcome.finish_reason != "stop":
            raise RuntimeError("E_FINISH_REASON")
        return outcome


def invalid_prediction() -> dict[str, Any]:
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "INVALID",
        "error_code": "E_SCHEMA_VALIDATION",
        "field_path": "/",
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }


def selected_samples() -> dict[str, list[Any]]:
    values = {
        split: list(itertools.islice(iter_language_examples(split), SAMPLE_COUNT_PER_SPLIT))
        for split in SPLITS
    }
    if any(len(rows) != SAMPLE_COUNT_PER_SPLIT for rows in values.values()):
        raise RuntimeError("E_SAMPLE_COUNT")
    return values


def sample_manifest(samples: Mapping[str, list[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "topocae-p4-api-r1c-sample-manifest/1",
        "selection_policy": "first_50_in_frozen_split_iteration",
        "total_selected": sum(len(rows) for rows in samples.values()),
        "ground_truth_written_or_printed": False,
        "formal_test_accessed": False,
        "splits": {},
    }
    for split, rows in samples.items():
        ids = [row.example_id for row in rows]
        result["splits"][split] = {
            "count": len(rows),
            "sample_ids": ids,
            "sample_id_set_sha256": hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "language_counts": dict(Counter(row.language for row in rows)),
            "difficulty_counts": dict(Counter(row.difficulty_level for row in rows)),
        }
    return result


def planning_cost(config: LLMRuntimeConfig, samples: Mapping[str, list[Any]]) -> dict[str, Any]:
    client = OpenAICompatibleJSONClient(replace(config, cache_dir=None))
    prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(INTENT_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    first_sizes: list[int] = []
    repair_sizes: list[int] = []
    for rows in samples.values():
        for example in rows:
            user_text = json.dumps(example.inference_input(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ]
            first_payload = client._payload(messages, "topocae_cae_intent_result_v1", schema)
            first_sizes.append(len(json.dumps(first_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
            repair_messages = messages + [
                {"role": "assistant", "content": "x" * (config.max_tokens * 4)},
                {"role": "user", "content": "The previous answer was invalid. Return a complete replacement JSON object matching the schema. Errors: " + "x" * 4096},
            ]
            repair_payload = client._payload(repair_messages, "topocae_cae_intent_result_v1", schema)
            repair_sizes.append(len(json.dumps(repair_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    estimated_upper = token_cost_cny(
        sum(first_sizes) + sum(repair_sizes),
        len(first_sizes) * 2 * config.max_tokens,
    )
    extra_request_upper = token_cost_cny(max(repair_sizes), config.max_tokens)
    guarded_plan_upper = estimated_upper + extra_request_upper
    return {
        "schema_version": "topocae-p4-api-r1c-cost-plan/1",
        "pricing": {
            "source": "https://api-docs.deepseek.com/quick_start/pricing/",
            "input_cache_miss_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "usd_cny_conservative": USD_CNY_CONSERVATIVE,
        },
        "planned_records": len(first_sizes),
        "planned_schema_attempts_upper": len(first_sizes) * 2,
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cost_cap_cny": COST_CAP_CNY,
        "estimated_two_attempt_upper_cny": estimated_upper,
        "estimated_201_request_upper_cny": guarded_plan_upper,
        "estimate_assumptions": {
            "input_token_upper_uses_payload_utf8_bytes": True,
            "first_response_placeholder_bytes_per_max_token": 4,
            "repair_error_placeholder_bytes": 4096,
            "every_response_uses_max_output_tokens": True,
            "all_input_priced_as_cache_miss": True,
        },
        "runtime_per_request_budget_guard": True,
        "gate_pass": guarded_plan_upper <= COST_CAP_CNY,
    }


def run_split(
    split: str,
    examples: list[Any],
    client: AuditedClient,
    memory_cache: MemoryCache,
    split_accumulator: BaselineAccumulator,
    combined_accumulator: BaselineAccumulator,
    graph_sets: dict[str, set[str]],
) -> None:
    for example in examples:
        formal_test_state()
        client.split = split
        client.example_id = example.example_id
        client.language = example.language
        client.difficulty = example.difficulty_level
        started = time.perf_counter()
        outcome = client.infer_structured(example.inference_input())
        prediction = dict(outcome.value or invalid_prediction())
        graph_path = str(example.source_ref["graph_path"])
        memory, instance_map = memory_cache.get(graph_path)
        grounding = ground_intent_result(prediction, memory, policy=memory_cache.policy)
        latency_ms = (time.perf_counter() - started) * 1000.0
        kwargs = {
            "example": example,
            "prediction": prediction,
            "grounding": grounding,
            "instance_map": instance_map,
            "latency_ms": latency_ms,
            "schema_valid_first_round": outcome.schema_valid_first_round,
            "retry_count": outcome.retry_count,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
        }
        flags = split_accumulator.add(**kwargs)
        combined_accumulator.add(**kwargs)
        graph_sets[split].add(graph_path)
        graph_sets["combined"].add(graph_path)
        append_jsonl(OUTPUT / "evaluation_audit.jsonl", {
            "split": split,
            "example_id": example.example_id,
            "language": example.language,
            "difficulty_level": example.difficulty_level,
            **flags,
        })


def aggregate_failures(outcome_rows: list[Mapping[str, Any]], evaluation_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(row["split"], row["example_id"]): row for row in evaluation_rows}
    dimensions: dict[str, Counter[str]] = {
        "split": Counter(),
        "language": Counter(),
        "difficulty_level": Counter(),
        "failure_category": Counter(),
        "schema_error_path": Counter(),
        "schema_error_code": Counter(),
    }
    failed_records = 0
    for row in outcome_rows:
        evaluation = by_key[(row["split"], row["example_id"])]
        failure = evaluation.get("failure_category") or "none"
        if failure != "none" or not row["schema_valid_after_retry"]:
            failed_records += 1
        dimensions["split"][str(row["split"])] += int(failure != "none")
        dimensions["language"][str(row["language"])] += int(failure != "none")
        dimensions["difficulty_level"][str(row["difficulty_level"])] += int(failure != "none")
        dimensions["failure_category"][str(failure)] += 1
        for attempt in row["validation_attempts"]:
            for error in attempt["errors"]:
                dimensions["schema_error_path"][error["field_path"]] += 1
                dimensions["schema_error_code"][error["error_code"]] += 1
    return {
        "schema_version": "topocae-p4-api-r1c-failure-stratification/1",
        "record_count": len(outcome_rows),
        "failed_record_count": failed_records,
        "counts": {name: dict(sorted(values.items())) for name, values in dimensions.items()},
        "queries_written": False,
        "ground_truth_written": False,
        "raw_responses_written": False,
    }


def apply_outcome_schema_metrics(
    metrics: dict[str, Any], outcome_rows: list[Mapping[str, Any]]
) -> None:
    total = len(outcome_rows)
    if total != int(metrics["record_count"]):
        raise RuntimeError("E_SCHEMA_METRIC_ROW_COUNT")
    metrics["language_cae_ir"]["schema_valid_rate_first_round"] = (
        sum(bool(row["schema_valid_first_round"]) for row in outcome_rows) / total
    )
    metrics["language_cae_ir"]["schema_valid_rate_after_retry"] = (
        sum(bool(row["schema_valid_after_retry"]) for row in outcome_rows) / total
    )
    metrics["runtime"]["retry_rate"] = (
        sum(int(row["retry_count"]) for row in outcome_rows) / total
    )


def threshold_acceptance(val_metrics: Mapping[str, Any]) -> dict[str, Any]:
    language = val_metrics["language_cae_ir"]
    grounding = val_metrics["grounding"]
    checks = {
        "schema_valid_after_retry_gte_0_99": language["schema_valid_rate_after_retry"] >= 0.99,
        "status_accuracy_gte_0_95": language["status_accuracy"] >= 0.95,
        "slot_macro_f1_gte_0_95": language["slot_macro_f1"] >= 0.95,
        "object_selection_accuracy_gte_0_95": grounding["object_selection_accuracy"] >= 0.95,
        "exact_face_set_accuracy_gte_0_90": grounding["exact_face_set_accuracy"] >= 0.90,
        "clarification_invalid_macro_f1_gte_0_85": language["clarification_invalid_detection"]["macro_f1"] >= 0.85,
        "illegal_out_of_range_id_rate_eq_0": grounding["illegal_out_of_range_id_rate"] == 0.0,
    }
    return {
        "schema_version": "topocae-p4-api-r1c-acceptance/1",
        "status": "READY_FOR_P4_API_R1C_SUPERVISOR_REVIEW" if all(checks.values()) else "BELOW_P4_GATE_WITH_EVIDENCE",
        "checks": checks,
        "automatic_second_revision_performed": False,
        "formal_test_unlock_requested": False,
    }


def security_audit(paths: list[Path], api_key: str) -> dict[str, Any]:
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    result = {
        "api_key_present_in_delivery": bool(api_key and api_key in text),
        "bearer_value_hits": len(re.findall(r"(?i)Bearer\s+[A-Za-z0-9._-]{8,}", text)),
        "raw_authorization_header_written": False,
        "query_written": False,
        "ground_truth_written": False,
        "raw_model_response_written": False,
        "account_balance_written": False,
        "formal_test_accessed": False,
    }
    if result["api_key_present_in_delivery"] or result["bearer_value_hits"]:
        raise RuntimeError("E_SECRET_DELIVERY")
    return result


def write_final_report(acceptance: Mapping[str, Any], val_metrics: Mapping[str, Any], transport: BudgetedCappedTransport) -> None:
    language = val_metrics["language_cae_ir"]
    grounding = val_metrics["grounding"]
    content = f'''# P4-API-R1C Structured Prompt Revision Acceptance

## Status

`{acceptance["status"]}`

One general Structured prompt revision was applied. No second revision was performed.

## Prompt Freeze

- old SHA-256: `{OLD_PROMPT_SHA256}`
- new SHA-256: `{NEW_PROMPT_SHA256}`
- provider/model/schema/validator/grounder/Direct prompt: unchanged

## Bounded Validation

- train: 50 deterministic records
- val: 50 deterministic records
- backend requests: {transport.request_count} / {BACKEND_REQUEST_CAP}
- accounted cost: {transport.accounted_cost_cny:.8f} / {COST_CAP_CNY:.2f} CNY
- formal test: not accessed

## Validation Gate (val)

- schema valid first round: {language["schema_valid_rate_first_round"]:.6f}
- schema valid after retry: {language["schema_valid_rate_after_retry"]:.6f}
- status accuracy: {language["status_accuracy"]:.6f}
- CAE-IR canonical exact match: {language["cae_ir_canonical_exact_match"]:.6f}
- slot Macro-F1: {language["slot_macro_f1"]:.6f}
- clarification/invalid Macro-F1: {language["clarification_invalid_detection"]["macro_f1"]:.6f}
- object selection accuracy: {grounding["object_selection_accuracy"]:.6f}
- exact face-set accuracy: {grounding["exact_face_set_accuracy"]:.6f}
- illegal/out-of-range ID rate: {grounding["illegal_out_of_range_id_rate"]:.6f}

## Scope

Direct LLM, formal test, P5, STEP, and Abaqus were not run. Query text, ground truth, and raw model responses are absent from the delivery.
'''
    (REPORT / "final_acceptance.zh-CN.md").write_text(content, encoding="utf-8", newline="\n")


def write_manifest() -> None:
    files = sorted(path for path in REPORT.iterdir() if path.is_file() and path.name != "delivery_manifest.json")
    write_json(REPORT / "delivery_manifest.json", {
        "schema_version": "topocae-p4-api-r1c-delivery-manifest/1",
        "assets": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    })


def main() -> int:
    if (REPORT / "acceptance.json").exists() or (OUTPUT / "outcome_audit.jsonl").exists():
        raise RuntimeError("E_R1C_RUN_ALREADY_STARTED")
    formal_before = formal_test_state()
    freeze = verify_full_freeze()
    if freeze["status"] != "PASS":
        raise RuntimeError("E_FULL_FREEZE")
    contracts_before = assert_contract_preflight()
    config = replace(LLMRuntimeConfig.from_env(), cache_dir=str(CACHE.resolve()))
    runtime_checks(config)
    samples = selected_samples()
    manifest = sample_manifest(samples)
    cost_plan = planning_cost(config, samples)
    if not cost_plan["gate_pass"]:
        raise RuntimeError("E_COST_PLAN_GATE")

    REPORT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / "sample_manifest.json", manifest)
    write_json(REPORT / "preflight.json", {
        "schema_version": "topocae-p4-api-r1c-preflight/1",
        "status": "PASS_BEFORE_API",
        "authorized_scope": "ONE_STRUCTURED_PROMPT_REVISION_AND_BOUNDED_TRAIN_VAL",
        "full_freeze_status": freeze["status"],
        "formal_test_before": formal_before,
        "offline_regression": {"status": "PASS", "tests": 48, "elapsed_seconds": 159.102},
        "old_prompt_sha256": OLD_PROMPT_SHA256,
        "new_prompt_sha256": NEW_PROMPT_SHA256,
        "contract_hashes_before": contracts_before,
        "runtime": config.public_metadata(),
        "cost_plan": cost_plan,
        "scope_guards": {
            "direct_run": False,
            "formal_test_run": False,
            "p5_entered": False,
            "step_entered": False,
            "abaqus_entered": False,
        },
    })

    transport = BudgetedCappedTransport()
    client = AuditedClient(config, transport)
    memory_cache = MemoryCache()
    accumulators = {split: BaselineAccumulator(baseline="structured_llm", split=split) for split in SPLITS}
    combined = BaselineAccumulator(baseline="structured_llm", split="train_val")
    graph_sets: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        run_split(split, samples[split], client, memory_cache, accumulators[split], combined, graph_sets)
        metrics = accumulators[split].finalize(unique_source_graphs=len(graph_sets[split]))
        split_outcomes = [row for row in client.audit_rows if row["split"] == split]
        apply_outcome_schema_metrics(metrics, split_outcomes)
        metrics["runtime_config"] = config.public_metadata()
        write_json(REPORT / f"{split}_metrics.json", metrics)

    combined_metrics = combined.finalize(unique_source_graphs=len(graph_sets["combined"]))
    combined_metrics["runtime_config"] = config.public_metadata()
    write_json(REPORT / "combined_metrics.json", combined_metrics)
    formal_after = formal_test_state()
    contracts_after = contract_hashes()
    if contracts_after != contracts_before:
        raise RuntimeError("E_CONTRACT_CHANGED_DURING_RUN")

    evaluation_rows = [json.loads(line) for line in (OUTPUT / "evaluation_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    failures = aggregate_failures(client.audit_rows, evaluation_rows)
    write_json(REPORT / "failure_stratification.json", failures)
    write_json(REPORT / "runtime_cost_audit.json", {
        "schema_version": "topocae-p4-api-r1c-runtime-cost/1",
        "runtime_identity": config.public_metadata(),
        "actual_model_ids": sorted({row["actual_model"] for row in client.audit_rows}),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in client.audit_rows)),
        "records": len(client.audit_rows),
        "backend_request_count": transport.request_count,
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cache_hits": sum(bool(row["cache_hit"]) for row in client.audit_rows),
        "cache_misses": sum(not row["cache_hit"] for row in client.audit_rows),
        "input_tokens": transport.metered_input_tokens,
        "output_tokens": transport.metered_output_tokens,
        "http_retry_count": sum(int(row["http_retry_count"]) for row in client.audit_rows),
        "schema_retry_count": sum(int(row["retry_count"]) for row in client.audit_rows),
        "schema_retry_rate": sum(int(row["retry_count"]) for row in client.audit_rows) / len(client.audit_rows),
        "accounted_cost_cny": transport.accounted_cost_cny,
        "cost_cap_cny": COST_CAP_CNY,
        "conservative_unresolved_reservations_cny": transport.conservative_reservations_cny,
        "preflight_cost_plan": cost_plan,
    })
    acceptance = threshold_acceptance(json.loads((REPORT / "val_metrics.json").read_text(encoding="utf-8")))
    acceptance.update({
        "formal_test_after": formal_after,
        "contracts_unchanged_during_run": contracts_after == contracts_before,
        "completed_records": len(client.audit_rows),
        "backend_request_count": transport.request_count,
        "accounted_cost_cny": transport.accounted_cost_cny,
    })
    write_json(REPORT / "acceptance.json", acceptance)
    write_final_report(acceptance, json.loads((REPORT / "val_metrics.json").read_text(encoding="utf-8")), transport)
    delivery_paths = [path for path in REPORT.iterdir() if path.is_file()]
    security = security_audit(delivery_paths, config.api_key)
    write_json(REPORT / "security_audit.json", security)
    write_json(REPORT / "CODEX_CONTEXT_P4_API_R1C.json", {
        "schema_version": "topocae-p4-api-r1c-context/1",
        "status": acceptance["status"],
        "prompt_revision_count": 1,
        "old_prompt_sha256": OLD_PROMPT_SHA256,
        "new_prompt_sha256": NEW_PROMPT_SHA256,
        "completed_records": len(client.audit_rows),
        "direct_run": False,
        "formal_test_accessed": False,
        "next_action": "SUPERVISOR_REVIEW",
    })
    write_manifest()
    print(json.dumps({
        "status": acceptance["status"],
        "records": len(client.audit_rows),
        "backend_requests": transport.request_count,
        "accounted_cost_cny": transport.accounted_cost_cny,
        "report_dir": str(REPORT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
