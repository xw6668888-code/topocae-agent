"""Bounded real-API smoke runner for P4-API-R1A.

This runner is intentionally limited to the deterministic first 50 validation
examples. It reuses the frozen P4 prompts, schemas, grounder, verifier, and
evaluators without modifying them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import run_p4_a1_llm_baselines as baseline_runner
from utils.cae_language_grounding.direct_llm import (
    DIRECT_PROMPT_PATH,
    DIRECT_SCHEMA_PATH,
    make_direct_inference_input,
)
from utils.cae_language_grounding.evaluation import MemoryCache
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
OUTPUT = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r1a_smoke_v1"
CACHE = (
    ROOT
    / "output_p4_a1_language_grounding_v1"
    / "p4_api_r1a_smoke_cache_v1"
)
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
SAMPLE_COUNT = 50
BACKEND_REQUEST_CAP = 201
COST_CAP_CNY = 20.0
USD_CNY_CONSERVATIVE = 8.0
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
HEALTH_BACKEND_REQUESTS = 1
HEALTH_INPUT_TOKENS = 63
HEALTH_OUTPUT_TOKENS = 5
HEALTH_REQUEST_HASH = (
    "a14c5e76d32f6cd75cb1f951298785dccfdac73a98eadf16991df656982090bc"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def assert_formal_test_locked() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    run_count = sum(len(rows) for rows in ledger["runs"].values())
    if run_count != 0 or ledger["test_content_accessed"]:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    if ledger["test_unlock_authorized"]:
        raise RuntimeError("E_FORMAL_TEST_UNLOCK_STATE")


class CappedTransport:
    def __init__(self, *, initial_backend_requests: int) -> None:
        self.total = initial_backend_requests
        self.by_stage: Counter[str] = Counter()
        self.stage = "unassigned"
        self.audit_path = OUTPUT / "backend_request_counter.json"
        self._persist()

    def _persist(self) -> None:
        write_json(
            self.audit_path,
            {
                "schema_version": "topocae-p4-api-r1a-backend-counter/1",
                "backend_request_count": self.total,
                "backend_requests_by_stage": dict(self.by_stage),
                "backend_request_cap": BACKEND_REQUEST_CAP,
            },
        )

    def __call__(self, url, headers, payload, timeout):
        if self.total >= BACKEND_REQUEST_CAP:
            raise RuntimeError("E_BACKEND_REQUEST_CAP")
        self.total += 1
        self.by_stage[self.stage] += 1
        self._persist()
        return _default_transport(url, headers, payload, timeout)


class AuditedClient(OpenAICompatibleJSONClient):
    def __init__(self, config: LLMRuntimeConfig, transport: CappedTransport) -> None:
        super().__init__(config, transport=transport)
        self.audit_rows: list[dict[str, Any]] = []
        self.stage = "unassigned"
        self.example_id = "unassigned"

    def infer_json(self, **kwargs) -> JSONInferenceOutcome:
        outcome = super().infer_json(**kwargs)
        row = {
            "stage": self.stage,
            "example_id": self.example_id,
            "schema_valid_first_round": outcome.schema_valid_first_round,
            "schema_valid_after_retry": outcome.schema_valid_after_retry,
            "retry_count": outcome.retry_count,
            "failure_code": outcome.failure_code,
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
        audit_path = OUTPUT / "outcome_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
        if outcome.retry_count > 1:
            raise RuntimeError("E_SCHEMA_RETRY_CAP")
        if not outcome.schema_valid_after_retry:
            raise RuntimeError("E_SCHEMA_INVALID_AFTER_RETRY")
        if outcome.actual_model != self.config.model:
            raise RuntimeError("E_ACTUAL_MODEL_MISMATCH")
        if outcome.finish_reason != "stop":
            raise RuntimeError("E_FINISH_REASON")
        return outcome


def cost_cny(rows: list[Mapping[str, Any]]) -> float:
    input_tokens = HEALTH_INPUT_TOKENS
    output_tokens = HEALTH_OUTPUT_TOKENS
    for row in rows:
        if not row["cache_hit"]:
            input_tokens += int(row["input_tokens"])
            output_tokens += int(row["output_tokens"])
    return (
        input_tokens * INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def payload_size(
    client: OpenAICompatibleJSONClient,
    *,
    inference_input: Mapping[str, Any],
    prompt: str,
    schema_name: str,
    schema: Mapping[str, Any],
) -> int:
    user_text = json.dumps(
        dict(inference_input),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = client._payload(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        schema_name,
        schema,
    )
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def build_plan(config: LLMRuntimeConfig, examples) -> dict[str, Any]:
    planning_client = OpenAICompatibleJSONClient(config)
    memory_cache = MemoryCache()
    structured_prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
    structured_schema = json.loads(
        INTENT_RESULT_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    direct_prompt = DIRECT_PROMPT_PATH.read_text(encoding="utf-8")
    direct_schema = json.loads(DIRECT_SCHEMA_PATH.read_text(encoding="utf-8"))
    sizes: dict[str, list[int]] = {"structured": [], "direct": []}
    for example in examples:
        sizes["structured"].append(
            payload_size(
                planning_client,
                inference_input=example.inference_input(),
                prompt=structured_prompt,
                schema_name="topocae_cae_intent_result_v1",
                schema=structured_schema,
            )
        )
        memory, _ = memory_cache.get(str(example.source_ref["graph_path"]))
        direct_input = make_direct_inference_input(
            text=example.text,
            language=example.language,
            memory=memory,
        )
        sizes["direct"].append(
            payload_size(
                planning_client,
                inference_input=direct_input,
                prompt=direct_prompt,
                schema_name="topocae_direct_selection_v1",
                schema=direct_schema,
            )
        )
    max_output = config.max_tokens
    first_attempt_input_upper = sum(sum(values) for values in sizes.values())
    repair_input_upper = sum(
        sum(size + max_output + 1024 for size in values)
        for values in sizes.values()
    )
    output_upper = 2 * len(examples) * 2 * max_output
    cost_upper = (
        (first_attempt_input_upper + repair_input_upper)
        * INPUT_USD_PER_MILLION
        + output_upper * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE
    ids = [example.example_id for example in examples]
    return {
        "schema_version": "topocae-p4-api-r1a-run-plan/1",
        "selection_policy": "first_50_in_frozen_val_iteration",
        "selected_count": len(examples),
        "sample_ids": ids,
        "sample_id_set_sha256": hashlib.sha256(
            json.dumps(
                ids, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "language_counts": dict(Counter(e.language for e in examples)),
        "difficulty_counts": dict(
            Counter(e.difficulty_level for e in examples)
        ),
        "payload_bytes": {
            name: {
                "total": sum(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in sizes.items()
        },
        "conservative_cost_upper_cny": cost_upper,
        "cost_cap_cny": COST_CAP_CNY,
        "cost_gate_pass": cost_upper <= COST_CAP_CNY,
        "pricing": {
            "source": "https://api-docs.deepseek.com/quick_start/pricing/",
            "input_cache_miss_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "usd_cny_conservative_upper": USD_CNY_CONSERVATIVE,
            "all_input_priced_as_cache_miss": True,
            "schema_attempts_per_item": 2,
            "max_output_tokens_per_attempt": max_output,
        },
        "ground_truth_printed": False,
        "formal_test_accessed": False,
    }


def write_audit(
    *,
    config: LLMRuntimeConfig,
    plan: Mapping[str, Any],
    client: AuditedClient,
    transport: CappedTransport,
    completed_stages: list[str],
) -> None:
    write_json(
        OUTPUT / "run_audit.json",
        {
            "schema_version": "topocae-p4-api-r1a-run-audit/1",
            "status": "IN_PROGRESS" if len(completed_stages) < 2 else "COMPLETE",
            "completed_stages": completed_stages,
            "runtime": config.public_metadata(),
            "health": {
                "backend_request_count": HEALTH_BACKEND_REQUESTS,
                "input_tokens": HEALTH_INPUT_TOKENS,
                "output_tokens": HEALTH_OUTPUT_TOKENS,
                "request_hash": HEALTH_REQUEST_HASH,
                "cache_replay_hit": True,
                "cache_replay_backend_request_count": 0,
            },
            "smoke_outcomes": client.audit_rows,
            "backend_request_count": transport.total,
            "backend_requests_by_stage": dict(transport.by_stage),
            "backend_request_cap": BACKEND_REQUEST_CAP,
            "conservative_actual_cost_cny": cost_cny(client.audit_rows),
            "cost_cap_cny": COST_CAP_CNY,
            "plan_cost_upper_cny": plan["conservative_cost_upper_cny"],
            "formal_test_accessed": False,
        },
    )


def main() -> int:
    assert_formal_test_locked()
    config = replace(
        LLMRuntimeConfig.from_env(),
        cache_dir=str(CACHE.resolve()),
    )
    if not config.available:
        raise RuntimeError("E_RUNTIME_MISSING")
    if config.provider_profile != "deepseek_json_object":
        raise RuntimeError("E_PROVIDER_PROFILE")
    examples = list(
        itertools.islice(iter_language_examples("val"), SAMPLE_COUNT)
    )
    if len(examples) != SAMPLE_COUNT:
        raise RuntimeError("E_SAMPLE_COUNT")
    plan = build_plan(config, examples)
    if not plan["cost_gate_pass"]:
        raise RuntimeError("E_COST_UPPER_BOUND")
    write_json(OUTPUT / "run_plan.json", plan)

    transport = CappedTransport(
        initial_backend_requests=HEALTH_BACKEND_REQUESTS
    )
    client = AuditedClient(config, transport)
    original_iterator = baseline_runner.iter_language_examples
    original_output = baseline_runner.OUTPUT_ROOT
    completed_stages: list[str] = []

    def selected_examples(split: str):
        if split != "val":
            raise RuntimeError("E_UNAUTHORIZED_SPLIT")
        for example in examples:
            client.example_id = example.example_id
            yield example

    baseline_runner.iter_language_examples = selected_examples
    baseline_runner.OUTPUT_ROOT = OUTPUT / "baseline_outputs"
    try:
        client.stage = "structured"
        transport.stage = "structured"
        structured_result = baseline_runner.run_structured("val", client)
        if len([r for r in client.audit_rows if r["stage"] == "structured"]) != 50:
            raise RuntimeError("E_STRUCTURED_COUNT")
        completed_stages.append("structured")
        write_audit(
            config=config,
            plan=plan,
            client=client,
            transport=transport,
            completed_stages=completed_stages,
        )
        if cost_cny(client.audit_rows) > COST_CAP_CNY:
            raise RuntimeError("E_COST_CAP_AFTER_STRUCTURED")

        assert_formal_test_locked()
        client.stage = "direct"
        transport.stage = "direct"
        direct_result = baseline_runner.run_direct("val", client)
        if len([r for r in client.audit_rows if r["stage"] == "direct"]) != 50:
            raise RuntimeError("E_DIRECT_COUNT")
        completed_stages.append("direct")
        write_audit(
            config=config,
            plan=plan,
            client=client,
            transport=transport,
            completed_stages=completed_stages,
        )
    finally:
        baseline_runner.iter_language_examples = original_iterator
        baseline_runner.OUTPUT_ROOT = original_output

    assert_formal_test_locked()
    if transport.total > BACKEND_REQUEST_CAP:
        raise RuntimeError("E_BACKEND_REQUEST_CAP")
    if cost_cny(client.audit_rows) > COST_CAP_CNY:
        raise RuntimeError("E_COST_CAP")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "selected_count": len(examples),
                "sample_id_set_sha256": plan["sample_id_set_sha256"],
                "structured": structured_result,
                "direct": direct_result,
                "backend_request_count": transport.total,
                "backend_request_cap": BACKEND_REQUEST_CAP,
                "conservative_actual_cost_cny": cost_cny(client.audit_rows),
                "cost_cap_cny": COST_CAP_CNY,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
