"""Bounded single-sample schema diagnosis for P4-API-R1B."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.cae_intent import validate_intent_result
from utils.cae_language_grounding.llm_adapter import (
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
OUTPUT = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r1b_schema_diagnosis_v1"
REPORT = ROOT / "reports" / "p4_api_r1b_schema_diagnosis_v1"
CACHE = OUTPUT / "cache"
LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
TARGET_ID = "RF.e8bd1f616e8424683c1c.en-1"
BACKEND_REQUEST_CAP = 6
COST_CAP_CNY = 1.0
USD_CNY_CONSERVATIVE = 8.0
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
EXPECTED_HASHES = {
    STRUCTURED_PROMPT_PATH: "2f6285afc985afe94295df80e44107dc72692d118ca5410a9a74401f964dcc2b",
    INTENT_RESULT_SCHEMA_PATH: "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _formal_test_state() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    run_count = sum(len(rows) for rows in ledger["runs"].values())
    state = {
        "run_count": run_count,
        "content_accessed": bool(ledger["test_content_accessed"]),
        "unlock_authorized": bool(ledger["test_unlock_authorized"]),
    }
    if state != {
        "run_count": 0,
        "content_accessed": False,
        "unlock_authorized": False,
    }:
        raise RuntimeError("E_FORMAL_TEST_GUARD")
    return state


def _frozen_hashes() -> dict[str, str]:
    actual = {str(path.relative_to(ROOT)): _sha256(path) for path in EXPECTED_HASHES}
    expected = {str(path.relative_to(ROOT)): value for path, value in EXPECTED_HASHES.items()}
    if actual != expected:
        raise RuntimeError("E_FROZEN_HASH_MISMATCH")
    return actual


def _sanitize_errors(errors: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    sanitized = []
    for error in errors:
        sanitized.append({
            "error_code": str(error.get("error_code") or error.get("code") or "E_UNKNOWN"),
            "field_path": str(error.get("field_path") or "/"),
        })
    return sanitized


class CappedTransport:
    def __init__(self, transport=_default_transport) -> None:
        self.transport = transport
        self.count = 0
        self.audit_path = OUTPUT / "backend_request_counter.json"
        self._persist()

    def _persist(self) -> None:
        _write_json(self.audit_path, {
            "schema_version": "topocae-p4-api-r1b-backend-counter/1",
            "backend_request_count": self.count,
            "backend_request_cap": BACKEND_REQUEST_CAP,
        })

    def __call__(self, url, headers, payload, timeout):
        if self.count >= BACKEND_REQUEST_CAP:
            raise RuntimeError("E_BACKEND_REQUEST_CAP")
        self.count += 1
        self._persist()
        return self.transport(url, headers, payload, timeout)


class ValidationRecorder:
    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []

    def __call__(self, value: Any) -> Mapping[str, Any]:
        validation = validate_intent_result(value)
        self.attempts.append({
            "attempt": len(self.attempts) + 1,
            "valid": bool(validation.get("valid")),
            "errors": _sanitize_errors(list(validation.get("errors", []))),
        })
        return validation


def _cost_upper_cny(config: LLMRuntimeConfig, inference_input: Mapping[str, Any]) -> float:
    client = OpenAICompatibleJSONClient(config)
    schema = json.loads(INTENT_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
    user_text = json.dumps(dict(inference_input), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    first_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    first = client._payload(first_messages, "topocae_cae_intent_result_v1", schema)
    first_bytes = len(json.dumps(first, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    # One byte per token and a full max-token response are conservative bounds.
    repair_bytes = first_bytes + config.max_tokens * 4 + 4096
    input_tokens_upper = config.max_http_retries + 1
    input_tokens_upper *= first_bytes + repair_bytes
    output_tokens_upper = 2 * (config.max_http_retries + 1) * config.max_tokens
    return (
        input_tokens_upper * INPUT_USD_PER_MILLION
        + output_tokens_upper * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def _actual_cost_cny(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_CONSERVATIVE


def _runtime_checks(config: LLMRuntimeConfig) -> None:
    if not config.available or not config.api_key:
        raise RuntimeError("E_RUNTIME_MISSING")
    if config.provider_profile != "deepseek_json_object":
        raise RuntimeError("E_PROVIDER_PROFILE")
    if config.model != "deepseek-v4-pro":
        raise RuntimeError("E_CONFIGURED_MODEL_MISMATCH")
    if config.max_http_retries != 2:
        raise RuntimeError("E_HTTP_RETRY_POLICY_CHANGED")


def _security_audit(paths: list[Path], api_key: str) -> dict[str, Any]:
    texts = [path.read_text(encoding="utf-8") for path in paths]
    joined = "\n".join(texts)
    result = {
        "api_key_present_in_artifacts": bool(api_key and api_key in joined),
        "authorization_header_present": "authorization" in joined.lower(),
        "ground_truth_written": False,
        "query_written": False,
        "raw_model_response_written": False,
    }
    if result["api_key_present_in_artifacts"] or result["authorization_header_present"]:
        raise RuntimeError("E_SECRET_ARTIFACT")
    return result


def main() -> int:
    result_path = REPORT / "diagnosis.json"
    if result_path.exists():
        raise RuntimeError("E_DIAGNOSIS_ALREADY_EXISTS")
    formal_before = _formal_test_state()
    hashes = _frozen_hashes()
    config = replace(LLMRuntimeConfig.from_env(), cache_dir=str(CACHE.resolve()))
    _runtime_checks(config)

    matches = [item for item in iter_language_examples("val") if item.example_id == TARGET_ID]
    if len(matches) != 1:
        raise RuntimeError("E_TARGET_SAMPLE")
    example = matches[0]
    inference_input = example.inference_input()
    cost_upper = _cost_upper_cny(config, inference_input)
    if cost_upper > COST_CAP_CNY:
        raise RuntimeError("E_COST_UPPER_BOUND")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": "topocae-p4-api-r1b-preflight/1",
        "authorized_scope": "SINGLE_SAMPLE_SCHEMA_DIAGNOSIS_ONLY",
        "target_example_id": TARGET_ID,
        "target_count": 1,
        "provider_profile": config.provider_profile,
        "configured_model": config.model,
        "prompt_schema_hashes": hashes,
        "formal_test_before": formal_before,
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "schema_repair_cap": 1,
        "cost_upper_cny": cost_upper,
        "cost_cap_cny": COST_CAP_CNY,
        "query_written": False,
        "ground_truth_written": False,
    }
    _write_json(REPORT / "preflight.json", preflight)

    transport = CappedTransport()
    client = OpenAICompatibleJSONClient(config, transport=transport)
    recorder = ValidationRecorder()
    schema = json.loads(INTENT_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    prompt = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
    outcome = client.infer_json(
        inference_input=inference_input,
        system_prompt=prompt,
        schema_name="topocae_cae_intent_result_v1",
        schema=schema,
        validator=recorder,
    )
    formal_after = _formal_test_state()
    if outcome.retry_count > 1:
        raise RuntimeError("E_SCHEMA_RETRY_CAP")
    if transport.count > BACKEND_REQUEST_CAP:
        raise RuntimeError("E_BACKEND_REQUEST_CAP")
    if outcome.actual_model != config.model:
        raise RuntimeError("E_ACTUAL_MODEL_MISMATCH")

    diagnosis = {
        "schema_version": "topocae-p4-api-r1b-schema-diagnosis/1",
        "status": "SCHEMA_VALID_AFTER_RETRY" if outcome.schema_valid_after_retry else "SCHEMA_INVALID_AFTER_RETRY",
        "target_example_id": TARGET_ID,
        "runtime_identity": {
            "configured": config.public_metadata(),
            "actual_model_id": outcome.actual_model,
            "finish_reason": outcome.finish_reason,
        },
        "schema": {
            "valid_first_round": outcome.schema_valid_first_round,
            "valid_after_retry": outcome.schema_valid_after_retry,
            "repair_retry_count": outcome.retry_count,
            "failure_code": outcome.failure_code,
            "attempts": recorder.attempts,
        },
        "usage": {
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
        },
        "latency_ms": outcome.latency_ms,
        "http_retry_count": outcome.http_retry_count,
        "backend_request_count": transport.count,
        "backend_request_cap": BACKEND_REQUEST_CAP,
        "cache_hit": outcome.cache_hit,
        "request_hash": outcome.request_hash,
        "metered_cost_cny": _actual_cost_cny(outcome.input_tokens, outcome.output_tokens),
        "conservative_preflight_cost_upper_cny": cost_upper,
        "cost_cap_cny": COST_CAP_CNY,
        "formal_test_after": formal_after,
        "scope": {
            "other_val_samples_run": 0,
            "direct_run": False,
            "formal_test_run": False,
            "p5_entered": False,
            "step_entered": False,
            "abaqus_entered": False,
            "prompt_or_config_modified": False,
        },
    }
    _write_json(result_path, diagnosis)
    security = _security_audit([REPORT / "preflight.json", result_path], config.api_key)
    _write_json(REPORT / "security_audit.json", security)

    files = [REPORT / "preflight.json", result_path, REPORT / "security_audit.json"]
    manifest = {
        "schema_version": "topocae-p4-api-r1b-delivery-manifest/1",
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    _write_json(REPORT / "delivery_manifest.json", manifest)
    print(json.dumps({
        "status": diagnosis["status"],
        "backend_request_count": transport.count,
        "cost_upper_cny": cost_upper,
        "report_dir": str(REPORT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
