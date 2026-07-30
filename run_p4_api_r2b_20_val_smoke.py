"""Bounded P4-API-R2B Compact QueryFrame smoke runner."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import run_p4_api_r2a_1_runtime_prompt as r2a1
from utils.cae_language_grounding.evaluation import canonical_json
from utils.cae_language_grounding.records import LanguageExample, iter_language_examples
from utils.cae_query_frame_runtime import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    NORMAL_ENDPOINT,
    PROFILE_NAME,
    QUERY_FRAME_SCHEMA_SHA256,
    build_deepseek_json_object_request,
    build_query_frame_repair_prompt,
    runtime_prompt_audit,
)


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = (
    ROOT / "reports" / "p4_api_r2a_1_runtime_prompt_v1" / "smoke_sample_manifest.json"
)
OUTPUT = ROOT / "reports" / "p4_api_r2b_20_val_smoke_v2"
WORK = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r2b_20_val_smoke_v2"
AUTHORIZED_BASE_URL = "https://api.deepseek.com"
AUTHORIZED_MODEL = "deepseek-v4-pro"
TEMPERATURE = 0.0
SEED = 20260401
MAX_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS
FIRST_ROUND_CAP = 20
REPAIR_CAP = 20
BACKEND_CAP = 40
HTTP_RETRY_CAP = 0
COST_CAP_CNY = 4.0


class R2BError(RuntimeError):
    pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _frozen_examples() -> tuple[list[LanguageExample], list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("split") != "val" or manifest.get("sample_count") != 20:
        raise R2BError("R2B_E_MANIFEST_CONTRACT")
    candidates = {row["sample_id"]: row for row in r2a1._val_candidates()[0]}
    examples_by_id = {item.example_id: item for item in iter_language_examples("val")}
    examples: list[LanguageExample] = []
    checks: list[dict[str, Any]] = []
    for index, sample in enumerate(manifest["samples"]):
        sample_id = str(sample["sample_id"])
        candidate = candidates.get(sample_id)
        example = examples_by_id.get(sample_id)
        check = {
            "index": index,
            "sample_id": sample_id,
            "found": candidate is not None and example is not None,
            "query_sha256_match": bool(
                candidate and candidate["query_sha256"] == sample["query_sha256"]
            ),
            "ground_truth_sha256_match": bool(
                candidate
                and candidate["ground_truth_sha256"] == sample["ground_truth_sha256"]
            ),
        }
        checks.append(check)
        if not all(check[key] for key in (
            "found", "query_sha256_match", "ground_truth_sha256_match"
        )):
            raise R2BError("R2B_E_SAMPLE_HASH_MISMATCH")
        examples.append(example)
    if len({item.example_id for item in examples}) != 20:
        raise R2BError("R2B_E_SAMPLE_ID_DUPLICATE")
    return examples, checks


def build_preflight(
    environment: Mapping[str, str], *, api_key_configured: bool
) -> tuple[dict[str, Any], list[LanguageExample]]:
    examples, checks = _frozen_examples()
    prompt_audit = runtime_prompt_audit()
    profile = environment.get("TOPOCAE_LLM_PROVIDER_PROFILE", "")
    base_url = environment.get("TOPOCAE_LLM_BASE_URL", "")
    model = environment.get("TOPOCAE_LLM_MODEL", "")
    identity_checks = {
        "provider_profile_match": profile == PROFILE_NAME,
        "base_url_match": base_url == AUTHORIZED_BASE_URL,
        "model_match": model == AUTHORIZED_MODEL,
        "api_key_configured": bool(api_key_configured),
        "beta_absent": "beta" not in base_url.lower(),
        "strict_disabled": True,
        "tool_calling_disabled": True,
        "fallback_disabled": True,
    }
    stable_error_code = None
    if not identity_checks["provider_profile_match"]:
        stable_error_code = "R2B_PREFLIGHT_E_PROVIDER_PROFILE"
    elif not identity_checks["base_url_match"] or not identity_checks["beta_absent"]:
        stable_error_code = "R2B_PREFLIGHT_E_BASE_URL"
    elif not identity_checks["model_match"]:
        stable_error_code = "R2B_PREFLIGHT_E_MODEL"
    elif not identity_checks["api_key_configured"]:
        stable_error_code = "R2B_PREFLIGHT_E_API_KEY_MISSING"
    audit = {
        "schema_version": "topocae-p4-api-r2b-preflight/2",
        "gate_pass": stable_error_code is None,
        "stable_error_code": stable_error_code,
        "provider_profile": profile,
        "base_url": base_url,
        "endpoint": NORMAL_ENDPOINT,
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": TEMPERATURE,
        "seed": SEED,
        "seed_sent": False,
        "max_tokens": MAX_TOKENS,
        "http_retry": HTTP_RETRY_CAP,
        "identity_checks": identity_checks,
        "api_key_configured": bool(api_key_configured),
        "api_key_value_persisted": False,
        "runtime_prompt_sha256": prompt_audit["runtime_prompt_sha256"],
        "query_frame_schema_sha256": QUERY_FRAME_SCHEMA_SHA256,
        "smoke_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "sample_count": len(examples),
        "sample_order_preserved": [item.example_id for item in examples]
        == [item["sample_id"] for item in checks],
        "all_query_sha256_match": all(item["query_sha256_match"] for item in checks),
        "all_ground_truth_sha256_match": all(
            item["ground_truth_sha256_match"] for item in checks
        ),
        "sample_hash_checks": checks,
        "request_caps": {
            "first_round": FIRST_ROUND_CAP,
            "repair": REPAIR_CAP,
            "backend": BACKEND_CAP,
        },
        "cost_cap_cny": COST_CAP_CNY,
    }
    return audit, examples


def build_request(
    inference_input: Mapping[str, Any],
    *,
    repair: tuple[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    request = build_deepseek_json_object_request(
        inference_input, model=AUTHORIZED_MODEL, max_output_tokens=MAX_TOKENS
    )
    if repair is not None:
        error_code, summary = repair
        request["payload"]["messages"][1]["content"] = build_query_frame_repair_prompt(
            inference_input, error_code, summary
        )
    return request


def repair_allowed(phase: str, error_code: str) -> bool:
    return phase in {"json_parse", "frame_validation", "frame_compile"} and (
        error_code.startswith("QF_E_")
    )


class RequestLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.value = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.value = {
                "schema_version": "topocae-p4-api-r2b-request-ledger/2",
                "counters": {
                    "first_round_requests": 0,
                    "repair_requests": 0,
                    "backend_requests": 0,
                    "http_retries": 0,
                },
                "caps": {
                    "first_round_requests": FIRST_ROUND_CAP,
                    "repair_requests": REPAIR_CAP,
                    "backend_requests": BACKEND_CAP,
                    "http_retries": HTTP_RETRY_CAP,
                },
                "entries": [],
            }
            _atomic_write_json(self.path, self.value)

    def reserve(self, *, sample_id: str, stage: str, request_hash: str) -> None:
        if stage not in {"first_round", "repair"}:
            raise R2BError("R2B_E_REQUEST_STAGE")
        if len(request_hash) != 64:
            raise R2BError("R2B_E_REQUEST_HASH")
        if any(
            item["sample_id"] == sample_id and item["stage"] == stage
            for item in self.value["entries"]
        ):
            code = (
                "R2B_E_FIRST_ROUND_SAMPLE_DUPLICATE"
                if stage == "first_round"
                else "R2B_E_REPAIR_SAMPLE_DUPLICATE"
            )
            raise R2BError(code)
        counters = self.value["counters"]
        key = f"{stage}_requests"
        if counters[key] >= self.value["caps"][key]:
            raise R2BError(f"R2B_E_{stage.upper()}_CAP")
        if counters["backend_requests"] >= BACKEND_CAP:
            raise R2BError("R2B_E_BACKEND_CAP")
        counters[key] += 1
        counters["backend_requests"] += 1
        self.value["entries"].append(
            {
                "sequence": counters["backend_requests"],
                "sample_id": sample_id,
                "stage": stage,
                "request_hash": request_hash,
                "reserved_before_call": True,
            }
        )
        _atomic_write_json(self.path, self.value)


def _operations(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("status") != "VALID":
        return []
    return list(value["intent"]["operations"])


def _labels(value: Mapping[str, Any], field: str) -> set[str]:
    result = set()
    for operation in _operations(value):
        if field == "operation_type":
            result.add(str(operation["operation_type"]))
        elif operation.get("target", {}).get(field) is not None:
            result.add(str(operation["target"][field]))
    return result


def _macro_f1(
    expected: Sequence[set[str]], predicted: Sequence[set[str]], classes: Iterable[str]
) -> float:
    scores = []
    for label in classes:
        tp = sum(label in e and label in p for e, p in zip(expected, predicted))
        fp = sum(label not in e and label in p for e, p in zip(expected, predicted))
        fn = sum(label in e and label not in p for e, p in zip(expected, predicted))
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 1.0


def _aligned_operation_pairs(
    expected: Mapping[str, Any], predicted: Mapping[str, Any]
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    actual = {str(item["operation_id"]): item for item in _operations(predicted)}
    for item in _operations(expected):
        yield item, actual.get(str(item["operation_id"]))


def _quantity_signature(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        if {"value", "unit"} <= set(value):
            result.append(
                f"{prefix}:{canonical_json({'value': value['value'], 'unit': value['unit']})}"
            )
        for key, child in value.items():
            result.extend(_quantity_signature(child, f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_quantity_signature(child, f"{prefix}/{index}"))
    return sorted(result)


def compute_semantic_metrics(
    examples: Sequence[LanguageExample], predictions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected_values = [item.oracle_intent_result for item in examples]
    predicted_values = [predictions[item.example_id] for item in examples]
    expected_actions = [_labels(item, "operation_type") for item in expected_values]
    predicted_actions = [_labels(item, "operation_type") for item in predicted_values]
    expected_features = [_labels(item, "feature_type") for item in expected_values]
    predicted_features = [_labels(item, "feature_type") for item in predicted_values]
    action_classes = sorted(set().union(*expected_actions, *predicted_actions))
    feature_classes = sorted(set().union(*expected_features, *predicted_features))
    status_correct = [
        expected.get("status") == predicted.get("status")
        for expected, predicted in zip(expected_values, predicted_values)
    ]
    canonical_exact = [
        canonical_json(expected) == canonical_json(predicted)
        for expected, predicted in zip(expected_values, predicted_values)
    ]
    role_hits = selector_hits = quantity_hits = aligned = 0
    operation_set_hits = 0
    for expected, predicted in zip(expected_values, predicted_values):
        operation_set_hits += int(
            sorted(_labels(expected, "operation_type"))
            == sorted(_labels(predicted, "operation_type"))
        )
        for left, right in _aligned_operation_pairs(expected, predicted):
            if "feature_type" not in left.get("target", {}):
                continue
            aligned += 1
            role_hits += int(
                right is not None
                and right.get("target", {}).get("face_role")
                == left["target"].get("face_role")
            )
            selector_hits += int(
                right is not None
                and right.get("target", {}).get("quantifier")
                == left["target"].get("quantifier")
            )
            quantity_hits += int(
                right is not None
                and _quantity_signature(left) == _quantity_signature(right)
            )
    multi_indices = [
        index for index, value in enumerate(expected_values) if len(_operations(value)) > 1
    ]
    invalid_index = next(
        index for index, value in enumerate(expected_values) if value.get("status") == "INVALID"
    )
    clarification_index = next(
        index
        for index, value in enumerate(expected_values)
        if value.get("status") == "NEEDS_CLARIFICATION"
    )
    total = len(examples)
    return {
        "schema_version": "topocae-p4-api-r2b-semantic-metrics/2",
        "sample_count": total,
        "status_accuracy": sum(status_correct) / total,
        "status_correct_count": sum(status_correct),
        "action_macro_f1": _macro_f1(
            expected_actions, predicted_actions, action_classes
        ),
        "feature_class_macro_f1": _macro_f1(
            expected_features, predicted_features, feature_classes
        ),
        "face_role_accuracy": role_hits / aligned if aligned else 1.0,
        "selector_accuracy": selector_hits / aligned if aligned else 1.0,
        "quantity_value_unit_exact": quantity_hits / aligned if aligned else 1.0,
        "operation_set_exact": operation_set_hits / total,
        "canonical_semantic_exact": sum(canonical_exact) / total,
        "canonical_semantic_exact_count": sum(canonical_exact),
        "multi_operation_exact": (
            sum(canonical_exact[index] for index in multi_indices) / len(multi_indices)
        ),
        "multi_operation_exact_count": sum(
            canonical_exact[index] for index in multi_indices
        ),
        "multi_operation_count": len(multi_indices),
        "invalid_prediction_correct": status_correct[invalid_index],
        "needs_clarification_prediction_correct": status_correct[clarification_index],
    }

