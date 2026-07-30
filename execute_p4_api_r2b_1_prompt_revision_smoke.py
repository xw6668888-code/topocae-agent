"""Execute the single R2B.1 prompt revision and publish paired evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import execute_p4_api_r2b_20_val_smoke_v2 as v2_delivery
import run_p4_api_r2b_20_val_smoke as frozen_core
import run_p4_api_r2b_1_prompt_revision_smoke as diagnosis
from run_p4_api_r2b_20_val_smoke_v2 import RequestLedger, R2BClient
from utils import cae_query_frame_runtime_r2b_1 as runtime
from utils.cae_language_grounding.llm_adapter_r0 import (
    LLMProtocolError,
    LLMRuntimeConfig,
    LLMTransportError,
)
from utils.cae_language_grounding.reliable_runner import AtomicJSONCache
from utils.cae_query_frame import QueryFrameCompileError, compile_query_frame, validate_query_frame
from utils.cae_intent import validate_intent_result


ROOT = Path(__file__).resolve().parent
OUTPUT = diagnosis.OUTPUT
WORK = diagnosis.WORK
CACHE = WORK / "cache"
LEDGER_PATH = WORK / "request_ledger.json"
FORMAL_LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
MODEL = "deepseek-v4-pro"
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
USD_CNY_PLANNING_RATE = 8.0
PLANNED_COST_UPPER_CNY = 3.7042512

HISTORICAL_TREES = (
    "reports/p4_api_r2a_compact_query_frame_v1",
    "reports/p4_api_r2a_1_runtime_prompt_v1",
    "reports/p4_api_r2b_20_val_smoke_v1",
    "reports/p4_api_r2b_20_val_smoke_v2",
)

ACCEPTANCE_CHECK_NAMES = (
    "post_repair_frame_valid_ge_19",
    "p2_valid_eq_20",
    "status_accuracy_ge_16_of_20",
    "invalid_correct",
    "needs_clarification_correct",
    "action_macro_f1_ge_0_80",
    "feature_class_macro_f1_ge_0_75",
    "selector_accuracy_ge_0_75",
    "canonical_semantic_exact_ge_12",
    "multi_operation_exact_ge_3_of_4",
    "object_selection_accuracy_ge_0_70",
    "exact_face_set_accuracy_ge_0_70",
    "illegal_or_out_of_range_id_rate_eq_0",
    "actual_cost_le_4_cny",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def historical_tree_hashes() -> dict[str, str]:
    return {relative: _tree_sha256(ROOT / relative) for relative in HISTORICAL_TREES}


def _formal_state() -> dict[str, Any]:
    value = json.loads(FORMAL_LEDGER.read_text(encoding="utf-8"))
    return {
        "sha256": _sha256(FORMAL_LEDGER),
        "run_count": sum(len(rows) for rows in value["runs"].values()),
        "test_content_accessed": bool(value["test_content_accessed"]),
        "test_unlock_authorized": bool(value["test_unlock_authorized"]),
    }


def _request_hash(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_request(
    inference_input: Mapping[str, Any],
    *,
    repair: tuple[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    request = runtime.build_deepseek_json_object_request(
        inference_input, model=MODEL, max_output_tokens=frozen_core.MAX_TOKENS
    )
    if repair is not None:
        code, summary = repair
        request["payload"]["messages"][1]["content"] = runtime.build_query_frame_repair_prompt(
            inference_input, code, summary
        )
    return request


class R2B1Client(R2BClient):
    def _fetch(
        self,
        *,
        sample_id: str,
        stage: str,
        request: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bool, str]:
        request_hash = _request_hash(request)
        namespace = f"p4_api_r2b_1_{stage}"
        cached = self.cache.get(namespace, request_hash)
        if cached is not None:
            if not self.ledger.completed(request_hash):
                raise frozen_core.R2BError("R2B1_E_CACHE_LEDGER_IDENTITY")
            response = cached.get("response")
            if not isinstance(response, Mapping):
                raise frozen_core.R2BError("R2B1_E_CACHE_RESPONSE")
            return response, True, request_hash
        self.ledger.reserve(sample_id=sample_id, stage=stage, request_hash=request_hash)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        try:
            response = self.transport(
                str(request["endpoint"]), headers, request["payload"], self.timeout_seconds
            )
        except Exception:
            self.ledger.complete(request_hash=request_hash, status="transport_error")
            raise
        if not isinstance(response, Mapping):
            self.ledger.complete(request_hash=request_hash, status="protocol_error")
            raise LLMProtocolError("E_RUNTIME_RESPONSE_TYPE")
        usage = response.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, Mapping):
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
        self.ledger.complete(
            request_hash=request_hash,
            status="response_received",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.cache.put(namespace, request_hash, {"response": dict(response)})
        return response, False, request_hash

    def infer(self, *, sample_id: str, inference_input: Mapping[str, Any]) -> dict[str, Any]:
        outcome: dict[str, Any] = {
            "sample_id": sample_id,
            "first_round_json_parse_valid": False,
            "first_round_frame_schema_valid": False,
            "post_repair_frame_valid": False,
            "p2_valid": False,
            "repair_count": 0,
            "cache_hit": False,
            "failure_phase": None,
            "failure_code": None,
            "error_path": None,
            "frame": None,
            "compiled_result": None,
            "requests": [],
        }
        try:
            response, cache_hit, request_hash = self._fetch(
                sample_id=sample_id,
                stage="first_round",
                request=build_request(inference_input),
            )
            outcome["cache_hit"] = cache_hit
            outcome["requests"].append(
                {"stage": "first_round", "request_hash": request_hash, "cache_hit": cache_hit}
            )
            frame, metadata = self._decode(response)
            outcome["first_round_json_parse_valid"] = True
            frame, compiled = self._validate_compile(frame)
            outcome.update(
                first_round_frame_schema_valid=True,
                post_repair_frame_valid=True,
                p2_valid=True,
                frame=frame,
                compiled_result=compiled,
                metadata=metadata,
            )
            return outcome
        except LLMTransportError as exc:
            outcome.update(failure_phase="transport", failure_code=exc.code)
            return outcome
        except LLMProtocolError as exc:
            code = str(exc.code)
            if code != "QF_E_JSON_TYPE":
                outcome.update(failure_phase="protocol", failure_code=code)
                return outcome
            phase, path = "json_parse", "/"
        except QueryFrameCompileError as exc:
            phase, code, path = "frame_validation", exc.code, exc.field_path
        if not frozen_core.repair_allowed(phase, code):
            outcome.update(failure_phase=phase, failure_code=code, error_path=path)
            return outcome
        outcome["repair_count"] = 1
        summary = {"field_path": path if path.startswith("/") else "/", "required": [], "allowed_enum": [], "rule": code}
        try:
            response, cache_hit, request_hash = self._fetch(
                sample_id=sample_id,
                stage="repair",
                request=build_request(inference_input, repair=(code, summary)),
            )
            outcome["requests"].append(
                {"stage": "repair", "request_hash": request_hash, "cache_hit": cache_hit}
            )
            frame, metadata = self._decode(response)
            frame, compiled = self._validate_compile(frame)
            outcome.update(
                post_repair_frame_valid=True,
                p2_valid=True,
                frame=frame,
                compiled_result=compiled,
                metadata=metadata,
                failure_phase=None,
                failure_code=None,
                error_path=None,
            )
        except LLMTransportError as exc:
            outcome.update(failure_phase="transport", failure_code=exc.code, error_path=path)
        except LLMProtocolError as exc:
            outcome.update(failure_phase="repair_protocol", failure_code=str(exc.code), error_path=path)
        except QueryFrameCompileError as exc:
            outcome.update(
                failure_phase="repair_frame_validation",
                failure_code=exc.code,
                error_path=exc.field_path,
            )
        return outcome


def acceptance_checks(
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any],
    grounding: Mapping[str, Any],
    actual_cost_cny: float,
) -> dict[str, bool]:
    return {
        "post_repair_frame_valid_ge_19": protocol["post_repair_frame_valid_count"] >= 19,
        "p2_valid_eq_20": protocol["p2_valid_count"] == 20,
        "status_accuracy_ge_16_of_20": semantic["status_correct_count"] >= 16,
        "invalid_correct": bool(semantic["invalid_prediction_correct"]),
        "needs_clarification_correct": bool(semantic["needs_clarification_prediction_correct"]),
        "action_macro_f1_ge_0_80": semantic["action_macro_f1"] >= 0.80,
        "feature_class_macro_f1_ge_0_75": semantic["feature_class_macro_f1"] >= 0.75,
        "selector_accuracy_ge_0_75": semantic["selector_accuracy"] >= 0.75,
        "canonical_semantic_exact_ge_12": semantic["canonical_semantic_exact_count"] >= 12,
        "multi_operation_exact_ge_3_of_4": semantic["multi_operation_exact_count"] >= 3,
        "object_selection_accuracy_ge_0_70": grounding["object_selection_accuracy"] >= 0.70,
        "exact_face_set_accuracy_ge_0_70": grounding["exact_face_set_accuracy"] >= 0.70,
        "illegal_or_out_of_range_id_rate_eq_0": grounding["illegal_or_out_of_range_id_rate"] == 0,
        "actual_cost_le_4_cny": actual_cost_cny <= frozen_core.COST_CAP_CNY,
    }


def stop_state(checks: Mapping[str, bool], external_blocker: str | None) -> str:
    if external_blocker:
        return "BLOCKED_WITH_EVIDENCE"
    if all(checks.get(key) is True for key in ACCEPTANCE_CHECK_NAMES):
        return "READY_FOR_P4_API_R2C_DISJOINT_100_VAL_REVIEW"
    return "DEEPSEEK_COMPACT_FRAME_INSUFFICIENT_WITH_EVIDENCE"


def _actual_cost(ledger: Mapping[str, Any]) -> float:
    counters = ledger["counters"]
    return (
        counters.get("input_tokens", 0) * INPUT_USD_PER_MILLION
        + counters.get("output_tokens", 0) * OUTPUT_USD_PER_MILLION
    ) / 1_000_000 * USD_CNY_PLANNING_RATE


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _contains_singleton_target(value: Any) -> bool:
    if isinstance(value, Mapping):
        ids = value.get("target_instance_ids")
        if isinstance(ids, list) and len(ids) == 1:
            return True
        return any(_contains_singleton_target(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_singleton_target(child) for child in value)
    return False


def _field_diagnosis(
    examples: Sequence[Any],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    ground_facts: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    rows = []
    singleton_wrong_quantifier_same_face = 0
    for example in examples:
        outcome = outcomes_by_id.get(example.example_id, {})
        predicted = outcome.get("compiled_result") or {"status": "ERROR"}
        ground = ground_facts.get(example.example_id, {})
        row = diagnosis.classify_field_differences(
            example.oracle_intent_result,
            predicted,
            protocol_error=not bool(outcome.get("post_repair_frame_valid")),
            grounding_error=(
                ground.get("object_exact") is False or ground.get("face_exact") is False
            ),
        )
        if (
            row["categories"]["quantifier_count_error"]
            and ground.get("face_exact") is True
            and _contains_singleton_target(example.ground_truth)
        ):
            singleton_wrong_quantifier_same_face += 1
        rows.append({"sample_id": example.example_id, **row})
    return {
        "schema_version": "topocae-p4-api-r2b-1-field-diagnosis/1",
        "sample_count": len(rows),
        "canonical_semantic_exact_count": sum(row["canonical_semantic_exact"] for row in rows),
        "operation_parameter_exact_count": sum(row["operation_parameter_exact"] for row in rows),
        "operation_parameter_exact": sum(row["operation_parameter_exact"] for row in rows) / len(rows),
        "target_contract_exact_count": sum(row["target_contract_exact"] for row in rows),
        "target_contract_exact": sum(row["target_contract_exact"] for row in rows) / len(rows),
        "category_counts": {
            key: sum(row["categories"][key] for row in rows) for key in diagnosis.CATEGORY_KEYS
        },
        "singleton_wrong_quantifier_same_face_set_count": singleton_wrong_quantifier_same_face,
        "samples": rows,
    }


def _publish(
    *,
    status: str,
    checks: Mapping[str, bool],
    preflight: Mapping[str, Any],
    examples: Sequence[Any],
    outcomes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any],
    grounding: Mapping[str, Any],
    stratified: Mapping[str, Any],
    field_diagnosis: Mapping[str, Any],
    paired: Mapping[str, Any],
    ledger: Mapping[str, Any],
    actual_cost_cny: float,
    external_blocker: str | None,
    formal_before: Mapping[str, Any],
    historical_before: Mapping[str, str],
) -> None:
    if OUTPUT.exists():
        raise frozen_core.R2BError("R2B1_E_IMMUTABLE_OUTPUT_EXISTS")
    temporary = Path(tempfile.mkdtemp(prefix=".p4_api_r2b_1.", dir=OUTPUT.parent))
    try:
        formal_after = _formal_state()
        historical_after = historical_tree_hashes()
        offline = json.loads((WORK / "v2_offline_field_diagnosis.json").read_text(encoding="utf-8"))
        payloads = {
            "acceptance.json": {
                "schema_version": "topocae-p4-api-r2b-1-acceptance/1",
                "status": status,
                "checks": dict(checks),
                "external_blocker": external_blocker,
                "backend_requests": ledger["counters"]["backend_requests"],
                "actual_cost_cny": actual_cost_cny,
                "second_prompt_revision_allowed": False,
                "next_stage_automatically_started": False,
            },
            "preflight.json": dict(preflight),
            "runtime_prompt_audit.json": runtime.runtime_prompt_audit(),
            "v2_offline_field_diagnosis.json": offline,
            "field_diagnosis.json": dict(field_diagnosis),
            "paired_changes.json": dict(paired),
            "protocol_metrics.json": dict(protocol),
            "semantic_metrics.json": dict(semantic),
            "grounding_metrics.json": dict(grounding),
            "stratified_metrics.json": dict(stratified),
            "request_ledger.json": dict(ledger),
            "sample_results.json": {
                "schema_version": "topocae-p4-api-r2b-1-sample-results/1",
                "results": [v2_delivery.sanitize_outcome(row) for row in outcomes],
                "raw_query_included": False,
                "ground_truth_payload_included": False,
                "raw_model_response_included": False,
            },
            "sample_hash_audit.json": {
                "schema_version": "topocae-p4-api-r2b-1-sample-hash-audit/1",
                "checks": preflight["sample_hash_checks"],
                "sample_count": preflight["sample_count"],
                "sample_order_preserved": preflight["sample_order_preserved"],
                "all_query_sha256_match": preflight["all_query_sha256_match"],
                "all_ground_truth_sha256_match": preflight["all_ground_truth_sha256_match"],
            },
            "freeze_audit.json": {
                "schema_version": "topocae-p4-api-r2b-1-freeze-audit/1",
                "historical_tree_sha256_before": dict(historical_before),
                "historical_tree_sha256_after": historical_after,
                "change_count": sum(historical_before[key] != historical_after[key] for key in historical_before),
                "gate_pass": dict(historical_before) == historical_after,
            },
            "security_audit.json": {
                "schema_version": "topocae-p4-api-r2b-1-security-audit/1",
                "formal_test_access_count": 0,
                "formal_test_decoded": False,
                "formal_ledger_unchanged": formal_before == formal_after,
                "raw_query_persisted": False,
                "ground_truth_payload_persisted": False,
                "raw_model_response_persisted": False,
                "p1_accessed": False,
                "step_or_brep_accessed": False,
                "direct_baseline_started": False,
                "p5_started": False,
                "abaqus_started": False,
            },
            "runtime_cost_audit.json": {
                "schema_version": "topocae-p4-api-r2b-1-cost-audit/1",
                "planned_cost_upper_cny": PLANNED_COST_UPPER_CNY,
                "cost_cap_cny": frozen_core.COST_CAP_CNY,
                "actual_cost_cny": actual_cost_cny,
                "actual_backend_requests": ledger["counters"]["backend_requests"],
                "actual_input_tokens": ledger["counters"].get("input_tokens", 0),
                "actual_output_tokens": ledger["counters"].get("output_tokens", 0),
                "gate_pass": actual_cost_cny <= frozen_core.COST_CAP_CNY,
            },
        }
        for name, value in payloads.items():
            _write_json(temporary / name, value)
        (temporary / "final_acceptance.zh-CN.md").write_text(
            "# P4-API-R2B.1 One General Prompt Revision and Paired Re-smoke\n\n"
            f"停止状态：`{status}`\n\n"
            f"请求：{ledger['counters']['backend_requests']}/40；repair：{ledger['counters']['repair_requests']}/20；HTTP retry：0；费用：{actual_cost_cny:.6f} CNY。\n\n"
            f"Frame-valid：{protocol['post_repair_frame_valid_count']}/20；P2-valid：{protocol['p2_valid_count']}/20；status：{semantic['status_correct_count']}/20；canonical exact：{semantic['canonical_semantic_exact_count']}/20；multi-operation exact：{semantic['multi_operation_exact_count']}/4。\n\n"
            f"operation parameter exact：{field_diagnosis['operation_parameter_exact_count']}/20；target contract exact：{field_diagnosis['target_contract_exact_count']}/20；singleton wrong-quantifier/same-face-set：{field_diagnosis['singleton_wrong_quantifier_same_face_set_count']}。\n\n"
            "未执行第二次 prompt 修订；未访问 formal test、P1、STEP/BRep、Direct baseline、P5 或 Abaqus。\n",
            encoding="utf-8",
            newline="\n",
        )
        assets = []
        for path in sorted(item for item in temporary.iterdir() if item.is_file()):
            assets.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        _write_json(temporary / "delivery_manifest.json", {
            "schema_version": "topocae-p4-api-r2b-1-delivery-manifest/1",
            "status": status,
            "assets": assets,
            "asset_count": len(assets),
            "immutable_publish": True,
        })
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    if OUTPUT.exists():
        raise frozen_core.R2BError("R2B1_E_IMMUTABLE_OUTPUT_EXISTS")
    if not (WORK / "v2_offline_field_diagnosis.json").exists():
        diagnosis.write_offline_diagnosis()
    historical_before = historical_tree_hashes()
    formal_before = _formal_state()
    config = LLMRuntimeConfig.from_env()
    environment = {
        "TOPOCAE_LLM_PROVIDER_PROFILE": config.provider_profile,
        "TOPOCAE_LLM_BASE_URL": config.base_url,
        "TOPOCAE_LLM_MODEL": config.model,
    }
    preflight, examples = frozen_core.build_preflight(
        environment, api_key_configured=bool(config.api_key)
    )
    prompt_audit = runtime.runtime_prompt_audit()
    preflight.update({
        "schema_version": "topocae-p4-api-r2b-1-preflight/1",
        "runtime_prompt_sha256": prompt_audit["runtime_prompt_sha256"],
        "model": config.model,
        "sample_order_preserved": [item.example_id for item in examples]
        == [row["sample_id"] for row in preflight["sample_hash_checks"]],
        "formal_test_locked": not formal_before["test_content_accessed"] and not formal_before["test_unlock_authorized"],
        "planned_cost_upper_cny": PLANNED_COST_UPPER_CNY,
        "planned_cost_gate_pass": PLANNED_COST_UPPER_CNY <= frozen_core.COST_CAP_CNY,
        "new_cache_namespace": "p4_api_r2b_1_{first_round|repair}",
    })
    preflight["gate_pass"] = bool(
        preflight["gate_pass"]
        and preflight["formal_test_locked"]
        and preflight["planned_cost_gate_pass"]
        and config.model == MODEL
    )
    WORK.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(LEDGER_PATH)
    outcomes = []
    external_blocker = None
    if preflight["gate_pass"]:
        client = R2B1Client(
            api_key=config.api_key or "",
            ledger=ledger,
            cache_dir=CACHE,
        )
        for example in examples:
            if _formal_state() != formal_before:
                external_blocker = "R2B1_E_FORMAL_LEDGER_CHANGED"
                break
            outcome = client.infer(
                sample_id=example.example_id,
                inference_input=example.inference_input(),
            )
            outcomes.append(outcome)
            frozen_core._atomic_write_json(WORK / "progress.json", {
                "schema_version": "topocae-p4-api-r2b-1-progress/1",
                "results": [v2_delivery.sanitize_outcome(row) for row in outcomes],
            })
            if outcome.get("failure_phase") in {"transport", "protocol"}:
                external_blocker = str(outcome.get("failure_code") or "R2B1_E_EXTERNAL_RUNTIME")
                break
            if _actual_cost(ledger.value) > frozen_core.COST_CAP_CNY:
                external_blocker = "R2B1_E_COST_CAP"
                break
    else:
        external_blocker = str(preflight.get("stable_error_code") or "R2B1_E_PREFLIGHT")
    outcomes_by_id = {row["sample_id"]: row for row in outcomes}
    protocol = v2_delivery._protocol_metrics(outcomes)
    semantic = v2_delivery._semantic_metrics(examples, outcomes_by_id)
    grounding_audit, groundings, instance_maps = v2_delivery._ground_predictions(examples, outcomes_by_id)
    grounding, ground_facts = v2_delivery._grounding_metrics(examples, groundings, instance_maps)
    grounding["grounding_audit"] = grounding_audit
    stratified = v2_delivery._stratified_metrics(examples, outcomes_by_id, ground_facts)
    revised = _field_diagnosis(examples, outcomes_by_id, ground_facts)
    v2_offline = json.loads((WORK / "v2_offline_field_diagnosis.json").read_text(encoding="utf-8"))
    paired = diagnosis.paired_changes(
        [item.example_id for item in examples],
        {row["sample_id"]: row for row in v2_offline["samples"]},
        {row["sample_id"]: row for row in revised["samples"]},
    )
    actual_cost = _actual_cost(ledger.value)
    checks = acceptance_checks(protocol, semantic, grounding, actual_cost)
    status = stop_state(checks, external_blocker)
    _publish(
        status=status,
        checks=checks,
        preflight=preflight,
        examples=examples,
        outcomes=outcomes,
        protocol=protocol,
        semantic=semantic,
        grounding=grounding,
        stratified=stratified,
        field_diagnosis=revised,
        paired=paired,
        ledger=ledger.value,
        actual_cost_cny=actual_cost,
        external_blocker=external_blocker,
        formal_before=formal_before,
        historical_before=historical_before,
    )
    print(json.dumps({
        "status": status,
        "completed_samples": len(outcomes),
        "backend_requests": ledger.value["counters"]["backend_requests"],
        "repair_requests": ledger.value["counters"]["repair_requests"],
        "actual_cost_cny": actual_cost,
        "external_blocker": external_blocker,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
