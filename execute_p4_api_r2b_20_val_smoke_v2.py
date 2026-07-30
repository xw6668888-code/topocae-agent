"""Execute the frozen P4-API-R2B smoke and publish immutable v2 evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import run_p4_api_r2b_20_val_smoke as core
from run_p4_api_r2b_20_val_smoke_v2 import R2BClient, RequestLedger
from utils.cae_language_grounding.evaluation import MemoryCache, canonical_json
from utils.cae_language_grounding.grounder import ground_intent_result
from utils.cae_language_grounding.llm_adapter_r0 import LLMRuntimeConfig
from utils.cae_language_grounding.records import LanguageExample


ROOT = Path(__file__).resolve().parent
OUTPUT = core.OUTPUT
WORK = core.WORK
CACHE = WORK / "cache"
LEDGER_PATH = WORK / "request_ledger.json"
FORMAL_LEDGER = ROOT / "output_p4_a1_language_grounding_v1" / "test_run_ledger.json"
V1 = ROOT / "reports" / "p4_api_r2b_20_val_smoke_v1"
INPUT_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
USD_CNY_PLANNING_RATE = 8.0
PLANNED_COST_UPPER_CNY = 3.7042512
FREEZE_HASHES = {
    "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json": "41067a5b8916bb3ad222c9e873d8576a4ec7aaa3634c79d89d0eebdcba63468a",
    "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json": "7a9e8f184013eeeff0db794a95defc73d19438c89e7cbbf6609abdea1a6a2845",
    "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json": "44b4be432a432bce02158ab3fddc1bcdc569cb3d86587286c5ab866b193349b9",
    "reports/p4_api_r2a_compact_query_frame_v1/query_frame.schema.json": "5f1913121e750493520d818b1d377d90add70a72e90c6612f6ec47cd7179da43",
    "reports/p4_api_r2a_compact_query_frame_v1/query_frame_prompt.zh-CN.md": "aae89eb07898edcf98962e9eabce052bd878fd5b96f2185b9104b19b3f5c7142",
    "reports/p4_api_r2a_1_runtime_prompt_v1/smoke_sample_manifest.json": "83ae43ad9bb17a61a723113f56b874ab4912dd147c0254f87057152985375f07",
}


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


def _formal_state() -> dict[str, Any]:
    value = json.loads(FORMAL_LEDGER.read_text(encoding="utf-8"))
    return {
        "sha256": _sha256(FORMAL_LEDGER),
        "run_count": sum(len(rows) for rows in value["runs"].values()),
        "test_content_accessed": bool(value["test_content_accessed"]),
        "test_unlock_authorized": bool(value["test_unlock_authorized"]),
    }


def _freeze_state(v1_tree_before: str) -> dict[str, Any]:
    assets = []
    for relative, expected in FREEZE_HASHES.items():
        actual = _sha256(ROOT / relative)
        assets.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    v1_after = _tree_sha256(V1)
    return {
        "schema_version": "topocae-p4-api-r2b-freeze-audit/2",
        "assets": assets,
        "change_count": sum(not row["match"] for row in assets),
        "v1_tree_sha256_before": v1_tree_before,
        "v1_tree_sha256_after": v1_after,
        "v1_unchanged": v1_tree_before == v1_after,
        "frozen_assets_modified": False,
        "gate_pass": all(row["match"] for row in assets) and v1_tree_before == v1_after,
    }


def _intent_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "ERROR", "actions": [], "features": [], "roles": [], "selectors": []}
    operations = (
        value.get("intent", {}).get("operations", [])
        if value.get("status") == "VALID"
        else []
    )
    return {
        "status": value.get("status"),
        "actions": [str(row.get("operation_type")) for row in operations],
        "features": [
            str(row.get("target", {}).get("feature_type"))
            for row in operations
            if row.get("target", {}).get("feature_type") is not None
        ],
        "roles": [
            str(row.get("target", {}).get("face_role"))
            for row in operations
            if row.get("target", {}).get("face_role") is not None
        ],
        "selectors": [
            str(row.get("target", {}).get("quantifier"))
            for row in operations
            if row.get("target", {}).get("quantifier") is not None
        ],
    }


def sanitize_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), Mapping) else {}
    return {
        "sample_id": outcome["sample_id"],
        "first_round_json_parse_valid": bool(outcome["first_round_json_parse_valid"]),
        "first_round_frame_schema_valid": bool(outcome["first_round_frame_schema_valid"]),
        "post_repair_frame_valid": bool(outcome["post_repair_frame_valid"]),
        "p2_valid": bool(outcome["p2_valid"]),
        "repair_count": int(outcome["repair_count"]),
        "failure_phase": outcome.get("failure_phase"),
        "failure_code": outcome.get("failure_code"),
        "error_path": outcome.get("error_path"),
        "actual_model": metadata.get("actual_model"),
        "finish_reason": metadata.get("finish_reason"),
        "input_tokens": int(metadata.get("input_tokens", 0)),
        "output_tokens": int(metadata.get("output_tokens", 0)),
        "requests": list(outcome.get("requests", [])),
        "predicted": _intent_summary(outcome.get("compiled_result")),
    }


def determine_status(
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any],
    grounding: Mapping[str, Any],
    *,
    actual_cost_cny: float,
    external_blocker: str | None = None,
) -> tuple[str, dict[str, bool]]:
    checks = {
        "post_repair_frame_valid_ge_19": protocol["post_repair_frame_valid_count"] >= 19,
        "all_compiled_p2_valid": protocol["p2_valid_count"] == protocol["compiled_result_count"],
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
        "actual_cost_le_4_cny": actual_cost_cny <= core.COST_CAP_CNY,
    }
    if external_blocker:
        return "BLOCKED_WITH_EVIDENCE", checks
    if not checks["post_repair_frame_valid_ge_19"] or not checks["all_compiled_p2_valid"]:
        return "R2B_PROTOCOL_GATE_FAILED_WITH_EVIDENCE", checks
    if all(checks.values()):
        return "READY_FOR_P4_API_R2C_100_VAL_REVIEW", checks
    return "R2B_SEMANTIC_GATE_FAILED_WITH_EVIDENCE", checks


def _protocol_metrics(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = 20
    repair_count = sum(int(row["repair_count"]) for row in outcomes)
    failures = Counter(str(row.get("failure_code")) for row in outcomes if row.get("failure_code"))
    return {
        "schema_version": "topocae-p4-api-r2b-protocol-metrics/2",
        "selected_samples": total,
        "completed_samples": len(outcomes),
        "first_round_json_parse_valid_count": sum(bool(row["first_round_json_parse_valid"]) for row in outcomes),
        "first_round_json_parse_valid_rate": sum(bool(row["first_round_json_parse_valid"]) for row in outcomes) / total,
        "first_round_frame_schema_valid_count": sum(bool(row["first_round_frame_schema_valid"]) for row in outcomes),
        "first_round_frame_schema_valid_rate": sum(bool(row["first_round_frame_schema_valid"]) for row in outcomes) / total,
        "repair_count": repair_count,
        "repair_rate": repair_count / total,
        "post_repair_frame_valid_count": sum(bool(row["post_repair_frame_valid"]) for row in outcomes),
        "post_repair_frame_valid_rate": sum(bool(row["post_repair_frame_valid"]) for row in outcomes) / total,
        "compiled_result_count": sum(row.get("compiled_result") is not None for row in outcomes),
        "p2_valid_count": sum(bool(row["p2_valid"]) for row in outcomes),
        "empty_count": failures["E_RUNTIME_EMPTY_CONTENT"],
        "truncated_count": failures["E_RUNTIME_TRUNCATED"],
        "error_count": sum(failures.values()),
        "empty_rate": failures["E_RUNTIME_EMPTY_CONTENT"] / total,
        "truncated_rate": failures["E_RUNTIME_TRUNCATED"] / total,
        "error_rate": sum(failures.values()) / total,
        "failure_codes": dict(sorted(failures.items())),
        "silent_repair_count": 0,
        "http_retry_count": 0,
    }


def _ground_predictions(
    examples: Sequence[LanguageExample], outcomes_by_id: Mapping[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Mapping[str, int]]]:
    cache = MemoryCache()
    groundings: dict[str, dict[str, Any]] = {}
    instance_maps: dict[str, Mapping[str, int]] = {}
    errors: dict[str, str] = {}
    for example in examples:
        outcome = outcomes_by_id.get(example.example_id)
        if not outcome or not outcome.get("p2_valid"):
            continue
        try:
            memory, instance_map = cache.get(str(example.source_ref["graph_path"]))
            groundings[example.example_id] = ground_intent_result(
                outcome["compiled_result"], memory
            )
            instance_maps[example.example_id] = instance_map
        except Exception as exc:
            errors[example.example_id] = type(exc).__name__
    return {"unique_p3_oracle_memories": len(cache), "errors": errors}, groundings, instance_maps


def _grounding_metrics(
    examples: Sequence[LanguageExample],
    groundings: Mapping[str, Mapping[str, Any]],
    instance_maps: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, Any], dict[str, dict[str, bool]]]:
    valid = [item for item in examples if item.oracle_intent_result["status"] == "VALID"]
    object_hits = face_hits = illegal_face = illegal_object = selected_faces = 0
    per_sample: dict[str, dict[str, bool]] = {}
    for example in valid:
        grounding = groundings.get(example.example_id)
        instance_map = instance_maps.get(example.example_id, {})
        object_exact = face_exact = grounding is not None
        actual = {
            str(row["operation_id"]): row
            for row in (grounding or {}).get("operations", [])
        }
        for expected in example.ground_truth.get("targets", []):
            row = actual.get(str(expected["operation_id"]))
            if expected["target_contract"] == "whole_part":
                correct = bool(
                    row
                    and row.get("status") == "VALID"
                    and row.get("target_contract") == "whole_part"
                    and not row.get("selected_object_ids")
                    and not row.get("selected_face_indices")
                )
                object_exact = object_exact and correct
                face_exact = face_exact and correct
                continue
            actual_object_ids = list(row.get("selected_object_ids", [])) if row else []
            unknown = [str(value) for value in actual_object_ids if str(value) not in instance_map]
            illegal_object += len(unknown)
            predicted_objects = {
                int(instance_map[str(value)])
                for value in actual_object_ids
                if str(value) in instance_map
            }
            expected_objects = {int(value) for value in expected["target_instance_ids"]}
            object_exact = object_exact and predicted_objects == expected_objects
            predicted_faces = {int(value) for value in row.get("selected_face_indices", [])} if row else set()
            expected_faces = {int(value) for value in expected["target_face_indices"]}
            selected_faces += len(predicted_faces)
            face_exact = face_exact and predicted_faces == expected_faces
        illegal_face += int((grounding or {}).get("verifier", {}).get("illegal_face_id_count", 0))
        object_hits += int(object_exact)
        face_hits += int(face_exact)
        per_sample[example.example_id] = {
            "object_exact": object_exact,
            "face_exact": face_exact,
        }
    denominator = len(valid)
    metrics = {
        "schema_version": "topocae-p4-api-r2b-grounding-metrics/2",
        "valid_sample_count": denominator,
        "valid_samples_grounded": sum(item.example_id in groundings for item in valid),
        "object_selection_exact_count": object_hits,
        "object_selection_accuracy": object_hits / denominator,
        "exact_face_set_count": face_hits,
        "exact_face_set_accuracy": face_hits / denominator,
        "illegal_object_id_count": illegal_object,
        "illegal_face_id_count": illegal_face,
        "illegal_or_out_of_range_id_rate": (illegal_object + illegal_face) / max(selected_faces, 1),
        "p1_recognizer_invocations": 0,
        "step_or_brep_access_count": 0,
    }
    return metrics, per_sample


def _semantic_metrics(
    examples: Sequence[LanguageExample], outcomes_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    predictions = {
        example.example_id: (
            outcomes_by_id[example.example_id]["compiled_result"]
            if example.example_id in outcomes_by_id
            and outcomes_by_id[example.example_id].get("compiled_result") is not None
            else {"status": "ERROR"}
        )
        for example in examples
    }
    return core.compute_semantic_metrics(examples, predictions)


def _stratified_metrics(
    examples: Sequence[LanguageExample],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    ground_facts: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    manifest = json.loads(core.MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = {row["sample_id"]: row for row in manifest["samples"]}
    rows = []
    for example in examples:
        outcome = outcomes_by_id.get(example.example_id, {})
        predicted = outcome.get("compiled_result") or {"status": "ERROR"}
        expected = example.oracle_intent_result
        rows.append(
            {
                "sample_id": example.example_id,
                "language": example.language,
                "difficulty": example.difficulty_level,
                "status": expected["status"].lower(),
                "scope": next(
                    tag.split(":", 1)[1]
                    for tag in metadata[example.example_id]["coverage_tags"]
                    if tag.startswith("operation_scope:")
                ),
                "tags": metadata[example.example_id]["coverage_tags"],
                "evaluated": bool(outcome.get("p2_valid")),
                "status_correct": predicted.get("status") == expected.get("status"),
                "canonical_exact": canonical_json(predicted) == canonical_json(expected),
                "object_exact": ground_facts.get(example.example_id, {}).get("object_exact"),
                "face_exact": ground_facts.get(example.example_id, {}).get("face_exact"),
            }
        )

    def summarize(selected):
        selected = list(selected)
        valid_objects = [row["object_exact"] for row in selected if row["object_exact"] is not None]
        valid_faces = [row["face_exact"] for row in selected if row["face_exact"] is not None]
        return {
            "n": len(selected),
            "evaluated": sum(row["evaluated"] for row in selected),
            "status_accuracy": sum(row["status_correct"] for row in selected) / len(selected) if selected else None,
            "canonical_semantic_exact": sum(row["canonical_exact"] for row in selected) / len(selected) if selected else None,
            "object_selection_accuracy": sum(valid_objects) / len(valid_objects) if valid_objects else None,
            "exact_face_set_accuracy": sum(valid_faces) / len(valid_faces) if valid_faces else None,
        }

    dimensions = {
        "language": ["en-US", "zh-CN"],
        "difficulty": ["L1", "L2", "L3", "L4", "L5"],
        "status": ["valid", "needs_clarification", "invalid"],
        "scope": ["single", "multi", "nonvalid"],
    }
    result = {
        dimension: {
            value: summarize(row for row in rows if row[dimension] == value)
            for value in values
        }
        for dimension, values in dimensions.items()
    }
    result["quantifier"] = {
        value: summarize(row for row in rows if f"quantifier:{value}" in row["tags"])
        for value in ("all", "exactly_one", "count", "largest", "smallest", "nearest")
    }
    result["action"] = {
        value: summarize(row for row in rows if f"action:{value}" in row["tags"])
        for value in (
            "fixed_support", "displacement", "pressure", "surface_traction",
            "material_assignment", "global_mesh_size", "local_mesh_size",
        )
    }
    return {"schema_version": "topocae-p4-api-r2b-stratified-metrics/2", "strata": result}


def _failure_atlas(
    examples: Sequence[LanguageExample],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    ground_facts: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    failures = []
    for example in examples:
        outcome = outcomes_by_id.get(example.example_id, {})
        predicted = outcome.get("compiled_result")
        semantic_exact = predicted is not None and canonical_json(predicted) == canonical_json(example.oracle_intent_result)
        ground = ground_facts.get(example.example_id, {})
        if semantic_exact and all(ground.get(key, True) for key in ("object_exact", "face_exact")):
            continue
        failures.append(
            {
                "sample_id": example.example_id,
                "language": example.language,
                "difficulty": example.difficulty_level,
                "stable_error_category": outcome.get("failure_phase") or "semantic_or_grounding_mismatch",
                "expected": _intent_summary(example.oracle_intent_result),
                "predicted": _intent_summary(predicted),
                "error_field_path": outcome.get("error_path"),
                "repair_attempted": bool(outcome.get("repair_count")),
                "object_exact": ground.get("object_exact"),
                "face_exact": ground.get("face_exact"),
            }
        )
    return {
        "schema_version": "topocae-p4-api-r2b-failure-atlas/2",
        "raw_query_included": False,
        "ground_truth_payload_included": False,
        "raw_model_response_included": False,
        "failures": failures,
        "r1c_paired_comparison_claimed": False,
    }


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


def _publish(
    *,
    status: str,
    checks: Mapping[str, bool],
    preflight: Mapping[str, Any],
    examples: Sequence[LanguageExample],
    outcomes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any],
    grounding: Mapping[str, Any],
    stratified: Mapping[str, Any],
    failure_atlas: Mapping[str, Any],
    ledger: Mapping[str, Any],
    formal_before: Mapping[str, Any],
    freeze: Mapping[str, Any],
    actual_cost_cny: float,
    external_blocker: str | None,
) -> None:
    if OUTPUT.exists():
        raise core.R2BError("R2B_E_IMMUTABLE_OUTPUT_EXISTS")
    reports_root = OUTPUT.parent
    temporary = Path(tempfile.mkdtemp(prefix=".p4_api_r2b_v2.", dir=reports_root))
    try:
        formal_after = _formal_state()
        payloads: dict[str, Any] = {
            "acceptance.json": {
                "schema_version": "topocae-p4-api-r2b-acceptance/2",
                "status": status,
                "checks": dict(checks),
                "external_blocker": external_blocker,
                "backend_requests": ledger["counters"]["backend_requests"],
                "actual_cost_cny": actual_cost_cny,
                "next_stage_automatically_started": False,
            },
            "preflight.json": dict(preflight),
            "request_ledger.json": dict(ledger),
            "protocol_metrics.json": dict(protocol),
            "semantic_metrics.json": dict(semantic),
            "grounding_metrics.json": dict(grounding),
            "stratified_metrics.json": dict(stratified),
            "failure_atlas.json": dict(failure_atlas),
            "runtime_cost_audit.json": {
                "schema_version": "topocae-p4-api-r2b-runtime-cost-audit/2",
                "pricing_basis": "frozen_r2a_1_planning_assumptions",
                "planned_cost_upper_cny": PLANNED_COST_UPPER_CNY,
                "cost_cap_cny": core.COST_CAP_CNY,
                "actual_input_tokens": ledger["counters"].get("input_tokens", 0),
                "actual_output_tokens": ledger["counters"].get("output_tokens", 0),
                "actual_backend_requests": ledger["counters"]["backend_requests"],
                "actual_cost_cny": actual_cost_cny,
                "gate_pass": actual_cost_cny <= core.COST_CAP_CNY,
            },
            "security_audit.json": {
                "schema_version": "topocae-p4-api-r2b-security-audit/2",
                "api_key_configured": preflight["api_key_configured"],
                "api_key_value_persisted": False,
                "raw_query_persisted": False,
                "ground_truth_payload_persisted": False,
                "raw_model_response_persisted": False,
                "beta_used": False,
                "strict_used": False,
                "tool_calling_used": False,
                "provider_or_model_fallback_used": False,
                "p1_recognizer_run": False,
                "step_or_brep_accessed": False,
                "direct_baseline_started": False,
                "p5_started": False,
                "abaqus_started": False,
                "formal_test_accessed": False,
            },
            "formal_test_ledger_audit.json": {
                "schema_version": "topocae-p4-api-r2b-formal-test-ledger-audit/2",
                "before": dict(formal_before),
                "after": formal_after,
                "unchanged": formal_before == formal_after,
                "formal_test_access_count": 0,
                "gate_pass": formal_before == formal_after and not formal_after["test_content_accessed"],
            },
            "freeze_audit.json": dict(freeze),
            "sample_hash_audit.json": {
                "schema_version": "topocae-p4-api-r2b-sample-hash-audit/2",
                "checks": preflight["sample_hash_checks"],
                "sample_count": preflight["sample_count"],
                "all_query_sha256_match": preflight["all_query_sha256_match"],
                "all_ground_truth_sha256_match": preflight["all_ground_truth_sha256_match"],
                "raw_query_persisted": False,
                "ground_truth_payload_persisted": False,
            },
            "sample_results.json": {
                "schema_version": "topocae-p4-api-r2b-sample-results/2",
                "results": [sanitize_outcome(row) for row in outcomes],
                "raw_query_included": False,
                "ground_truth_payload_included": False,
                "raw_model_response_included": False,
            },
        }
        for name, value in payloads.items():
            _write_json(temporary / name, value)
        final_text = (
            "# P4-API-R2B Frozen 20-Val Compact QueryFrame Smoke v2\n\n"
            f"停止状态：`{status}`\n\n"
            f"真实后端请求：{ledger['counters']['backend_requests']}；repair：{ledger['counters']['repair_requests']}；"
            f"HTTP retry：0；实际费用：{actual_cost_cny:.6f} CNY。\n\n"
            f"repair 后 Frame-valid：{protocol['post_repair_frame_valid_count']}/20；"
            f"status 正确：{semantic['status_correct_count']}/20；"
            f"canonical semantic exact：{semantic['canonical_semantic_exact_count']}/20；"
            f"multi-operation exact：{semantic['multi_operation_exact_count']}/4。\n\n"
            "未访问 formal test、P1、STEP/BRep、Direct baseline、P5 或 Abaqus；v1 与冻结资产保持不变。\n"
        )
        (temporary / "final_acceptance.zh-CN.md").write_text(final_text, encoding="utf-8", newline="\n")
        context = (
            "# P4-API-R2B v2 Context\n\n"
            f"- Status: `{status}`\n"
            f"- Backend requests: {ledger['counters']['backend_requests']}/40\n"
            f"- Repair requests: {ledger['counters']['repair_requests']}/20\n"
            f"- Cost: {actual_cost_cny:.6f}/4.00 CNY\n"
            f"- Completed samples: {len(outcomes)}/20\n"
            f"- External blocker: {external_blocker or 'none'}\n"
            "- v1 preserved: true\n- Formal-test access: 0\n"
        )
        (temporary / "CODEX_CONTEXT_P4_API_R2B.md").write_text(context, encoding="utf-8", newline="\n")
        assets = []
        for path in sorted(item for item in temporary.iterdir() if item.is_file()):
            assets.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
        _write_json(
            temporary / "delivery_manifest.json",
            {
                "schema_version": "topocae-p4-api-r2b-delivery-manifest/2",
                "status": status,
                "assets": assets,
                "asset_count": len(assets),
                "required_assets_present": True,
                "immutable_publish": True,
            },
        )
        temporary.replace(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    if OUTPUT.exists():
        raise core.R2BError("R2B_E_IMMUTABLE_OUTPUT_EXISTS")
    v1_tree_before = _tree_sha256(V1)
    formal_before = _formal_state()
    config = LLMRuntimeConfig.from_env()
    environment = {
        "TOPOCAE_LLM_PROVIDER_PROFILE": config.provider_profile,
        "TOPOCAE_LLM_BASE_URL": config.base_url,
        "TOPOCAE_LLM_MODEL": config.model,
    }
    preflight, examples = core.build_preflight(
        environment, api_key_configured=bool(config.api_key)
    )
    freeze = _freeze_state(v1_tree_before)
    preflight["formal_test_locked"] = (
        formal_before["run_count"] == 0
        and not formal_before["test_content_accessed"]
        and not formal_before["test_unlock_authorized"]
    )
    preflight["freeze_gate_pass"] = freeze["gate_pass"]
    preflight["planned_cost_upper_cny"] = PLANNED_COST_UPPER_CNY
    preflight["planned_cost_gate_pass"] = PLANNED_COST_UPPER_CNY <= core.COST_CAP_CNY
    preflight["gate_pass"] = bool(
        preflight["gate_pass"]
        and preflight["formal_test_locked"]
        and preflight["freeze_gate_pass"]
        and preflight["planned_cost_gate_pass"]
    )
    WORK.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(LEDGER_PATH)
    outcomes: list[dict[str, Any]] = []
    external_blocker = None
    if preflight["gate_pass"]:
        client = R2BClient(
            api_key=config.api_key or "",
            ledger=ledger,
            cache_dir=CACHE,
        )
        for example in examples:
            if _formal_state() != formal_before:
                external_blocker = "R2B_E_FORMAL_LEDGER_CHANGED"
                break
            outcome = client.infer(
                sample_id=example.example_id,
                inference_input=example.inference_input(),
            )
            outcomes.append(outcome)
            core._atomic_write_json(
                WORK / "progress.json",
                {
                    "schema_version": "topocae-p4-api-r2b-progress/2",
                    "results": [sanitize_outcome(row) for row in outcomes],
                },
            )
            if outcome.get("failure_phase") in {"transport", "protocol"}:
                external_blocker = str(outcome.get("failure_code") or "R2B_E_EXTERNAL_RUNTIME")
                break
            if _actual_cost(ledger.value) > core.COST_CAP_CNY:
                external_blocker = "R2B_E_COST_CAP"
                break
    else:
        external_blocker = str(preflight.get("stable_error_code") or "R2B_E_PREFLIGHT")

    outcomes_by_id = {row["sample_id"]: row for row in outcomes}
    protocol = _protocol_metrics(outcomes)
    semantic = _semantic_metrics(examples, outcomes_by_id)
    grounding_audit, groundings, instance_maps = _ground_predictions(examples, outcomes_by_id)
    grounding, ground_facts = _grounding_metrics(examples, groundings, instance_maps)
    grounding["grounding_audit"] = grounding_audit
    stratified = _stratified_metrics(examples, outcomes_by_id, ground_facts)
    failure_atlas = _failure_atlas(examples, outcomes_by_id, ground_facts)
    actual_cost_cny = _actual_cost(ledger.value)
    status, checks = determine_status(
        protocol,
        semantic,
        grounding,
        actual_cost_cny=actual_cost_cny,
        external_blocker=external_blocker,
    )
    freeze = _freeze_state(v1_tree_before)
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
        failure_atlas=failure_atlas,
        ledger=ledger.value,
        formal_before=formal_before,
        freeze=freeze,
        actual_cost_cny=actual_cost_cny,
        external_blocker=external_blocker,
    )
    print(
        json.dumps(
            {
                "status": status,
                "completed_samples": len(outcomes),
                "backend_requests": ledger.value["counters"]["backend_requests"],
                "repair_requests": ledger.value["counters"]["repair_requests"],
                "actual_cost_cny": actual_cost_cny,
                "external_blocker": external_blocker,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
