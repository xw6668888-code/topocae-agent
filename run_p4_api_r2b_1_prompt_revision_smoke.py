"""Offline R2B v2 diagnosis and shared R2B.1 paired metrics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_p4_api_r2b_20_val_smoke as r2b_core
from utils.cae_query_frame import canonical_to_query_frame, compile_query_frame, validate_query_frame


ROOT = Path(__file__).resolve().parent
V2_REPORT = ROOT / "reports" / "p4_api_r2b_20_val_smoke_v2"
V2_WORK = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r2b_20_val_smoke_v2"
WORK = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r2b_1_prompt_revision_smoke_v1"
OUTPUT = ROOT / "reports" / "p4_api_r2b_1_prompt_revision_smoke_v1"

CATEGORY_KEYS = (
    "status_decision_error",
    "action_error",
    "feature_error",
    "face_role_error",
    "quantifier_count_error",
    "parameter_value_or_unit_error",
    "multi_operation_omission_or_order_error",
    "frame_or_p2_protocol_error",
    "grounding_error",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _operations(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("status") != "VALID":
        return []
    return list(value.get("intent", {}).get("operations", []))


def _operation_signature(value: Mapping[str, Any], field: str) -> list[Any]:
    result = []
    for operation in _operations(value):
        if field == "action":
            result.append(operation.get("operation_type"))
        else:
            result.append(operation.get("target", {}).get(field))
    return result


def _target_contract_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    expected_ops = _operations(expected)
    predicted_ops = _operations(predicted)
    return len(expected_ops) == len(predicted_ops) and all(
        canonical_json(left.get("target")) == canonical_json(right.get("target"))
        for left, right in zip(expected_ops, predicted_ops)
    )


def _operation_parameter_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    expected_ops = _operations(expected)
    predicted_ops = _operations(predicted)
    return len(expected_ops) == len(predicted_ops) and all(
        canonical_json(left.get("parameters")) == canonical_json(right.get("parameters"))
        for left, right in zip(expected_ops, predicted_ops)
    )


def classify_field_differences(
    expected: Mapping[str, Any],
    predicted: Mapping[str, Any],
    *,
    protocol_error: bool = False,
    grounding_error: bool = False,
) -> dict[str, Any]:
    expected_actions = _operation_signature(expected, "action")
    predicted_actions = _operation_signature(predicted, "action")
    expected_features = _operation_signature(expected, "feature_type")
    predicted_features = _operation_signature(predicted, "feature_type")
    expected_roles = _operation_signature(expected, "face_role")
    predicted_roles = _operation_signature(predicted, "face_role")
    expected_quantifiers = _operation_signature(expected, "quantifier")
    predicted_quantifiers = _operation_signature(predicted, "quantifier")
    expected_counts = _operation_signature(expected, "count")
    predicted_counts = _operation_signature(predicted, "count")
    canonical_exact = canonical_json(expected) == canonical_json(predicted)
    categories = {
        "status_decision_error": expected.get("status") != predicted.get("status"),
        "action_error": expected_actions != predicted_actions,
        "feature_error": expected_features != predicted_features,
        "face_role_error": expected_roles != predicted_roles,
        "quantifier_count_error": (
            expected_quantifiers != predicted_quantifiers
            or expected_counts != predicted_counts
        ),
        "parameter_value_or_unit_error": not _operation_parameter_exact(expected, predicted),
        "multi_operation_omission_or_order_error": (
            len(expected_actions) > 1 and expected_actions != predicted_actions
        ),
        "frame_or_p2_protocol_error": bool(protocol_error),
        "grounding_error": bool(grounding_error),
    }
    return {
        "canonical_semantic_exact": canonical_exact,
        "operation_parameter_exact": _operation_parameter_exact(expected, predicted),
        "target_contract_exact": _target_contract_exact(expected, predicted),
        "categories": categories,
    }


def _cache_response(request_hash: str, stage: str) -> Mapping[str, Any]:
    path = V2_WORK / "cache" / f"p4_api_r2b_{stage}" / f"{request_hash}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("request_hash") != request_hash:
        raise RuntimeError("R2B1_E_V2_CACHE_HASH")
    response = value.get("outcome", {}).get("response")
    if not isinstance(response, Mapping):
        raise RuntimeError("R2B1_E_V2_CACHE_RESPONSE")
    return response


def _frame_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    raw = response["choices"][0]["message"]["content"]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("R2B1_E_V2_FRAME_TYPE")
    return value


def _frame_category_overrides(
    expected: Mapping[str, Any], frame: Mapping[str, Any], categories: dict[str, bool]
) -> None:
    if expected.get("status") != "VALID" or frame.get("status") != "valid":
        return
    expected_frame = canonical_to_query_frame(expected)
    expected_ops = expected_frame.get("operations", [])
    predicted_ops = frame.get("operations", []) if isinstance(frame.get("operations"), list) else []
    if len(expected_ops) != len(predicted_ops):
        categories["action_error"] = True
        categories["multi_operation_omission_or_order_error"] = len(expected_ops) > 1
        return
    categories["status_decision_error"] = False
    categories["action_error"] = any(
        left.get("action") != right.get("action")
        for left, right in zip(expected_ops, predicted_ops)
    )
    categories["feature_error"] = any(
        left.get("feature") != right.get("feature")
        for left, right in zip(expected_ops, predicted_ops)
    )
    categories["face_role_error"] = any(
        left.get("face_role") != right.get("face_role")
        for left, right in zip(expected_ops, predicted_ops)
    )
    categories["quantifier_count_error"] = any(
        (left.get("selector"), left.get("instance_count"))
        != (right.get("selector"), right.get("instance_count"))
        for left, right in zip(expected_ops, predicted_ops)
    )


def load_v2_cached_diagnosis() -> dict[str, Any]:
    """Replay cached responses only; this function has no transport dependency."""

    progress = json.loads((V2_WORK / "progress.json").read_text(encoding="utf-8"))
    results = progress["results"]
    examples, _ = r2b_core._frozen_examples()
    if [row["sample_id"] for row in results] != [item.example_id for item in examples]:
        raise RuntimeError("R2B1_E_V2_SAMPLE_ORDER")
    failure_atlas = json.loads((V2_REPORT / "failure_atlas.json").read_text(encoding="utf-8"))
    grounding_by_id = {
        row["sample_id"]: not (
            row.get("object_exact", True) is False or row.get("face_exact", True) is False
        )
        for row in failure_atlas["failures"]
    }
    rows = []
    for example, summary in zip(examples, results):
        requests = summary["requests"]
        selected = requests[-1]
        response = _cache_response(selected["request_hash"], selected["stage"])
        frame = _frame_from_response(response)
        predicted: Mapping[str, Any] = {"status": "ERROR"}
        validation = validate_query_frame(frame)
        if summary["p2_valid"] and validation["valid"]:
            predicted = compile_query_frame(frame)
        expected = example.oracle_intent_result
        row = classify_field_differences(
            expected,
            predicted,
            protocol_error=not summary["post_repair_frame_valid"],
            grounding_error=not grounding_by_id.get(example.example_id, True),
        )
        _frame_category_overrides(expected, frame, row["categories"])
        rows.append({
            "sample_id": example.example_id,
            "repair_count": summary["repair_count"],
            "failure_phase": summary.get("failure_phase"),
            **row,
        })
    counts = {
        key: sum(row["categories"][key] for row in rows) for key in CATEGORY_KEYS
    }
    return {
        "schema_version": "topocae-p4-api-r2b-1-offline-field-diagnosis/1",
        "source": "r2b_v2_cached_responses_only",
        "api_request_count": 0,
        "sample_count": len(rows),
        "canonical_semantic_exact_count": sum(row["canonical_semantic_exact"] for row in rows),
        "operation_parameter_exact_count": sum(row["operation_parameter_exact"] for row in rows),
        "target_contract_exact_count": sum(row["target_contract_exact"] for row in rows),
        "category_counts": counts,
        "samples": rows,
    }


def paired_changes(
    sample_ids: Sequence[str],
    v2_rows: Mapping[str, Mapping[str, Any]],
    revised_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    transition_counts = Counter()
    for sample_id in sample_ids:
        before = v2_rows[sample_id]
        after = revised_rows[sample_id]
        before_exact = bool(before["canonical_semantic_exact"])
        after_exact = bool(after["canonical_semantic_exact"])
        transition = (
            "improved" if not before_exact and after_exact
            else "regressed" if before_exact and not after_exact
            else "unchanged_exact" if before_exact
            else "unchanged_inexact"
        )
        transition_counts[transition] += 1
        rows.append({
            "sample_id": sample_id,
            "transition": transition,
            "v2_canonical_exact": before_exact,
            "revised_canonical_exact": after_exact,
            "v2_categories": [key for key, value in before["categories"].items() if value],
            "revised_categories": [key for key, value in after["categories"].items() if value],
        })
    return {
        "schema_version": "topocae-p4-api-r2b-1-paired-changes/1",
        "sample_count": len(rows),
        "transition_counts": dict(sorted(transition_counts.items())),
        "samples": rows,
    }


def write_offline_diagnosis() -> Path:
    value = load_v2_cached_diagnosis()
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "v2_offline_field_diagnosis.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main() -> int:
    path = write_offline_diagnosis()
    value = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "path": str(path),
        "sample_count": value["sample_count"],
        "canonical_semantic_exact_count": value["canonical_semantic_exact_count"],
        "category_counts": value["category_counts"],
        "api_request_count": value["api_request_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
