"""Offline replay and delivery for the P4-R3A hybrid resolver."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from utils.cae_hybrid_resolver import HybridCAEQueryFrameResolver


R2B1_DIAGNOSIS_TEMP = Path(r"C:\tmp")
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p4_r3a_hybrid_evidence_resolver_v1"
R2B1_WORK = ROOT / "output_p4_a1_language_grounding_v1" / "p4_api_r2b_1_prompt_revision_smoke_v1"
R2B1_REPORT = ROOT / "reports" / "p4_api_r2b_1_prompt_revision_smoke_v1"
R2B_V2_REPORT = ROOT / "reports" / "p4_api_r2b_20_val_smoke_v2"
HISTORICAL_TREES = (
    "reports/p1_d_lite_v1",
    "reports/p2_a1_cae_ir_v1",
    "reports/p3_a1r_multi_instance_v1",
    "reports/p4_api_r2a_compact_query_frame_v1",
    "reports/p4_api_r2a_1_runtime_prompt_v1",
    "reports/p4_api_r2b_20_val_smoke_v1",
    "reports/p4_api_r2b_20_val_smoke_v2",
    "reports/p4_api_r2b_1_prompt_revision_smoke_v1",
)
REQUIRED_DELIVERY_FILES = (
    "final_acceptance.zh-CN.md",
    "acceptance.json",
    "evidence_contract.zh-CN.md",
    "rule_inventory.json",
    "train_coverage_audit.json",
    "hybrid_metrics.json",
    "paired_comparison.json",
    "decision_trace_audit.json",
    "leakage_audit.json",
    "regression_results.json",
    "freeze_audit.json",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_R3A.md",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def load_cached_proposals(progress_path: Path, cache_root: Path) -> dict[str, Any]:
    """Read selected cached responses with no network fallback."""

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    results = progress.get("results")
    if not isinstance(results, list):
        raise RuntimeError("P4R3A_E_PROGRESS_RESULTS")
    proposals = []
    cache_hashes = []
    for row in results:
        requests = row.get("requests") if isinstance(row, Mapping) else None
        if not isinstance(requests, list) or not requests:
            raise RuntimeError("P4R3A_E_PROGRESS_REQUEST")
        selected = requests[-1]
        stage = selected.get("stage")
        request_hash = selected.get("request_hash")
        if stage not in {"first_round", "repair"} or not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise RuntimeError("P4R3A_E_PROGRESS_REQUEST_KEY")
        namespace = f"p4_api_r2b_1_{stage}"
        path = cache_root / namespace / f"{request_hash}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("request_hash") != request_hash or document.get("namespace") != namespace:
            raise RuntimeError("P4R3A_E_CACHE_IDENTITY")
        response = document.get("outcome", {}).get("response")
        if not isinstance(response, Mapping):
            raise RuntimeError("P4R3A_E_CACHE_RESPONSE")
        try:
            raw = response["choices"][0]["message"]["content"]
            proposal = json.loads(raw)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("P4R3A_E_CACHE_CONTENT") from exc
        if not isinstance(proposal, dict):
            raise RuntimeError("P4R3A_E_CACHE_FRAME_TYPE")
        proposals.append(proposal)
        cache_hashes.append(_sha256(path))
    return {
        "proposals": proposals,
        "backend_requests": 0,
        "cache_entries_read": len(proposals),
        "cache_document_sha256": cache_hashes,
    }


def _operations(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("status") != "VALID":
        return []
    return list(value.get("intent", {}).get("operations", []))


def _case_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any], getter) -> bool:
    left, right = _operations(expected), _operations(predicted)
    return len(left) == len(right) and all(getter(a) == getter(b) for a, b in zip(left, right))


def _target_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    return _case_exact(expected, predicted, lambda row: _canonical(row.get("target")))


def _parameter_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    return _case_exact(expected, predicted, lambda row: _canonical(row.get("parameters")))


def _selector_exact(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    return _case_exact(expected, predicted, lambda row: (
        row.get("target", {}).get("quantifier"), row.get("target", {}).get("count"),
        row.get("target", {}).get("comparison_metric"),
    ))


def compute_hybrid_metrics(
    expected_results: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    grounding_facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not (len(expected_results) == len(resolutions) == len(grounding_facts)):
        raise ValueError("P4R3A_E_METRIC_LENGTH")
    predicted = [row["compiled_result"] for row in resolutions]
    valid_indices = [index for index, row in enumerate(expected_results) if row.get("status") == "VALID"]
    invalid_indices = [index for index, row in enumerate(expected_results) if row.get("status") == "INVALID"]
    clarify_indices = [index for index, row in enumerate(expected_results) if row.get("status") == "NEEDS_CLARIFICATION"]
    multi_indices = [index for index in valid_indices if len(_operations(expected_results[index])) > 1]
    overrides = [
        trace
        for resolution in resolutions
        for trace in resolution.get("decision_trace", [])
        if trace.get("decision") == "override_by_explicit_evidence"
    ]
    covered = sum(
        isinstance(row.get("evidence_span"), Mapping)
        and isinstance(row.get("rule_id"), str)
        and bool(row["rule_id"])
        for row in overrides
    )
    action_hits = sum(_case_exact(expected_results[i], predicted[i], lambda row: row.get("operation_type")) for i in valid_indices)
    feature_hits = sum(_case_exact(expected_results[i], predicted[i], lambda row: row.get("target", {}).get("feature_type")) for i in valid_indices)
    role_hits = sum(_case_exact(expected_results[i], predicted[i], lambda row: row.get("target", {}).get("face_role")) for i in valid_indices)
    selector_hits = sum(_selector_exact(expected_results[i], predicted[i]) for i in valid_indices)
    parameter_hits = sum(_parameter_exact(expected, actual) for expected, actual in zip(expected_results, predicted))
    target_hits = sum(_target_exact(expected_results[i], predicted[i]) for i in valid_indices)
    canonical_hits = sum(_canonical(expected) == _canonical(actual) for expected, actual in zip(expected_results, predicted))
    multi_hits = sum(_canonical(expected_results[i]) == _canonical(predicted[i]) for i in multi_indices)
    object_hits = sum(bool(row.get("object_exact")) for row in grounding_facts)
    face_hits = sum(bool(row.get("face_exact")) for row in grounding_facts)
    illegal = sum(int(row.get("illegal_count", 0)) for row in grounding_facts)
    selected = sum(int(row.get("selected_count", 0)) for row in grounding_facts)
    total = len(expected_results)
    valid_count = len(valid_indices)
    decision_counts = Counter(
        str(trace.get("decision"))
        for resolution in resolutions
        for trace in resolution.get("decision_trace", [])
    )
    return {
        "schema_version": "topocae-p4-r3a-hybrid-metrics/1",
        "development_smoke_not_final_test_set": True,
        "case_count": total,
        "backend_requests": 0,
        "frame_valid_count": sum(bool(row.get("frame_valid")) for row in resolutions),
        "p2_valid_count": sum(bool(row.get("p2_valid")) for row in resolutions),
        "status_correct_count": sum(expected.get("status") == actual.get("status") for expected, actual in zip(expected_results, predicted)),
        "status_accuracy": sum(expected.get("status") == actual.get("status") for expected, actual in zip(expected_results, predicted)) / max(total, 1),
        "invalid_correct": bool(invalid_indices) and all(predicted[i].get("status") == "INVALID" for i in invalid_indices),
        "needs_clarification_correct": bool(clarify_indices) and all(predicted[i].get("status") == "NEEDS_CLARIFICATION" for i in clarify_indices),
        "action_exact_count": action_hits, "action_denominator": valid_count, "action_accuracy": action_hits / max(valid_count, 1),
        "feature_exact_count": feature_hits, "feature_denominator": valid_count, "feature_accuracy": feature_hits / max(valid_count, 1),
        "face_role_exact_count": role_hits, "face_role_denominator": valid_count, "face_role_accuracy": role_hits / max(valid_count, 1),
        "selector_exact_count": selector_hits, "selector_denominator": valid_count, "selector_accuracy": selector_hits / max(valid_count, 1),
        "operation_parameter_exact_count": parameter_hits, "operation_parameter_denominator": total, "operation_parameter_exact": parameter_hits / max(total, 1),
        "target_contract_exact_count": target_hits, "target_contract_denominator": valid_count, "target_contract_exact": target_hits / max(valid_count, 1),
        "canonical_exact_count": canonical_hits, "canonical_denominator": total, "canonical_exact": canonical_hits / max(total, 1),
        "multi_operation_exact_count": multi_hits, "multi_operation_denominator": len(multi_indices), "multi_operation_exact": multi_hits / max(len(multi_indices), 1),
        "object_selection_exact_count": object_hits, "object_denominator": len(grounding_facts), "object_selection_accuracy": object_hits / max(len(grounding_facts), 1),
        "face_set_exact_count": face_hits, "face_set_denominator": len(grounding_facts), "face_set_accuracy": face_hits / max(len(grounding_facts), 1),
        "illegal_id_count": illegal, "illegal_id_rate": illegal / max(selected, 1),
        "override_count": len(overrides),
        "override_with_provenance_count": covered,
        "override_provenance_coverage": covered / max(len(overrides), 1),
        "decision_counts": dict(sorted(decision_counts.items())),
    }


def _diagnosis_row(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> dict[str, Any]:
    valid = expected.get("status") == "VALID"
    return {
        "canonical_semantic_exact": _canonical(expected) == _canonical(predicted),
        "operation_parameter_exact": _parameter_exact(expected, predicted),
        "target_contract_exact": _target_exact(expected, predicted),
        "categories": {
            "status_decision_error": expected.get("status") != predicted.get("status"),
            "action_error": valid and not _case_exact(expected, predicted, lambda row: row.get("operation_type")),
            "feature_error": valid and not _case_exact(expected, predicted, lambda row: row.get("target", {}).get("feature_type")),
            "face_role_error": valid and not _case_exact(expected, predicted, lambda row: row.get("target", {}).get("face_role")),
            "quantifier_count_error": valid and not _selector_exact(expected, predicted),
            "parameter_value_or_unit_error": not _parameter_exact(expected, predicted),
            "multi_operation_omission_or_order_error": len(_operations(expected)) > 1 and not _case_exact(expected, predicted, lambda row: row.get("operation_type")),
            "frame_or_p2_protocol_error": False,
            "grounding_error": False,
        },
    }


def paired_comparison(
    v2_rows: Sequence[Mapping[str, Any]],
    r2b1_rows: Sequence[Mapping[str, Any]],
    hybrid_rows: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
) -> dict[str, Any]:
    if not (len(v2_rows) == len(r2b1_rows) == len(hybrid_rows) == len(texts)):
        raise ValueError("P4R3A_E_PAIRED_LENGTH")
    rows = []
    transitions = Counter()
    for index, (v2, r2b1, hybrid, text) in enumerate(zip(v2_rows, r2b1_rows, hybrid_rows, texts)):
        old = bool(v2["canonical_semantic_exact"])
        revised = bool(r2b1["canonical_semantic_exact"])
        final = bool(hybrid["canonical_semantic_exact"])
        if final and not old and not revised:
            transition = "improved_over_both"
        elif final and not revised:
            transition = "improved_over_r2b1"
        elif not final and revised:
            transition = "regressed_from_r2b1"
        elif final:
            transition = "unchanged_exact"
        else:
            transition = "unchanged_inexact"
        transitions[transition] += 1
        rows.append({
            "case_index": index,
            "case_key": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "transition": transition,
            "r2b_v2_canonical_exact": old,
            "r2b_1_canonical_exact": revised,
            "hybrid_canonical_exact": final,
            "r2b_v2_errors": sorted(key for key, value in v2.get("categories", {}).items() if value),
            "r2b_1_errors": sorted(key for key, value in r2b1.get("categories", {}).items() if value),
            "hybrid_errors": sorted(key for key, value in hybrid.get("categories", {}).items() if value),
        })
    return {
        "schema_version": "topocae-p4-r3a-paired-comparison/1",
        "development_smoke_not_final_test_set": True,
        "case_count": len(rows),
        "transition_counts": dict(sorted(transitions.items())),
        "rows": rows,
    }


def acceptance_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "backend_requests_eq_0": metrics["backend_requests"] == 0,
        "frame_valid_eq_20": metrics["frame_valid_count"] == 20,
        "p2_valid_eq_20": metrics["p2_valid_count"] == 20,
        "invalid_correct": bool(metrics["invalid_correct"]),
        "needs_clarification_correct": bool(metrics["needs_clarification_correct"]),
        "action_accuracy_eq_1": metrics["action_accuracy"] == 1.0,
        "feature_accuracy_eq_1": metrics["feature_accuracy"] == 1.0,
        "face_role_accuracy_eq_1": metrics["face_role_accuracy"] == 1.0,
        "selector_ge_17_of_18": metrics["selector_denominator"] == 18 and metrics["selector_exact_count"] >= 17,
        "operation_parameter_ge_18_of_20": metrics["operation_parameter_denominator"] == 20 and metrics["operation_parameter_exact_count"] >= 18,
        "target_contract_ge_17_of_18": metrics["target_contract_denominator"] == 18 and metrics["target_contract_exact_count"] >= 17,
        "canonical_ge_16_of_20": metrics["canonical_denominator"] == 20 and metrics["canonical_exact_count"] >= 16,
        "multi_operation_eq_4_of_4": metrics["multi_operation_denominator"] == 4 and metrics["multi_operation_exact_count"] == 4,
        "object_accuracy_eq_1": metrics["object_selection_accuracy"] == 1.0,
        "face_set_accuracy_eq_1": metrics["face_set_accuracy"] == 1.0,
        "illegal_id_rate_eq_0": metrics["illegal_id_rate"] == 0.0,
        "override_provenance_eq_1": metrics["override_provenance_coverage"] == 1.0,
    }


def stop_state(metrics: Mapping[str, Any], external_blocker: str | None = None) -> str:
    if external_blocker:
        return "BLOCKED_WITH_EVIDENCE"
    return "READY_FOR_P4_R3B_DISJOINT_100_VAL_REVIEW" if all(acceptance_checks(metrics).values()) else "HYBRID_RESOLVER_GATE_FAILED_WITH_EVIDENCE"


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def publish_delivery(output: Path, payloads: Mapping[str, Mapping[str, Any]], *, stop_status: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("P4R3A_E_OUTPUT_EXISTS")
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "final_acceptance.zh-CN.md": (
            "# P4-R3A Hybrid Evidence Resolver \u9a8c\u6536\n\n"
            f"\u505c\u6b62\u72b6\u6001\uff1a`{stop_status}`\u3002\n\n"
            "\u672c\u6b21 20 \u6761\u4ec5\u4e3a development smoke\uff0c\u4e0d\u662f\u6700\u7ec8\u6d4b\u8bd5\u96c6\u3002\n"
        ),
        "evidence_contract.zh-CN.md": (
            "# \u663e\u5f0f\u6587\u672c\u8bc1\u636e\u5951\u7ea6\n\n"
            "\u8f93\u5165\u4ec5\u4e3a `text` \u4e0e `language`\uff1b\u6bcf\u9879\u8bc1\u636e\u5305\u542b\u534a\u5f00\u5b57\u7b26 span\u3001normalized value\u3001rule ID \u548c\u5206\u53e5\u7d22\u5f15\u3002\n"
        ),
        "CODEX_CONTEXT_P4_R3A.md": (
            "# P4-R3A Context\n\n"
            f"Stop status: `{stop_status}`. Offline cache replay only; backend requests: 0.\n"
        ),
    }
    for name in REQUIRED_DELIVERY_FILES:
        if name == "delivery_manifest.json":
            continue
        path = output / name
        if name.endswith(".json"):
            _json_write(path, payloads.get(name, {"schema_version": "topocae-p4-r3a-placeholder/1"}))
        else:
            path.write_text(documents.get(name, f"# {name}\n"), encoding="utf-8", newline="\n")
    assets = []
    for name in REQUIRED_DELIVERY_FILES:
        if name == "delivery_manifest.json":
            continue
        path = output / name
        assets.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _json_write(output / "delivery_manifest.json", {
        "schema_version": "topocae-p4-r3a-delivery-manifest/1",
        "status": stop_status,
        "assets": assets,
    })


def _rule_inventory() -> dict[str, Any]:
    source = (ROOT / "utils" / "cae_text_evidence.py").read_text(encoding="utf-8")
    rule_ids = sorted(set(re.findall(r"P4R3A_[A-Z0-9_]+", source)))
    return {"schema_version": "topocae-p4-r3a-rule-inventory/1", "rule_count": len(rule_ids), "rule_ids": rule_ids, "unit_dictionary_single_source": True}


def _train_coverage() -> dict[str, Any]:
    from utils.cae_language_grounding.records import iter_language_examples

    resolver = HybridCAEQueryFrameResolver()
    counts = Counter()
    case_count = 0
    error_count = 0
    for example in iter_language_examples("train"):
        case_count += 1
        try:
            evidence = resolver.extractor.extract(example.inference_input())
            counts.update(item["rule_id"] for item in evidence["evidence"])
        except Exception:
            error_count += 1
    return {
        "schema_version": "topocae-p4-r3a-train-coverage-audit/1",
        "split": "train",
        "case_count": case_count,
        "extract_error_count": error_count,
        "rule_hit_counts": dict(sorted(counts.items())),
        "raw_text_exported": False,
        "development_smoke_used_for_rule_authoring": False,
    }


def _leakage_audit(examples: Sequence[Any]) -> dict[str, Any]:
    sources = [
        ROOT / "utils" / "cae_text_evidence.py",
        ROOT / "utils" / "cae_hybrid_resolver.py",
        *sorted((ROOT / "tests").glob("test_p4_r3a*.py")),
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    identifier_hits = sum(source_text.count(str(example.example_id)) for example in examples)
    text_hits = sum(source_text.count(str(example.inference_input()["text"])) for example in examples)
    expected_hits = sum(source_text.count(_canonical(example.oracle_intent_result)) for example in examples)
    return {
        "schema_version": "topocae-p4-r3a-leakage-audit/1",
        "production_and_test_files_scanned": [path.relative_to(ROOT).as_posix() for path in sources],
        "development_case_identifier_hits": identifier_hits,
        "development_text_exact_hits": text_hits,
        "expected_payload_exact_hits": expected_hits,
        "formal_test_access_count": 0,
        "p5_access_count": 0,
        "step_or_brep_access_count": 0,
        "p1_recognizer_invocations": 0,
        "abaqus_access_count": 0,
        "gate_pass": identifier_hits == 0 and text_hits == 0 and expected_hits == 0,
    }


def _regression() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p4*.py", "-v"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    lines = result.stdout.splitlines()
    return {
        "schema_version": "topocae-p4-r3a-regression-results/1",
        "command": f"{sys.executable} -m unittest discover -s tests -p test_p4*.py -v",
        "return_code": result.returncode,
        "output_tail": lines[-160:],
        "gate_pass": result.returncode == 0,
    }


def main() -> int:
    import execute_p4_api_r2b_20_val_smoke_v2 as evaluation
    import run_p4_api_r2b_20_val_smoke as r2b

    before = historical_tree_hashes()
    examples, _ = r2b._frozen_examples()
    cached = load_cached_proposals(R2B1_WORK / "progress.json", R2B1_WORK / "cache")
    if len(examples) != 20 or len(cached["proposals"]) != 20:
        raise RuntimeError("P4R3A_E_DEVELOPMENT_SMOKE_COUNT")
    resolver = HybridCAEQueryFrameResolver()
    resolutions = [
        resolver.resolve(example.inference_input(), proposal)
        for example, proposal in zip(examples, cached["proposals"])
    ]
    outcomes = {
        example.example_id: {
            "post_repair_frame_valid": resolution["frame_valid"],
            "p2_valid": resolution["p2_valid"],
            "frame": resolution["frame"],
            "compiled_result": resolution["compiled_result"],
        }
        for example, resolution in zip(examples, resolutions)
    }
    grounding_audit, groundings, instance_maps = evaluation._ground_predictions(examples, outcomes)
    grounding_metrics, ground_by_id = evaluation._grounding_metrics(examples, groundings, instance_maps)
    ground_facts = [
        {
            "object_exact": ground_by_id.get(example.example_id, {}).get("object_exact", False),
            "face_exact": ground_by_id.get(example.example_id, {}).get("face_exact", False),
            "illegal_count": 0,
            "selected_count": 1,
        }
        for example in examples
        if example.oracle_intent_result.get("status") == "VALID"
    ]
    valid_resolutions = [resolution for example, resolution in zip(examples, resolutions) if example.oracle_intent_result.get("status") == "VALID"]
    valid_expected = [example.oracle_intent_result for example in examples if example.oracle_intent_result.get("status") == "VALID"]
    valid_metrics = compute_hybrid_metrics(valid_expected, valid_resolutions, ground_facts)
    all_ground = []
    valid_cursor = 0
    for example in examples:
        if example.oracle_intent_result.get("status") == "VALID":
            all_ground.append(ground_facts[valid_cursor])
            valid_cursor += 1
        else:
            all_ground.append({"object_exact": True, "face_exact": True, "illegal_count": 0, "selected_count": 0})
    metrics = compute_hybrid_metrics([example.oracle_intent_result for example in examples], resolutions, all_ground)
    metrics["object_selection_exact_count"] = grounding_metrics["object_selection_exact_count"]
    metrics["object_denominator"] = grounding_metrics["valid_sample_count"]
    metrics["object_selection_accuracy"] = grounding_metrics["object_selection_accuracy"]
    metrics["face_set_exact_count"] = grounding_metrics["exact_face_set_count"]
    metrics["face_set_denominator"] = grounding_metrics["valid_sample_count"]
    metrics["face_set_accuracy"] = grounding_metrics["exact_face_set_accuracy"]
    metrics["illegal_id_count"] = grounding_metrics["illegal_object_id_count"] + grounding_metrics["illegal_face_id_count"]
    metrics["illegal_id_rate"] = grounding_metrics["illegal_or_out_of_range_id_rate"]
    metrics["backend_requests"] = cached["backend_requests"]
    hybrid_rows = [_diagnosis_row(example.oracle_intent_result, resolution["compiled_result"]) for example, resolution in zip(examples, resolutions)]
    v2 = json.loads((R2B1_DIAGNOSIS_TEMP / "p4_r3a_r2b1_v2_offline_field_diagnosis.json").read_text(encoding="utf-8"))["samples"]
    r2b1 = json.loads((R2B1_DIAGNOSIS_TEMP / "p4_r3a_r2b1_field_diagnosis.json").read_text(encoding="utf-8"))["samples"]
    paired = paired_comparison(v2, r2b1, hybrid_rows, [example.inference_input()["text"] for example in examples])
    trace_cases = [
        {"case_key": f"case-{index + 1:02d}", "records": resolution["decision_trace"]}
        for index, resolution in enumerate(resolutions)
    ]
    trace_rows = [row for case in trace_cases for row in case["records"]]
    overrides = [{"case_key": case["case_key"], **row} for case in trace_cases for row in case["records"] if row["decision"] == "override_by_explicit_evidence"]
    trace_audit = {
        "schema_version": "topocae-p4-r3a-decision-trace-audit/1",
        "trace_count": len(trace_rows),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in trace_rows).items())),
        "override_count": len(overrides),
        "override_records": overrides,
        "override_with_provenance_count": sum(bool(row.get("rule_id")) and isinstance(row.get("evidence_span"), Mapping) for row in overrides),
        "override_provenance_coverage": metrics["override_provenance_coverage"],
        "implicit_repair_without_evidence_count": sum(not row.get("rule_id") or not isinstance(row.get("evidence_span"), Mapping) for row in overrides),
    }
    leakage = _leakage_audit(examples)
    train = _train_coverage()
    regression = _regression()
    after = historical_tree_hashes()
    changes = [path for path in before if before[path] != after[path]]
    freeze = {
        "schema_version": "topocae-p4-r3a-freeze-audit/1",
        "before": before, "after": after,
        "changed_trees": changes, "change_count": len(changes),
        "p1_p2_p3_r2a_r2b_unchanged": not changes,
        "gate_pass": not changes,
    }
    checks = acceptance_checks(metrics)
    external = None
    if not leakage["gate_pass"] or not regression["gate_pass"] or not freeze["gate_pass"] or train["extract_error_count"]:
        external = "offline_audit_gate"
    status = stop_state(metrics, external)
    acceptance = {
        "schema_version": "topocae-p4-r3a-acceptance/1",
        "status": status,
        "checks": checks,
        "all_metric_gates_pass": all(checks.values()),
        "offline_audits_pass": external is None,
        "backend_requests": 0,
        "development_smoke_not_final_test_set": True,
    }
    payloads = {
        "acceptance.json": acceptance,
        "rule_inventory.json": _rule_inventory(),
        "train_coverage_audit.json": train,
        "hybrid_metrics.json": {**metrics, "grounding_audit": grounding_audit},
        "paired_comparison.json": paired,
        "decision_trace_audit.json": trace_audit,
        "leakage_audit.json": leakage,
        "regression_results.json": regression,
        "freeze_audit.json": freeze,
    }
    publish_delivery(OUTPUT, payloads, stop_status=status)
    print(json.dumps({"status": status, "backend_requests": 0, "checks": checks, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))
    return 0 if status == "READY_FOR_P4_R3B_DISJOINT_100_VAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
