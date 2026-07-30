"""Formal train/val evaluation helpers for P4-A1 language baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from utils.cae_intent import load_feature_role_policy, validate_intent_result
from utils.workspace_paths import resolve_workspace_path

from .grounder import ground_intent_result
from .memory import build_feature_object_memory
from .records import LanguageExample


SLOT_NAMES = (
    "operation_type",
    "feature_type",
    "face_role",
    "quantifier",
    "count",
    "filter",
    "parameter",
    "unit",
    "direction",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _semantic_signature(value: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(value))
    intent = normalized.get("intent")
    if isinstance(intent, dict):
        intent["source_language"] = "*"
        intent["normalization_audit"] = []
    return canonical_json(normalized)


def _flatten_parameters(
    value: Any,
    *,
    prefix: str,
    slots: dict[str, set[str]],
) -> None:
    if isinstance(value, Mapping):
        if "kind" in value and (
            "vector" in value
            or value.get("kind")
            in {
                "surface_normal",
                "opposite_surface_normal",
                "global_cartesian",
                "part_local_cartesian",
            }
        ):
            slots["direction"].add(f"{prefix}:{canonical_json(value)}")
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}"
            if key == "unit":
                slots["unit"].add(f"{prefix}:{child}")
            elif key not in {"direction", "reference_frame"}:
                _flatten_parameters(child, prefix=child_prefix, slots=slots)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _flatten_parameters(
                child,
                prefix=f"{prefix}/{index}",
                slots=slots,
            )
    elif isinstance(value, (str, int, float, bool)) or value is None:
        slots["parameter"].add(f"{prefix}:{canonical_json(value)}")


def slot_items(result: Mapping[str, Any]) -> dict[str, set[str]]:
    slots = {name: set() for name in SLOT_NAMES}
    if result.get("status") != "VALID":
        return slots
    for index, operation in enumerate(result["intent"]["operations"]):
        prefix = f"op{index + 1}"
        slots["operation_type"].add(
            f"{prefix}:{operation['operation_type']}"
        )
        target = operation["target"]
        if "feature_type" in target:
            slots["feature_type"].add(f"{prefix}:{target['feature_type']}")
            slots["face_role"].add(f"{prefix}:{target['face_role']}")
            slots["quantifier"].add(f"{prefix}:{target['quantifier']}")
            if "count" in target:
                slots["count"].add(f"{prefix}:{target['count']}")
            for name, spec in sorted(target.get("filters", {}).items()):
                slots["filter"].add(
                    f"{prefix}:{name}:{canonical_json(spec)}"
                )
        _flatten_parameters(
            operation.get("parameters", {}),
            prefix=f"{prefix}/parameters",
            slots=slots,
        )
    return slots


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _set_counts(expected: set[Any], predicted: set[Any]) -> tuple[int, int, int]:
    return (
        len(expected & predicted),
        len(predicted - expected),
        len(expected - predicted),
    )


def _canonical_feature(
    policy: Mapping[str, Any],
    source_feature: str,
) -> str:
    for canonical, feature in policy["features"].items():
        if source_feature == canonical or source_feature in feature.get("aliases", []):
            return str(canonical)
    return source_feature


class MemoryCache:
    def __init__(self, policy: Mapping[str, Any] | None = None):
        self.policy = policy if policy is not None else load_feature_role_policy()
        self._cache: dict[
            str, tuple[dict[str, Any], dict[str, int]]
        ] = {}

    def get(self, graph_path: str) -> tuple[dict[str, Any], dict[str, int]]:
        path = resolve_workspace_path(graph_path)
        cache_key = str(path)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        raw = path.read_bytes()
        graph = json.loads(raw.decode("utf-8"))
        memory = build_feature_object_memory(
            graph,
            source_graph_sha256=hashlib.sha256(raw).hexdigest(),
            policy=self.policy,
        )
        instance_map: dict[str, int] = {}
        for obj in memory["objects"]:
            feature = str(obj["canonical_feature_type"])
            faces = set(obj["topology"]["complete_instance_face_indices"])
            matches = [
                item
                for item in graph["feature_sets"]
                if _canonical_feature(
                    self.policy, str(item["macro_feature"])
                )
                == feature
                and set(int(index) for index in item["node_indices"]) == faces
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{obj['object_id']} maps to {len(matches)} source instances"
                )
            instance_map[str(obj["object_id"])] = int(matches[0]["instance_id"])
        cached = (memory, instance_map)
        self._cache[cache_key] = cached
        return cached

    def __len__(self) -> int:
        return len(self._cache)


class BaselineAccumulator:
    def __init__(self, *, baseline: str, split: str):
        self.baseline = baseline
        self.split = split
        self.counts = Counter()
        self.status_confusion = Counter()
        self.slot_counts = {name: Counter() for name in SLOT_NAMES}
        self.class_counts = {
            status: Counter() for status in ("NEEDS_CLARIFICATION", "INVALID")
        }
        self.object_counts = Counter()
        self.face_counts = Counter()
        self.strata: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.paraphrase_ir: dict[str, set[str]] = defaultdict(set)
        self.paraphrase_faces: dict[str, set[str]] = defaultdict(set)
        self.latencies_ms: list[float] = []
        self.failure_taxonomy = Counter()
        self.failure_atlas: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _record_stratum(
        self,
        dimension: str,
        value: str,
        *,
        status_correct: bool,
        ir_exact: bool,
        executable_success: bool,
        face_exact: bool,
    ) -> None:
        row = self.strata[dimension][value]
        row["total"] += 1
        row["status_correct"] += int(status_correct)
        row["ir_exact"] += int(ir_exact)
        row["executable_success"] += int(executable_success)
        row["face_exact"] += int(face_exact)

    def add(
        self,
        *,
        example: LanguageExample,
        prediction: Mapping[str, Any],
        grounding: Mapping[str, Any],
        instance_map: Mapping[str, int],
        latency_ms: float,
        schema_valid_first_round: bool | None = None,
        retry_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, Any]:
        oracle = example.oracle_intent_result
        validation = validate_intent_result(prediction)
        schema_valid = bool(validation["valid"])
        status_correct = prediction.get("status") == oracle.get("status")
        grounding_status_correct = grounding.get("status") == oracle.get("status")
        ir_exact = canonical_json(prediction) == canonical_json(oracle)
        self.counts["records"] += 1
        first_round_valid = (
            schema_valid
            if schema_valid_first_round is None
            else bool(schema_valid_first_round)
        )
        self.counts["schema_valid_first_round"] += int(first_round_valid)
        self.counts["schema_valid_after_retry"] += int(schema_valid)
        self.counts["retry_count"] += int(retry_count)
        self.counts["input_tokens"] += int(input_tokens)
        self.counts["output_tokens"] += int(output_tokens)
        self.counts["status_correct"] += int(status_correct)
        self.counts["grounding_status_correct"] += int(grounding_status_correct)
        self.counts["canonical_ir_exact"] += int(ir_exact)
        self.status_confusion[
            (str(oracle.get("status")), str(prediction.get("status")))
        ] += 1
        self.latencies_ms.append(float(latency_ms))
        self.paraphrase_ir[example.canonical_record_id].add(
            _semantic_signature(prediction)
        )

        expected_slots = slot_items(oracle)
        predicted_slots = slot_items(prediction)
        for name in SLOT_NAMES:
            tp, fp, fn = _set_counts(expected_slots[name], predicted_slots[name])
            self.slot_counts[name]["tp"] += tp
            self.slot_counts[name]["fp"] += fp
            self.slot_counts[name]["fn"] += fn
        expected_operations = sorted(expected_slots["operation_type"])
        predicted_operations = sorted(predicted_slots["operation_type"])
        operation_exact = expected_operations == predicted_operations
        self.counts["operation_type_exact"] += int(operation_exact)

        for status in ("NEEDS_CLARIFICATION", "INVALID"):
            expected_positive = oracle.get("status") == status
            predicted_positive = prediction.get("status") == status
            if expected_positive and predicted_positive:
                self.class_counts[status]["tp"] += 1
            elif predicted_positive:
                self.class_counts[status]["fp"] += 1
            elif expected_positive:
                self.class_counts[status]["fn"] += 1

        expected_targets = {
            str(item["operation_id"]): item
            for item in example.ground_truth.get("targets", [])
        }
        actual_targets = {
            str(item["operation_id"]): item
            for item in grounding.get("operations", [])
        }
        executable_success = (
            oracle.get("status") != "VALID"
            and grounding.get("status") == oracle.get("status")
        )
        feature_target_count = 0
        exact_feature_targets = 0
        face_signature: list[Any] = []
        grounding_error_code = grounding.get("error_code")
        if oracle.get("status") == "VALID":
            executable_success = True
            for operation_id, expected in expected_targets.items():
                actual = actual_targets.get(operation_id)
                actual_valid = actual is not None and actual.get("status") == "VALID"
                if expected["target_contract"] == "whole_part":
                    if (
                        not actual_valid
                        or actual.get("target_contract") != "whole_part"
                        or actual.get("selected_object_ids")
                        or actual.get("selected_face_indices")
                    ):
                        executable_success = False
                    continue
                feature_target_count += 1
                expected_objects = {
                    int(value) for value in expected["target_instance_ids"]
                }
                actual_objects = (
                    {
                        instance_map[str(value)]
                        for value in actual["selected_object_ids"]
                        if str(value) in instance_map
                    }
                    if actual_valid
                    else set()
                )
                otp, ofp, ofn = _set_counts(expected_objects, actual_objects)
                self.object_counts["tp"] += otp
                self.object_counts["fp"] += ofp
                self.object_counts["fn"] += ofn
                object_exact = actual_objects == expected_objects
                self.object_counts["exact"] += int(object_exact)
                self.object_counts["targets"] += 1

                expected_faces = {
                    int(value) for value in expected["target_face_indices"]
                }
                actual_faces = (
                    {int(value) for value in actual["selected_face_indices"]}
                    if actual_valid
                    else set()
                )
                ftp, ffp, ffn = _set_counts(expected_faces, actual_faces)
                self.face_counts["tp"] += ftp
                self.face_counts["fp"] += ffp
                self.face_counts["fn"] += ffn
                union = expected_faces | actual_faces
                self.face_counts["iou_sum"] += (
                    len(expected_faces & actual_faces) / len(union)
                    if union
                    else 1.0
                )
                face_exact = actual_faces == expected_faces
                self.face_counts["exact"] += int(face_exact)
                self.face_counts["targets"] += 1
                exact_feature_targets += int(face_exact)
                executable_success = (
                    executable_success
                    and actual_valid
                    and object_exact
                    and face_exact
                )
                face_signature.append(
                    [operation_id, sorted(actual_objects), sorted(actual_faces)]
                )
        self.paraphrase_faces[example.canonical_record_id].add(
            canonical_json(face_signature)
        )
        self.counts["executable_success"] += int(executable_success)
        self.counts["feature_target_records"] += int(feature_target_count > 0)
        face_exact_record = (
            feature_target_count > 0
            and exact_feature_targets == feature_target_count
        )
        self.counts["exact_face_set_records"] += int(face_exact_record)
        self.counts["illegal_face_ids"] += int(
            grounding.get("verifier", {}).get("illegal_face_id_count", 0)
        )
        self.counts["silent_fallbacks"] += int(
            grounding.get("verifier", {}).get("silent_fallback_count", 0)
        )

        strata_values = {
            "language": [example.language],
            "difficulty_level": [example.difficulty_level],
        }
        candidate_counts = set()
        for target in example.ground_truth.get("targets", []):
            if "canonical_feature_type" in target:
                strata_values.setdefault("feature_type", []).append(
                    str(target["canonical_feature_type"])
                )
                strata_values.setdefault("operation_type", []).append(
                    str(target["operation_type"])
                )
                strata_values.setdefault("quantifier", []).append(
                    str(target["quantifier"])
                )
                uniqueness = target.get("uniqueness_contract", {})
                if "candidate_count" in uniqueness:
                    candidate_counts.add(str(uniqueness["candidate_count"]))
            else:
                strata_values.setdefault("operation_type", []).append(
                    str(target.get("operation_type", ""))
                )
        if candidate_counts:
            strata_values["candidate_count"] = sorted(candidate_counts)
        for dimension, values in strata_values.items():
            for value in sorted(set(values)):
                self._record_stratum(
                    dimension,
                    value,
                    status_correct=status_correct,
                    ir_exact=ir_exact,
                    executable_success=executable_success,
                    face_exact=face_exact_record,
                )

        category = None
        if not schema_valid or not ir_exact:
            category = "parser_error"
        if grounding.get("status") != oracle.get("status") or not executable_success:
            code = str(grounding_error_code or "")
            if code.startswith("E_ROLE"):
                category = "face_role_expansion_error"
            elif code == "E_VERIFIER_REJECTION":
                category = "verifier_rejection"
            elif code in {
                "E_DESCRIPTOR_TIE",
                "E_SPATIAL_TIE",
                "E_TARGET_NOT_UNIQUE",
                "E_COUNT_MISMATCH",
                "E_COMPARISON_DESCRIPTOR_UNAVAILABLE",
            }:
                category = "quantifier_descriptor_selection_error"
            elif code in {
                "E_NO_CANDIDATE",
                "E_FILTER_UNSUPPORTED",
                "E_SPATIAL_RELATION_UNSUPPORTED",
            }:
                category = "candidate_filtering_error"
            elif category is None:
                category = "parser_error"
        if category is not None:
            self.failure_taxonomy[category] += 1
            if self.split == "val" and len(self.failure_atlas[category]) < 10:
                self.failure_atlas[category].append(
                    {
                        "example_id": example.example_id,
                        "canonical_record_id": example.canonical_record_id,
                        "language": example.language,
                        "difficulty_level": example.difficulty_level,
                        "query": example.text,
                        "oracle_status": oracle.get("status"),
                        "predicted_status": prediction.get("status"),
                        "oracle_summary": _intent_summary(oracle),
                        "prediction_summary": _intent_summary(prediction),
                        "grounding_status": grounding.get("status"),
                        "grounding_error_code": grounding_error_code,
                    }
                )
        return {
            "schema_valid": schema_valid,
            "status_correct": status_correct,
            "ir_exact": ir_exact,
            "operation_type_exact": operation_exact,
            "executable_success": executable_success,
            "exact_face_set": face_exact_record,
            "failure_category": category,
        }

    def finalize(self, *, unique_source_graphs: int) -> dict[str, Any]:
        total = self.counts["records"]
        slot_metrics = {
            name: _prf(
                self.slot_counts[name]["tp"],
                self.slot_counts[name]["fp"],
                self.slot_counts[name]["fn"],
            )
            for name in SLOT_NAMES
        }
        slot_macro_f1 = statistics.fmean(
            value["f1"] for name, value in slot_metrics.items()
            if name != "operation_type"
        )
        class_metrics = {
            status: _prf(
                counts["tp"], counts["fp"], counts["fn"]
            )
            for status, counts in self.class_counts.items()
        }
        class_macro_f1 = statistics.fmean(
            value["f1"] for value in class_metrics.values()
        )
        object_prf = _prf(
            self.object_counts["tp"],
            self.object_counts["fp"],
            self.object_counts["fn"],
        )
        face_prf = _prf(
            self.face_counts["tp"],
            self.face_counts["fp"],
            self.face_counts["fn"],
        )
        face_targets = self.face_counts["targets"]
        object_targets = self.object_counts["targets"]
        latencies = sorted(self.latencies_ms)

        def quantile(fraction: float) -> float:
            if not latencies:
                return 0.0
            index = min(len(latencies) - 1, round((len(latencies) - 1) * fraction))
            return latencies[index]

        strata = {}
        for dimension, rows in sorted(self.strata.items()):
            strata[dimension] = {}
            for value, counts in sorted(rows.items()):
                row_total = counts["total"]
                strata[dimension][value] = {
                    "n": row_total,
                    "status_accuracy": counts["status_correct"] / row_total,
                    "cae_ir_exact_match": counts["ir_exact"] / row_total,
                    "executable_success": counts["executable_success"] / row_total,
                    "exact_face_set_record_rate": counts["face_exact"] / row_total,
                }
        paraphrase_total = len(self.paraphrase_ir)
        return {
            "schema_version": "topocae-p4-a1-baseline-metrics/1",
            "baseline": self.baseline,
            "split": self.split,
            "record_count": total,
            "unique_source_graphs": unique_source_graphs,
            "language_cae_ir": {
                "schema_valid_rate_first_round": self.counts[
                    "schema_valid_first_round"
                ]
                / total,
                "schema_valid_rate_after_retry": self.counts[
                    "schema_valid_after_retry"
                ]
                / total,
                "retry_count": self.counts["retry_count"],
                "status_accuracy": self.counts["status_correct"] / total,
                "cae_ir_canonical_exact_match": self.counts[
                    "canonical_ir_exact"
                ]
                / total,
                "operation_type_accuracy": self.counts[
                    "operation_type_exact"
                ]
                / total,
                "slot_metrics": slot_metrics,
                "slot_macro_f1": slot_macro_f1,
                "clarification_invalid_detection": {
                    **class_metrics,
                    "macro_f1": class_macro_f1,
                },
                "status_confusion": {
                    f"{expected}->{predicted}": count
                    for (expected, predicted), count in sorted(
                        self.status_confusion.items()
                    )
                },
                "paraphrase_consistency": {
                    "canonical_groups": paraphrase_total,
                    "cae_ir_consistent_groups": sum(
                        len(values) == 1 for values in self.paraphrase_ir.values()
                    ),
                    "cae_ir_consistency_rate": (
                        sum(
                            len(values) == 1
                            for values in self.paraphrase_ir.values()
                        )
                        / paraphrase_total
                        if paraphrase_total
                        else 1.0
                    ),
                    "grounding_consistent_groups": sum(
                        len(values) == 1
                        for values in self.paraphrase_faces.values()
                    ),
                    "grounding_consistency_rate": (
                        sum(
                            len(values) == 1
                            for values in self.paraphrase_faces.values()
                        )
                        / paraphrase_total
                        if paraphrase_total
                        else 1.0
                    ),
                },
            },
            "grounding": {
                "object_selection_accuracy": (
                    self.object_counts["exact"] / object_targets
                    if object_targets
                    else 1.0
                ),
                "object_prf": object_prf,
                "exact_instance_set_accuracy": (
                    self.object_counts["exact"] / object_targets
                    if object_targets
                    else 1.0
                ),
                "face_precision": face_prf["precision"],
                "face_recall": face_prf["recall"],
                "face_f1": face_prf["f1"],
                "face_mean_iou": (
                    self.face_counts["iou_sum"] / face_targets
                    if face_targets
                    else 1.0
                ),
                "exact_face_set_accuracy": (
                    self.face_counts["exact"] / face_targets
                    if face_targets
                    else 1.0
                ),
                "executable_success": self.counts["executable_success"] / total,
                "clarification_abstention_accuracy": self.counts[
                    "grounding_status_correct"
                ]
                / total,
                "illegal_out_of_range_id_rate": self.counts[
                    "illegal_face_ids"
                ]
                / max(face_targets, 1),
                "silent_fallback_count": self.counts["silent_fallbacks"],
            },
            "stratified": strata,
            "runtime": {
                "latency_ms_mean": statistics.fmean(latencies)
                if latencies
                else 0.0,
                "latency_ms_p50": quantile(0.50),
                "latency_ms_p95": quantile(0.95),
                "latency_ms_max": max(latencies) if latencies else 0.0,
                "input_tokens": self.counts["input_tokens"],
                "output_tokens": self.counts["output_tokens"],
                "failure_count": sum(self.failure_taxonomy.values()),
                "retry_rate": self.counts["retry_count"] / total,
            },
            "failure_taxonomy": {
                name: self.failure_taxonomy.get(name, 0)
                for name in (
                    "parser_error",
                    "p2_schema_policy_rejection",
                    "candidate_filtering_error",
                    "quantifier_descriptor_selection_error",
                    "face_role_expansion_error",
                    "verifier_rejection",
                    "direct_llm_invalid_id_error",
                )
            },
        }


def _intent_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "status": result.get("status"),
        "error_code": result.get("error_code"),
        "operations": [],
    }
    if result.get("status") == "VALID":
        for operation in result["intent"]["operations"]:
            target = operation["target"]
            summary["operations"].append(
                {
                    "operation_type": operation["operation_type"],
                    "feature_type": target.get("feature_type"),
                    "face_role": target.get("face_role"),
                    "quantifier": target.get("quantifier"),
                    "count": target.get("count"),
                    "filters": target.get("filters"),
                    "parameters": operation.get("parameters"),
                }
            )
    return summary


__all__ = [
    "BaselineAccumulator",
    "MemoryCache",
    "canonical_json",
    "slot_items",
]
