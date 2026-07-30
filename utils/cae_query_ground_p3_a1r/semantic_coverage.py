from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


EXPECTED_QUANTIFIERS = ("largest", "smallest", "nearest")


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def _feature_targets(record: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = record.get("ground_truth", {})
    if ground_truth.get("intent_status") != "VALID":
        return []
    return [
        target
        for target in ground_truth.get("targets", [])
        if target.get("grounding_required") is True
        and target.get("target_contract") == "feature_face_set"
    ]


def _target_operation(record: dict[str, Any], operation_id: str) -> dict[str, Any] | None:
    intent = record.get("intent_result", {}).get("intent", {})
    for operation in intent.get("operations", []):
        if operation.get("operation_id") == operation_id:
            return operation
    return None


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    l4_quantifiers: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    def fail(kind: str, record: dict[str, Any]) -> None:
        failures[kind] += 1
        bucket = examples.setdefault(kind, [])
        if len(bucket) < 5:
            source_ref = record.get("source_ref", {})
            bucket.append(str(source_ref.get("canonical_record_id", "<unknown>")))

    for record in records:
        counts["canonical_records"] += 1
        level = record.get("query", {}).get("difficulty_level")
        status = record.get("intent_result", {}).get("status")
        feature_targets = _feature_targets(record)
        counts["valid_feature_targets"] += len(feature_targets)

        if level == "L4" and status == "VALID":
            counts["valid_L4"] += 1

        for target in feature_targets:
            uniqueness = target.get("uniqueness_contract", {})
            candidate_count = int(uniqueness.get("candidate_count", 0) or 0)
            selected_count = int(uniqueness.get("selected_count", 0) or 0)
            quantifier = str(target.get("quantifier", ""))
            operation = _target_operation(record, str(target.get("operation_id", "")))
            intent_target = operation.get("target", {}) if operation else {}

            if candidate_count >= 2:
                counts["candidate_count_ge_2"] += 1
            if quantifier == "count":
                counts["count_quantifier"] += 1

            if level == "L2" and candidate_count >= 2:
                counts["executable_L2_multi"] += 1
                if quantifier == "all":
                    counts["all_multi"] += 1
                    if selected_count < 2:
                        fail("all_selected_count_lt_2", record)
                elif quantifier == "count":
                    counts["count_multi"] += 1
                    declared = intent_target.get("count")
                    if not isinstance(declared, int) or declared != selected_count:
                        fail("count_value_mismatch", record)

            if level == "L4" and status == "VALID":
                l4_quantifiers[quantifier] += 1
                if quantifier not in EXPECTED_QUANTIFIERS:
                    fail("unexpected_valid_L4_quantifier", record)
                if candidate_count < 2:
                    fail("L4_candidate_count_lt_2", record)
                if selected_count != 1:
                    fail("L4_selected_count_not_1", record)
                if quantifier in {"largest", "smallest"} and not intent_target.get("comparison_metric"):
                    fail("L4_comparison_metric_missing", record)
                if quantifier == "nearest":
                    filters = intent_target.get("filters", {})
                    if not filters.get("nearest_to") or not intent_target.get("reference_frame"):
                        fail("L4_nearest_reference_missing", record)

    semantic_gate_failures: list[str] = []
    if counts["candidate_count_ge_2"] == 0:
        semantic_gate_failures.append("no_multi_instance_feature_target")
    if counts["count_quantifier"] == 0:
        semantic_gate_failures.append("no_count_quantifier")
    if counts["valid_L4"] == 0:
        semantic_gate_failures.append("no_valid_L4")
    for quantifier in EXPECTED_QUANTIFIERS:
        if l4_quantifiers[quantifier] == 0:
            semantic_gate_failures.append(f"no_valid_L4_{quantifier}")
    semantic_gate_failures.extend(sorted(failures))

    return {
        "schema_version": "p3-a1r-semantic-coverage-audit/1",
        "status": "PASS" if not semantic_gate_failures else "SEMANTIC-COVERAGE-FAIL",
        "counts": {
            "canonical_records": counts["canonical_records"],
            "valid_feature_targets": counts["valid_feature_targets"],
            "candidate_count_ge_2": counts["candidate_count_ge_2"],
            "count_quantifier": counts["count_quantifier"],
            "valid_L4": counts["valid_L4"],
            "executable_L2_multi": counts["executable_L2_multi"],
            "all_multi": counts["all_multi"],
            "count_multi": counts["count_multi"],
            "valid_L4_quantifiers": {name: l4_quantifiers[name] for name in EXPECTED_QUANTIFIERS},
        },
        "contract_failure_counts": dict(sorted(failures.items())),
        "failure_examples": examples,
        "semantic_gate_failures": semantic_gate_failures,
    }


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    paths = [dataset_root / "canonical" / f"{split}.jsonl" for split in ("train", "val", "test")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical files: {missing}")
    result = audit_records(iter_jsonl(paths))
    result["dataset_root"] = str(dataset_root.resolve())
    result["canonical_files"] = [str(path.resolve()) for path in paths]
    return result
