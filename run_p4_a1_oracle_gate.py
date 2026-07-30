"""Run the P4-A1 Oracle-IR deterministic-grounder Gate on train/val only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from utils.cae_intent import load_feature_role_policy
from utils.cae_language_grounding.dataset import DecodeAudit, iter_canonical_records
from utils.cae_language_grounding.grounder import ground_intent_result
from utils.cae_language_grounding.memory import build_feature_object_memory
from utils.workspace_paths import resolve_workspace_path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "output_p4_a1_language_grounding_v1"
    / "task_b"
    / "oracle_ir_train_val_gate.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_feature(
    policy: Mapping[str, Any],
    source_feature: str,
) -> str:
    for canonical, feature in policy["features"].items():
        if source_feature == canonical or source_feature in feature.get("aliases", []):
            return str(canonical)
    return source_feature


def _object_instance_map(
    memory: Mapping[str, Any],
    graph: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, int]:
    feature_sets = [
        item for item in graph["feature_sets"] if isinstance(item, Mapping)
    ]
    result: dict[str, int] = {}
    for obj in memory["objects"]:
        feature = str(obj["canonical_feature_type"])
        faces = set(obj["topology"]["complete_instance_face_indices"])
        matches = [
            item
            for item in feature_sets
            if _canonical_feature(policy, str(item["macro_feature"])) == feature
            and set(int(index) for index in item["node_indices"]) == faces
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Object {obj['object_id']} has {len(matches)} source-instance matches"
            )
        result[str(obj["object_id"])] = int(matches[0]["instance_id"])
    return result


def _load_memory(
    graph_path: str,
    *,
    policy: Mapping[str, Any],
    cache: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, int]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    path = resolve_workspace_path(graph_path)
    cache_key = str(path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    raw = path.read_bytes()
    graph = json.loads(raw.decode("utf-8"))
    memory = build_feature_object_memory(
        graph,
        source_graph_sha256=_sha256(raw),
        policy=policy,
    )
    instance_map = _object_instance_map(memory, graph, policy)
    cached = (memory, graph, instance_map)
    cache[cache_key] = cached
    return cached


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    policy = load_feature_role_policy()
    memory_cache: dict[
        str, tuple[dict[str, Any], dict[str, Any], dict[str, int]]
    ] = {}
    counts = Counter()
    by_split = Counter()
    by_status = Counter()
    by_quantifier = Counter()
    by_operation = Counter()
    failures: list[dict[str, Any]] = []
    decode_audits: dict[str, Any] = {}

    for split in ("train", "val"):
        decode = DecodeAudit("canonical", split)
        for record in iter_canonical_records(split, audit=decode):
            counts["records"] += 1
            by_split[split] += 1
            record_id = str(
                record.get("source_ref", {}).get("canonical_record_id", "")
            )
            memory, _, instance_map = _load_memory(
                str(record["source_ref"]["graph_path"]),
                policy=policy,
                cache=memory_cache,
            )
            prediction = ground_intent_result(
                record["intent_result"],
                memory,
                policy=policy,
            )
            expected_status = str(record["intent_result"]["status"])
            by_status[expected_status] += 1
            status_match = prediction["status"] == expected_status
            counts["status_matches"] += int(status_match)
            counts["illegal_face_ids"] += int(
                prediction["verifier"]["illegal_face_id_count"]
            )
            counts["silent_fallbacks"] += int(
                prediction["verifier"]["silent_fallback_count"]
            )
            record_errors: list[str] = []
            if not status_match:
                record_errors.append(
                    f"status:{prediction['status']}!={expected_status}"
                )
            if expected_status != "VALID":
                counts["non_executable_records"] += 1
                if prediction["operations"]:
                    record_errors.append("nonvalid_prediction_has_operations")
                if not record_errors:
                    counts["non_executable_status_passed"] += 1
            else:
                counts["executable_records"] += 1
                expected_targets = {
                    str(item["operation_id"]): item
                    for item in record["ground_truth"]["targets"]
                }
                actual_targets = {
                    str(item["operation_id"]): item
                    for item in prediction["operations"]
                }
                if set(expected_targets) != set(actual_targets):
                    record_errors.append("operation_id_set_mismatch")
                for operation_id, expected in expected_targets.items():
                    counts["operation_targets"] += 1
                    operation_type = str(expected["operation_type"])
                    by_operation[operation_type] += 1
                    actual = actual_targets.get(operation_id)
                    if actual is None:
                        record_errors.append(f"{operation_id}:missing")
                        continue
                    if actual["status"] != "VALID":
                        record_errors.append(
                            f"{operation_id}:status={actual['status']}:"
                            f"{actual['error_code']}"
                        )
                        continue
                    if actual["target_contract"] != expected["target_contract"]:
                        record_errors.append(f"{operation_id}:contract_mismatch")
                        continue
                    if expected["target_contract"] == "whole_part":
                        counts["whole_part_targets"] += 1
                        if actual["selected_object_ids"] or actual["selected_face_indices"]:
                            record_errors.append(
                                f"{operation_id}:whole_part_has_geometry_ids"
                            )
                        else:
                            counts["whole_part_targets_passed"] += 1
                        continue
                    counts["feature_targets"] += 1
                    quantifier = str(expected["quantifier"])
                    by_quantifier[quantifier] += 1
                    actual_instances = sorted(
                        instance_map[object_id]
                        for object_id in actual["selected_object_ids"]
                    )
                    expected_instances = sorted(
                        int(item) for item in expected["target_instance_ids"]
                    )
                    expected_faces = sorted(
                        int(item) for item in expected["target_face_indices"]
                    )
                    if actual_instances == expected_instances:
                        counts["exact_instance_sets"] += 1
                    else:
                        record_errors.append(
                            f"{operation_id}:instance_set_mismatch"
                        )
                    if actual["selected_face_indices"] == expected_faces:
                        counts["exact_face_sets"] += 1
                    else:
                        record_errors.append(f"{operation_id}:face_set_mismatch")
                if not record_errors:
                    counts["executable_target_reconstruction_passed"] += 1
            if not record_errors:
                counts["records_passed"] += 1
            elif len(failures) < 100:
                failures.append(
                    {
                        "split": split,
                        "record_id": record_id,
                        "errors": record_errors,
                    }
                )
        decode_audits[split] = decode.as_dict()

    def ratio(numerator: str, denominator: str) -> float:
        total = counts[denominator]
        return float(counts[numerator] / total) if total else 1.0

    status_accuracy = ratio("status_matches", "records")
    reconstruction_accuracy = ratio(
        "executable_target_reconstruction_passed", "executable_records"
    )
    instance_accuracy = ratio("exact_instance_sets", "feature_targets")
    face_accuracy = ratio("exact_face_sets", "feature_targets")
    passed = (
        status_accuracy == 1.0
        and reconstruction_accuracy == 1.0
        and instance_accuracy == 1.0
        and face_accuracy == 1.0
        and counts["illegal_face_ids"] == 0
        and counts["silent_fallbacks"] == 0
        and not failures
        and all(audit["decoded_test_lines"] == 0 for audit in decode_audits.values())
    )
    return {
        "schema_version": "topocae-p4-a1-oracle-ir-gate/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "train_val_only",
        "status": "PASS" if passed else "FAIL",
        "dataset_counts": {
            "records": counts["records"],
            "by_split": dict(sorted(by_split.items())),
            "by_status": dict(sorted(by_status.items())),
            "unique_source_graphs": len(memory_cache),
            "executable_records": counts["executable_records"],
            "non_executable_records": counts["non_executable_records"],
            "operation_targets": counts["operation_targets"],
            "feature_targets": counts["feature_targets"],
            "whole_part_targets": counts["whole_part_targets"],
            "by_operation": dict(sorted(by_operation.items())),
            "by_quantifier": dict(sorted(by_quantifier.items())),
        },
        "gate_metrics": {
            "status_accuracy": status_accuracy,
            "executable_target_reconstruction": reconstruction_accuracy,
            "exact_instance_set_accuracy": instance_accuracy,
            "exact_face_set_accuracy": face_accuracy,
            "illegal_out_of_range_face_id_count": counts["illegal_face_ids"],
            "silent_fallback_count": counts["silent_fallbacks"],
        },
        "raw_counts": dict(sorted(counts.items())),
        "decode_audit": decode_audits,
        "failure_count": len(failures),
        "failure_examples": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gate_contract": {
            "status_accuracy_required": 1.0,
            "executable_target_reconstruction_required": 1.0,
            "exact_face_set_accuracy_required": 1.0,
            "illegal_face_id_count_required": 0,
            "silent_fallback_count_required": 0
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run P4-A1 Oracle-IR train/val Gate"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_gate()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "gate_metrics": result["gate_metrics"],
                "failure_count": result["failure_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
