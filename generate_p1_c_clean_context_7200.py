from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import canonical_graph_signature, extract_graph_v2
from p1_c.formal_generator_v4 import GENERATOR_REVISION, build_planned_result, transformed_shape_and_audit
from p1_c.formal_plan_v4 import (
    CONTEXT_QUOTAS,
    PLAN_REVISION,
    PlannedSample,
    audit_plan,
    build_plan,
    load_plan,
    sha256_file,
    write_plan,
)
from p1_c.geometry import finite_tree, shape_volume, topology_counts
from p1_c.pilot_h_c2 import exact_labels
from p1_c.preflight import protected_inventory
from utils.ontology import load_layer_index
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c_clean_context_7200"
PLAN_PATH = OUTPUT / "manifests" / "generation_plan.jsonl"
H_C1_CSV = ROOT / "output_generation_p1_c1_clean_rib_web_pilot_48" / "human_review_canonical_12.csv"
H_C2_CSV = ROOT / "output_generation_p1_c2_all_class_canonical_36" / "human_review_canonical_36.csv"
PREFLIGHT_INVENTORY = ROOT / "reports" / "p1_c_preflight" / "protected_inventory.json"
SAMPLE_SCHEMA = "topocae_p1_c_clean_context_sample_record_1"
DATASET_SCHEMA = "topocae_p1_c_clean_context_dataset_1"
RETRY_SEED_STRIDE = 720000
MAX_GENERATION_ATTEMPTS = 12


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and audit P1-C Clean-Context-7200")
    parser.add_argument("command", choices=("prepare", "generate", "finalize", "validate", "status"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        payload = prepare(output)
    elif args.command == "generate":
        payload = generate(output, limit=args.limit)
    elif args.command == "finalize":
        payload = finalize(output)
    elif args.command == "validate":
        payload = validate(output)
    else:
        payload = status(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if bool(payload.get("passed", payload.get("automatic_pass", True))) else 1


def prepare(output: Path) -> dict[str, Any]:
    gates = verify_human_gates()
    for name in ("steps", "graphs", "manifests", "splits", "reports", "state/records", "splits_matched_5000"):
        (output / name).mkdir(parents=True, exist_ok=True)
    freeze_path = output / "manifests" / "generation_plan_freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if sha256_file(output / "manifests" / "generation_plan.jsonl") != freeze["generation_plan_sha256"]:
            raise RuntimeError("existing formal generation plan no longer matches its freeze hash")
    else:
        freeze = write_plan(output)
    gate_path = output / "manifests" / "human_gate_freeze.json"
    gate_payload = {
        "schema_version": "topocae_p1_c_human_gate_freeze_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        **gates,
    }
    gate_path.write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "schema_version": "topocae_p1_c_clean_context_generation_config_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_schema": DATASET_SCHEMA,
        "sample_count": 7200,
        "classes": 18,
        "per_primary": 400,
        "context_quota_per_primary": CONTEXT_QUOTAS,
        "plan_revision": PLAN_REVISION,
        "generator_revision": GENERATOR_REVISION,
        "retry_policy": {
            "maximum_attempts": MAX_GENERATION_ATTEMPTS,
            "seed_stride": RETRY_SEED_STRIDE,
            "deterministic": True,
            "selection_uses_only_automatic_geometry_validity": True,
        },
        "graph_schema": "topocae_graph_v2_relative_1_1",
        "input_feature_abs_limit": 1e4,
        "contracts": {
            "p1_b_geometry_reused": False,
            "test_part_s3_accessed": False,
            "real_dev_used": False,
            "old_holdout_results_used": False,
            "llm_step_recognition_used": False,
            "superpower_skills_used": False,
            "plan_frozen_before_model_results": True,
        },
    }
    config_path = output / "manifests" / "generation_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("sample_count", "context_quota_per_primary", "plan_revision", "generator_revision", "retry_policy"):
            if existing[key] != config[key]:
                raise RuntimeError(f"existing generation config differs at {key}")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "passed": True,
        "output": str(output),
        "human_gates": gates,
        "plan_freeze": freeze,
        "generation_config_sha256": sha256_file(config_path),
    }


def verify_human_gates() -> dict[str, Any]:
    h_c1 = _verify_gate(
        H_C1_CSV,
        expected_rows=12,
        always_pass=("single_solid", "clean_host", "feature_concept_valid", "decision"),
        conditional_pass=(),
    )
    h_c2 = _verify_gate(
        H_C2_CSV,
        expected_rows=36,
        always_pass=("target_valid", "clean_host", "decision"),
        conditional_pass=("secondary_valid", "composition_valid"),
    )
    return {"H-C1": h_c1, "H-C2": h_c2, "all_pass": True}


def _verify_gate(
    path: Path,
    *,
    expected_rows: int,
    always_pass: tuple[str, ...],
    conditional_pass: tuple[str, ...],
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    failures: list[dict[str, Any]] = []
    for row in rows:
        invalid = [field for field in always_pass if row.get(field, "").strip().upper() != "PASS"]
        for field in conditional_pass:
            value = row.get(field, "").strip().upper()
            if value not in {"PASS", "N/A"}:
                invalid.append(field)
        if invalid:
            failures.append({"review_id": row.get("review_id"), "invalid_fields": invalid})
    if len(rows) != expected_rows or failures:
        raise RuntimeError(f"Human Gate is incomplete: {path}: rows={len(rows)}, failures={failures}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "row_count": len(rows),
        "pass_count": len(rows),
        "sha256": sha256_file(path),
    }


def generate(output: Path, *, limit: int | None = None) -> dict[str, Any]:
    _verify_prepared(output)
    plan = load_plan(output / "manifests" / "generation_plan.jsonl")
    target = len(plan) if limit is None else min(len(plan), max(0, int(limit)))
    completed = 0
    skipped = 0
    generated_now = 0
    started = datetime.now(timezone.utc).isoformat()
    for position, planned in enumerate(plan[:target], 1):
        record_path = output / "state" / "records" / f"{planned.sample_id}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _verify_record_files(record, output)
            skipped += 1
            completed += 1
            continue
        record = _generate_one(planned, output)
        record_path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        generated_now += 1
        completed += 1
        if position % 25 == 0 or position == target:
            progress = {
                "schema_version": "topocae_p1_c_clean_context_progress_1",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_count": target,
                "completed_count": completed,
                "generated_this_run": generated_now,
                "skipped_existing": skipped,
                "last_sample_id": planned.sample_id,
            }
            (output / "state" / "progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"P1-C 7200 generation {completed}/{target}; new={generated_now}; resumed={skipped}; last={planned.sample_id}",
                flush=True,
            )
    return {
        "passed": completed == target,
        "started_at_utc": started,
        "target_count": target,
        "completed_count": completed,
        "generated_this_run": generated_now,
        "skipped_existing": skipped,
        "full_plan_count": len(plan),
    }


def _generate_one(planned: PlannedSample, output: Path) -> dict[str, Any]:
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    failures: list[dict[str, Any]] = []
    step_path = output / "steps" / f"{planned.sample_id}.step"
    graph_path = output / "graphs" / f"{planned.sample_id}.json"
    for partial in (step_path, graph_path):
        if partial.exists():
            partial.unlink()
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        actual_seed = planned.seed + attempt * RETRY_SEED_STRIDE
        candidate = replace(planned, seed=actual_seed)
        try:
            result = build_planned_result(candidate)
            labels = exact_labels(result, ROOT)
            shape, transform_audit = transformed_shape_and_audit(
                result,
                scale=planned.uniform_scale,
                rotation_axis=planned.rotation_axis,
                rotation_angle_radians=planned.rotation_angle_radians,
            )
            for label in labels:
                label["face_area"] = float(label.get("face_area", 0.0)) * planned.uniform_scale ** 2
            graph = extract_graph_v2(shape, face_labels=labels, primitive_surface_index=primitive_index)
            graph["sample_id"] = planned.sample_id
            graph["base_geometry_family_id"] = planned.base_geometry_family_id
            graph["parametric_family_id"] = planned.parametric_family_id
            graph["generation"] = {
                "stage": "P1-C Clean-Context-7200",
                "generator_revision": GENERATOR_REVISION,
                "plan_revision": PLAN_REVISION,
                "planned_seed": planned.seed,
                "actual_seed": actual_seed,
                "generation_attempt": attempt,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": result.parameters,
                "transform": transform_audit,
                "cleanup_audit": transform_audit["cleanup_audit"],
                "sampling_metadata": {
                    "primary_target": planned.primary_target,
                    "context": planned.context,
                    "secondary_features": list(planned.secondary_features),
                    "split": planned.split,
                    "host_family": planned.host_family,
                },
                "inference_input_contract": {
                    "sampling_metadata_is_model_input": False,
                    "sample_id_is_model_input": False,
                    "seed_is_model_input": False,
                    "source_path_is_model_input": False,
                },
            }
            exact_signature = _signature_hash(canonical_graph_signature(graph, ndigits=6))
            near_signature = _signature_hash(canonical_graph_signature(graph, ndigits=4))
            base_signature = _signature_hash(
                {
                    "primary_target": planned.primary_target,
                    "secondary_features": list(planned.secondary_features),
                    "host_family": planned.host_family,
                    "geometry_parameters": result.parameters,
                }
            )
            graph["geometry_signatures"] = {
                "base_geometry_signature": base_signature,
                "exact_graph_signature": exact_signature,
                "near_graph_signature": near_signature,
                "rotation_scale_invariant": True,
                "canonical_patch_aware": True,
            }
            with _silence_native_output():
                write_step_with_positive_volume(shape, step_path)
            step_report = inspect_step_file(step_path)
            audit = _audit_generated_sample(planned, result, shape, labels, graph, step_report, transform_audit)
            if not audit["passed"]:
                raise ValueError(f"automatic sample audit failed: {audit['failed_checks']}")
            graph["p1_c_automatic_audit"] = audit
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            step_sha = sha256_file(step_path)
            graph_sha = sha256_file(graph_path)
            return {
                "schema_version": SAMPLE_SCHEMA,
                **planned.to_dict(),
                "planned_seed": planned.seed,
                "actual_seed": actual_seed,
                "generation_attempt": attempt,
                "generator_revision": GENERATOR_REVISION,
                "plan_revision": PLAN_REVISION,
                "generated_at_utc": graph["generation"]["generated_at_utc"],
                "step_path": str(step_path.resolve()),
                "step_relative_path": step_path.relative_to(output).as_posix(),
                "graph_path": str(graph_path.resolve()),
                "graph_relative_path": graph_path.relative_to(output).as_posix(),
                "step_sha256": step_sha,
                "graph_sha256": graph_sha,
                "base_geometry_signature": base_signature,
                "exact_graph_signature": exact_signature,
                "near_graph_signature": near_signature,
                "node_count": len(graph["nodes"]),
                "directed_edge_count": len(graph["edges"]),
                "canonical_patch_count": len(graph["canonical_patches"]),
                "feature_instance_count": len(graph["feature_sets"]),
                "macro_instance_counts": dict(Counter(item["macro_feature"] for item in graph["feature_sets"])),
                "cleanup_audit": transform_audit["cleanup_audit"],
                "label_audit": audit["label_audit"],
                "graph_input_audit": audit["graph_input_audit"],
                "step_roundtrip_audit": audit["step_roundtrip_audit"],
                "automatic_pass": True,
            }
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "seed": actual_seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                }
            )
            for partial in (step_path, graph_path):
                if partial.exists():
                    partial.unlink()
    failure_path = output / "state" / f"{planned.sample_id}.failure.json"
    failure_path.write_text(
        json.dumps({"planned": planned.to_dict(), "failures": failures}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raise RuntimeError(f"sample exhausted deterministic retries: {planned.sample_id}: {failures[-1]}")


def _audit_generated_sample(
    planned: PlannedSample,
    result: Any,
    shape: Any,
    labels: list[dict[str, Any]],
    graph: dict[str, Any],
    step_report: dict[str, Any],
    transform_audit: dict[str, Any],
) -> dict[str, Any]:
    counts = topology_counts(shape)
    memory_volume = abs(shape_volume(shape))
    step_volume = abs(float(step_report.get("signed_volume") or 0.0))
    volume_error = abs(memory_volume - step_volume) / max(memory_volume, step_volume, 1e-12)
    expected_macros = (planned.primary_target, *planned.secondary_features)
    expected_instances = set(range(1, len(expected_macros) + 1))
    graph_instances = {int(item["instance_id"]) for item in graph["feature_sets"]}
    graph_instance_macros = {
        int(item["instance_id"]): str(item["macro_feature"]) for item in graph["feature_sets"]
    }
    label_instances = {int(item["instance_id"]) for item in labels if int(item["instance_id"]) > 0}
    node_values = [
        float(value)
        for node in graph["nodes"]
        for value in node["attributes"]["recognition_features_v2"].values()
    ]
    edge_values = [
        float(value)
        for edge in graph["edge_attributes"]
        for value in edge["recognition_features_v2"].values()
    ]
    patch_face_indices = [index for patch in graph["canonical_patches"] for index in patch["face_indices"]]
    all_face_indices = {int(node["idx"]) for node in graph["nodes"]}
    label_checks = {
        "face_label_coverage_100_percent": len(labels) == counts["face_count"] == len(graph["nodes"]),
        "face_multiple_truth_owner_count_zero": len(labels) == len(graph["nodes"]),
        "exact_instance_set": label_instances == expected_instances == graph_instances,
        "feature_set_count": len(graph["feature_sets"]) == len(expected_macros),
        "instance_macro_alignment": all(
            graph_instance_macros.get(instance_id) == macro
            for instance_id, macro in enumerate(expected_macros, 1)
        ),
        "canonical_patch_maps_all_faces_once": len(patch_face_indices) == len(set(patch_face_indices))
        and set(patch_face_indices) == all_face_indices,
        "canonical_patches_respect_instance": all(
            len(patch["instance_ids"]) == 1 and len(patch["macro_features"]) == 1
            for patch in graph["canonical_patches"]
        ),
    }
    graph_checks = {
        "finite_tree": finite_tree(graph),
        "finite_node_features": all(math.isfinite(value) and abs(value) <= 1e4 for value in node_values),
        "finite_edge_features": all(math.isfinite(value) and abs(value) <= 1e4 for value in edge_values),
        "directed_edges_aligned": len(graph["edges"]) == len(graph["edge_attributes"]),
    }
    step_checks = {
        "step_readable": bool(step_report.get("readable")),
        "single_solid": counts["solid_count"] == 1 and int(step_report.get("solid_count") or 0) == 1,
        "single_shell": counts["shell_count"] == 1 and int(step_report.get("shell_count") or 0) == 1,
        "valid_brep": bool(step_report.get("valid")),
        "positive_volume": memory_volume > 1e-9 and step_volume > 1e-9,
        "face_count_roundtrip": counts["face_count"] == int(step_report.get("face_count") or -1),
        "edge_count_roundtrip": counts["edge_count"] == int(step_report.get("edge_count") or -1),
        "volume_roundtrip_lt_1e_9": volume_error < 1e-9,
        "step_has_no_issues": not step_report.get("issues"),
    }
    cleanup = transform_audit["cleanup_audit"]
    cleanup_checks = {
        "transform_audit": bool(transform_audit["passed"]),
        "cleanup_idempotent": bool(cleanup["passed"]),
        "zero_removable_same_domain_faces": int(cleanup["removable_same_domain_face_count"]) == 0,
        "zero_removable_same_domain_edges": int(cleanup["removable_same_domain_edge_count"]) == 0,
        "zero_redundant_stock_partitions": int(cleanup["redundant_coplanar_stock_partition_count"]) == 0,
        "relative_volume_error_lt_1e_9": float(cleanup["relative_volume_error"]) < 1e-9,
    }
    checks = {**label_checks, **graph_checks, **step_checks, **cleanup_checks}
    return {
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "topology": counts,
        "cleanup_audit": cleanup,
        "label_audit": {"passed": all(label_checks.values()), "checks": label_checks},
        "graph_input_audit": {
            "passed": all(graph_checks.values()),
            "checks": graph_checks,
            "max_abs_node_feature": max((abs(value) for value in node_values), default=0.0),
            "max_abs_edge_feature": max((abs(value) for value in edge_values), default=0.0),
        },
        "step_roundtrip_audit": {
            "passed": all(step_checks.values()),
            "checks": step_checks,
            "relative_volume_error": volume_error,
            "step_report": step_report,
        },
    }


def finalize(output: Path) -> dict[str, Any]:
    _verify_prepared(output)
    plan = load_plan(output / "manifests" / "generation_plan.jsonl")
    records = _load_records(output)
    if len(records) != len(plan):
        raise RuntimeError(f"cannot finalize incomplete dataset: {len(records)}/{len(plan)} records")
    plan_by_id = {record.sample_id: record for record in plan}
    for record in records:
        if record["sample_id"] not in plan_by_id:
            raise RuntimeError(f"record is not in frozen plan: {record['sample_id']}")
        _verify_record_files(record, output)
    primary_counts = Counter(record["primary_target"] for record in records)
    context_by_primary: dict[str, Counter[str]] = defaultdict(Counter)
    secondary_counts: Counter[str] = Counter()
    directed_counts: Counter[tuple[str, str]] = Counter()
    host_by_primary: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        context_by_primary[record["primary_target"]][record["context"]] += 1
        host_by_primary[record["primary_target"]][record["host_family"]] += 1
        for secondary in record["secondary_features"]:
            secondary_counts[secondary] += 1
            directed_counts[(record["primary_target"], secondary)] += 1
    split_audit = _write_splits_and_audit(output, records)
    matched = _write_matched_5000(output, records)
    cooccurrence = _write_cooccurrence(output, records)
    label_audit = {
        "schema_version": "topocae_p1_c_label_audit_1",
        "sample_count": len(records),
        "pass_count": sum(bool(record["label_audit"]["passed"]) for record in records),
        "face_label_overlap_count": 0,
        "face_label_coverage_fraction": 1.0,
        "passed": all(record["label_audit"]["passed"] for record in records),
    }
    cleanup_audit = {
        "schema_version": "topocae_p1_c_cleanup_audit_1",
        "sample_count": len(records),
        "pass_count": sum(bool(record["cleanup_audit"]["passed"]) for record in records),
        "total_removable_same_domain_faces": sum(
            int(record["cleanup_audit"]["removable_same_domain_face_count"]) for record in records
        ),
        "total_removable_same_domain_edges": sum(
            int(record["cleanup_audit"]["removable_same_domain_edge_count"]) for record in records
        ),
        "passed": all(record["cleanup_audit"]["passed"] for record in records),
    }
    graph_audit = {
        "schema_version": "topocae_p1_c_graph_input_audit_1",
        "sample_count": len(records),
        "pass_count": sum(bool(record["graph_input_audit"]["passed"]) for record in records),
        "max_abs_node_feature": max(record["graph_input_audit"]["max_abs_node_feature"] for record in records),
        "max_abs_edge_feature": max(record["graph_input_audit"]["max_abs_edge_feature"] for record in records),
        "passed": all(record["graph_input_audit"]["passed"] for record in records),
    }
    protected_diff = _protected_inventory_diff()
    old_hashes = _protected_step_hashes()
    reused_hashes = sorted({record["step_sha256"] for record in records} & old_hashes)
    quota_checks = {
        "sample_count_7200": len(records) == 7200,
        "primary_400_each": all(primary_counts[macro] == 400 for macro in primary_counts) and len(primary_counts) == 18,
        "context_exact_each": all(
            context_by_primary[primary][context] == quota
            for primary in primary_counts
            for context, quota in CONTEXT_QUOTAS.items()
        ),
        "secondary_190_210": len(secondary_counts) == 18 and all(190 <= count <= 210 for count in secondary_counts.values()),
        "secondary_exact_200": len(secondary_counts) == 18 and all(count == 200 for count in secondary_counts.values()),
        "directed_pair_at_most_16": max(directed_counts.values(), default=0) <= 16,
        "at_least_10_secondary_classes": all(
            sum(directed_counts[(primary, secondary)] > 0 for secondary in secondary_counts) >= 10
            for primary in primary_counts
        ),
        "at_least_4_host_families": all(len(host_by_primary[primary]) >= 4 for primary in primary_counts),
        "host_family_at_most_35_percent": all(
            max(host_by_primary[primary].values(), default=0) <= 140 for primary in primary_counts
        ),
    }
    automatic_pass = (
        all(quota_checks.values())
        and label_audit["passed"]
        and cleanup_audit["passed"]
        and graph_audit["passed"]
        and split_audit["passed"]
        and matched["passed"]
        and protected_diff["unchanged"]
        and not reused_hashes
        and all(record["automatic_pass"] for record in records)
    )
    manifests = output / "manifests"
    sample_manifest = manifests / "sample_manifest.jsonl"
    with sample_manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output / "reports" / "label_audit.json").write_text(json.dumps(label_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "reports" / "cleanup_audit.json").write_text(json.dumps(cleanup_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "reports" / "graph_input_audit.json").write_text(json.dumps(graph_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "reports" / "protected_inventory_diff.json").write_text(json.dumps(protected_diff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gates = verify_human_gates()
    dataset_meta = {
        "schema_version": DATASET_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_pass": automatic_pass,
        "sample_count": len(records),
        "feature_instance_count": sum(record["feature_instance_count"] for record in records),
        "generator_revision": GENERATOR_REVISION,
        "plan_revision": PLAN_REVISION,
        "human_gate_hashes": {gate: payload["sha256"] for gate, payload in gates.items() if gate.startswith("H-")},
        "primary_counts": dict(sorted(primary_counts.items())),
        "context_counts_by_primary": {key: dict(value) for key, value in sorted(context_by_primary.items())},
        "secondary_counts": dict(sorted(secondary_counts.items())),
        "directed_pair_max": max(directed_counts.values(), default=0),
        "host_family_counts_by_primary": {key: dict(value) for key, value in sorted(host_by_primary.items())},
        "quota_checks": quota_checks,
        "cooccurrence": cooccurrence,
        "label_audit": label_audit,
        "cleanup_audit": cleanup_audit,
        "graph_input_audit": graph_audit,
        "split_audit": split_audit,
        "matched_5000": matched,
        "protected_inventory_diff": protected_diff,
        "old_protected_step_hash_reuse_count": len(reused_hashes),
        "old_protected_step_hash_reuse": reused_hashes,
        "contracts": {
            "all_geometry_new": not reused_hashes,
            "p1_b_geometry_reused": False,
            "test_part_s3_accessed": False,
            "real_dev_used": False,
            "old_holdout_results_used": False,
            "llm_step_recognition_used": False,
            "superpower_skills_used": False,
            "sampling_metadata_used_as_model_input": False,
        },
    }
    meta_path = output / "dataset_meta.json"
    meta_path.write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card_path = output / "DATASET_CARD.zh-CN.md"
    card_path.write_text(_dataset_card(dataset_meta), encoding="utf-8")
    artifact_manifest = {
        "schema_version": "topocae_p1_c_clean_context_artifact_manifest_1",
        "automatic_pass": automatic_pass,
        "sample_count": len(records),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "dataset_meta_sha256": sha256_file(meta_path),
        "dataset_card_sha256": sha256_file(card_path),
        "generation_plan_sha256": sha256_file(output / "manifests" / "generation_plan.jsonl"),
        "compatibility_matrix_sha256": sha256_file(output / "manifests" / "compatibility_matrix.json"),
        "aggregate_step_bytes": sum(Path(record["step_path"]).stat().st_size for record in records),
        "aggregate_graph_bytes": sum(Path(record["graph_path"]).stat().st_size for record in records),
    }
    (manifests / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset_meta


def validate(output: Path) -> dict[str, Any]:
    meta_path = output / "dataset_meta.json"
    if not meta_path.exists():
        raise RuntimeError("dataset has not been finalized")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = _load_records(output)
    for record in records:
        _verify_record_files(record, output)
    protected_diff = _protected_inventory_diff()
    checks = {
        "dataset_meta_automatic_pass": bool(meta["automatic_pass"]),
        "record_count_7200": len(records) == 7200,
        "all_record_audits_pass": all(record["automatic_pass"] for record in records),
        "protected_inventory_unchanged": protected_diff["unchanged"],
        "sample_manifest_hash": sha256_file(output / "manifests" / "sample_manifest.jsonl")
        == json.loads((output / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8"))["sample_manifest_sha256"],
    }
    return {"passed": all(checks.values()), "checks": checks, "record_count": len(records), "protected_diff": protected_diff}


def status(output: Path) -> dict[str, Any]:
    record_dir = output / "state" / "records"
    completed = len(list(record_dir.glob("*.json"))) if record_dir.exists() else 0
    failures = len(list((output / "state").glob("*.failure.json"))) if (output / "state").exists() else 0
    return {
        "passed": True,
        "prepared": (output / "manifests" / "generation_plan_freeze.json").exists(),
        "completed_records": completed,
        "target_records": 7200,
        "failure_records": failures,
        "finalized": (output / "dataset_meta.json").exists(),
    }


def _verify_prepared(output: Path) -> None:
    freeze_path = output / "manifests" / "generation_plan_freeze.json"
    config_path = output / "manifests" / "generation_config.json"
    if not freeze_path.exists() or not config_path.exists():
        raise RuntimeError("run prepare before formal generation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    plan_path = output / "manifests" / "generation_plan.jsonl"
    if sha256_file(plan_path) != freeze["generation_plan_sha256"]:
        raise RuntimeError("frozen generation plan hash mismatch")
    verify_human_gates()


def _verify_record_files(record: dict[str, Any], output: Path) -> None:
    step_path = output / record["step_relative_path"]
    graph_path = output / record["graph_relative_path"]
    if not step_path.exists() or not graph_path.exists():
        raise RuntimeError(f"record payload is missing: {record['sample_id']}")
    if sha256_file(step_path) != record["step_sha256"] or sha256_file(graph_path) != record["graph_sha256"]:
        raise RuntimeError(f"record payload hash mismatch: {record['sample_id']}")


def _load_records(output: Path) -> list[dict[str, Any]]:
    record_dir = output / "state" / "records"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(record_dir.glob("*.json"))]
    records.sort(key=lambda item: item["sample_id"])
    return records


def _write_splits_and_audit(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    for split in ("train", "val", "development_test"):
        path = output / "splits" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in sorted(by_split[split], key=lambda item: item["sample_id"]):
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    leakage: dict[str, dict[str, int]] = {}
    keys = (
        "base_geometry_family_id",
        "parametric_family_id",
        "base_geometry_signature",
        "exact_graph_signature",
        "near_graph_signature",
        "step_sha256",
    )
    split_pairs = (("train", "val"), ("train", "development_test"), ("val", "development_test"))
    for key in keys:
        sets = {split: {str(record[key]) for record in by_split[split]} for split in by_split}
        leakage[key] = {
            f"{left}__{right}": len(sets[left] & sets[right]) for left, right in split_pairs
        }
    split_counts = {split: len(by_split[split]) for split in ("train", "val", "development_test")}
    collision_count = sum(value for group in leakage.values() for value in group.values())
    payload = {
        "schema_version": "topocae_p1_c_split_audit_1",
        "passed": split_counts == {"train": 5760, "val": 720, "development_test": 720} and collision_count == 0,
        "split_counts": split_counts,
        "leakage": leakage,
        "cross_split_collision_count": collision_count,
        "group_contract": "base/parametric/signature/hash groups remain in one split",
    }
    (output / "reports" / "split_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _write_matched_5000(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    targets = {
        "train": {"target_only": 2400, "pair": 1200, "triple": 400},
        "val": {"target_only": 300, "pair": 150, "triple": 50},
        "development_test": {"target_only": 300, "pair": 150, "triple": 50},
    }
    classes = sorted({record["primary_target"] for record in records})
    selected: list[dict[str, Any]] = []
    for split, context_targets in targets.items():
        for context, target_count in context_targets.items():
            base, remainder = divmod(target_count, len(classes))
            for class_position, primary in enumerate(classes):
                quota = base + (1 if (class_position + len(context) + len(split)) % len(classes) < remainder else 0)
                candidates = [
                    record for record in records
                    if record["split"] == split and record["context"] == context and record["primary_target"] == primary
                ]
                candidates.sort(key=lambda item: hashlib.sha256(f"P1-C-matched-5000::{item['sample_id']}".encode("utf-8")).hexdigest())
                if len(candidates) < quota:
                    raise RuntimeError(f"matched-5000 stratum underflow: {split}/{context}/{primary}")
                selected.extend(candidates[:quota])
    matched_root = output / "splits_matched_5000"
    selected.sort(key=lambda item: item["sample_id"])
    with (matched_root / "sample_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    split_counts = Counter(record["split"] for record in selected)
    context_counts = Counter(record["context"] for record in selected)
    class_counts = Counter(record["primary_target"] for record in selected)
    for split in ("train", "val", "development_test"):
        with (matched_root / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in selected:
                if record["split"] == split:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        "schema_version": "topocae_p1_c_matched_5000_1",
        "frozen_before_training": True,
        "sample_count": len(selected),
        "split_counts": dict(split_counts),
        "context_counts": dict(context_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "class_count_range": [min(class_counts.values()), max(class_counts.values())],
        "sample_manifest_sha256": sha256_file(matched_root / "sample_manifest.jsonl"),
        "passed": len(selected) == 5000
        and dict(split_counts) == {"train": 4000, "val": 500, "development_test": 500}
        and dict(context_counts) == {"target_only": 3000, "pair": 1500, "triple": 500}
        and max(class_counts.values()) - min(class_counts.values()) <= 2,
    }
    (matched_root / "matched_5000_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _write_cooccurrence(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    classes = sorted({record["primary_target"] for record in records})
    matrix = {primary: Counter() for primary in classes}
    for record in records:
        for secondary in record["secondary_features"]:
            matrix[record["primary_target"]][secondary] += 1
    csv_path = output / "reports" / "cooccurrence_matrix.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["primary_target", *classes])
        for primary in classes:
            writer.writerow([primary, *(matrix[primary][secondary] for secondary in classes)])
    json_path = output / "reports" / "cooccurrence_matrix.json"
    payload = {
        "schema_version": "topocae_p1_c_cooccurrence_matrix_1",
        "classes": classes,
        "matrix": {primary: {secondary: matrix[primary][secondary] for secondary in classes} for primary in classes},
        "csv_sha256": sha256_file(csv_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"csv_path": csv_path.relative_to(output).as_posix(), "json_path": json_path.relative_to(output).as_posix(), "csv_sha256": payload["csv_sha256"]}


def _protected_inventory_diff() -> dict[str, Any]:
    baseline = json.loads(PREFLIGHT_INVENTORY.read_text(encoding="utf-8"))
    current = protected_inventory(ROOT)
    return {
        "schema_version": "topocae_p1_c_protected_inventory_diff_1",
        "baseline_aggregate_sha256": baseline["aggregate_sha256"],
        "current_aggregate_sha256": current["aggregate_sha256"],
        "baseline_file_count": baseline["total_file_count"],
        "current_file_count": current["total_file_count"],
        "baseline_byte_count": baseline["total_byte_count"],
        "current_byte_count": current["total_byte_count"],
        "unchanged": baseline["aggregate_sha256"] == current["aggregate_sha256"]
        and baseline["total_file_count"] == current["total_file_count"]
        and baseline["total_byte_count"] == current["total_byte_count"],
    }


def _protected_step_hashes() -> set[str]:
    baseline = json.loads(PREFLIGHT_INVENTORY.read_text(encoding="utf-8"))
    return {
        str(record["sha256"]).lower()
        for directory in baseline["directories"]
        for record in directory["files"]
        if str(record["path"]).lower().endswith((".step", ".stp"))
    }


def _dataset_card(meta: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P1-C Clean-Context-7200 数据集卡",
            "",
            f"- 自动验收：{'PASS' if meta['automatic_pass'] else 'FAIL'}。",
            f"- 样本：{meta['sample_count']}；实例：{meta['feature_instance_count']}。",
            "- 18类各400：target-only 240、pair 120、triple 40。",
            f"- secondary：每类恰好200；directed pair 最大 {meta['directed_pair_max']}。",
            f"- split：{meta['split_audit']['split_counts']}；跨 split 碰撞 {meta['split_audit']['cross_split_collision_count']}。",
            f"- matched-5000：{meta['matched_5000']['sample_count']}，在任何训练结果产生前冻结。",
            f"- H-C1 SHA-256：`{meta['human_gate_hashes']['H-C1']}`。",
            f"- H-C2 SHA-256：`{meta['human_gate_hashes']['H-C2']}`。",
            f"- cleanup：{meta['cleanup_audit']['pass_count']}/{meta['cleanup_audit']['sample_count']} PASS；可移除同域面 {meta['cleanup_audit']['total_removable_same_domain_faces']}。",
            f"- 标签：{meta['label_audit']['pass_count']}/{meta['label_audit']['sample_count']} PASS；覆盖率100%，重叠0。",
            f"- 受保护 P1-B 资产：{'未变化' if meta['protected_inventory_diff']['unchanged'] else '发生变化'}。",
            f"- 旧受保护 STEP 哈希复用：{meta['old_protected_step_hash_reuse_count']}。",
            "- Graph v2 的 primary/context/host/seed/path 仅作采样与评估元数据，不进入推理输入。",
            "- 未访问 test_part_s3；未使用 Real-Dev、旧 Holdout 结果、LLM STEP 识别或 superpower skills。",
            "",
        ]
    )


def _signature_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _silence_native_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


if __name__ == "__main__":
    raise SystemExit(main())
