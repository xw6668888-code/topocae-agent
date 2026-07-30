from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepTools import breptools
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import canonical_graph_signature
from p1_c.geometry import shape_volume, topology_counts
from utils.cae_query_ground_p3_a1r.pilot_geometry import (
    CANONICAL_FEATURES,
    COMPARISON_METRIC,
    FEATURE_FAMILY,
    build_graph_with_metric_evidence,
    build_multi_instance_result,
    metric_table,
)
from utils.step_solid_validation import inspect_step_file, write_step_with_positive_volume


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output_generation_p3_a1r_multi_instance_v1"
PILOT_SCHEMA = "topocae_p3_a1r_multi_instance_pilot/1"
PILOT_SEED = 202607220000
TIE_FEATURES = {
    "through_hole_round",
    "blind_hole_round",
    "obround_hole",
    "slot_rect",
    "slot_o",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def signature_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_plan() -> list[dict[str, Any]]:
    plan = []
    sequence = 0
    for feature_index, feature_type in enumerate(CANONICAL_FEATURES):
        counts = (2, 3, 4, 2)
        if feature_type == "fillet":
            counts = (2, 2, 2, 2)
        for variant_index, instance_count in enumerate(counts):
            tie = feature_type in TIE_FEATURES and variant_index == 3
            sample_id = f"p3a1r_{feature_index + 1:02d}_{feature_type}_{variant_index + 1:02d}"
            remainder = sequence % 10
            split = "train" if remainder < 8 else ("val" if remainder == 8 else "test")
            plan.append({
                "sample_id": sample_id,
                "feature_type": feature_type,
                "feature_family": FEATURE_FAMILY[feature_type],
                "instance_count": instance_count,
                "tie_design": tie,
                "comparison_metric": COMPARISON_METRIC[feature_type],
                "planned_seed": PILOT_SEED + feature_index * 1000 + variant_index * 100,
                "split": split,
                "base_geometry_family_id": f"p3a1r::{feature_type}::{variant_index + 1:02d}",
            })
            sequence += 1
    if not 60 <= len(plan) <= 100:
        raise AssertionError(f"pilot plan must contain 60-100 models, got {len(plan)}")
    return plan


def generate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    for name in ("steps", "geometries", "graphs", "manifests", "splits", "reports", "state/records"):
        (output / name).mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    plan_path = output / "manifests" / "generation_plan.jsonl"
    plan_payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in plan)
    if plan_path.exists() and plan_path.read_text(encoding="utf-8") != plan_payload:
        raise RuntimeError("existing pilot plan differs from the deterministic P3-A1R plan")
    plan_path.write_text(plan_payload, encoding="utf-8", newline="\n")

    records = []
    for position, planned in enumerate(plan, start=1):
        record_path = output / "state" / "records" / f"{planned['sample_id']}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _verify_record_files(record)
        else:
            record = _generate_one(planned, output)
            record_path.write_text(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        records.append(record)
        if position % 10 == 0 or position == len(plan):
            print(f"P3-A1R pilot geometry {position}/{len(plan)}: {planned['sample_id']}", flush=True)
    return finalize(output, plan, records)


def _generate_one(planned: dict[str, Any], output: Path) -> dict[str, Any]:
    failures = []
    for attempt in range(40):
        actual_seed = int(planned["planned_seed"]) + attempt * 7919
        try:
            result = build_multi_instance_result(
                str(planned["feature_type"]),
                int(planned["instance_count"]),
                actual_seed,
                tie=bool(planned["tie_design"]),
            )
            graph, labels = build_graph_with_metric_evidence(result, ROOT)
            metrics = metric_table(graph, str(planned["feature_type"]))
            stability = _metric_stability(graph, metrics, tie=bool(planned["tie_design"]))
            if not stability["passed"]:
                raise ValueError(f"metric stability contract failed: {stability}")
            sample_id = str(planned["sample_id"])
            step_path = output / "steps" / f"{sample_id}.step"
            brep_path = output / "geometries" / f"{sample_id}.brep"
            graph_path = output / "graphs" / f"{sample_id}.json"
            with _silence_native_output():
                write_step_with_positive_volume(result.shape, step_path)
            brep_path.write_text(breptools.WriteToString(result.shape), encoding="utf-8", newline="\n")
            step_report = inspect_step_file(step_path)
            brep_shape = breptools.ReadFromString(brep_path.read_text(encoding="utf-8"))
            geometry_audit = _geometry_audit(result, graph, labels, metrics, step_report, brep_shape)
            if not geometry_audit["passed"]:
                raise ValueError(f"geometry audit failed: {geometry_audit['failed_checks']}")

            graph["sample_id"] = sample_id
            graph["dataset_split"] = planned["split"]
            graph["base_geometry_family_id"] = planned["base_geometry_family_id"]
            graph["source_geometry"] = {
                "step_path": str(step_path.resolve()),
                "brep_path": str(brep_path.resolve()),
                "step_sha256": sha256_file(step_path),
                "brep_sha256": sha256_file(brep_path),
            }
            graph["metric_table"] = metrics
            graph["metric_stability"] = stability
            graph["p3_a1r_geometry_audit"] = geometry_audit
            exact_signature = signature_hash(canonical_graph_signature(graph, ndigits=6))
            near_signature = signature_hash(canonical_graph_signature(graph, ndigits=4))
            graph["geometry_signatures"] = {
                "exact_graph_signature": exact_signature,
                "near_graph_signature": near_signature,
                "base_signature": signature_hash({
                    "feature_type": planned["feature_type"],
                    "instance_count": planned["instance_count"],
                    "tie_design": planned["tie_design"],
                    "feature_parameters": result.parameters["features"],
                }),
            }
            graph_path.write_text(
                json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            return {
                "schema_version": PILOT_SCHEMA,
                **planned,
                "actual_seed": actual_seed,
                "generation_attempt": attempt,
                "step_path": str(step_path.resolve()),
                "brep_path": str(brep_path.resolve()),
                "graph_path": str(graph_path.resolve()),
                "step_sha256": sha256_file(step_path),
                "brep_sha256": sha256_file(brep_path),
                "graph_sha256": sha256_file(graph_path),
                "exact_graph_signature": exact_signature,
                "near_graph_signature": near_signature,
                "node_count": len(graph["nodes"]),
                "feature_instance_count": len(graph["feature_sets"]),
                "metric_stability": stability,
                "geometry_audit": geometry_audit,
                "automatic_pass": True,
            }
        except Exception as exc:
            failures.append({
                "attempt": attempt,
                "actual_seed": actual_seed,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            })
    raise RuntimeError(f"pilot sample exhausted deterministic retries: {planned['sample_id']}: {failures[-3:]}")


def _metric_stability(graph: dict[str, Any], rows: list[dict[str, Any]], *, tie: bool) -> dict[str, Any]:
    values = sorted(float(item["comparison_value"]) for item in rows)
    distances = sorted(float(item["nearest_distance"]) for item in rows)
    value_gaps = [right - left for left, right in zip(values, values[1:])]
    distance_gaps = [right - left for left, right in zip(distances, distances[1:])]
    part_scale = float(graph["part_metadata_v2"]["normalization"]["part_bbox_diagonal"])
    value_scale = max(max(abs(value) for value in values), 1e-9)
    if tie:
        comparison_tied = max(values) - min(values) <= max(1e-7, value_scale * 1e-8)
        nearest_tied = max(distances) - min(distances) <= max(1e-6, part_scale * 1e-7)
        checks = {
            "comparison_tied": comparison_tied,
            "nearest_tied": nearest_tied,
            "clarification_required": comparison_tied or nearest_tied,
        }
    else:
        min_value_gap = min(value_gaps)
        min_distance_gap = min(distance_gaps)
        checks = {
            "comparison_unique_stable": min_value_gap > max(1e-6, value_scale * 0.015),
            "nearest_unique_stable": min_distance_gap > max(0.25, part_scale * 0.002),
        }
    passed = (
        checks["clarification_required"]
        if tie
        else checks["comparison_unique_stable"] or checks["nearest_unique_stable"]
    )
    return {
        "passed": passed,
        "tie_design": tie,
        "checks": checks,
        "comparison_values": values,
        "nearest_distances": distances,
        "minimum_comparison_gap": min(value_gaps) if value_gaps else 0.0,
        "minimum_nearest_gap": min(distance_gaps) if distance_gaps else 0.0,
        "part_scale": part_scale,
    }


def _geometry_audit(
    result: Any,
    graph: dict[str, Any],
    labels: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    step_report: dict[str, Any],
    brep_shape: Any,
) -> dict[str, Any]:
    counts = topology_counts(result.shape)
    brep_counts = topology_counts(brep_shape)
    expected_instances = set(range(1, len(result.parameters["features"]) + 1))
    graph_instances = {int(item["instance_id"]) for item in graph["feature_sets"]}
    label_instances = {int(item["instance_id"]) for item in labels if int(item["instance_id"]) > 0}
    node_sets = [set(int(value) for value in item["node_indices"]) for item in graph["feature_sets"]]
    shared_nodes = sum(
        bool(left & right)
        for index, left in enumerate(node_sets)
        for right in node_sets[index + 1 :]
    )
    centroid_pairs = [
        math.dist(left["centroid"], right["centroid"])
        for index, left in enumerate(metrics)
        for right in metrics[index + 1 :]
    ]
    memory_volume = abs(shape_volume(result.shape))
    brep_volume = abs(shape_volume(brep_shape))
    checks = {
        "memory_valid_single_solid": counts["solid_count"] == 1 and BRepCheck_Analyzer(result.shape).IsValid(),
        "memory_positive_volume": memory_volume > 1e-9,
        "step_valid_single_solid": bool(step_report.get("valid")) and int(step_report.get("solid_count") or 0) == 1,
        "step_face_count_roundtrip": int(step_report.get("face_count") or -1) == counts["face_count"],
        "step_edge_count_roundtrip": int(step_report.get("edge_count") or -1) == counts["edge_count"],
        "brep_valid_single_solid": brep_counts["solid_count"] == 1 and BRepCheck_Analyzer(brep_shape).IsValid(),
        "brep_face_count_roundtrip": brep_counts["face_count"] == counts["face_count"],
        "brep_volume_roundtrip": abs(memory_volume - brep_volume) / max(memory_volume, brep_volume, 1e-12) < 1e-10,
        "label_count_matches_faces": len(labels) == counts["face_count"] == len(graph["nodes"]),
        "exact_instance_set": expected_instances == graph_instances == label_instances,
        "two_to_four_same_type_instances": 2 <= len(graph["feature_sets"]) <= 4
        and len({item["macro_feature"] for item in graph["feature_sets"]}) == 1,
        "feature_sets_nonempty": all(item["node_indices"] for item in graph["feature_sets"]),
        "instance_face_sets_disjoint": shared_nodes == 0,
        "instance_centroids_distinct": min(centroid_pairs, default=0.0) > 1e-3,
        "semantic_roles_complete": all(item.get("semantic_roles") for item in graph["feature_sets"]),
        "metric_evidence_complete": all(
            item.get("metric_evidence", {}).get("source_node_indices")
            and len(item.get("geometry_parameters", {}).get("centroid", [])) == 3
            for item in graph["feature_sets"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "memory_topology": counts,
        "brep_topology": brep_counts,
        "step_report": step_report,
        "shared_instance_face_set_pairs": shared_nodes,
        "minimum_instance_centroid_distance": min(centroid_pairs, default=0.0),
    }


def finalize(output: Path, plan: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = output / "manifests" / "sample_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
        newline="\n",
    )
    for split in ("train", "val", "test"):
        split_records = [item for item in records if item["split"] == split]
        (output / "splits" / f"{split}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in split_records),
            encoding="utf-8",
            newline="\n",
        )
    geometry_audit = _aggregate_geometry_audit(records)
    leakage_audit = _split_leakage_audit(records)
    (output / "reports" / "geometry_validity_audit.json").write_text(
        json.dumps(geometry_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "reports" / "split_leakage_audit.json").write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    meta = {
        "schema_version": PILOT_SCHEMA,
        "status": "PILOT_AWAITING_H3R_01",
        "sample_count": len(records),
        "feature_count": len({item["feature_type"] for item in records}),
        "feature_family_count": len({item["feature_family"] for item in records}),
        "feature_counts": dict(sorted(Counter(item["feature_type"] for item in records).items())),
        "family_counts": dict(sorted(Counter(item["feature_family"] for item in records).items())),
        "instance_count_distribution": dict(sorted(Counter(str(item["instance_count"]) for item in records).items())),
        "split_counts": dict(sorted(Counter(item["split"] for item in records).items())),
        "tie_sample_count": sum(bool(item["tie_design"]) for item in records),
        "all_count_eligible_models": len(records),
        "stable_comparison_and_nearest_models": sum(not bool(item["tie_design"]) for item in records),
        "automatic_pass": geometry_audit["automatic_pass"] and leakage_audit["automatic_pass"],
        "plan_sha256": sha256_file(output / "manifests" / "generation_plan.jsonl"),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "scope_guards": {
            "pilot_only_60_to_100": 60 <= len(records) <= 100,
            "formal_510_plus_generated": False,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "forbidden_data_accessed": False,
        },
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return meta


def _aggregate_geometry_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [item["sample_id"] for item in records if not item["geometry_audit"]["passed"]]
    features = defaultdict(int)
    for record in records:
        features[record["feature_type"]] += 1
    return {
        "schema_version": "p3-a1r-pilot-geometry-validity-audit/1",
        "sample_count": len(records),
        "passed_sample_count": len(records) - len(failures),
        "failed_sample_count": len(failures),
        "failure_samples": failures,
        "feature_counts": dict(sorted(features.items())),
        "automatic_pass": not failures and len(features) == 17 and all(count >= 4 for count in features.values()),
    }


def _split_leakage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("base_geometry_family_id", "exact_graph_signature", "near_graph_signature")
    collisions: dict[str, dict[str, list[str]]] = {}
    for field in fields:
        owners: dict[str, set[str]] = defaultdict(set)
        for record in records:
            owners[str(record[field])].add(str(record["split"]))
        collisions[field] = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
    split_counts = Counter(item["split"] for item in records)
    return {
        "schema_version": "p3-a1r-pilot-split-leakage-audit/1",
        "split_counts": dict(sorted(split_counts.items())),
        "collisions": collisions,
        "total_collisions": sum(len(value) for value in collisions.values()),
        "every_split_has_true_multi_instance": all(split_counts[split] > 0 for split in ("train", "val", "test")),
        "automatic_pass": not any(collisions.values()) and all(split_counts[split] > 0 for split in ("train", "val", "test")),
    }


def _verify_record_files(record: dict[str, Any]) -> None:
    for path_key, hash_key in (("step_path", "step_sha256"), ("brep_path", "brep_sha256"), ("graph_path", "graph_sha256")):
        path = Path(record[path_key])
        if not path.is_file() or sha256_file(path) != record[hash_key]:
            raise RuntimeError(f"existing pilot record artifact mismatch: {path}")


def validate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    plan = build_plan()
    records = []
    for item in plan:
        path = output / "state" / "records" / f"{item['sample_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        _verify_record_files(record)
        records.append(record)
    return finalize(output, plan, records)


@contextmanager
def _silence_native_output() -> Iterable[None]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the P3-A1R 60-100 model multi-instance pilot")
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = generate(args.output) if args.command == "generate" else validate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
