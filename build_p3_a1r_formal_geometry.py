from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from core.graph_v2_1 import canonical_graph_signature
from build_p3_a1r_multi_instance_pilot import (
    ROOT,
    _aggregate_geometry_audit,
    _generate_one,
    _split_leakage_audit,
    _verify_record_files,
    sha256_file,
    signature_hash,
)
from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest
from utils.cae_query_ground_p3_a1r.pilot_geometry import (
    CANONICAL_FEATURES,
    COMPARISON_METRIC,
    COMPLETE_DIMENSION_KEYS,
    FEATURE_FAMILY,
)


DEFAULT_OUTPUT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"
FORMAL_SCHEMA = "topocae_p3_a1r_multi_instance_supplement/1"
FORMAL_SEED = 202607230000
VARIANTS_PER_FEATURE = 36
TIE_VARIANTS = frozenset({6, 15, 29, 33})


def build_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for feature_index, feature_type in enumerate(CANONICAL_FEATURES):
        for variant_index in range(VARIANTS_PER_FEATURE):
            if variant_index < 28:
                split = "train"
            elif variant_index < 32:
                split = "val"
            else:
                split = "test"
            tie_design = variant_index in TIE_VARIANTS
            instance_count = 2 if tie_design or feature_type == "fillet" else 2 + (variant_index % 3)
            sample_id = f"p3a1r_formal_{feature_index + 1:02d}_{feature_type}_{variant_index + 1:03d}"
            plan.append({
                "sample_id": sample_id,
                "feature_type": feature_type,
                "feature_family": FEATURE_FAMILY[feature_type],
                "instance_count": instance_count,
                "tie_design": tie_design,
                "comparison_metric": COMPARISON_METRIC[feature_type],
                "planned_seed": FORMAL_SEED + feature_index * 100_000 + variant_index * 100,
                "split": split,
                "base_geometry_family_id": f"p3a1r-formal::{feature_type}::{variant_index + 1:03d}",
            })
    if len(plan) != 612:
        raise AssertionError(f"formal plan must contain 612 models, got {len(plan)}")
    return plan


def _upgrade_record(record: dict[str, Any]) -> dict[str, Any]:
    graph_path = Path(record["graph_path"])
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["generation"]["stage"] = "P3-A1R formal multi-instance supplement"
    graph["generation"]["metric_descriptor_revision"] = "p3_a1r_metric_descriptor_contract_r3"
    exact_signature = signature_hash(canonical_graph_signature(graph, ndigits=6))
    near_signature = signature_hash(canonical_graph_signature(graph, ndigits=4))
    graph["geometry_signatures"] = {
        "exact_graph_signature": exact_signature,
        "near_graph_signature": near_signature,
        "base_signature": graph["geometry_signatures"]["base_signature"],
    }
    graph_path.write_text(
        json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        **record,
        "schema_version": FORMAL_SCHEMA,
        "graph_sha256": sha256_file(graph_path),
        "exact_graph_signature": exact_signature,
        "near_graph_signature": near_signature,
        "metric_descriptor_revision": "p3_a1r_metric_descriptor_contract_r3",
    }


def _descriptor_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    recomputed = 0
    for record in records:
        graph = json.loads(Path(record["graph_path"]).read_text(encoding="utf-8"))
        reference = [float(value) for value in graph["part_reference"]["centroid"]]
        metric_rows = sorted(graph["metric_table"], key=lambda item: int(item["instance_id"]))
        feature_sets = sorted(graph["feature_sets"], key=lambda item: int(item["instance_id"]))
        for feature_set, metric_row in zip(feature_sets, metric_rows, strict=True):
            params = feature_set["geometry_parameters"]
            contract = feature_set["descriptor_contract"]
            metric = str(metric_row["comparison_metric"])
            required = COMPLETE_DIMENSION_KEYS[str(record["feature_type"])]
            checks = {
                "ordering_scope": contract["scope"] == "same_type_instance_ordering_only",
                "not_complete": contract["is_complete_feature_description"] is False,
                "complete_dimensions_present": all(key in params for key in required),
                "metric_matches_contract": metric == contract["comparison_metric"],
                "metric_recomputes": math.isclose(
                    float(metric_row["comparison_value"]),
                    float(params[metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "nearest_recomputes": math.isclose(
                    float(metric_row["nearest_distance"]),
                    math.dist([float(value) for value in params["centroid"]], reference),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
            }
            recomputed += 1
            if not all(checks.values()):
                failures.append({
                    "sample_id": str(record["sample_id"]),
                    "instance_id": str(feature_set["instance_id"]),
                    "failed_checks": ",".join(name for name, passed in checks.items() if not passed),
                })
    return {
        "schema_version": "p3-a1r-formal-descriptor-audit/1",
        "instances_recomputed": recomputed,
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "automatic_pass": not failures,
    }


def finalize(output: Path, plan: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = output / "manifests" / "sample_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
        newline="\n",
    )
    for split in ("train", "val", "test"):
        items = [item for item in records if item["split"] == split]
        (output / "splits" / f"{split}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items),
            encoding="utf-8",
            newline="\n",
        )
    geometry_audit = _aggregate_geometry_audit(records)
    leakage_audit = _split_leakage_audit(records)
    descriptor_audit = _descriptor_audit(records)
    freeze = verify_freeze_manifest(
        ROOT,
        ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
    )
    reports = {
        "geometry_validity_audit.json": geometry_audit,
        "split_leakage_audit.json": leakage_audit,
        "descriptor_and_metric_audit.json": descriptor_audit,
        "old_p3_freeze_audit.json": freeze,
    }
    for name, payload in reports.items():
        (output / "reports" / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    counts = Counter(item["feature_type"] for item in records)
    split_counts = Counter(item["split"] for item in records)
    tie_count = sum(bool(item["tie_design"]) for item in records)
    meta = {
        "schema_version": FORMAL_SCHEMA,
        "status": "FORMAL_GEOMETRY_COMPLETE",
        "sample_count": len(records),
        "feature_counts": dict(sorted(counts.items())),
        "family_counts": dict(sorted(Counter(item["feature_family"] for item in records).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "tie_sample_count": tie_count,
        "stable_comparison_and_nearest_models": len(records) - tie_count,
        "instances_per_model": "2-4",
        "sample_manifest_sha256": sha256_file(manifest_path),
        "automatic_pass": (
            len(records) >= 510
            and len(counts) == 17
            and all(count >= 20 for count in counts.values())
            and all(split_counts[name] > 0 for name in ("train", "val", "test"))
            and geometry_audit["automatic_pass"]
            and leakage_audit["automatic_pass"]
            and descriptor_audit["automatic_pass"]
            and freeze["automatic_pass"]
            and freeze["hash_changes"] == 0
        ),
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "forbidden_data_accessed": False,
        },
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not meta["automatic_pass"]:
        raise RuntimeError(f"formal geometry finalization failed: {meta}")
    return meta


def generate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    for name in ("steps", "geometries", "graphs", "manifests", "splits", "reports", "state/records"):
        (output / name).mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    plan_path = output / "manifests" / "generation_plan.jsonl"
    payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in plan)
    if plan_path.exists() and plan_path.read_text(encoding="utf-8") != payload:
        raise RuntimeError("existing formal plan differs from deterministic plan")
    plan_path.write_text(payload, encoding="utf-8", newline="\n")

    records: list[dict[str, Any]] = []
    for position, planned in enumerate(plan, start=1):
        state_path = output / "state" / "records" / f"{planned['sample_id']}.json"
        if state_path.exists():
            record = json.loads(state_path.read_text(encoding="utf-8"))
            _verify_record_files(record)
        else:
            record = _upgrade_record(_generate_one(planned, output))
            state_path.write_text(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        records.append(record)
        if position % 10 == 0 or position == len(plan):
            print(f"P3-A1R formal geometry {position}/{len(plan)}: {planned['sample_id']}", flush=True)
    return finalize(output, plan, records)


def validate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    plan = build_plan()
    records: list[dict[str, Any]] = []
    for item in plan:
        state_path = output / "state" / "records" / f"{item['sample_id']}.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        record = json.loads(state_path.read_text(encoding="utf-8"))
        _verify_record_files(record)
        records.append(record)
    return finalize(output, plan, records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P3-A1R formal 612-model multi-instance supplement")
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = generate(args.output) if args.command == "generate" else validate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
