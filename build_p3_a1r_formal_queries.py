from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

from build_p3_a1r_pilot_queries import (
    QUANTIFIERS,
    _build_record,
    _record_quantifier,
    read_jsonl,
)
from utils.cae_intent import validate_intent_result


ROOT = Path(__file__).resolve().parent
DEFAULT_GEOMETRY_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_formal_v1"
FORMAL_QUERY_SCHEMA = "p3-a1r-formal-query-supplement/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _formalize_record(
    record: dict[str, Any],
    variants: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = str(record["source_ref"]["dataset_split"])
    quantifier = _record_quantifier(record)
    record["record_schema_version"] = "p3-a1r-formal-query-record/1"
    record["query"]["template_family_id"] = f"p3a1r.formal.{split}.{quantifier}.en1"
    record["query"]["template_id"] = f"p3a1r.formal.{split}.{quantifier}.en1.v1"
    for variant in variants:
        variant["record_schema_version"] = "p3-a1r-formal-query-rendering/1"
        variant["template_family_id"] = (
            f"p3a1r.formal.{split}.{quantifier}.{variant['rendering_variant']}"
        )
    return record, variants


def _stability(graph: dict[str, Any]) -> dict[str, bool]:
    rows = sorted(graph["metric_table"], key=lambda item: int(item["instance_id"]))
    comparison_values = sorted(float(item["comparison_value"]) for item in rows)
    nearest_distances = sorted(float(item["nearest_distance"]) for item in rows)
    comparison_scale = max(max(abs(value) for value in comparison_values), 1e-9)
    part_scale = float(graph["part_metadata_v2"]["normalization"]["part_bbox_diagonal"])
    comparison_stable = min(
        right - left for left, right in zip(comparison_values, comparison_values[1:])
    ) > max(1e-6, comparison_scale * 0.015)
    nearest_stable = min(
        right - left for left, right in zip(nearest_distances, nearest_distances[1:])
    ) > max(0.25, part_scale * 0.002)
    return {
        "comparison_stable": comparison_stable,
        "nearest_stable": nearest_stable,
    }


def _split_leakage_audit(
    canonical: list[dict[str, Any]],
    renderings: list[dict[str, Any]],
) -> dict[str, Any]:
    geometry_groups = {
        "base_geometry_family_id": defaultdict(set),
        "exact_graph_signature": defaultdict(set),
        "near_graph_signature": defaultdict(set),
    }
    texts: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for record in canonical:
        split = str(record["source_ref"]["dataset_split"])
        for field, groups in geometry_groups.items():
            groups[str(record["source_ref"][field])].add(split)
        texts[split].add(_normalize_text(record["query"]["text"]))
        families[split].add(str(record["query"]["template_family_id"]))
    canonical_by_id = {
        str(record["source_ref"]["canonical_record_id"]): record for record in canonical
    }
    for rendering in renderings:
        split = str(canonical_by_id[str(rendering["canonical_record_id"])]["source_ref"]["dataset_split"])
        texts[split].add(_normalize_text(rendering["text"]))
        families[split].add(str(rendering["template_family_id"]))

    def pairwise(groups: dict[str, set[str]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        splits = sorted(groups)
        for index, left in enumerate(splits):
            for right in splits[index + 1:]:
                overlap = sorted(groups[left] & groups[right])
                if overlap:
                    result[f"{left}:{right}"] = overlap[:20]
        return result

    geometry = {
        field: {
            "collision_count": sum(len(splits) > 1 for splits in groups.values()),
            "examples": {
                value: sorted(splits)
                for value, splits in list(
                    (item for item in groups.items() if len(item[1]) > 1)
                )[:20]
            },
        }
        for field, groups in geometry_groups.items()
    }
    text_collisions = pairwise(texts)
    family_collisions = pairwise(families)
    total = (
        sum(item["collision_count"] for item in geometry.values())
        + sum(len(items) for items in text_collisions.values())
        + sum(len(items) for items in family_collisions.values())
    )
    return {
        "schema_version": "p3-a1r-formal-query-split-leakage/1",
        "geometry": geometry,
        "normalized_exact_text": {
            "collision_count": sum(len(items) for items in text_collisions.values()),
            "examples": text_collisions,
        },
        "template_family": {
            "collision_count": sum(len(items) for items in family_collisions.values()),
            "examples": family_collisions,
        },
        "total_collisions": total,
        "automatic_pass": total == 0,
    }


def _audit(
    canonical: list[dict[str, Any]],
    renderings: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    status_counts = Counter()
    quantifier_counts = Counter()
    executable = Counter()
    executable_by_split = Counter()
    executable_features: dict[str, set[str]] = defaultdict(set)
    feature_quantifier = Counter()
    level_status = Counter()
    candidate_selected = Counter()
    operation_counts = Counter()
    canonical_splits = Counter()
    metric_counts = Counter()
    reconstruction_passed = 0
    metric_passed = 0
    identity_pattern = re.compile(
        r"(?:p3a1r_|face[_ -]?id|instance[_ -]?id|\\graphs\\|\\steps\\)",
        re.IGNORECASE,
    )
    cached_graph_path = ""
    cached_graph: dict[str, Any] | None = None

    for record in canonical:
        record_id = str(record["source_ref"]["canonical_record_id"])
        validation = validate_intent_result(record["intent_result"])
        if not validation["valid"]:
            failures.append({"record_id": record_id, "reason": "p2_schema_or_policy_invalid"})
            continue
        if identity_pattern.search(str(record["query"]["text"])):
            failures.append({"record_id": record_id, "reason": "internal_identity_in_query"})
            continue
        graph_path = str(record["source_ref"]["graph_path"])
        if graph_path != cached_graph_path:
            cached_graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
            cached_graph_path = graph_path
        if cached_graph is None:
            raise AssertionError("graph cache was not initialized")
        graph = cached_graph
        feature = str(graph["feature_sets"][0]["macro_feature"])
        quantifier = _record_quantifier(record)
        status = str(record["intent_result"]["status"])
        split = str(record["source_ref"]["dataset_split"])
        candidates = sorted(graph["feature_sets"], key=lambda item: int(item["instance_id"]))
        rows = sorted(graph["metric_table"], key=lambda item: int(item["instance_id"]))
        stability = _stability(graph)
        status_counts[status] += 1
        quantifier_counts[quantifier] += 1
        feature_quantifier[(feature, quantifier)] += 1
        level = "L2" if quantifier in {"all", "count"} else "L4"
        level_status[(level, status)] += 1
        canonical_splits[split] += 1
        operation_counts["local_mesh_size"] += 1
        if quantifier in {"largest", "smallest"}:
            metric_counts[str(rows[0]["comparison_metric"])] += 1

        expected_clarification = (
            quantifier in {"largest", "smallest"} and not stability["comparison_stable"]
            or quantifier == "nearest" and not stability["nearest_stable"]
        )
        if status != "VALID":
            if not expected_clarification:
                failures.append({"record_id": record_id, "reason": "unnecessary_clarification"})
            else:
                metric_passed += 1
                candidate_selected[(len(candidates), 0)] += 1
            continue
        if expected_clarification:
            failures.append({"record_id": record_id, "reason": "unstable_quantifier_marked_valid"})
            continue

        target = record["ground_truth"]["targets"][0]
        if quantifier in {"all", "count"}:
            expected_ids = [int(item["instance_id"]) for item in candidates]
        elif quantifier == "largest":
            expected_ids = [int(max(rows, key=lambda item: float(item["comparison_value"]))["instance_id"])]
        elif quantifier == "smallest":
            expected_ids = [int(min(rows, key=lambda item: float(item["comparison_value"]))["instance_id"])]
        else:
            expected_ids = [int(min(rows, key=lambda item: float(item["nearest_distance"]))["instance_id"])]
        expected_faces = sorted({
            int(index)
            for item in candidates
            if int(item["instance_id"]) in expected_ids
            for index in item["node_indices"]
        })
        if (
            target["target_instance_ids"] == expected_ids
            and target["target_face_indices"] == expected_faces
        ):
            reconstruction_passed += 1
        else:
            failures.append({"record_id": record_id, "reason": "target_reconstruction_mismatch"})
        metric_passed += 1
        candidate_selected[(len(candidates), len(expected_ids))] += 1
        if level == "L4":
            executable[quantifier] += 1
            executable_by_split[(split, quantifier)] += 1
            executable_features[quantifier].add(feature)

    rendering_counts = Counter(item["language"] for item in renderings)
    rendering_cardinality = Counter(item["canonical_record_id"] for item in renderings)
    unique_rendering_intents: dict[str, dict[str, Any]] = {}
    for item in renderings:
        payload = json.dumps(item["intent_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        unique_rendering_intents.setdefault(payload, item["intent_result"])
    rendering_intent_failures = sum(
        not validate_intent_result(intent)["valid"]
        for intent in unique_rendering_intents.values()
    )
    if rendering_intent_failures:
        failures.append({"record_id": "*", "reason": "rendering_p2_schema_or_policy_invalid"})
    if any(count != 3 for count in rendering_cardinality.values()):
        failures.append({"record_id": "*", "reason": "rendering_cardinality_not_three"})
    if any(identity_pattern.search(str(item["text"])) for item in renderings):
        failures.append({"record_id": "*", "reason": "internal_identity_in_rendering"})
    leakage = _split_leakage_audit(canonical, renderings)
    if not leakage["automatic_pass"]:
        failures.append({"record_id": "*", "reason": "split_leakage"})

    valid_total = status_counts["VALID"]
    l2_valid = level_status[("L2", "VALID")]
    result = {
        "schema_version": "p3-a1r-formal-query-audit/1",
        "canonical_records": len(canonical),
        "renderings": len(renderings),
        "status_counts": dict(sorted(status_counts.items())),
        "quantifier_counts": dict(sorted(quantifier_counts.items())),
        "L2_executable_multi": l2_valid,
        "L2_all": quantifier_counts["all"],
        "L2_count": quantifier_counts["count"],
        "L4_executable": sum(executable.values()),
        "L4_by_quantifier": dict(sorted(executable.items())),
        "L4_feature_types": {
            key: sorted(value) for key, value in sorted(executable_features.items())
        },
        "L4_by_split": {
            f"{split}:{quantifier}": count
            for (split, quantifier), count in sorted(executable_by_split.items())
        },
        "comparison_metric_counts": dict(sorted(metric_counts.items())),
        "target_reconstruction": {
            "passed": reconstruction_passed,
            "total": valid_total,
        },
        "metric_recomputation": {
            "passed": metric_passed,
            "total": len(canonical),
        },
        "feature_quantifier_distribution": {
            f"{feature}:{quantifier}": count
            for (feature, quantifier), count in sorted(feature_quantifier.items())
        },
        "level_status_distribution": {
            f"{level}:{status}": count
            for (level, status), count in sorted(level_status.items())
        },
        "candidate_selected_distribution": {
            f"{candidate}:{selected}": count
            for (candidate, selected), count in sorted(candidate_selected.items())
        },
        "operation_distribution": dict(sorted(operation_counts.items())),
        "canonical_split_distribution": dict(sorted(canonical_splits.items())),
        "rendering_language_distribution": dict(sorted(rendering_counts.items())),
        "rendering_p2_validation": {
            "records_covered": len(renderings),
            "unique_intent_result_payloads": len(unique_rendering_intents),
            "failures": rendering_intent_failures,
        },
        "split_leakage": leakage,
        "failure_count": len(failures),
        "failure_examples": failures[:50],
    }
    result["automatic_pass"] = (
        not failures
        and len(canonical) == 3060
        and len(renderings) == 9180
        and l2_valid >= 900
        and quantifier_counts["all"] >= 450
        and quantifier_counts["count"] >= 450
        and sum(executable.values()) >= 900
        and all(executable[name] >= 300 for name in ("largest", "smallest", "nearest"))
        and all(len(executable_features[name]) >= 12 for name in ("largest", "smallest", "nearest"))
        and all(
            executable_by_split[(split, quantifier)] > 0
            for split in ("train", "val", "test")
            for quantifier in ("largest", "smallest", "nearest")
        )
        and len(metric_counts) >= 4
        and reconstruction_passed == valid_total
        and metric_passed == len(canonical)
        and leakage["automatic_pass"]
    )
    return result


def build(geometry_root: Path) -> dict[str, Any]:
    geometry_root = geometry_root.resolve()
    query_root = geometry_root / "queries"
    report_root = geometry_root / "reports"
    query_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(geometry_root / "manifests" / "sample_manifest.jsonl")
    canonical: list[dict[str, Any]] = []
    renderings: list[dict[str, Any]] = []
    for sample in samples:
        graph = json.loads(Path(sample["graph_path"]).read_text(encoding="utf-8"))
        for quantifier in QUANTIFIERS:
            record, variants = _build_record(sample, graph, quantifier)
            record, variants = _formalize_record(record, variants)
            canonical.append(record)
            renderings.extend(variants)
    audit = _audit(canonical, renderings)
    if not audit["automatic_pass"]:
        raise RuntimeError(f"formal query audit failed: {audit['failure_examples'][:10]}")

    canonical_path = query_root / "canonical.jsonl"
    renderings_path = query_root / "renderings.jsonl"
    canonical_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in canonical),
        encoding="utf-8",
        newline="\n",
    )
    renderings_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in renderings),
        encoding="utf-8",
        newline="\n",
    )
    audit["canonical_sha256"] = sha256_file(canonical_path)
    audit["renderings_sha256"] = sha256_file(renderings_path)
    (report_root / "query_semantic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "schema_version": FORMAL_QUERY_SCHEMA,
        "status": "FORMAL_QUERY_SUPPLEMENT_COMPLETE",
        "canonical_records": len(canonical),
        "renderings": len(renderings),
        "canonical_sha256": audit["canonical_sha256"],
        "renderings_sha256": audit["renderings_sha256"],
        "query_audit": audit,
        "scope_guards": {
            "manual_templates_only": True,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p4_entered": False,
            "forbidden_data_accessed": False,
        },
    }
    (geometry_root / "query_dataset_meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P3-A1R formal deterministic query supplement")
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    args = parser.parse_args()
    result = build(args.geometry_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
