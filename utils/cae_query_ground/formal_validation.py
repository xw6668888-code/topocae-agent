from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from . import validation as base
from .constants import (
    CANONICAL_FEATURES,
    LEVELS,
    OPERATIONS,
    P2_GROUNDING_SCHEMA,
    P2_INTENT_SCHEMA,
    P2_RESULT_SCHEMA,
    STATUSES,
)
from .formal_contract import FINAL_MANIFEST_SCHEMA, FINAL_RECORD_SCHEMA
from .formal_planner import FINAL_RECORD_VERSION
from .formal_renderer_v2 import audit_formal_template_manifest, normalize_exact_text
from .io import canonical_sha256, read_json
from .validation_v2 import cached_sha256_file


base.sha256_file = cached_sha256_file
QUERY_KEYS = {
    "text",
    "language",
    "template_family_id",
    "template_id",
    "rendering_id",
    "rendering_variant",
    "difficulty_level",
}
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


@lru_cache(maxsize=2)
def _validator(kind: str) -> Draft202012Validator:
    paths = (
        P2_INTENT_SCHEMA,
        P2_RESULT_SCHEMA,
        P2_GROUNDING_SCHEMA,
        FINAL_RECORD_SCHEMA,
        FINAL_MANIFEST_SCHEMA,
    )
    schemas = [read_json(path) for path in paths]
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas]
    )
    target = read_json(FINAL_RECORD_SCHEMA if kind == "record" else FINAL_MANIFEST_SCHEMA)
    return Draft202012Validator(target, registry=registry)


def schema_errors(value: Any, kind: str) -> list[str]:
    return [
        f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(
            _validator(kind).iter_errors(value), key=lambda item: list(item.absolute_path)
        )
    ]


@lru_cache(maxsize=65536)
def _p2_cached(payload: str) -> dict[str, Any]:
    return base.p2_api().validate_intent_result(json.loads(payload))


def _p2(value: Any) -> dict[str, Any]:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return copy.deepcopy(_p2_cached(payload))


def validate_formal_record(record: dict[str, Any]) -> list[str]:
    errors = schema_errors(record, "record")
    if errors:
        return errors
    if record["record_schema_version"] != FINAL_RECORD_VERSION:
        errors.append("formal record version mismatch")
    p2 = _p2(record["intent_result"])
    if not p2["valid"]:
        errors.extend(f"P2: {item}" for item in p2["errors"])
    if record["ground_truth"]["intent_status"] != record["intent_result"]["status"]:
        errors.append("intent-result and ground-truth status differ")
    if record["intent_result"]["status"] == "VALID":
        if record["intent_result"]["intent"]["source_language"] != record["query"]["language"]:
            errors.append("VALID source_language differs from query language")
    errors.extend(base._query_identity_errors(record))
    errors.extend(base._query_semantic_errors(record))
    errors.extend(base._target_reconstruction_errors(record))
    if "placeholder" in record["query"]["text"].casefold() or "{" in record["query"]["text"]:
        errors.append("query contains placeholder")
    return errors


def _rendering_errors(record: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_top = {
        "record_schema_version",
        "record_kind",
        "source_ref",
        "query",
        "intent_result",
        "ground_truth",
    }
    if set(record) != expected_top:
        return ["rendering top-level fields violate closed schema"]
    if record["record_kind"] != "rendering":
        errors.append("derived record is not rendering")
    if record["record_schema_version"] != FINAL_RECORD_VERSION:
        errors.append("rendering version mismatch")
    if record["source_ref"] != canonical["source_ref"]:
        errors.append("rendering source_ref differs from canonical")
    if record["ground_truth"] != canonical["ground_truth"]:
        errors.append("rendering ground_truth differs from canonical")
    if base.semantic_identity(record) != base.semantic_identity(canonical):
        errors.append("rendering semantic identity differs from canonical")
    query = record.get("query")
    if not isinstance(query, dict) or set(query) != QUERY_KEYS:
        errors.append("rendering query violates closed schema")
    else:
        expected_language = "zh-CN" if query["rendering_variant"] == "zh-1" else "en-US"
        if query["rendering_variant"] not in {"en-1", "en-2", "zh-1"}:
            errors.append("invalid rendering variant")
        if query["language"] != expected_language:
            errors.append("rendering language/variant mismatch")
        if query["difficulty_level"] not in LEVELS:
            errors.append("invalid rendering level")
        if not isinstance(query["text"], str) or len(query["text"]) < 5:
            errors.append("empty or short rendering text")
        for field in ("template_family_id", "template_id", "rendering_id"):
            if not isinstance(query[field], str) or not IDENTIFIER.fullmatch(query[field]):
                errors.append(f"invalid rendering {field}")
    p2 = _p2(record["intent_result"])
    if not p2["valid"]:
        errors.extend(f"P2: {item}" for item in p2["errors"])
    if record["intent_result"]["status"] == "VALID":
        if record["intent_result"]["intent"]["source_language"] != record["query"]["language"]:
            errors.append("rendering source_language mismatch")
    errors.extend(base._query_identity_errors(record))
    errors.extend(base._query_semantic_errors(record))
    if "placeholder" in record["query"]["text"].casefold() or "{" in record["query"]["text"]:
        errors.append("rendering contains placeholder")
    return errors


def split_leakage_audit(
    canonical_records: list[dict[str, Any]], rendering_records: list[dict[str, Any]]
) -> dict[str, Any]:
    texts: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    geometry_groups: dict[str, dict[str, set[str]]] = {
        key: defaultdict(set)
        for key in ("base_geometry_family_id", "exact_graph_signature", "near_graph_signature")
    }
    for record in canonical_records + rendering_records:
        split = record["source_ref"]["dataset_split"]
        texts[split].add(normalize_exact_text(record["query"]["text"]))
        families[split].add(record["query"]["template_family_id"])
    for record in canonical_records:
        split = record["source_ref"]["dataset_split"]
        for key, groups in geometry_groups.items():
            groups[record["source_ref"][key]].add(split)

    def pairwise(groups: dict[str, set[str]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        splits = sorted(groups)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                overlap = sorted(groups[left] & groups[right])
                if overlap:
                    result[f"{left}:{right}"] = overlap[:100]
        return result

    geometry = {
        key: {
            "collision_count": sum(len(splits) > 1 for splits in groups.values()),
            "examples": {
                value: sorted(splits)
                for value, splits in list(
                    (item for item in groups.items() if len(item[1]) > 1)
                )[:20]
            },
        }
        for key, groups in geometry_groups.items()
    }
    text_collisions = pairwise(texts)
    family_collisions = pairwise(families)
    template_manifest = audit_formal_template_manifest()
    total = (
        sum(item["collision_count"] for item in geometry.values())
        + sum(len(values) for values in text_collisions.values())
        + sum(len(values) for values in family_collisions.values())
        + sum(len(values) for values in template_manifest["cross_split_collisions"].values())
    )
    return {
        "schema_version": "topocae_cae_query_split_leakage_audit_formal_1",
        "automatic_pass": total == 0,
        "geometry": geometry,
        "rotation_scale_family": geometry["base_geometry_family_id"],
        "normalized_exact_text": {
            "collision_count": sum(len(values) for values in text_collisions.values()),
            "examples": text_collisions,
        },
        "template_family": {
            "collision_count": sum(len(values) for values in family_collisions.values()),
            "examples": family_collisions,
        },
        "template_manifest": template_manifest,
        "total_collisions": total,
    }


def audit_formal(
    plans: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    rendering_records: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = 0
    examples: list[dict[str, Any]] = []
    target_total = 0
    target_passed = 0
    for record in canonical_records:
        errors = validate_formal_record(record)
        if errors:
            failures += 1
            if len(examples) < 100:
                examples.append({"rendering_id": record["query"]["rendering_id"], "errors": errors})
        feature_targets = [
            target
            for target in record["ground_truth"]["targets"]
            if target.get("target_contract") == "feature_face_set"
        ]
        target_total += len(feature_targets)
        target_errors = base._target_reconstruction_errors(record)
        if not target_errors:
            target_passed += len(feature_targets)

    canonical_by_id = {
        record["source_ref"]["canonical_record_id"]: record for record in canonical_records
    }
    for record in rendering_records:
        canonical = canonical_by_id.get(record["source_ref"]["canonical_record_id"])
        errors = ["unknown canonical"] if canonical is None else _rendering_errors(record, canonical)
        if errors:
            failures += 1
            if len(examples) < 100:
                examples.append({"rendering_id": record["query"]["rendering_id"], "errors": errors})

    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical_records + rendering_records:
        by_canonical[record["source_ref"]["canonical_record_id"]].append(record)
    identity_failures = 0
    cardinality_failures = 0
    for records in by_canonical.values():
        if len({base.semantic_identity(record) for record in records}) != 1:
            identity_failures += 1
        kinds = Counter(record["record_kind"] for record in records)
        languages = Counter(
            record["query"]["language"] for record in records if record["record_kind"] == "rendering"
        )
        if kinds != {"canonical": 1, "rendering": 3} or languages != {"en-US": 2, "zh-CN": 1}:
            cardinality_failures += 1

    counts = {
        "canonical_records": len(canonical_records),
        "renderings": len(rendering_records),
        "source_graphs": len({record["source_ref"]["sample_id"] for record in canonical_records}),
        "splits": dict(sorted(Counter(record["source_ref"]["dataset_split"] for record in canonical_records).items())),
        "languages": dict(sorted(Counter(record["query"]["language"] for record in rendering_records).items())),
        "levels": dict(sorted(Counter(plan["level"] for plan in plans).items())),
        "statuses": dict(sorted(Counter(plan["status"] for plan in plans).items())),
        "features": dict(sorted(Counter(plan["verbalization"]["feature_type"] for plan in plans).items())),
        "operations": dict(sorted(Counter(
            operation["operation_type"]
            for plan in plans
            for operation in plan["verbalization"]["operations"]
        ).items())),
        "target_face_counts": dict(sorted(Counter(
            len(target["target_face_indices"])
            for record in canonical_records
            for target in record["ground_truth"]["targets"]
            if target.get("target_contract") == "feature_face_set"
        ).items())),
    }
    source_cardinality = Counter(record["source_ref"]["sample_id"] for record in canonical_records)
    duplicate_payloads = len(canonical_records) - len({canonical_sha256({
        "source_graph_sha256": record["source_ref"]["graph_sha256"],
        "query": record["query"],
        "intent_result": record["intent_result"],
        "ground_truth": record["ground_truth"],
    }) for record in canonical_records})
    coverage = {
        "features": set(counts["features"]) == set(CANONICAL_FEATURES),
        "operations": set(counts["operations"]) == set(OPERATIONS),
        "levels": set(counts["levels"]) == set(LEVELS),
        "statuses": set(counts["statuses"]) == set(STATUSES),
        "languages": set(counts["languages"]) == {"en-US", "zh-CN"},
    }
    leakage = split_leakage_audit(canonical_records, rendering_records)
    automatic_pass = (
        len(canonical_records) >= 20000
        and len(rendering_records) >= 60000
        and failures == 0
        and target_passed == target_total
        and identity_failures == 0
        and cardinality_failures == 0
        and set(source_cardinality.values()) == {7}
        and duplicate_payloads == 0
        and all(coverage.values())
        and leakage["automatic_pass"]
    )
    return {
        "schema_version": "topocae_cae_query_quality_audit_formal_1",
        "automatic_pass": automatic_pass,
        "validation_proof": {
            "canonical_full_schema_p2_semantics_and_target_reconstruction": len(canonical_records),
            "rendering_closed_schema_p2_and_canonical_invariant": len(rendering_records),
        },
        "counts": counts,
        "coverage": coverage,
        "record_schema_and_semantic_failures": failures,
        "failure_examples": examples,
        "target_reconstruction": {"passed": target_passed, "total": target_total},
        "rendering_identity_failures": identity_failures,
        "rendering_cardinality_failures": cardinality_failures,
        "plans_per_source": dict(sorted(Counter(source_cardinality.values()).items())),
        "duplicate_canonical_payloads": duplicate_payloads,
        "split_leakage": leakage,
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
            "test_part_s3_accessed": False,
            "p4_entered": False,
        },
    }


def validate_formal_manifest(manifest: dict[str, Any]) -> list[str]:
    return schema_errors(manifest, "manifest")

