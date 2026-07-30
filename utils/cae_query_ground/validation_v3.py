"""Equivalent grouped validation for canonical records and invariant renderings."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Any

from .constants import CANONICAL_FEATURES, LEVELS, OPERATIONS, STATUSES
from . import validation as base
from .validation_v2 import cached_sha256_file  # installs immutable graph-hash cache


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


def _rendering_invariant_errors(record: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(record) != {"record_schema_version", "record_kind", "source_ref", "query", "intent_result", "ground_truth"}:
        errors.append("rendering top-level fields violate closed record schema")
        return errors
    if record["record_kind"] != "rendering":
        errors.append("derived language record is not marked rendering")
    if record["record_schema_version"] != canonical["record_schema_version"]:
        errors.append("rendering record schema version differs from canonical")
    if record["source_ref"] != canonical["source_ref"]:
        errors.append("rendering source_ref differs from canonical")
    if record["ground_truth"] != canonical["ground_truth"]:
        errors.append("rendering ground_truth differs from canonical")
    if base.semantic_identity(record) != base.semantic_identity(canonical):
        errors.append("rendering semantic identity differs from canonical")

    query = record.get("query")
    if not isinstance(query, dict) or set(query) != QUERY_KEYS:
        errors.append("rendering query violates closed query schema")
    else:
        if not isinstance(query["text"], str) or len(query["text"]) < 5:
            errors.append("rendering query text is empty/short")
        if query["language"] not in {"en-US", "zh-CN"}:
            errors.append("rendering query language invalid")
        if query["rendering_variant"] not in {"en-1", "en-2", "zh-1"}:
            errors.append("rendering variant invalid")
        if query["difficulty_level"] not in LEVELS:
            errors.append("rendering difficulty level invalid")
        for field in ("template_family_id", "template_id", "rendering_id"):
            if not isinstance(query[field], str) or not IDENTIFIER.fullmatch(query[field]):
                errors.append(f"rendering {field} invalid")
        expected_language = "zh-CN" if query["rendering_variant"] == "zh-1" else "en-US"
        if query["language"] != expected_language:
            errors.append("rendering variant/language mismatch")

    p2_validation = base.p2_api().validate_intent_result(record["intent_result"])
    if not p2_validation["valid"]:
        errors.extend(f"P2: {item}" for item in p2_validation["errors"])
    if record["intent_result"]["status"] == "VALID":
        if record["intent_result"]["intent"]["source_language"] != record["query"]["language"]:
            errors.append("rendering source_language does not match query")
    errors.extend(base._query_identity_errors(record))
    errors.extend(base._query_semantic_errors(record))
    if "placeholder" in record["query"]["text"].casefold() or "{" in record["query"]["text"]:
        errors.append("rendering query contains placeholder")
    return errors


def audit_pilot(
    plans: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    rendering_records: list[dict[str, Any]],
) -> dict[str, Any]:
    record_errors: list[dict[str, Any]] = []
    canonical_by_id = {
        record["source_ref"]["canonical_record_id"]: record for record in canonical_records
    }
    reconstruction_total = 0
    reconstruction_passed = 0
    for record in canonical_records:
        errors = base.validate_record(record)
        if errors and len(record_errors) < 100:
            record_errors.append({"rendering_id": record["query"]["rendering_id"], "errors": errors})
        feature_targets = [
            target
            for target in record["ground_truth"]["targets"]
            if target.get("target_contract") == "feature_face_set"
        ]
        reconstruction_total += len(feature_targets)
        target_errors = [
            error
            for error in errors
            if any(token in error for token in ("target", "instance", "face", "crosswalk"))
        ]
        if not target_errors:
            reconstruction_passed += len(feature_targets)

    for record in rendering_records:
        canonical_id = record["source_ref"]["canonical_record_id"]
        canonical = canonical_by_id.get(canonical_id)
        errors = ["rendering references unknown canonical"] if canonical is None else _rendering_invariant_errors(record, canonical)
        if errors and len(record_errors) < 100:
            record_errors.append({"rendering_id": record["query"]["rendering_id"], "errors": errors})

    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical_records + rendering_records:
        by_canonical[record["source_ref"]["canonical_record_id"]].append(record)
    identity_failures = 0
    rendering_cardinality_failures = 0
    for records in by_canonical.values():
        if len({base.semantic_identity(record) for record in records}) != 1:
            identity_failures += 1
        kinds = Counter(record["record_kind"] for record in records)
        languages = Counter(
            record["query"]["language"] for record in records if record["record_kind"] == "rendering"
        )
        if kinds != {"canonical": 1, "rendering": 3} or languages != {"en-US": 2, "zh-CN": 1}:
            rendering_cardinality_failures += 1

    counts = {
        "canonical_records": len(canonical_records),
        "renderings": len(rendering_records),
        "splits": dict(sorted(Counter(record["source_ref"]["dataset_split"] for record in canonical_records).items())),
        "languages": dict(sorted(Counter(record["query"]["language"] for record in rendering_records).items())),
        "levels": dict(sorted(Counter(plan["level"] for plan in plans).items())),
        "statuses": dict(sorted(Counter(plan["status"] for plan in plans).items())),
        "features": dict(sorted(Counter(plan["verbalization"]["feature_type"] for plan in plans).items())),
        "operations": dict(
            sorted(
                Counter(
                    operation["operation_type"]
                    for plan in plans
                    for operation in plan["verbalization"]["operations"]
                ).items()
            )
        ),
    }
    coverage = {
        "features": sorted(counts["features"]) == sorted(CANONICAL_FEATURES),
        "operations": sorted(counts["operations"]) == sorted(OPERATIONS),
        "levels": sorted(counts["levels"]) == sorted(LEVELS),
        "statuses": sorted(counts["statuses"]) == sorted(STATUSES),
        "languages": sorted(counts["languages"]) == ["en-US", "zh-CN"],
    }
    leakage = base.split_leakage_audit(rendering_records)
    automatic_pass = (
        len(canonical_records) >= 800
        and len(rendering_records) >= 2400
        and not record_errors
        and reconstruction_passed == reconstruction_total
        and identity_failures == 0
        and rendering_cardinality_failures == 0
        and all(coverage.values())
        and leakage["automatic_pass"]
    )
    return {
        "schema_version": "topocae_cae_query_pilot_quality_audit_1",
        "automatic_pass": automatic_pass,
        "validation_proof": {
            "canonical_full_record_schema_and_oracle_reconstruction": len(canonical_records),
            "rendering_closed_query_p2_and_canonical_invariant": len(rendering_records),
            "reason": "Renderings may change only query metadata/text and VALID source_language; source_ref and ground_truth are byte-equivalent to the fully validated canonical record.",
        },
        "counts": counts,
        "coverage": coverage,
        "record_schema_and_semantic_failures": len(record_errors),
        "record_error_examples": record_errors,
        "target_reconstruction": {"passed": reconstruction_passed, "total": reconstruction_total},
        "rendering_identity_failures": identity_failures,
        "rendering_cardinality_failures": rendering_cardinality_failures,
        "split_leakage": leakage,
        "scope_guards": {
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
            "formal_scale_generated": False,
            "p4_entered": False,
        },
    }


validate_manifest = base.validate_manifest

__all__ = ["audit_pilot", "validate_manifest"]
