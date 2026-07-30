from __future__ import annotations

import copy
import importlib.util
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .constants import (
    CANONICAL_FEATURES,
    LEVELS,
    OPERATIONS,
    P2_GROUNDING_SCHEMA,
    P2_INTENT_SCHEMA,
    P2_POLICY,
    P2_RESULT_SCHEMA,
    RECORD_VERSION,
    ROOT,
    STATUSES,
    SOURCE_MACRO_TO_CANONICAL,
    assert_source_path,
)
from .io import canonical_sha256, read_json, read_yaml, sha256_file
from .renderer import audit_template_manifest, normalize_exact_text


RECORD_SCHEMA = ROOT / "schemas" / "cae_query_ground_record.schema.json"
DATASET_MANIFEST_SCHEMA = ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json"
P2_API = ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "utils" / "cae_intent.py"


@lru_cache(maxsize=1)
def p2_api():
    spec = importlib.util.spec_from_file_location("topocae_p2_frozen_cae_intent", P2_API)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen P2 API: {P2_API}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _validator(kind: str) -> Draft202012Validator:
    paths = [P2_INTENT_SCHEMA, P2_RESULT_SCHEMA, P2_GROUNDING_SCHEMA, RECORD_SCHEMA, DATASET_MANIFEST_SCHEMA]
    schemas = [read_json(path) for path in paths]
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas]
    )
    target = read_json(RECORD_SCHEMA if kind == "record" else DATASET_MANIFEST_SCHEMA)
    return Draft202012Validator(target, registry=registry)


def _schema_errors(value: Any, kind: str) -> list[str]:
    return [
        f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(_validator(kind).iter_errors(value), key=lambda item: list(item.absolute_path))
    ]


@lru_cache(maxsize=512)
def _graph(path: str) -> dict[str, Any]:
    return read_json(assert_source_path(path))


@lru_cache(maxsize=1)
def _crosswalk() -> dict[str, Any]:
    return read_yaml(ROOT / "configs" / "paper_v1" / "cae_query_role_crosswalk.yaml")


def semantic_identity(record: dict[str, Any]) -> str:
    result = copy.deepcopy(record["intent_result"])
    if result.get("status") == "VALID":
        result["intent"]["source_language"] = "__LANGUAGE__"
    return canonical_sha256({"intent_result": result, "ground_truth": record["ground_truth"]})


def _number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _typed_quantities(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"dimension", "value", "unit"}.issubset(value):
            yield value
            return
        for child in value.values():
            yield from _typed_quantities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _typed_quantities(child)


def _query_identity_errors(record: dict[str, Any]) -> list[str]:
    text = record["query"]["text"]
    source = record["source_ref"]
    forbidden_literals = {
        str(source["sample_id"]),
        Path(source["graph_path"]).name,
        Path(source["geometry_path"]).name,
        str(source["graph_path"]),
        str(source["geometry_path"]),
    }
    errors = [f"query leaks internal literal {literal!r}" for literal in forbidden_literals if literal and literal in text]
    lowered = text.casefold()
    for token in ("sample_id", "face_id", "instance_id", "graph_sha256", "geometry_source", "area_truth"):
        if token in lowered:
            errors.append(f"query leaks forbidden field name {token}")

    serialized_intent = json.dumps(record["intent_result"], ensure_ascii=False, sort_keys=True)
    for literal in forbidden_literals:
        if literal and literal in serialized_intent:
            errors.append(f"intent leaks internal literal {literal!r}")
    return errors


def _query_semantic_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = record["query"]["text"]
    lowered = text.casefold()
    result = record["intent_result"]
    if result["status"] == "VALID":
        for quantity in _typed_quantities(result["intent"]):
            number = _number(quantity["value"])
            if number not in text or str(quantity["unit"]) not in text:
                errors.append(f"query omits or changes typed quantity {number} {quantity['unit']}")
        for operation in result["intent"]["operations"]:
            target = operation["target"]
            if target.get("quantifier") == "count" and str(target["count"]) not in text:
                errors.append("query omits count quantifier value")
            direction = operation.get("parameters", {}).get("direction")
            if isinstance(direction, dict) and direction.get("kind") == "surface_normal":
                if "surface normal" not in lowered and "表面法向" not in text:
                    errors.append("query omits surface-normal direction")
            if isinstance(direction, dict) and direction.get("kind") == "opposite_surface_normal":
                if "opposite" not in lowered and "反向" not in text:
                    errors.append("query omits opposite-normal direction")
    elif result["status"] == "NEEDS_CLARIFICATION" and result.get("field_path", "").endswith("/direction"):
        if "surface normal" in lowered or "表面法向" in text:
            errors.append("missing-direction query accidentally supplies a direction")
    elif result["status"] == "INVALID":
        if "except" not in lowered and "排除" not in text:
            errors.append("unsupported-negation query does not retain negation")
    return errors


def _target_reconstruction_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source = record["source_ref"]
    graph = _graph(source["graph_path"])
    crosswalk = _crosswalk()
    if sha256_file(Path(source["graph_path"])) != source["graph_sha256"]:
        errors.append("record graph hash does not match source file")
        return errors
    node_by_idx = {int(node["idx"]): node for node in graph["nodes"]}
    feature_sets = {int(item["instance_id"]): item for item in graph["feature_sets"]}
    ground = record["ground_truth"]
    if ground["contract_status"] == "non_executable":
        if ground["targets"]:
            errors.append("non-executable record carries targets")
        return errors

    intent_ops = {operation["operation_id"]: operation for operation in record["intent_result"]["intent"]["operations"]}
    if set(intent_ops) != {target["operation_id"] for target in ground["targets"]}:
        errors.append("ground-truth operation IDs do not match intent")
    for target in ground["targets"]:
        operation = intent_ops.get(target["operation_id"])
        if operation is None or operation["operation_type"] != target["operation_type"]:
            errors.append("ground target operation mismatch")
            continue
        if target["target_contract"] == "whole_part":
            if target["grounding_required"] or "feature_type" in operation["target"]:
                errors.append("whole-part target has fake grounding")
            continue
        canonical_feature = target["canonical_feature_type"]
        role_name = target["face_role"]
        policy_roles = {
            role["face_role"]: role for role in crosswalk["features"][canonical_feature]["face_roles"]
        }
        role = policy_roles.get(role_name)
        if role is None or str(role["selection_status"]).startswith("unsupported"):
            errors.append(f"unsupported or unknown exact role {canonical_feature}.{role_name}")
            continue
        union_faces: set[int] = set()
        evidence_instances: list[int] = []
        for evidence in target["semantic_evidence"]:
            instance_id = int(evidence["instance_id"])
            evidence_instances.append(instance_id)
            feature_set = feature_sets.get(instance_id)
            if feature_set is None:
                errors.append(f"instance {instance_id} absent from graph")
                continue
            source_macro = str(feature_set["macro_feature"])
            if SOURCE_MACRO_TO_CANONICAL.get(source_macro) != canonical_feature:
                errors.append(f"instance {instance_id} canonical feature mismatch")
            instance_faces = sorted(int(index) for index in feature_set["node_indices"])
            if evidence["instance_face_indices"] != instance_faces:
                errors.append(f"instance {instance_id} face-set mismatch")
            selected = sorted(int(index) for index in evidence["selected_face_indices"])
            if not selected or not set(selected).issubset(instance_faces):
                errors.append(f"instance {instance_id} selected faces violate instance boundary")
            if role["requires_complete_instance"]:
                expected = instance_faces
            else:
                candidates = set(role["semantic_subtype_candidates"])
                expected = sorted(index for index in instance_faces if str(node_by_idx[index]["semantic_role"]) in candidates)
            if selected != expected:
                errors.append(f"instance {instance_id} target cannot be reconstructed from crosswalk")
            observed = sorted({str(node_by_idx[index]["semantic_role"]) for index in instance_faces})
            if evidence["observed_semantic_roles"] != observed:
                errors.append(f"instance {instance_id} semantic evidence mismatch")
            union_faces.update(selected)
        if sorted(evidence_instances) != target["target_instance_ids"]:
            errors.append("target instance list differs from evidence")
        if sorted(union_faces) != target["target_face_indices"]:
            errors.append("target face union differs from per-instance evidence")
    return errors


def validate_record(record: dict[str, Any]) -> list[str]:
    errors = _schema_errors(record, "record")
    if errors:
        return errors
    if record["record_schema_version"] != RECORD_VERSION:
        errors.append("record version mismatch")
    p2_validation = p2_api().validate_intent_result(record["intent_result"])
    if not p2_validation["valid"]:
        errors.extend(f"P2: {item}" for item in p2_validation["errors"])
    if record["ground_truth"]["intent_status"] != record["intent_result"]["status"]:
        errors.append("intent-result and ground-truth status differ")
    if record["query"]["language"] == "zh-CN" and record["intent_result"]["status"] == "VALID":
        if record["intent_result"]["intent"]["source_language"] != "zh-CN":
            errors.append("Chinese query source_language mismatch")
    errors.extend(_query_identity_errors(record))
    errors.extend(_query_semantic_errors(record))
    errors.extend(_target_reconstruction_errors(record))
    if "placeholder" in record["query"]["text"].casefold() or "{" in record["query"]["text"]:
        errors.append("query contains placeholder")
    return errors


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    return _schema_errors(manifest, "manifest")


def split_leakage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_text: dict[str, set[str]] = defaultdict(set)
    grouped_families: dict[str, set[str]] = defaultdict(set)
    geometry_groups: dict[str, dict[str, set[str]]] = {
        "base_geometry_family_id": defaultdict(set),
        "exact_graph_signature": defaultdict(set),
        "near_graph_signature": defaultdict(set),
    }
    for record in records:
        split = record["source_ref"]["dataset_split"]
        grouped_text[split].add(normalize_exact_text(record["query"]["text"]))
        grouped_families[split].add(record["query"]["template_family_id"])
        for key in geometry_groups:
            geometry_groups[key][record["source_ref"][key]].add(split)

    def pair_collisions(groups: dict[str, set[str]]) -> dict[str, list[str]]:
        collisions: dict[str, list[str]] = {}
        splits = sorted(groups)
        for index, left in enumerate(splits):
            for right in splits[index + 1:]:
                overlap = sorted(groups[left] & groups[right])
                if overlap:
                    collisions[f"{left}:{right}"] = overlap[:100]
        return collisions

    geometry = {
        key: {
            "collision_count": sum(1 for splits in values.values() if len(splits) > 1),
            "examples": {
                value: sorted(splits)
                for value, splits in list((item for item in values.items() if len(item[1]) > 1))[:20]
            },
        }
        for key, values in geometry_groups.items()
    }
    text_collisions = pair_collisions(grouped_text)
    template_collisions = pair_collisions(grouped_families)
    template_manifest = audit_template_manifest()
    total = (
        sum(item["collision_count"] for item in geometry.values())
        + sum(len(value) for value in text_collisions.values())
        + sum(len(value) for value in template_collisions.values())
        + sum(len(value) for value in template_manifest["cross_split_collisions"].values())
    )
    return {
        "schema_version": "topocae_cae_query_split_leakage_audit_1",
        "automatic_pass": total == 0,
        "geometry": geometry,
        "rotation_scale_family": geometry["base_geometry_family_id"],
        "normalized_exact_text": {"collision_count": sum(len(value) for value in text_collisions.values()), "examples": text_collisions},
        "template_family": {"collision_count": sum(len(value) for value in template_collisions.values()), "examples": template_collisions},
        "template_manifest": template_manifest,
        "total_collisions": total,
    }


def audit_pilot(
    plans: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    rendering_records: list[dict[str, Any]],
) -> dict[str, Any]:
    all_records = canonical_records + rendering_records
    record_errors: list[dict[str, Any]] = []
    reconstruction_total = 0
    reconstruction_passed = 0
    for record in all_records:
        errors = validate_record(record)
        if errors and len(record_errors) < 100:
            record_errors.append({"rendering_id": record["query"]["rendering_id"], "errors": errors})
        if record["record_kind"] == "canonical":
            feature_targets = [
                target
                for target in record["ground_truth"]["targets"]
                if target.get("target_contract") == "feature_face_set"
            ]
            reconstruction_total += len(feature_targets)
            if not [error for error in errors if "target" in error or "instance" in error or "face" in error]:
                reconstruction_passed += len(feature_targets)

    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_canonical[record["source_ref"]["canonical_record_id"]].append(record)
    identity_failures = 0
    rendering_cardinality_failures = 0
    for records in by_canonical.values():
        identities = {semantic_identity(record) for record in records}
        if len(identities) != 1:
            identity_failures += 1
        kinds = Counter(record["record_kind"] for record in records)
        languages = Counter(record["query"]["language"] for record in records if record["record_kind"] == "rendering")
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
    leakage = split_leakage_audit(rendering_records)
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
