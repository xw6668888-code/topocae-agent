from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CANONICAL_FEATURES,
    DATASET_SPLIT_MAP,
    GRAPH_SCHEMA,
    MODEL_AREA_UNIT,
    MODEL_LENGTH_UNIT,
    P2_FREEZE_MANIFEST,
    P2_POLICY,
    P2_RUNNER,
    P2_RUNNER_EXPECTED_SHA256,
    SOURCE_ARTIFACT_MANIFEST,
    SOURCE_MACRO_TO_CANONICAL,
    SOURCE_MANIFEST,
    SOURCE_META,
    SOURCE_SPLITS,
    UNIT_CONTRACT,
    assert_source_path,
)
from .io import read_json, read_jsonl, read_yaml, sha256_file


def build_role_crosswalk(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != "topocae_cae_feature_role_policy_1":
        raise ValueError("Unexpected P2 role-policy schema")
    features = policy.get("features")
    if not isinstance(features, dict) or tuple(features) != CANONICAL_FEATURES:
        raise ValueError("P2 canonical feature order/content mismatch")

    semantic_inverse: dict[str, list[dict[str, str]]] = defaultdict(list)
    feature_entries: dict[str, Any] = {}
    unsupported: list[str] = []
    for canonical, feature in features.items():
        role_entries: list[dict[str, Any]] = []
        for role in feature["selectable_roles"]:
            entry = {
                "face_role": role["role"],
                "selection_status": role["selection_status"],
                "semantic_subtype_candidates": list(role["semantic_subtype_candidates"]),
                "region_kind": role["region_kind"],
                "allowed_operations": list(role["allowed_operations"]),
                "requires_complete_instance": bool(role["requires_complete_instance"]),
            }
            role_entries.append(entry)
            qualified = f"{canonical}.{role['role']}"
            if str(role["selection_status"]).startswith("unsupported"):
                unsupported.append(qualified)
            for semantic_role in role["semantic_subtype_candidates"]:
                semantic_inverse[str(semantic_role)].append(
                    {"canonical_feature": canonical, "face_role": str(role["role"])}
                )
        source_macros = [canonical]
        if canonical == "rib":
            source_macros.append("web")
        feature_entries[canonical] = {
            "family": feature["family"],
            "source_macros": source_macros,
            "face_roles": role_entries,
        }

    return {
        "schema_version": "topocae_cae_query_role_crosswalk_1",
        "derived_from": {
            "p2_policy_schema": policy["schema_version"],
            "p2_policy_revision": policy.get("policy_revision"),
            "p2_policy_sha256": sha256_file(P2_POLICY),
        },
        "unit_provenance": {
            "length_unit": MODEL_LENGTH_UNIT,
            "area_unit": MODEL_AREA_UNIT,
            "contract": UNIT_CONTRACT,
            "brep_note": "OpenCASCADE BREP text is unitless; P1 synthetic model-coordinate convention is explicit here.",
        },
        "source_macro_aliases": {
            "web": {
                "canonical_feature": "rib",
                "preserve_source_macro": True,
                "semantic_role_map": {
                    "web_side_plane": "web_side_plane",
                    "web_top_plane": "web_top_plane",
                },
            }
        },
        "features": feature_entries,
        "semantic_role_to_p2_face_roles": {
            key: sorted(value, key=lambda item: (item["canonical_feature"], item["face_role"]))
            for key, value in sorted(semantic_inverse.items())
        },
        "unsupported_exact_roles": sorted(unsupported),
    }


def _collision_audit(records: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[str(record[key])].add(str(record["split"]))
    collisions = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    return {
        "group_count": len(groups),
        "collision_count": len(collisions),
        "examples": dict(list(sorted(collisions.items()))[:20]),
    }


def _verify_p2_freeze() -> dict[str, Any]:
    manifest = read_json(P2_FREEZE_MANIFEST)
    rows = []
    for asset in manifest["assets"]:
        path = Path(asset["path"])
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        rows.append(
            {
                "role": asset["role"],
                "path": str(path),
                "expected_sha256": str(asset["sha256"]).lower(),
                "actual_sha256": actual,
                "matches": exists and actual == str(asset["sha256"]).lower(),
            }
        )
    runner_actual = sha256_file(P2_RUNNER) if P2_RUNNER.is_file() else None
    return {
        "assets": rows,
        "asset_count": len(rows),
        "matching_count": sum(bool(row["matches"]) for row in rows),
        "mismatch_count": sum(not bool(row["matches"]) for row in rows),
        "supervisor_added_runner": {
            "path": str(P2_RUNNER),
            "expected_sha256": P2_RUNNER_EXPECTED_SHA256,
            "actual_sha256": runner_actual,
            "matches": runner_actual == P2_RUNNER_EXPECTED_SHA256,
        },
    }


def _role_lookup(crosswalk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        feature: {
            role["face_role"]: role
            for role in entry["face_roles"]
        }
        for feature, entry in crosswalk["features"].items()
    }


def audit_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_yaml(P2_POLICY)
    crosswalk = build_role_crosswalk(policy)
    role_lookup = _role_lookup(crosswalk)
    artifact_manifest = read_json(SOURCE_ARTIFACT_MANIFEST)
    dataset_meta = read_json(SOURCE_META)
    manifest_records = read_jsonl(SOURCE_MANIFEST)
    failures: list[dict[str, Any]] = []

    actual_meta_hash = sha256_file(SOURCE_META)
    actual_sample_manifest_hash = sha256_file(SOURCE_MANIFEST)
    for name, expected, actual in (
        ("dataset_meta", str(artifact_manifest["dataset_meta_sha256"]).lower(), actual_meta_hash),
        ("sample_manifest", str(artifact_manifest["sample_manifest_sha256"]).lower(), actual_sample_manifest_hash),
    ):
        if expected != actual:
            failures.append({"kind": "source_hash_mismatch", "asset": name, "expected": expected, "actual": actual})

    if len(manifest_records) != 5000 or dataset_meta.get("sample_count") != 5000:
        failures.append(
            {
                "kind": "sample_count_mismatch",
                "manifest": len(manifest_records),
                "dataset_meta": dataset_meta.get("sample_count"),
                "expected": 5000,
            }
        )
    sample_ids = [str(record["sample_id"]) for record in manifest_records]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append({"kind": "duplicate_sample_ids", "duplicate_count": len(sample_ids) - len(set(sample_ids))})

    by_id = {str(record["sample_id"]): record for record in manifest_records}
    split_counts: Counter[str] = Counter()
    split_file_hashes: dict[str, str] = {}
    split_membership_failures = 0
    for source_split, split_path in SOURCE_SPLITS.items():
        split_file_hashes[source_split] = sha256_file(split_path)
        split_records = read_jsonl(split_path)
        split_counts[source_split] = len(split_records)
        for split_record in split_records:
            sample_id = str(split_record.get("sample_id"))
            if sample_id not in by_id or split_record != by_id[sample_id] or split_record.get("split") != source_split:
                split_membership_failures += 1
    if sum(split_counts.values()) != len(manifest_records) or split_membership_failures:
        failures.append(
            {
                "kind": "split_membership_mismatch",
                "split_total": sum(split_counts.values()),
                "manifest_total": len(manifest_records),
                "record_mismatches": split_membership_failures,
            }
        )

    graph_hash_mismatches = 0
    graph_missing = 0
    geometry_missing = 0
    graph_schema_mismatches = 0
    feature_consistency_failures = 0
    feature_instance_count = 0
    source_macro_counts: Counter[str] = Counter()
    canonical_feature_counts: Counter[str] = Counter()
    semantic_role_counts: Counter[str] = Counter()
    observed_semantic_roles: set[str] = set()
    crosswalk_semantic_roles = set(crosswalk["semantic_role_to_p2_face_roles"])
    graph_error_examples: list[dict[str, Any]] = []

    def graph_failure(sample_id: str, code: str, details: Any) -> None:
        nonlocal feature_consistency_failures
        feature_consistency_failures += 1
        if len(graph_error_examples) < 50:
            graph_error_examples.append({"sample_id": sample_id, "code": code, "details": details})

    for record in manifest_records:
        sample_id = str(record["sample_id"])
        graph_path = assert_source_path(record["graph_path"])
        geometry_path = assert_source_path(record["geometry_path"])
        if not graph_path.is_file():
            graph_missing += 1
            continue
        actual_graph_hash = sha256_file(graph_path)
        if actual_graph_hash != str(record["graph_sha256"]).lower():
            graph_hash_mismatches += 1
            if len(graph_error_examples) < 50:
                graph_error_examples.append(
                    {
                        "sample_id": sample_id,
                        "code": "graph_hash_mismatch",
                        "expected": str(record["graph_sha256"]).lower(),
                        "actual": actual_graph_hash,
                    }
                )
            continue
        if not geometry_path.is_file():
            geometry_missing += 1

        graph = read_json(graph_path)
        if graph.get("schema_version") != GRAPH_SCHEMA:
            graph_schema_mismatches += 1
        if graph.get("sample_id") != sample_id:
            graph_failure(sample_id, "sample_id_mismatch", graph.get("sample_id"))
        signatures = graph.get("geometry_signatures", {})
        if signatures.get("exact_graph_signature") != record.get("exact_graph_signature"):
            graph_failure(sample_id, "exact_signature_mismatch", signatures.get("exact_graph_signature"))
        if signatures.get("near_graph_signature") != record.get("near_graph_signature"):
            graph_failure(sample_id, "near_signature_mismatch", signatures.get("near_graph_signature"))

        node_by_idx = {int(node["idx"]): node for node in graph.get("nodes", [])}
        feature_sets = graph.get("feature_sets", [])
        if len(feature_sets) != int(record.get("feature_instance_count", -1)):
            graph_failure(sample_id, "feature_instance_count_mismatch", len(feature_sets))
        seen_instances: set[int] = set()
        for feature_set in feature_sets:
            feature_instance_count += 1
            instance_id = int(feature_set.get("instance_id", 0))
            source_macro = str(feature_set.get("macro_feature", ""))
            canonical = SOURCE_MACRO_TO_CANONICAL.get(source_macro)
            if canonical is None:
                graph_failure(sample_id, "unknown_source_macro", source_macro)
                continue
            source_macro_counts[source_macro] += 1
            canonical_feature_counts[canonical] += 1
            if instance_id <= 0 or instance_id in seen_instances:
                graph_failure(sample_id, "invalid_or_duplicate_instance_id", instance_id)
            seen_instances.add(instance_id)
            indices = [int(index) for index in feature_set.get("node_indices", [])]
            if not indices or len(indices) != len(set(indices)):
                graph_failure(sample_id, "empty_or_duplicate_feature_set", indices)
                continue
            if any(index not in node_by_idx for index in indices):
                graph_failure(sample_id, "feature_node_missing", indices)
                continue
            members = [node_by_idx[index] for index in indices]
            all_instance_nodes = {
                index for index, node in node_by_idx.items() if int(node.get("instance_id", 0) or 0) == instance_id
            }
            if set(indices) != all_instance_nodes:
                graph_failure(
                    sample_id,
                    "instance_membership_mismatch",
                    {"feature_set": sorted(indices), "nodes": sorted(all_instance_nodes)},
                )
            for node in members:
                if int(node.get("instance_id", 0) or 0) != instance_id:
                    graph_failure(sample_id, "node_instance_mismatch", node.get("idx"))
                if str(node.get("macro_feature")) != source_macro:
                    graph_failure(sample_id, "node_macro_mismatch", node.get("idx"))
                semantic_role = str(node.get("semantic_role", ""))
                semantic_role_counts[semantic_role] += 1
                observed_semantic_roles.add(semantic_role)
                if semantic_role not in crosswalk_semantic_roles:
                    graph_failure(sample_id, "semantic_role_not_in_policy", semantic_role)
            # Every complete-instance role must reconstruct exactly the source feature set.
            for face_role, role in role_lookup[canonical].items():
                if role["selection_status"].startswith("unsupported"):
                    continue
                candidates = set(role["semantic_subtype_candidates"])
                selected = sorted(
                    int(node["idx"]) for node in members if str(node.get("semantic_role")) in candidates
                )
                if role["requires_complete_instance"] and selected != sorted(indices):
                    graph_failure(
                        sample_id,
                        "complete_instance_role_mismatch",
                        {"role": face_role, "selected": selected, "instance": sorted(indices)},
                    )
                if not selected:
                    graph_failure(sample_id, "empty_supported_role", {"role": face_role, "instance": instance_id})

    if graph_missing or graph_hash_mismatches:
        failures.append(
            {
                "kind": "graph_source_failure",
                "missing": graph_missing,
                "hash_mismatches": graph_hash_mismatches,
            }
        )
    if geometry_missing or graph_schema_mismatches or feature_consistency_failures:
        failures.append(
            {
                "kind": "graph_contract_failure",
                "geometry_missing": geometry_missing,
                "schema_mismatches": graph_schema_mismatches,
                "feature_consistency_failures": feature_consistency_failures,
            }
        )

    split_leakage = {
        "base_geometry_family_id": _collision_audit(manifest_records, "base_geometry_family_id"),
        "rotation_scale_family": _collision_audit(manifest_records, "base_geometry_family_id"),
        "exact_graph_signature": _collision_audit(manifest_records, "exact_graph_signature"),
        "near_graph_signature": _collision_audit(manifest_records, "near_graph_signature"),
    }
    leakage_total = sum(item["collision_count"] for item in split_leakage.values())
    if leakage_total:
        failures.append({"kind": "source_split_leakage", "collision_total": leakage_total})

    p2_freeze = _verify_p2_freeze()
    if p2_freeze["mismatch_count"] or not p2_freeze["supervisor_added_runner"]["matches"]:
        failures.append({"kind": "p2_freeze_hash_mismatch", "details": p2_freeze})

    invented_roles = sorted(observed_semantic_roles - crosswalk_semantic_roles)
    if invented_roles:
        failures.append({"kind": "ontology_extension_required", "roles": invented_roles})

    audit = {
        "schema_version": "topocae_cae_query_source_role_audit_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_pass": not failures,
        "source": {
            "dataset_root": str(SOURCE_META.parent.resolve()),
            "sample_count": len(manifest_records),
            "feature_instance_count": feature_instance_count,
            "dataset_meta_sha256": actual_meta_hash,
            "sample_manifest_sha256": actual_sample_manifest_hash,
            "artifact_manifest_sha256": sha256_file(SOURCE_ARTIFACT_MANIFEST),
            "split_file_sha256": split_file_hashes,
            "split_counts": dict(sorted(split_counts.items())),
            "dataset_split_counts": {
                DATASET_SPLIT_MAP[key]: value for key, value in sorted(split_counts.items())
            },
            "graph_count_expected": len(manifest_records),
            "graph_missing": graph_missing,
            "graph_hash_mismatches": graph_hash_mismatches,
            "geometry_missing": geometry_missing,
            "graph_schema_mismatches": graph_schema_mismatches,
            "feature_consistency_failures": feature_consistency_failures,
        },
        "role_audit": {
            "source_macro_instance_counts": dict(sorted(source_macro_counts.items())),
            "canonical_feature_instance_counts": dict(sorted(canonical_feature_counts.items())),
            "semantic_role_face_counts": dict(sorted(semantic_role_counts.items())),
            "observed_semantic_role_count": len(observed_semantic_roles),
            "ontology_extension_required": invented_roles,
            "unsupported_exact_roles": crosswalk["unsupported_exact_roles"],
            "web_to_rib": {
                "source_web_instances": source_macro_counts.get("web", 0),
                "canonical_rib_instances": canonical_feature_counts.get("rib", 0),
                "provenance_preserved": True,
            },
        },
        "split_leakage": split_leakage,
        "p2_freeze": p2_freeze,
        "failures": failures,
        "error_examples": graph_error_examples,
        "scope_guards": {
            "allowed_source_root_only": True,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "holdout_run": False,
            "real_part_content_accessed": False,
        },
    }
    return audit, crosswalk


def render_audit_markdown(audit: dict[str, Any]) -> str:
    source = audit["source"]
    role = audit["role_audit"]
    leakage = audit["split_leakage"]
    lines = [
        "# P3-A1 源数据与 Role Crosswalk 审计",
        "",
        f"状态：`{'PASS' if audit['automatic_pass'] else 'FAIL'}`  ",
        f"源样本：{source['sample_count']}  ",
        f"feature instances：{source['feature_instance_count']}  ",
        f"split：{source['split_counts']}  ",
        "",
        "## 完整性",
        "",
        f"- graph 缺失：{source['graph_missing']}",
        f"- graph SHA-256 不匹配：{source['graph_hash_mismatches']}",
        f"- geometry 缺失：{source['geometry_missing']}",
        f"- graph schema 不匹配：{source['graph_schema_mismatches']}",
        f"- feature-set/node/role 一致性失败：{source['feature_consistency_failures']}",
        f"- P2 freeze hash mismatch：{audit['p2_freeze']['mismatch_count']}",
        "",
        "## Split 泄漏",
        "",
        f"- base family：{leakage['base_geometry_family_id']['collision_count']}",
        f"- rotation/scale family：{leakage['rotation_scale_family']['collision_count']}",
        f"- exact graph signature：{leakage['exact_graph_signature']['collision_count']}",
        f"- near graph signature：{leakage['near_graph_signature']['collision_count']}",
        "",
        "## Role Crosswalk",
        "",
        f"- observed semantic roles：{role['observed_semantic_role_count']}",
        f"- ontology extension required：{role['ontology_extension_required']}",
        f"- unsupported exact roles：{', '.join(role['unsupported_exact_roles'])}",
        f"- source web instances：{role['web_to_rib']['source_web_instances']}",
        f"- canonical rib instances（rib + web）：{role['web_to_rib']['canonical_rib_instances']}",
        "- `web` 的 source macro 与 `web_side_plane`/`web_top_plane` provenance 保留；CAE-IR 输出仅使用 `rib`。",
        "",
        "## 结论",
        "",
        "审计通过后才允许进入 schema/planner/rendering；任一 source hash mismatch 或 ontology extension requirement 都会硬停止。",
    ]
    if audit["failures"]:
        lines.extend(["", "## 失败", "", "```json", str(audit["failures"]), "```"])
    return "\n".join(lines) + "\n"
