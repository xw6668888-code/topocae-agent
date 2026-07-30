"""Refined source-role audit with explicit complete-instance eligibility rules."""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import audit as base
from .constants import SOURCE_MACRO_TO_CANONICAL, SOURCE_MANIFEST, assert_source_path
from .io import read_json, read_jsonl


SOURCE_ONLY_COMPLETE_INSTANCE_ROLES = {
    "slot_terminal_face": {
        "canonical_feature_families": ["slot"],
        "p2_face_roles": ["feature_region"],
        "mapping_rule": "complete_instance_membership_only",
        "exact_subrole_target_allowed": False,
        "reason": (
            "The source graph contains this generic terminal face but frozen P2 has no "
            "distinct selectable subtype. It is retained only inside the complete feature instance."
        ),
    }
}


def _recheck_graph_roles(crosswalk: dict[str, Any]) -> dict[str, Any]:
    role_lookup = {
        feature: {role["face_role"]: role for role in entry["face_roles"]}
        for feature, entry in crosswalk["features"].items()
    }
    known_roles = set(crosswalk["semantic_role_to_p2_face_roles"])
    source_only_roles = set(SOURCE_ONLY_COMPLETE_INSTANCE_ROLES)
    hard_failures = 0
    hard_examples: list[dict[str, Any]] = []
    ineligible = 0
    ineligible_reasons: Counter[str] = Counter()
    eligible = 0
    source_only_counts: Counter[str] = Counter()

    def fail(sample_id: str, code: str, details: Any) -> None:
        nonlocal hard_failures
        hard_failures += 1
        if len(hard_examples) < 50:
            hard_examples.append({"sample_id": sample_id, "code": code, "details": details})

    for record in read_jsonl(SOURCE_MANIFEST):
        sample_id = str(record["sample_id"])
        graph = read_json(assert_source_path(record["graph_path"]))
        node_by_idx = {int(node["idx"]): node for node in graph.get("nodes", [])}
        seen_instances: set[int] = set()
        for feature_set in graph.get("feature_sets", []):
            instance_id = int(feature_set.get("instance_id", 0) or 0)
            source_macro = str(feature_set.get("macro_feature", ""))
            canonical = SOURCE_MACRO_TO_CANONICAL.get(source_macro)
            if canonical is None:
                fail(sample_id, "unknown_source_macro", source_macro)
                continue
            if instance_id <= 0 or instance_id in seen_instances:
                fail(sample_id, "invalid_or_duplicate_instance_id", instance_id)
            seen_instances.add(instance_id)
            indices = [int(index) for index in feature_set.get("node_indices", [])]
            if not indices or len(indices) != len(set(indices)) or any(index not in node_by_idx for index in indices):
                fail(sample_id, "invalid_feature_set_indices", indices)
                continue
            members = [node_by_idx[index] for index in indices]
            all_instance_nodes = {
                index for index, node in node_by_idx.items() if int(node.get("instance_id", 0) or 0) == instance_id
            }
            if set(indices) != all_instance_nodes:
                fail(
                    sample_id,
                    "instance_membership_mismatch",
                    {"feature_set": sorted(indices), "nodes": sorted(all_instance_nodes)},
                )
            for node in members:
                if int(node.get("instance_id", 0) or 0) != instance_id:
                    fail(sample_id, "node_instance_mismatch", node.get("idx"))
                if str(node.get("macro_feature")) != source_macro:
                    fail(sample_id, "node_macro_mismatch", node.get("idx"))
                semantic_role = str(node.get("semantic_role", ""))
                if semantic_role in source_only_roles:
                    source_only_counts[semantic_role] += 1
                elif semantic_role not in known_roles:
                    fail(sample_id, "semantic_role_not_in_frozen_policy", semantic_role)

            member_roles = {str(node.get("semantic_role", "")) for node in members}
            for face_role, role in role_lookup[canonical].items():
                status = str(role["selection_status"])
                if status.startswith("unsupported"):
                    continue
                if role["requires_complete_instance"]:
                    # Task contract: feature_region is the complete instance face set.
                    eligible += 1
                    continue
                candidates = set(role["semantic_subtype_candidates"])
                present = member_roles & candidates
                if not present:
                    ineligible += 1
                    ineligible_reasons[f"{canonical}.{face_role}:empty"] += 1
                elif status == "composite" and present != candidates:
                    ineligible += 1
                    ineligible_reasons[f"{canonical}.{face_role}:partial_composite"] += 1
                else:
                    eligible += 1
    return {
        "hard_failure_count": hard_failures,
        "hard_failure_examples": hard_examples,
        "eligible_role_instance_count": eligible,
        "ineligible_role_instance_count": ineligible,
        "ineligible_role_instance_reasons": dict(sorted(ineligible_reasons.items())),
        "source_only_complete_instance_role_counts": dict(sorted(source_only_counts.items())),
    }


def audit_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    audit, crosswalk = base.audit_sources()
    crosswalk["source_only_instance_membership_roles"] = SOURCE_ONLY_COMPLETE_INSTANCE_ROLES
    refined = _recheck_graph_roles(crosswalk)

    retained_failures = [
        failure
        for failure in audit["failures"]
        if failure["kind"] not in {"graph_contract_failure", "ontology_extension_required"}
    ]
    source = audit["source"]
    source["feature_consistency_failures"] = refined["hard_failure_count"]
    if source["geometry_missing"] or source["graph_schema_mismatches"] or refined["hard_failure_count"]:
        retained_failures.append(
            {
                "kind": "graph_contract_failure",
                "geometry_missing": source["geometry_missing"],
                "schema_mismatches": source["graph_schema_mismatches"],
                "feature_consistency_failures": refined["hard_failure_count"],
            }
        )

    audit["role_audit"]["ontology_extension_required"] = []
    audit["role_audit"]["source_only_instance_membership_roles"] = sorted(
        refined["source_only_complete_instance_role_counts"]
    )
    audit["role_audit"].update(refined)
    audit["failures"] = retained_failures
    audit["error_examples"] = refined["hard_failure_examples"]
    audit["automatic_pass"] = not retained_failures
    return audit, crosswalk


def render_audit_markdown(audit: dict[str, Any]) -> str:
    source = audit["source"]
    role = audit["role_audit"]
    leakage = audit["split_leakage"]
    status = "PASS" if audit["automatic_pass"] else "FAIL"
    return f"""# P3-A1 源数据与 Role Crosswalk 审计

状态：`{status}`  
源样本：{source['sample_count']}  
feature instances：{source['feature_instance_count']}  
split：{source['split_counts']}

## 完整性

- graph 缺失：{source['graph_missing']}
- graph SHA-256 不匹配：{source['graph_hash_mismatches']}
- geometry 缺失：{source['geometry_missing']}
- graph schema 不匹配：{source['graph_schema_mismatches']}
- feature-set/node/macro/role 硬失败：{source['feature_consistency_failures']}
- P2 freeze hash mismatch：{audit['p2_freeze']['mismatch_count']}

## Split 泄漏

- base family：{leakage['base_geometry_family_id']['collision_count']}
- rotation/scale family：{leakage['rotation_scale_family']['collision_count']}
- exact graph signature：{leakage['exact_graph_signature']['collision_count']}
- near graph signature：{leakage['near_graph_signature']['collision_count']}

## Role Crosswalk

- observed semantic roles：{role['observed_semantic_role_count']}
- ontology extension required：{role['ontology_extension_required']}
- source-only complete-instance roles：{role['source_only_instance_membership_roles']}
- eligible role-instance combinations：{role['eligible_role_instance_count']}
- ineligible role-instance combinations（不生成精确样本）：{role['ineligible_role_instance_count']}
- unsupported exact roles：{', '.join(role['unsupported_exact_roles'])}
- source web instances：{role['web_to_rib']['source_web_instances']}
- canonical rib instances（rib + web）：{role['web_to_rib']['canonical_rib_instances']}

`slot_terminal_face` 不映射为新的 P2 selectable subtype，只能随完整 feature instance 进入 `feature_region`；不得作为独立 exact subrole target。`web` 的 source macro 与 semantic-role provenance 保留，CAE-IR 仅输出 `rib`。

## 结论

审计通过后才允许进入 schema/planner/rendering；任一 source hash mismatch、硬 graph contract 失败或 ontology extension requirement 都会停止生成。
"""
