from __future__ import annotations

from copy import deepcopy

from p1_c.all_features import CleanFeatureResult, FaceTag
from p1_c.rib_web_v3 import (
    GENERATOR_REVISION as RIB_WEB_GENERATOR_REVISION,
    RIB_HOST_FAMILIES,
    WEB_HOST_FAMILIES,
    build_rib_web_sample,
)


GENERATOR_REVISION = "p1_c_supported_wall_r2_strict_concept_hosts"
SUPPORTED_WALL_HOST_FAMILIES = RIB_HOST_FAMILIES + WEB_HOST_FAMILIES
HOST_FAMILIES_BY_MACRO = {"rib": RIB_HOST_FAMILIES, "web": WEB_HOST_FAMILIES}


def build_supported_wall_sample(
    primary: str,
    host_family: str,
    variant: str,
    seed: int,
) -> CleanFeatureResult:
    if primary not in HOST_FAMILIES_BY_MACRO:
        raise KeyError(primary)
    if host_family not in HOST_FAMILIES_BY_MACRO[primary]:
        raise ValueError(
            f"concept-incompatible supported-wall host: {primary}/{host_family}; "
            f"allowed={HOST_FAMILIES_BY_MACRO[primary]}"
        )
    if variant not in {"plain", "hole", "transition"}:
        raise KeyError(variant)

    scenario = f"{primary}_{host_family}_{variant}"
    raw = build_rib_web_sample(scenario, int(seed))
    raw_target_instance = int(raw.parameters["feature_instance_id"])
    cavity = host_family in {"rect_cavity", "circular_cavity"}
    pocket_macro = raw.parameters.get("host_metadata", {}).get("cavity_macro_feature")

    remapped_tags: list[FaceTag] = []
    for tag in raw.face_tags:
        if tag.instance_id == raw_target_instance:
            remapped_tags.append(FaceTag(tag.face, tag.semantic_role, primary, 1))
        elif cavity and tag.instance_id == 1:
            remapped_tags.append(FaceTag(tag.face, tag.semantic_role, tag.macro_feature, 2))
        else:
            remapped_tags.append(
                FaceTag(tag.face, tag.semantic_role, tag.macro_feature, 3 if cavity else 2)
            )

    features: list[dict[str, object]] = [
        {
            "macro_feature": primary,
            "instance_id": 1,
            "operation": "supported_thin_wall_fuse",
            "feature_face_count": sum(tag.instance_id == 1 for tag in remapped_tags),
        }
    ]
    secondary_features: list[str] = []
    if cavity:
        if pocket_macro not in {"pocket_rect", "pocket_circular"}:
            raise ValueError(f"invalid structural cavity macro: {pocket_macro}")
        secondary_features.append(str(pocket_macro))
        features.append(
            {
                "macro_feature": str(pocket_macro),
                "instance_id": 2,
                "operation": "host_cavity_cut",
                "feature_face_count": sum(tag.instance_id == 2 for tag in remapped_tags),
            }
        )
    if variant == "hole":
        hole_instance = 3 if cavity else 2
        secondary_features.append("through_hole_round")
        features.append(
            {
                "macro_feature": "through_hole_round",
                "instance_id": hole_instance,
                "operation": "thin_wall_hole_cut",
                "feature_face_count": sum(tag.instance_id == hole_instance for tag in remapped_tags),
            }
        )

    parameters = {
        "schema_version": "topocae_p1_c_clean_feature_parameters_2",
        "generator_revision": GENERATOR_REVISION,
        "rib_web_generator_revision": RIB_WEB_GENERATOR_REVISION,
        "seed": int(seed),
        "primary_target": primary,
        "secondary_features": secondary_features,
        "context": ("target_only", "pair", "triple")[len(secondary_features)],
        "host_family": host_family,
        "host_metadata": deepcopy(raw.parameters["host_metadata"]),
        "host_cleanup_before_features": deepcopy(raw.parameters["host_cleanup_before_feature"]),
        "features": features,
        "instance_ids": {
            str(record["macro_feature"]): int(record["instance_id"])
            for record in features
        },
        "source_engineering_scenario": scenario,
        "source_engineering_macro": primary,
        "source_parameters": deepcopy(raw.parameters),
        "concept_contract": {
            "target_macro_equals_source_engineering_macro": True,
            "source_engineering_macro": primary,
            "target_macro": primary,
            "semantic_relabel_used": False,
            "host_family_concept_compatible": True,
            "allowed_host_families": list(HOST_FAMILIES_BY_MACRO[primary]),
        },
    }
    result = CleanFeatureResult(raw.shape, tuple(remapped_tags), parameters, raw.cleanup)
    _validate_supported_wall_result(result)
    return result


def _validate_supported_wall_result(result: CleanFeatureResult) -> None:
    params = result.parameters
    primary = str(params["primary_target"])
    host_family = str(params["host_family"])
    if host_family not in HOST_FAMILIES_BY_MACRO[primary]:
        raise ValueError(f"concept-incompatible supported-wall result: {primary}/{host_family}")
    if params["source_engineering_macro"] != primary:
        raise ValueError("supported-wall source engineering macro differs from target")
    if not result.cleanup["passed"] or not params["host_cleanup_before_features"]["passed"]:
        raise ValueError("supported-wall sample is not cleanup-idempotent")
    expected_instances = set(range(1, len(params["features"]) + 1))
    actual_instances = {tag.instance_id for tag in result.face_tags}
    if actual_instances != expected_instances:
        raise ValueError(f"supported-wall instance mismatch: {expected_instances} != {actual_instances}")
    for record in params["features"]:
        macros = {
            tag.macro_feature
            for tag in result.face_tags
            if tag.instance_id == record["instance_id"]
        }
        if macros != {record["macro_feature"]}:
            raise ValueError(f"supported-wall macro mismatch: {record}, {macros}")
    target_roles = {tag.semantic_role for tag in result.face_tags if tag.instance_id == 1}
    if not target_roles or any(not role.startswith(f"{primary}_") for role in target_roles):
        raise ValueError(f"supported-wall target semantic-role mismatch: {target_roles}")


__all__ = [
    "GENERATOR_REVISION",
    "HOST_FAMILIES_BY_MACRO",
    "SUPPORTED_WALL_HOST_FAMILIES",
    "build_supported_wall_sample",
]
