from __future__ import annotations

from p1_c.all_features import CleanFeatureResult, FaceTag, GENERATOR_REVISION
from p1_c.rib_web_v2 import build_rib_web_sample


SUPPORTED_WALL_HOST_FAMILIES = ("l_corner", "rect_cavity", "circular_cavity", "two_supports")


def build_supported_wall_sample(
    primary: str,
    host_family: str,
    variant: str,
    seed: int,
) -> CleanFeatureResult:
    if primary not in {"rib", "web"}:
        raise KeyError(primary)
    if host_family not in SUPPORTED_WALL_HOST_FAMILIES:
        raise KeyError(host_family)
    if variant not in {"plain", "hole", "transition"}:
        raise KeyError(variant)
    source_macro = "web" if host_family == "two_supports" else "rib"
    scenario = f"{source_macro}_{host_family}_{variant}"
    raw = build_rib_web_sample(scenario, int(seed))
    raw_target_instance = int(raw.parameters["feature_instance_id"])
    cavity = host_family in {"rect_cavity", "circular_cavity"}
    pocket_macro = raw.parameters.get("host_metadata", {}).get("cavity_macro_feature")
    remapped_tags: list[FaceTag] = []
    for tag in raw.face_tags:
        if tag.instance_id == raw_target_instance:
            role = tag.semantic_role
            if source_macro != primary:
                role = role.replace(f"{source_macro}_", f"{primary}_")
            remapped_tags.append(FaceTag(tag.face, role, primary, 1))
        elif cavity and tag.instance_id == 1:
            remapped_tags.append(FaceTag(tag.face, tag.semantic_role, tag.macro_feature, 2))
        else:
            remapped_tags.append(FaceTag(tag.face, tag.semantic_role, tag.macro_feature, 3 if cavity else 2))
    features = [
        {
            "macro_feature": primary,
            "instance_id": 1,
            "operation": "supported_thin_wall_fuse",
            "feature_face_count": sum(tag.instance_id == 1 for tag in remapped_tags),
        }
    ]
    secondary_features: list[str] = []
    if cavity:
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
        "schema_version": "topocae_p1_c_clean_feature_parameters_1",
        "generator_revision": GENERATOR_REVISION,
        "seed": int(seed),
        "primary_target": primary,
        "secondary_features": secondary_features,
        "context": ("target_only", "pair", "triple")[len(secondary_features)],
        "host_family": host_family,
        "host_metadata": raw.parameters["host_metadata"],
        "host_cleanup_before_features": raw.parameters["host_cleanup_before_feature"],
        "features": features,
        "instance_ids": {record["macro_feature"]: record["instance_id"] for record in features},
        "source_engineering_scenario": scenario,
        "source_parameters": raw.parameters,
        "semantic_relabel": {"source": source_macro, "target": primary},
    }
    result = CleanFeatureResult(raw.shape, tuple(remapped_tags), parameters, raw.cleanup)
    _validate_supported_wall_result(result)
    return result


def _validate_supported_wall_result(result: CleanFeatureResult) -> None:
    if not result.cleanup["passed"] or not result.parameters["host_cleanup_before_features"]["passed"]:
        raise ValueError("supported wall sample is not clean")
    expected_instances = set(range(1, len(result.parameters["features"]) + 1))
    actual_instances = {tag.instance_id for tag in result.face_tags}
    if actual_instances != expected_instances:
        raise ValueError(f"supported wall instance mismatch: {expected_instances} != {actual_instances}")
    for record in result.parameters["features"]:
        macros = {tag.macro_feature for tag in result.face_tags if tag.instance_id == record["instance_id"]}
        if macros != {record["macro_feature"]}:
            raise ValueError(f"supported wall macro mismatch: {record}, {macros}")


__all__ = ["SUPPORTED_WALL_HOST_FAMILIES", "build_supported_wall_sample"]
