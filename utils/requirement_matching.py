from __future__ import annotations

from typing import Any


FeatureObject = dict[str, Any]
Requirement = dict[str, Any] | str


TARGET_RULES = [
    {
        "target": "stepped_hole",
        "patterns": ("阶梯孔",),
        "macro_features": ("stepped_hole_round",),
    },
    {
        "target": "blind_hole",
        "patterns": ("盲孔",),
        "macro_features": ("blind_hole_round",),
    },
    {
        "target": "through_hole",
        "patterns": ("通孔",),
        "macro_features": ("through_hole_round",),
    },
    {
        "target": "u_slot",
        "patterns": ("U型槽", "U 形槽", "U形槽", "U槽", "u型槽", "u形槽", "u槽"),
        "macro_features": ("slot_u",),
    },
    {
        "target": "rectangular_cavity",
        "patterns": ("矩形腔", "矩形槽", "矩形口袋"),
        "macro_features": ("pocket_rect", "slot_rect"),
    },
    {
        "target": "fillet",
        "patterns": ("圆角",),
        "macro_features": ("fillet",),
    },
    {
        "target": "chamfer",
        "patterns": ("倒角",),
        "macro_features": ("chamfer",),
    },
    {
        "target": "cylinder_boss",
        "patterns": ("圆柱凸台", "圆柱"),
        "macro_features": ("cylinder_boss",),
    },
    {
        "target": "circular_pocket",
        "patterns": ("圆形腔", "圆形口袋"),
        "macro_features": ("pocket_circular",),
    },
]


def match_requirements(requirements: list[Requirement], feature_objects_data: dict | list[FeatureObject]) -> dict:
    feature_objects = _coerce_feature_objects(feature_objects_data)
    results = []
    for index, requirement in enumerate(requirements, start=1):
        requirement_id, text = _coerce_requirement(requirement, index)
        result = match_requirement(text, feature_objects)
        result["requirement_id"] = requirement_id
        results.append(result)

    return {
        "matching_method": "deterministic_chinese_keyword_to_selectable_region_v1",
        "requirement_count": len(results),
        "requirements": results,
    }


def match_requirement(text: str, feature_objects_data: dict | list[FeatureObject]) -> dict:
    feature_objects = _coerce_feature_objects(feature_objects_data)
    action = detect_action(text)
    target_rules = detect_target_rules(text)
    matches = []
    warnings = []

    if not target_rules:
        return {
            "text": text,
            "action": action,
            "matches": [],
            "warnings": [{
                "reason": "target_not_understood",
                "message": "No supported CAE feature target was detected in the requirement text.",
            }],
        }

    for target_rule in target_rules:
        macro_features = tuple(target_rule["macro_features"])
        candidate_objects = [
            feature_object for feature_object in feature_objects
            if feature_object.get("macro_feature") in macro_features
        ]
        if not candidate_objects:
            warnings.append({
                "reason": "feature_not_found",
                "target": target_rule["target"],
                "macro_features": list(macro_features),
                "message": "The requested feature type does not exist in this model's feature objects.",
            })
            continue

        role_requests = requested_roles(text, target_rule["target"], action)
        for feature_object in candidate_objects:
            regions_by_role = {
                region.get("role"): region
                for region in feature_object.get("selectable_regions", [])
            }
            for role_request in role_requests:
                allowed_macro_features = role_request.get("macro_features")
                if allowed_macro_features and feature_object.get("macro_feature") not in allowed_macro_features:
                    continue
                role = role_request["role"]
                region = regions_by_role.get(role)
                if region is None:
                    warnings.append({
                        "reason": "region_not_available",
                        "feature_object_id": feature_object.get("feature_object_id"),
                        "macro_feature": feature_object.get("macro_feature"),
                        "requested_role": role,
                        "message": role_request.get("missing_message", "The requested selectable region is not available."),
                    })
                    continue

                match = {
                    "feature_object_id": feature_object.get("feature_object_id"),
                    "macro_feature": feature_object.get("macro_feature"),
                    "role": role,
                    "face_indices": list(region.get("face_indices", [])),
                    "geometry_parameters": dict(feature_object.get("geometry_parameters", {})),
                    "intended_uses": list(region.get("intended_uses", [])),
                    "is_default_for": list(region.get("is_default_for", [])),
                    "action": action,
                    "selection_reason": role_request["reason"],
                }
                if _is_intent_mismatch(action, region):
                    match["compatibility_warning"] = {
                        "reason": "intent_not_declared_for_region",
                        "message": "The region was selected by geometry wording, but its intended_uses do not explicitly include this action.",
                    }
                matches.append(match)

    return {
        "text": text,
        "action": action,
        "matches": matches,
        "warnings": warnings,
    }


def detect_action(text: str) -> str:
    if _contains_any(text, ("压力", "压强")):
        return "pressure_load"
    if _contains_any(text, ("固定", "约束")):
        return "fixed_support"
    if _contains_any(text, ("加载", "载荷", "施加")):
        return "surface_load"
    if _contains_any(text, ("选取", "选择")):
        return "selection"
    if _contains_any(text, ("圆角", "倒角", "网格", "细化")):
        return "mesh_refinement"
    return "selection"


def detect_target_rules(text: str) -> list[dict[str, Any]]:
    return [
        rule for rule in TARGET_RULES
        if _contains_any(text, rule["patterns"])
    ]


def requested_roles(text: str, target: str, action: str) -> list[dict[str, str]]:
    if target == "blind_hole":
        if _contains_any(text, ("底", "底面", "底部")):
            return [_role("hole_bottom_face", "blind hole bottom face explicitly requested")]
        return [_role("hole_cylindrical_wall", "default blind-hole constraint/load region")]

    if target == "through_hole":
        return [_role("hole_cylindrical_wall", "through-hole inner wall requested")]

    if target == "stepped_hole":
        roles = []
        if _contains_any(text, ("台阶面", "阶梯面", "肩面")):
            roles.append(_role("step_shoulder_face", "stepped-hole shoulder face requested"))
        if _contains_any(text, ("小孔壁", "小孔内壁")):
            roles.append(_role("small_hole_cylindrical_wall", "stepped-hole small wall requested"))
        if _contains_any(text, ("大孔壁", "大孔内壁")):
            roles.append(_role("large_hole_cylindrical_wall", "stepped-hole large wall requested"))
        return roles or [_role("large_hole_cylindrical_wall", "default stepped-hole wall region")]

    if target == "rectangular_cavity":
        if _contains_any(text, ("底部", "底面", "底")):
            return [
                _role("pocket_bottom_face", "rectangular cavity bottom requested", macro_features=("pocket_rect",)),
                _role("slot_bottom_face", "rectangular slot bottom requested", macro_features=("slot_rect",)),
            ]
        return [
            _role("pocket_side_faces", "rectangular cavity inner wall requested", macro_features=("pocket_rect",)),
            _role("slot_side_faces", "rectangular slot inner wall requested", macro_features=("slot_rect",)),
        ]

    if target == "u_slot":
        if _contains_any(text, ("底部", "底面", "底")):
            return [_role("slot_bottom_face", "U-slot bottom requested")]
        return [
            _role("slot_side_faces", "U-slot straight inner wall requested"),
            _role("slot_end_cylinders", "U-slot curved end wall requested"),
        ]

    if target == "fillet":
        return [_role("transition_faces", "fillet transition faces requested")]

    if target == "chamfer":
        return [_role("transition_faces", "chamfer transition faces requested")]

    if target == "cylinder_boss":
        roles = []
        if _contains_any(text, ("顶面", "顶部", "上表面")):
            roles.append(_role("boss_top_face", "cylindrical boss top face requested"))
        if _contains_any(text, ("底面", "底部", "下表面")):
            roles.append(_role(
                "boss_bottom_face",
                "cylindrical boss bottom face requested",
                "For a fused cylindrical boss, the bottom face is usually not an exposed B-Rep face.",
            ))
        return roles or [_role("boss_top_face", "default cylindrical boss selectable face")]

    if target == "circular_pocket":
        if _contains_any(text, ("底部", "底面", "底")):
            return [_role("pocket_bottom_face", "circular pocket bottom requested")]
        return [_role("pocket_cylindrical_wall", "circular pocket wall requested")]

    return []


def _role(
    role: str,
    reason: str,
    missing_message: str | None = None,
    macro_features: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    result = {
        "role": role,
        "reason": reason,
    }
    if missing_message:
        result["missing_message"] = missing_message
    if macro_features:
        result["macro_features"] = macro_features
    return result


def _coerce_feature_objects(feature_objects_data: dict | list[FeatureObject]) -> list[FeatureObject]:
    if isinstance(feature_objects_data, dict):
        return list(feature_objects_data.get("feature_objects", []))
    return list(feature_objects_data)


def _coerce_requirement(requirement: Requirement, index: int) -> tuple[str, str]:
    if isinstance(requirement, str):
        return f"req_{index:03d}", requirement
    return str(requirement.get("id", f"req_{index:03d}")), str(requirement["text"])


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _is_intent_mismatch(action: str, region: dict) -> bool:
    if action in {"selection", "mesh_refinement"}:
        return False
    intended_uses = set(region.get("intended_uses", []))
    if action == "surface_load":
        return not bool({"surface_load", "pressure_load"} & intended_uses)
    return action not in intended_uses
