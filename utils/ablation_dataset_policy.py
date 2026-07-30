from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy
from typing import Any

from utils.feature_footprint import capsule_footprint, circle_footprint, footprint_conflicts, polyline_capsule_footprint


TARGET_MACRO_FEATURES = [
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_v",
    "slot_o",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
    "rib",
    "web",
    "fillet",
    "chamfer",
]

TRANSITION_MACROS = {"fillet", "chamfer"}


def balanced_target_sequence(targets: list[str], count: int, seed: int | None = None) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not targets and count:
        raise ValueError("targets must be non-empty when count > 0")

    sequence = []
    rng = random.Random(seed)
    while len(sequence) < count:
        cycle = list(targets)
        rng.shuffle(cycle)
        sequence.extend(cycle)
    return sequence[:count]


def audit_ablation_sample(
    graph_data: dict[str, Any],
    *,
    policy: str,
    max_feature_sets: int = 4,
    min_feature_sets: int = 2,
    footprint_clearance: float = 1.0,
) -> dict[str, Any]:
    if policy not in {"single_feature", "nonintersecting_multi"}:
        raise ValueError(f"Unsupported ablation policy: {policy}")

    feature_sets = [
        feature_set
        for feature_set in graph_data.get("feature_sets", [])
        if feature_set.get("macro_feature") != "stock"
    ]
    report = {
        "policy": policy,
        "valid": True,
        "reasons": [],
        "feature_set_count": len(feature_sets),
        "macro_counts": dict(Counter(feature_set.get("macro_feature", "unknown") for feature_set in feature_sets)),
        "footprint_conflicts": [],
    }

    stock_type = graph_data.get("stock_profile", {}).get("stock_type")
    if stock_type == "composite":
        _invalidate(report, "composite_stock_disallowed")

    if policy == "single_feature":
        if len(feature_sets) != 1:
            _invalidate(report, "feature_set_count_not_1")
        return report

    if len(feature_sets) < min_feature_sets:
        _invalidate(report, "feature_set_count_below_minimum")
    if len(feature_sets) > max_feature_sets:
        _invalidate(report, "feature_set_count_above_maximum")

    footprints = []
    for feature_set in feature_sets:
        if feature_set.get("macro_feature") in TRANSITION_MACROS:
            continue
        footprint = feature_set_footprint(feature_set)
        if footprint is None:
            continue
        footprints.append((feature_set, _force_conflicting_operation(footprint)))

    for index, (left_set, left_footprint) in enumerate(footprints):
        for right_set, right_footprint in footprints[index + 1:]:
            if _is_allowed_cavity_rib_parent_overlap(left_set, right_set):
                continue
            if footprint_conflicts(left_footprint, [right_footprint], clearance=footprint_clearance):
                _invalidate(report, "feature_footprint_conflict")
                report["footprint_conflicts"].append({
                    "left_instance_id": left_set.get("instance_id"),
                    "left_macro_feature": left_set.get("macro_feature"),
                    "right_instance_id": right_set.get("instance_id"),
                    "right_macro_feature": right_set.get("macro_feature"),
                })

    return report


def feature_set_footprint(feature_set: dict[str, Any]) -> dict[str, Any] | None:
    macro_feature = feature_set.get("macro_feature", "")
    params = _feature_parameters(feature_set)

    if macro_feature in {
        "through_hole_round",
        "blind_hole_round",
        "stepped_hole_round",
        "counterbore_hole",
        "countersink_hole",
        "pocket_circular",
        "cylinder_boss",
    }:
        center_a, center_b, zone_plane = _axis_projected_center(params)
        radius = _max_numeric(params, ["R", "R1", "R2", "radius", "large_radius", "diameter"], diameter_keys={"diameter"})
        if radius <= 0.0:
            return None
        return circle_footprint(
            center_a,
            center_b,
            radius,
            operation=_operation_for_macro(macro_feature),
            zone_plane=zone_plane,
            feature_kind=macro_feature,
        )

    if macro_feature == "obround_hole":
        center_a, center_b, zone_plane = _axis_projected_center(params)
        radius = _max_numeric(params, ["R", "radius", "diameter"], diameter_keys={"diameter"})
        distance = float(params.get("dist", params.get("length", radius * 4.0)))
        return capsule_footprint(
            center_a,
            center_b,
            length=max(distance + 2.0 * radius, 2.0 * radius),
            width=2.0 * radius,
            angle=float(params.get("angle", 0.0)),
            operation="cut",
            zone_plane=zone_plane,
            feature_kind=macro_feature,
        )

    if macro_feature in {"slot_u", "slot_v", "slot_o", "slot_rect"}:
        points = params.get("path_points_xy")
        width = float(params.get("w", params.get("width", params.get("profile_width", 0.0))))
        if points and width > 0.0:
            return polyline_capsule_footprint(
                [(float(x), float(y)) for x, y in points],
                width,
                operation="cut",
                feature_kind=macro_feature,
            )
        length = float(params.get("l", params.get("length", params.get("sweep_path_length", 0.0))))
        if width <= 0.0:
            width = float(params.get("diameter", params.get("radius", 0.0) * 2.0))
        if length <= 0.0 or width <= 0.0:
            return None
        return capsule_footprint(
            float(params.get("cx", 0.0)),
            float(params.get("cy", 0.0)),
            length=length,
            width=width,
            angle=float(params.get("angle", params.get("path_angle", 0.0))),
            operation="cut",
            feature_kind=macro_feature,
        )

    if macro_feature in {"pocket_rect", "rectangular_boss", "rib", "web"}:
        width = float(params.get("w", params.get("width", params.get("thickness", 0.0))))
        length = float(params.get("l", params.get("length", params.get("height", 0.0))))
        radius = max(float(params.get("radius", 0.0)), 0.5 * math.hypot(width, length))
        if radius <= 0.0:
            return None
        return circle_footprint(
            float(params.get("cx", params.get("feature_cx", 0.0))),
            float(params.get("cy", params.get("feature_cy", 0.0))),
            radius,
            operation=_operation_for_macro(macro_feature),
            feature_kind=macro_feature,
        )

    return None


def _feature_parameters(feature_set: dict[str, Any]) -> dict[str, Any]:
    params = {}
    params.update(feature_set.get("geometry_parameters") or {})
    params.update(feature_set.get("raw_parameters") or {})
    return params


def _is_allowed_cavity_rib_parent_overlap(left_set: dict[str, Any], right_set: dict[str, Any]) -> bool:
    return _is_cavity_rib_child_of(left_set, right_set) or _is_cavity_rib_child_of(right_set, left_set)


def _is_cavity_rib_child_of(rib_set: dict[str, Any], parent_set: dict[str, Any]) -> bool:
    if rib_set.get("macro_feature") != "rib":
        return False
    parent_macro = parent_set.get("macro_feature")
    if parent_macro not in {"pocket_rect", "pocket_circular"}:
        return False
    params = _feature_parameters(rib_set)
    return (
        params.get("placement_mode") == "cavity"
        and params.get("rib_body_kind") == "cavity_wall_rib"
        and params.get("parent_macro_feature") == parent_macro
        and int(params.get("parent_instance_id", -1)) == int(parent_set.get("instance_id", -2))
    )


def _axis_projected_center(params: dict[str, Any]) -> tuple[float, float, str]:
    axis = params.get("axis", "z")
    if axis == "x":
        return float(params.get("cy", 0.0)), float(params.get("cz", 0.0)), "yz"
    if axis == "y":
        return float(params.get("cx", 0.0)), float(params.get("cz", 0.0)), "xz"
    return float(params.get("cx", 0.0)), float(params.get("cy", 0.0)), "xy"


def _max_numeric(params: dict[str, Any], keys: list[str], *, diameter_keys: set[str] | None = None) -> float:
    diameter_keys = diameter_keys or set()
    values = []
    for key in keys:
        if key not in params:
            continue
        try:
            value = float(params[key])
        except (TypeError, ValueError):
            continue
        if key in diameter_keys:
            value *= 0.5
        values.append(value)
    return max(values) if values else 0.0


def _operation_for_macro(macro_feature: str) -> str:
    if macro_feature in {"cylinder_boss", "rectangular_boss", "rib", "web"}:
        return "add"
    return "cut"


def _force_conflicting_operation(footprint: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(footprint)
    normalized["operation"] = "cut"
    return normalized


def _invalidate(report: dict[str, Any], reason: str) -> None:
    report["valid"] = False
    if reason not in report["reasons"]:
        report["reasons"].append(reason)
