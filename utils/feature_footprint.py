from __future__ import annotations

import math
from typing import Any


def circle_footprint(
    cx: float,
    cy: float,
    radius: float,
    *,
    operation: str = "cut",
    zone_plane: str = "xy",
    feature_kind: str = "feature",
) -> dict[str, Any]:
    return {
        "footprint_type": "circle",
        "cx": float(cx),
        "cy": float(cy),
        "radius": float(radius),
        "operation": operation,
        "zone_plane": zone_plane,
        "feature_kind": feature_kind,
    }


def capsule_footprint(
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle: float,
    *,
    operation: str = "cut",
    zone_plane: str = "xy",
    feature_kind: str = "slot",
) -> dict[str, Any]:
    radius = float(width) / 2.0
    return {
        "footprint_type": "capsule",
        "cx": float(cx),
        "cy": float(cy),
        "length": float(length),
        "width": float(width),
        "radius": radius,
        "angle": float(angle),
        "operation": operation,
        "zone_plane": zone_plane,
        "feature_kind": feature_kind,
    }


def polyline_capsule_footprint(
    points: list[tuple[float, float]],
    width: float,
    *,
    operation: str = "cut",
    zone_plane: str = "xy",
    feature_kind: str = "slot",
) -> dict[str, Any]:
    safe_points = [(float(x), float(y)) for x, y in points]
    if safe_points:
        cx = sum(point[0] for point in safe_points) / len(safe_points)
        cy = sum(point[1] for point in safe_points) / len(safe_points)
        bounding_radius = max(math.hypot(point[0] - cx, point[1] - cy) for point in safe_points) + float(width) / 2.0
    else:
        cx = cy = 0.0
        bounding_radius = float(width) / 2.0
    return {
        "footprint_type": "polyline_capsule",
        "points": safe_points,
        "cx": cx,
        "cy": cy,
        "width": float(width),
        "radius": bounding_radius,
        "tube_radius": float(width) / 2.0,
        "bounding_radius": bounding_radius,
        "operation": operation,
        "zone_plane": zone_plane,
        "feature_kind": feature_kind,
    }


def footprint_conflicts(candidate: dict[str, Any], existing_zones: list[dict[str, Any]], clearance: float = 2.0) -> bool:
    for existing in existing_zones:
        if existing.get("zone_plane", "xy") != candidate.get("zone_plane", "xy"):
            continue
        if not _operations_should_conflict(candidate, existing):
            continue
        if _footprint_distance(candidate, _normalize_zone(existing)) < float(clearance):
            return True
    return False


def _operations_should_conflict(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    candidate_operation = candidate.get("operation", "cut")
    existing_operation = existing.get("operation", "cut")
    return candidate_operation == "cut" and existing_operation == "cut"


def _normalize_zone(zone: dict[str, Any]) -> dict[str, Any]:
    if "footprint_type" in zone:
        return zone
    return circle_footprint(
        zone.get("cx", 0.0),
        zone.get("cy", 0.0),
        zone.get("radius", 0.0),
        operation=zone.get("operation", "cut"),
        zone_plane=zone.get("zone_plane", "xy"),
        feature_kind=zone.get("feature_kind", "legacy_feature"),
    )


def _footprint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_type = a.get("footprint_type", "circle")
    b_type = b.get("footprint_type", "circle")

    if a_type == "circle" and b_type == "circle":
        center_distance = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
        return center_distance - a["radius"] - b["radius"]

    if a_type == "capsule" and b_type == "circle":
        segment_distance = _point_to_capsule_segment_distance(b["cx"], b["cy"], a)
        return segment_distance - a["radius"] - b["radius"]

    if a_type == "circle" and b_type == "capsule":
        segment_distance = _point_to_capsule_segment_distance(a["cx"], a["cy"], b)
        return segment_distance - a["radius"] - b["radius"]

    if a_type == "capsule" and b_type == "capsule":
        a0, a1 = _capsule_segment(a)
        b0, b1 = _capsule_segment(b)
        segment_distance = _segment_distance(a0, a1, b0, b1)
        return segment_distance - a["radius"] - b["radius"]

    if a_type == "polyline_capsule" and b_type == "circle":
        return _point_to_polyline_distance(b["cx"], b["cy"], a["points"]) - _tube_radius(a) - b["radius"]

    if a_type == "circle" and b_type == "polyline_capsule":
        return _point_to_polyline_distance(a["cx"], a["cy"], b["points"]) - a["radius"] - _tube_radius(b)

    if a_type == "polyline_capsule" and b_type == "capsule":
        b0, b1 = _capsule_segment(b)
        return _polyline_segment_distance(a["points"], b0, b1) - _tube_radius(a) - b["radius"]

    if a_type == "capsule" and b_type == "polyline_capsule":
        a0, a1 = _capsule_segment(a)
        return _polyline_segment_distance(b["points"], a0, a1) - a["radius"] - _tube_radius(b)

    if a_type == "polyline_capsule" and b_type == "polyline_capsule":
        return _polyline_polyline_distance(a["points"], b["points"]) - _tube_radius(a) - _tube_radius(b)

    raise ValueError(f"Unsupported footprint pair: {a_type}, {b_type}")


def _capsule_segment(capsule: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    half_length = max(float(capsule["length"]) / 2.0 - float(capsule["radius"]), 0.0)
    dx = math.cos(float(capsule["angle"]))
    dy = math.sin(float(capsule["angle"]))
    cx = float(capsule["cx"])
    cy = float(capsule["cy"])
    return (
        (cx - dx * half_length, cy - dy * half_length),
        (cx + dx * half_length, cy + dy * half_length),
    )


def _point_to_capsule_segment_distance(px: float, py: float, capsule: dict[str, Any]) -> float:
    a, b = _capsule_segment(capsule)
    return _point_segment_distance((float(px), float(py)), a, b)


def _tube_radius(polyline_capsule: dict[str, Any]) -> float:
    return float(polyline_capsule.get("tube_radius", polyline_capsule.get("width", 0.0) / 2.0))


def _point_to_polyline_distance(px: float, py: float, points: list[tuple[float, float]]) -> float:
    if not points:
        return math.inf
    if len(points) == 1:
        return math.hypot(float(px) - points[0][0], float(py) - points[0][1])
    return min(
        _point_segment_distance((float(px), float(py)), points[index - 1], points[index])
        for index in range(1, len(points))
    )


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(max(t, 0.0), 1.0)
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _segment_distance(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> float:
    if _segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def _polyline_segment_distance(points: list[tuple[float, float]], seg_start, seg_end) -> float:
    if len(points) < 2:
        return _point_segment_distance(points[0], seg_start, seg_end) if points else math.inf
    return min(
        _segment_distance(points[index - 1], points[index], seg_start, seg_end)
        for index in range(1, len(points))
    )


def _polyline_polyline_distance(a_points: list[tuple[float, float]], b_points: list[tuple[float, float]]) -> float:
    if len(a_points) < 2 or len(b_points) < 2:
        if not a_points or not b_points:
            return math.inf
        return min(_point_to_polyline_distance(a_points[0][0], a_points[0][1], b_points), _point_to_polyline_distance(b_points[0][0], b_points[0][1], a_points))
    return min(
        _segment_distance(a_points[a_index - 1], a_points[a_index], b_points[b_index - 1], b_points[b_index])
        for a_index in range(1, len(a_points))
        for b_index in range(1, len(b_points))
    )


def _segments_intersect(a0, a1, b0, b1) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    return o1 * o2 <= 0.0 and o3 * o4 <= 0.0
