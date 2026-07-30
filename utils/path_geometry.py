from __future__ import annotations

import math


Point2D = tuple[float, float]


def polyline_length(points: list[Point2D]) -> float:
    return sum(
        math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    )


def point_to_polyline_distance(px: float, py: float, points: list[Point2D]) -> float:
    if not points:
        raise ValueError("Polyline must contain at least one point.")
    if len(points) == 1:
        return math.hypot(float(px) - points[0][0], float(py) - points[0][1])
    return min(
        point_to_segment_distance((float(px), float(py)), points[index - 1], points[index])
        for index in range(1, len(points))
    )


def point_to_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
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


def sample_circular_arc_points(
    start: Point2D,
    end: Point2D,
    sagitta: float,
    sample_count: int = 17,
) -> list[Point2D]:
    sample_count = max(int(sample_count), 3)
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    sagitta = float(sagitta)
    chord_dx = ex - sx
    chord_dy = ey - sy
    chord_length = math.hypot(chord_dx, chord_dy)
    if chord_length <= 1e-9:
        raise ValueError("Arc endpoints must be distinct.")
    if abs(sagitta) <= 1e-9:
        return [
            (
                sx + chord_dx * index / (sample_count - 1),
                sy + chord_dy * index / (sample_count - 1),
            )
            for index in range(sample_count)
        ]

    mid_x = (sx + ex) / 2.0
    mid_y = (sy + ey) / 2.0
    normal_x = -chord_dy / chord_length
    normal_y = chord_dx / chord_length
    radius = (chord_length * chord_length) / (8.0 * abs(sagitta)) + abs(sagitta) / 2.0
    center_offset = math.copysign(radius - abs(sagitta), sagitta)
    center_x = mid_x - normal_x * center_offset
    center_y = mid_y - normal_y * center_offset
    start_angle = math.atan2(sy - center_y, sx - center_x)
    end_angle = math.atan2(ey - center_y, ex - center_x)
    sweep = _normalize_arc_sweep(end_angle - start_angle, sagitta)

    points: list[Point2D] = []
    for index in range(sample_count):
        if index == 0:
            points.append((sx, sy))
            continue
        if index == sample_count - 1:
            points.append((ex, ey))
            continue
        theta = start_angle + sweep * index / (sample_count - 1)
        points.append((center_x + radius * math.cos(theta), center_y + radius * math.sin(theta)))
    return points


def path_tangent_angle(points: list[Point2D], at_end: bool = False) -> float:
    if len(points) < 2:
        raise ValueError("Path must contain at least two points.")
    if at_end:
        start, end = points[-2], points[-1]
    else:
        start, end = points[0], points[1]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _normalize_arc_sweep(raw_sweep: float, sagitta: float) -> float:
    if sagitta > 0.0:
        while raw_sweep <= 0.0:
            raw_sweep += 2.0 * math.pi
        if raw_sweep > math.pi:
            raw_sweep -= 2.0 * math.pi
        return raw_sweep

    while raw_sweep >= 0.0:
        raw_sweep -= 2.0 * math.pi
    if raw_sweep < -math.pi:
        raw_sweep += 2.0 * math.pi
    return raw_sweep
