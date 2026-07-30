from __future__ import annotations

import math

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

from utils.path_geometry import path_tangent_angle, polyline_length


def swept_profile_metadata(
    profile_shape: str,
    sweep_path_length: float,
    profile_width: float,
    profile_depth: float,
    *,
    path_type: str = "straight",
    path_curve_kind: str = "line",
    path_on_face: bool = True,
    path_start_on_boundary: bool = False,
    path_end_on_boundary: bool = False,
    anchor_face_kind: str = "existing_planar_face",
) -> dict:
    return {
        "construction_method": "swept_closed_profile",
        "profile_shape": profile_shape,
        "sweep_path_length": float(sweep_path_length),
        "profile_width": float(profile_width),
        "profile_depth": float(profile_depth),
        "sweep_to_profile_ratio": float(sweep_path_length) / max(float(profile_width), 1e-9),
        "path_type": path_type,
        "path_curve_kind": path_curve_kind,
        "path_on_face": bool(path_on_face),
        "path_start_on_boundary": bool(path_start_on_boundary),
        "path_end_on_boundary": bool(path_end_on_boundary),
        "anchor_face_kind": anchor_face_kind,
    }


def straight_boundary_path_on_box(seed_x: float, seed_y: float, angle: float, L: float, W: float, inset: float = 0.0) -> dict:
    half_l = max(float(L) / 2.0 - float(inset), 1e-6)
    half_w = max(float(W) / 2.0 - float(inset), 1e-6)
    sx = _clamp(float(seed_x), -half_l + 1e-6, half_l - 1e-6)
    sy = _clamp(float(seed_y), -half_w + 1e-6, half_w - 1e-6)
    dx = math.cos(angle)
    dy = math.sin(angle)
    candidates: list[tuple[float, tuple[float, float]]] = []

    if abs(dx) > 1e-9:
        for x_boundary in (-half_l, half_l):
            t = (x_boundary - sx) / dx
            y = sy + t * dy
            if -half_w - 1e-6 <= y <= half_w + 1e-6:
                candidates.append((t, (x_boundary, _clamp(y, -half_w, half_w))))

    if abs(dy) > 1e-9:
        for y_boundary in (-half_w, half_w):
            t = (y_boundary - sy) / dy
            x = sx + t * dx
            if -half_l - 1e-6 <= x <= half_l + 1e-6:
                candidates.append((t, (_clamp(x, -half_l, half_l), y_boundary)))

    unique_candidates: list[tuple[float, tuple[float, float]]] = []
    for t, point in sorted(candidates, key=lambda item: item[0]):
        if unique_candidates and abs(t - unique_candidates[-1][0]) < 1e-6:
            continue
        unique_candidates.append((t, point))

    if len(unique_candidates) < 2:
        raise ValueError("Could not find two boundary intersections for swept slot path.")

    t_start, start = unique_candidates[0]
    t_end, end = unique_candidates[-1]
    center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    length = math.hypot(end[0] - start[0], end[1] - start[1])

    return {
        "start": start,
        "end": end,
        "center": center,
        "length": length,
        "angle": math.atan2(end[1] - start[1], end[0] - start[0]),
        "seed": (sx, sy),
        "t_start": t_start,
        "t_end": t_end,
    }


def straight_boundary_path_on_circle(seed_x: float, seed_y: float, angle: float, radius: float) -> dict:
    r = max(float(radius), 1e-6)
    sx = float(seed_x)
    sy = float(seed_y)
    seed_radius = math.hypot(sx, sy)
    if seed_radius >= r:
        scale = (r - 1e-6) / max(seed_radius, 1e-9)
        sx *= scale
        sy *= scale

    dx = math.cos(angle)
    dy = math.sin(angle)
    b = 2.0 * (sx * dx + sy * dy)
    c = sx * sx + sy * sy - r * r
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        raise ValueError("Could not find circle intersections for swept slot path.")

    root = math.sqrt(discriminant)
    t_start = (-b - root) / 2.0
    t_end = (-b + root) / 2.0
    start = (sx + t_start * dx, sy + t_start * dy)
    end = (sx + t_end * dx, sy + t_end * dy)
    center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)

    return {
        "start": start,
        "end": end,
        "center": center,
        "length": math.hypot(end[0] - start[0], end[1] - start[1]),
        "angle": math.atan2(end[1] - start[1], end[0] - start[0]),
        "seed": (sx, sy),
        "t_start": t_start,
        "t_end": t_end,
    }


def build_rectangular_swept_profile_wire(cx: float, cy: float, z: float, width: float, path_length: float, angle: float):
    half_width = width / 2.0
    half_length = path_length / 2.0
    return _closed_polygon_from_local_points(cx, cy, z, [
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ], angle)


def build_capsule_swept_profile_wire(cx: float, cy: float, z: float, radius: float, path_length: float, angle: float):
    right_top = _rotated_point(cx, cy, z, path_length / 2.0, radius, angle)
    left_top = _rotated_point(cx, cy, z, -path_length / 2.0, radius, angle)
    left_bottom = _rotated_point(cx, cy, z, -path_length / 2.0, -radius, angle)
    right_bottom = _rotated_point(cx, cy, z, path_length / 2.0, -radius, angle)
    right_mid = _rotated_point(cx, cy, z, path_length / 2.0 + radius, 0.0, angle)
    left_mid = _rotated_point(cx, cy, z, -path_length / 2.0 - radius, 0.0, angle)

    return BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(right_bottom, right_mid, right_top).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(right_top, left_top).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(left_top, left_mid, left_bottom).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(left_bottom, right_bottom).Edge(),
    ).Wire()


def build_u_swept_profile_wire(cx: float, cy: float, z: float, radius: float, path_length: float, angle: float):
    return build_capsule_swept_profile_wire(cx, cy, z, radius, path_length, angle)


def build_v_swept_profile_wire(cx: float, cy: float, z: float, width: float, path_length: float, angle: float):
    half_width = width / 2.0
    half_length = path_length / 2.0
    return _closed_polygon_from_local_points(cx, cy, z, [
        (-half_length, -half_width),
        (half_length, 0.0),
        (-half_length, half_width),
    ], angle)


def build_dovetail_swept_profile_wire(
    cx: float,
    cy: float,
    z: float,
    bottom_width: float,
    top_width: float,
    path_length: float,
    angle: float,
):
    half_bottom = bottom_width / 2.0
    half_top = top_width / 2.0
    half_length = path_length / 2.0
    return _closed_polygon_from_local_points(cx, cy, z, [
        (-half_length, -half_bottom),
        (half_length, -half_top),
        (half_length, half_top),
        (-half_length, half_bottom),
    ], angle)


def build_o_swept_profile_wire(cx: float, cy: float, z: float, radius: float, path_length: float, angle: float):
    circle = GC_MakeCircle(gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0.0, 0.0, 1.0)), radius).Value()
    return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()


def vertical_swept_slot_profile_points(
    profile_shape: str,
    width: float,
    depth: float,
    surface_z: float,
    top_margin: float = 1.0,
) -> list[tuple[float, float]]:
    shape = profile_shape.lower()
    half_width = float(width) / 2.0
    top_z = float(surface_z) + float(top_margin)
    bottom_z = float(surface_z) - float(depth)

    if shape == "u":
        radius = min(half_width, max(float(depth) * 0.65, half_width * 0.55))
        center_z = bottom_z + radius
        points = [(-half_width, top_z), (-half_width, center_z)]
        for index in range(1, 12):
            theta = math.pi + math.pi * index / 12.0
            points.append((half_width * math.cos(theta), center_z + radius * math.sin(theta)))
        points.append((half_width, top_z))
        return points

    if shape == "v":
        return [(-half_width, top_z), (0.0, bottom_z), (half_width, top_z)]

    if shape == "dovetail":
        top_half = half_width * 0.62
        return [(-top_half, top_z), (-half_width, bottom_z), (half_width, bottom_z), (top_half, top_z)]

    if shape == "o":
        center_z = (top_z + bottom_z) / 2.0
        vertical_radius = (top_z - bottom_z) / 2.0
        return [
            (
                half_width * math.cos(2.0 * math.pi * index / 28.0),
                center_z + vertical_radius * math.sin(2.0 * math.pi * index / 28.0),
            )
            for index in range(28)
        ]

    raise ValueError(f"Unsupported swept slot profile shape: {profile_shape}")


def build_vertical_swept_slot_profile_wire(
    profile_shape: str,
    start_x: float,
    start_y: float,
    path_angle: float,
    width: float,
    depth: float,
    surface_z: float,
    top_margin: float = 1.0,
):
    shape = profile_shape.lower()
    path_dx = math.cos(path_angle)
    path_dy = math.sin(path_angle)
    lateral_dx = -math.sin(path_angle)
    lateral_dy = math.cos(path_angle)
    half_width = float(width) / 2.0
    top_z = float(surface_z) + float(top_margin)
    bottom_z = float(surface_z) - float(depth)

    def point(lateral: float, z_value: float) -> gp_Pnt:
        return gp_Pnt(
            float(start_x) + lateral_dx * lateral,
            float(start_y) + lateral_dy * lateral,
            z_value,
        )

    if shape == "u":
        radius = min(half_width, max(float(depth) - 1e-6, 1e-6))
        center_z = bottom_z + radius
        top_left = point(-half_width, top_z)
        top_right = point(half_width, top_z)
        right_arc_end = point(half_width, center_z)
        bottom_mid = point(0.0, center_z - radius)
        left_arc_end = point(-half_width, center_z)
        return BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(top_left, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, right_arc_end).Edge(),
            BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(right_arc_end, bottom_mid, left_arc_end).Value()).Edge(),
            BRepBuilderAPI_MakeEdge(left_arc_end, top_left).Edge(),
        ).Wire()

    if shape == "v":
        top_left = point(-half_width, top_z)
        bottom_mid = point(0.0, bottom_z)
        top_right = point(half_width, top_z)
        return BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(top_left, bottom_mid).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_mid, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        ).Wire()

    if shape == "dovetail":
        top_half = half_width * 0.62
        top_left = point(-top_half, top_z)
        bottom_left = point(-half_width, bottom_z)
        bottom_right = point(half_width, bottom_z)
        top_right = point(top_half, top_z)
        return BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(top_left, bottom_left).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_left, bottom_right).Edge(),
            BRepBuilderAPI_MakeEdge(bottom_right, top_right).Edge(),
            BRepBuilderAPI_MakeEdge(top_right, top_left).Edge(),
        ).Wire()

    if shape == "o":
        radius = min(half_width, max(float(depth), 1e-6))
        center = point(0.0, float(surface_z))
        circle = GC_MakeCircle(gp_Ax2(center, gp_Dir(path_dx, path_dy, 0.0)), radius).Value()
        return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()

    raise ValueError(f"Unsupported swept slot profile shape: {profile_shape}")


def build_vertical_swept_slot_tool(
    profile_shape: str,
    cx: float,
    cy: float,
    surface_z: float,
    path_length: float,
    path_angle: float,
    width: float,
    depth: float,
    top_margin: float = 1.0,
):
    path_dx = math.cos(path_angle)
    path_dy = math.sin(path_angle)
    start_x = cx - path_dx * path_length / 2.0
    start_y = cy - path_dy * path_length / 2.0

    profile_wire = build_vertical_swept_slot_profile_wire(
        profile_shape=profile_shape,
        start_x=start_x,
        start_y=start_y,
        path_angle=path_angle,
        width=width,
        depth=depth,
        surface_z=surface_z,
        top_margin=top_margin,
    )
    face = BRepBuilderAPI_MakeFace(profile_wire).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(path_dx * path_length, path_dy * path_length, 0.0)).Shape()


def build_path_wire_from_points(
    path_points_xy: list[tuple[float, float]],
    surface_z: float,
    path_curve_kind: str = "polyline",
):
    if len(path_points_xy) < 2:
        raise ValueError("Swept slot path requires at least two points.")
    if path_curve_kind == "circular_arc" and len(path_points_xy) >= 3:
        start = path_points_xy[0]
        mid = path_points_xy[len(path_points_xy) // 2]
        end = path_points_xy[-1]
        return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(
            gp_Pnt(float(start[0]), float(start[1]), float(surface_z)),
            gp_Pnt(float(mid[0]), float(mid[1]), float(surface_z)),
            gp_Pnt(float(end[0]), float(end[1]), float(surface_z)),
        ).Value()).Edge()).Wire()
    wire_builder = BRepBuilderAPI_MakeWire()
    for index in range(1, len(path_points_xy)):
        start = path_points_xy[index - 1]
        end = path_points_xy[index]
        wire_builder.Add(BRepBuilderAPI_MakeEdge(
            gp_Pnt(float(start[0]), float(start[1]), float(surface_z)),
            gp_Pnt(float(end[0]), float(end[1]), float(surface_z)),
        ).Edge())
    return wire_builder.Wire()


def build_vertical_swept_slot_tool_from_path(
    profile_shape: str,
    path_points_xy: list[tuple[float, float]],
    surface_z: float,
    width: float,
    depth: float,
    top_margin: float = 1.0,
    path_curve_kind: str = "polyline",
):
    if len(path_points_xy) < 2:
        raise ValueError("Swept slot path requires at least two points.")
    if len(path_points_xy) == 2:
        start, end = path_points_xy
        center_x = (float(start[0]) + float(end[0])) / 2.0
        center_y = (float(start[1]) + float(end[1])) / 2.0
        return build_vertical_swept_slot_tool(
            profile_shape=profile_shape,
            cx=center_x,
            cy=center_y,
            surface_z=surface_z,
            path_length=polyline_length(path_points_xy),
            path_angle=path_tangent_angle(path_points_xy),
            width=width,
            depth=depth,
            top_margin=top_margin,
        )

    start_x, start_y = path_points_xy[0]
    profile_wire = build_vertical_swept_slot_profile_wire(
        profile_shape=profile_shape,
        start_x=float(start_x),
        start_y=float(start_y),
        path_angle=path_tangent_angle(path_points_xy),
        width=width,
        depth=depth,
        surface_z=surface_z,
        top_margin=top_margin,
    )
    profile_face = BRepBuilderAPI_MakeFace(profile_wire).Face()
    path_wire = build_path_wire_from_points(path_points_xy, surface_z, path_curve_kind=path_curve_kind)
    pipe = BRepOffsetAPI_MakePipe(path_wire, profile_face)
    pipe.Build()
    if not pipe.IsDone():
        raise ValueError("Could not build swept slot pipe tool.")
    return pipe.Shape()


def _rotated_point(cx: float, cy: float, z: float, local_x: float, local_y: float, angle: float):
    rx = local_x * math.cos(angle) - local_y * math.sin(angle)
    ry = local_x * math.sin(angle) + local_y * math.cos(angle)
    return gp_Pnt(cx + rx, cy + ry, z)


def _closed_polygon_from_local_points(cx: float, cy: float, z: float, points: list[tuple[float, float]], angle: float):
    poly = BRepBuilderAPI_MakePolygon()
    for local_x, local_y in points:
        poly.Add(_rotated_point(cx, cy, z, local_x, local_y, angle))
    poly.Close()
    return poly.Wire()


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
