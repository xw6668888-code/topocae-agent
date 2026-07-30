from __future__ import annotations

from typing import Any, Iterable

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.gp import gp_Pnt, gp_Vec


def polygon_prism_xy(
    points: Iterable[tuple[float, float]],
    *,
    z_min: float,
    height: float,
) -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    materialized = list(points)
    if len(materialized) < 3:
        raise ValueError("an XY profile requires at least three points")
    for x_value, y_value in materialized:
        polygon.Add(gp_Pnt(float(x_value), float(y_value), float(z_min)))
    polygon.Close()
    face_builder = BRepBuilderAPI_MakeFace(polygon.Wire())
    if not face_builder.IsDone():
        raise ValueError("could not build XY polygon face")
    return BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0.0, 0.0, float(height))).Shape()


def rectangle_prism_xy(
    center: tuple[float, float],
    size_x: float,
    size_y: float,
    *,
    z_min: float,
    height: float,
) -> Any:
    cx, cy = center
    half_x = float(size_x) / 2.0
    half_y = float(size_y) / 2.0
    return polygon_prism_xy(
        [
            (cx - half_x, cy - half_y),
            (cx + half_x, cy - half_y),
            (cx + half_x, cy + half_y),
            (cx - half_x, cy + half_y),
        ],
        z_min=z_min,
        height=height,
    )


def capsule_prism_xy(
    center: tuple[float, float],
    center_distance: float,
    radius: float,
    *,
    z_min: float,
    height: float,
) -> Any:
    cx, cy = center
    half_distance = float(center_distance) / 2.0
    radius = float(radius)
    right_top = gp_Pnt(cx + half_distance, cy + radius, z_min)
    left_top = gp_Pnt(cx - half_distance, cy + radius, z_min)
    left_bottom = gp_Pnt(cx - half_distance, cy - radius, z_min)
    right_bottom = gp_Pnt(cx + half_distance, cy - radius, z_min)
    right_mid = gp_Pnt(cx + half_distance + radius, cy, z_min)
    left_mid = gp_Pnt(cx - half_distance - radius, cy, z_min)
    wire = BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(right_bottom, right_mid, right_top).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(right_top, left_top).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(left_top, left_mid, left_bottom).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(left_bottom, right_bottom).Edge(),
    ).Wire()
    face_builder = BRepBuilderAPI_MakeFace(wire)
    if not face_builder.IsDone():
        raise ValueError("could not build capsule face")
    return BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0.0, 0.0, float(height))).Shape()


def u_profile_prism_xy(
    center: tuple[float, float],
    straight_length: float,
    radius: float,
    *,
    z_min: float,
    height: float,
) -> Any:
    cx, cy = center
    left_x = cx - float(straight_length) / 2.0
    right_x = cx + float(straight_length) / 2.0
    radius = float(radius)
    right_top = gp_Pnt(right_x, cy + radius, z_min)
    left_top = gp_Pnt(left_x, cy + radius, z_min)
    left_bottom = gp_Pnt(left_x, cy - radius, z_min)
    right_bottom = gp_Pnt(right_x, cy - radius, z_min)
    right_mid = gp_Pnt(right_x + radius, cy, z_min)
    wire = BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(right_bottom, right_mid).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(right_bottom, right_mid, right_top).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(right_top, left_top).Edge(),
        BRepBuilderAPI_MakeEdge(left_top, left_bottom).Edge(),
        BRepBuilderAPI_MakeEdge(left_bottom, right_bottom).Edge(),
    ).Wire()
    face_builder = BRepBuilderAPI_MakeFace(wire)
    if not face_builder.IsDone():
        raise ValueError("could not build U profile face")
    return BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0.0, 0.0, float(height))).Shape()


__all__ = [
    "capsule_prism_xy",
    "polygon_prism_xy",
    "rectangle_prism_xy",
    "u_profile_prism_xy",
]
