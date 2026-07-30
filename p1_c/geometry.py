from __future__ import annotations

import math
from typing import Any, Iterable

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer


def prism_xz(points: Iterable[tuple[float, float]], *, y_min: float, depth: float) -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    materialized = list(points)
    if len(materialized) < 3:
        raise ValueError("a prism profile needs at least three points")
    for x_value, z_value in materialized:
        polygon.Add(gp_Pnt(float(x_value), float(y_min), float(z_value)))
    polygon.Close()
    face_builder = BRepBuilderAPI_MakeFace(polygon.Wire())
    if not face_builder.IsDone():
        raise ValueError("could not build XZ profile face")
    return BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0.0, float(depth), 0.0)).Shape()


def clean_fuse(first: Any, second: Any) -> tuple[Any, Any]:
    operation = BRepAlgoAPI_Fuse(first, second)
    operation.Build()
    if not operation.IsDone():
        raise ValueError("boolean fuse failed")
    operation.SimplifyResult(True, True)
    shape = operation.Shape()
    _require_basic_quality(shape, "fuse")
    return operation, shape


def clean_cut(stock: Any, tool: Any) -> tuple[Any, Any]:
    operation = BRepAlgoAPI_Cut(stock, tool)
    operation.Build()
    if not operation.IsDone():
        raise ValueError("boolean cut failed")
    operation.SimplifyResult(True, True)
    shape = operation.Shape()
    _require_basic_quality(shape, "cut")
    return operation, shape


def unified_probe(shape: Any) -> Any:
    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.Build()
    return unifier.Shape()


def cleanup_audit(shape: Any) -> dict[str, Any]:
    before = topology_counts(shape)
    before_volume = abs(shape_volume(shape))
    probe = unified_probe(shape)
    after = topology_counts(probe)
    after_volume = abs(shape_volume(probe))
    denominator = max(before_volume, after_volume, 1e-12)
    relative_volume_error = abs(before_volume - after_volume) / denominator
    removable_faces = max(0, before["face_count"] - after["face_count"])
    removable_edges = max(0, before["edge_count"] - after["edge_count"])
    checks = {
        "single_solid_before": before["solid_count"] == 1,
        "single_solid_after": after["solid_count"] == 1,
        "valid_before": bool(BRepCheck_Analyzer(shape).IsValid()),
        "valid_after": bool(BRepCheck_Analyzer(probe).IsValid()),
        "positive_volume": before_volume > 1e-9,
        "face_idempotent": before["face_count"] == after["face_count"],
        "edge_idempotent": before["edge_count"] == after["edge_count"],
        "relative_volume_error_lt_1e_9": relative_volume_error < 1e-9,
        "no_removable_same_domain_faces": removable_faces == 0,
        "no_redundant_coplanar_stock_partitions": removable_faces == 0 and removable_edges == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "before": before,
        "after": after,
        "relative_volume_error": relative_volume_error,
        "removable_same_domain_face_count": removable_faces,
        "removable_same_domain_edge_count": removable_edges,
        "redundant_coplanar_stock_partition_count": 0 if checks["no_redundant_coplanar_stock_partitions"] else removable_faces,
    }


def topology_counts(shape: Any) -> dict[str, int]:
    topology = TopologyExplorer(shape)
    return {
        "solid_count": len(list(topology.solids())),
        "shell_count": len(list(topology.shells())),
        "face_count": len(list(topology.faces())),
        "edge_count": len(list(topology.edges())),
    }


def shape_volume(shape: Any) -> float:
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return float(properties.Mass())


def face_area(face: Any) -> float:
    properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, properties)
    return float(properties.Mass())


def face_center(face: Any) -> tuple[float, float, float]:
    properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, properties)
    center = properties.CentreOfMass()
    return float(center.X()), float(center.Y()), float(center.Z())


def common_volume(first: Any, second: Any) -> float:
    operation = BRepAlgoAPI_Common(first, second)
    operation.Build()
    if not operation.IsDone():
        raise ValueError("boolean common failed")
    return abs(shape_volume(operation.Shape()))


def operation_faces(operation: Any, source_face: Any) -> list[Any]:
    final_faces = list(TopologyExplorer(operation.Shape()).faces())
    candidates: list[Any] = []
    for accessor in (operation.Modified, operation.Generated):
        for mapped in accessor(source_face):
            if mapped.ShapeType() == 4:
                _append_shape_unique(candidates, mapped)
    if not candidates:
        for face in final_faces:
            if source_face.IsSame(face):
                _append_shape_unique(candidates, face)
    return candidates


def propagated_faces(operation: Any, source_face: Any) -> list[Any]:
    final_faces = list(TopologyExplorer(operation.Shape()).faces())
    mapped = operation_faces(operation, source_face)
    if not mapped and any(source_face.IsSame(face) for face in final_faces):
        mapped = [source_face]
    return mapped


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(key) and finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def _append_shape_unique(items: list[Any], candidate: Any) -> None:
    if not any(candidate.IsSame(existing) for existing in items):
        items.append(candidate)


def _require_basic_quality(shape: Any, operation_name: str) -> None:
    counts = topology_counts(shape)
    if counts["solid_count"] != 1:
        raise ValueError(f"{operation_name} result is not one solid: {counts}")
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(f"{operation_name} result is not a valid B-Rep")
    if abs(shape_volume(shape)) <= 1e-9:
        raise ValueError(f"{operation_name} result has no positive volume")


__all__ = [
    "clean_cut",
    "clean_fuse",
    "cleanup_audit",
    "common_volume",
    "face_area",
    "face_center",
    "finite_tree",
    "operation_faces",
    "prism_xz",
    "propagated_faces",
    "shape_volume",
    "topology_counts",
    "unified_probe",
]
