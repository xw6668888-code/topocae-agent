from __future__ import annotations

"""Rotation/translation/scale-invariant B-Rep cylinder material polarity."""

import math
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.TopologyUtils import TopologyExplorer


FEATURE_NAME = "oriented_cylindrical_radial_alignment"
REVISION = "p1_c_oriented_cylinder_polarity_v1"


def augment_graph_with_oriented_cylinder_polarity(shape: Any, graph: dict[str, Any]) -> dict[str, Any]:
    faces = list(TopologyExplorer(shape).faces())
    nodes = list(graph.get("nodes", []))
    if len(faces) != len(nodes):
        raise RuntimeError("STEP/Graph face count mismatch while adding cylinder polarity")
    values = []
    for face, node in zip(faces, nodes):
        value = oriented_cylindrical_radial_alignment(face)
        recognition = node.setdefault("attributes", {}).setdefault("recognition_features_v2", {})
        recognition[FEATURE_NAME] = value
        values.append(value)
    protocol = graph.setdefault("graph_v2_protocol", {})
    protocol["oriented_cylinder_polarity_revision"] = REVISION
    protocol["oriented_cylinder_polarity_definition"] = "dot(oriented_face_normal,local_cylinder_radial)"
    protocol["oriented_cylinder_polarity_invariances"] = ["rotation", "translation", "uniform_scale"]
    protocol["oriented_cylinder_polarity_uses_labels"] = False
    protocol["oriented_cylinder_polarity_uses_sample_metadata"] = False
    graph["p1_c_oriented_cylinder_polarity_audit"] = {
        "revision": REVISION,
        "node_count": len(values),
        "nonzero_count": sum(abs(value) > 0.5 for value in values),
        "finite": all(math.isfinite(value) for value in values),
        "bounded": all(-1.0 <= value <= 1.0 for value in values),
    }
    if not graph["p1_c_oriented_cylinder_polarity_audit"]["finite"]:
        raise RuntimeError("non-finite cylinder polarity")
    if not graph["p1_c_oriented_cylinder_polarity_audit"]["bounded"]:
        raise RuntimeError("out-of-range cylinder polarity")
    return graph


def oriented_cylindrical_radial_alignment(face: Any) -> float:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Cylinder:
        return 0.0
    u = _finite_midpoint(surface.FirstUParameter(), surface.LastUParameter())
    v = _finite_midpoint(surface.FirstVParameter(), surface.LastVParameter())
    props = BRepLProp_SLProps(surface, u, v, 1, 1e-7)
    if not props.IsNormalDefined():
        raise RuntimeError("cylindrical surface normal is undefined")
    normal = _xyz(props.Normal())
    if face.Orientation() == TopAbs_REVERSED:
        normal = _scale(normal, -1.0)
    point = _xyz(surface.Value(u, v))
    cylinder = surface.Cylinder()
    origin = _xyz(cylinder.Location())
    axis = _unit(_xyz(cylinder.Axis().Direction()))
    offset = _subtract(point, origin)
    radial = _subtract(offset, _scale(axis, _dot(offset, axis)))
    value = _dot(_unit(normal), _unit(radial))
    if not math.isfinite(value):
        raise RuntimeError("cylindrical polarity is non-finite")
    return max(-1.0, min(1.0, value))


def _finite_midpoint(first: float, last: float) -> float:
    first_value, last_value = float(first), float(last)
    if not math.isfinite(first_value) or not math.isfinite(last_value):
        raise RuntimeError("unbounded cylindrical parameter range")
    return (first_value + last_value) / 2.0


def _xyz(value: Any) -> list[float]:
    return [float(value.X()), float(value.Y()), float(value.Z())]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _scale(vector: list[float], factor: float) -> list[float]:
    return [value * factor for value in vector]


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-12:
        raise RuntimeError("degenerate cylinder radial/normal vector")
    return [value / length for value in vector]

