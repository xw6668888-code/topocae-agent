from __future__ import annotations

from copy import deepcopy
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c import all_features as core
from p1_c import formal_generator_v4 as revision4
from p1_c.formal_plan_v4 import PLAN_REVISION, PlannedSample


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r5"
SUPPORTED_WALL_CLASSES = revision4.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision4.transformed_shape_and_audit


def _all_geometric_outer_linear_edges(
    shape: Any,
    host: dict[str, Any],
) -> list[tuple[float, tuple[float, float, float], Any]]:
    x_min = float(host.get("x_min", -float(host["length"]) / 2.0))
    x_max = float(host.get("x_max", float(host["length"]) / 2.0))
    y_min = float(host.get("y_min", -float(host["width"]) / 2.0))
    y_max = float(host.get("y_max", float(host["width"]) / 2.0))
    length = float(host["length"])
    width = float(host["width"])
    top_z = float(host["top_z"])
    tolerance = max(length, width) * 1e-5
    candidates: list[tuple[int, float, float, tuple[float, float, float], Any]] = []
    for edge in TopologyExplorer(shape).edges():
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line:
            continue
        first = curve.Value(curve.FirstParameter())
        last = curve.Value(curve.LastParameter())
        point = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2.0)
        midpoint = (float(point.X()), float(point.Y()), float(point.Z()))
        on_outer_xy = (
            abs(midpoint[0] - x_min) <= tolerance
            or abs(midpoint[0] - x_max) <= tolerance
            or abs(midpoint[1] - y_min) <= tolerance
            or abs(midpoint[1] - y_max) <= tolerance
            or abs(midpoint[0]) > 0.40 * length
            or abs(midpoint[1]) > 0.40 * width
        )
        if not on_outer_xy:
            continue
        top_preference = 0 if abs(midpoint[2] - top_z) <= tolerance else 1
        candidates.append(
            (top_preference, abs(midpoint[2] - top_z), float(first.Distance(last)), midpoint, edge)
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -abs(item[3][0]), -abs(item[3][1])))
    return [(length_value, midpoint, edge) for _top, _z_distance, length_value, midpoint, edge in candidates]


revision4._transition_candidates = _all_geometric_outer_linear_edges


def build_planned_result(plan: PlannedSample) -> core.CleanFeatureResult:
    result = revision4.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["plan_revision"] = PLAN_REVISION
    parameters["transition_edge_selection"] = "first_cleanup_idempotent_geometric_outer_linear_edge"
    return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
