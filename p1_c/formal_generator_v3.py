from __future__ import annotations

from copy import deepcopy
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c import all_features as core
from p1_c import formal_generator_v2 as revision2
from p1_c.formal_plan_v4 import PLAN_REVISION, PlannedSample


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r3"
SUPPORTED_WALL_CLASSES = revision2.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision2.transformed_shape_and_audit


def _select_clean_outer_top_edge(shape: Any, host: dict[str, Any]) -> Any:
    top = float(host["top_z"])
    candidates: list[tuple[float, float, float, Any]] = []
    for edge in TopologyExplorer(shape).edges():
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line:
            continue
        first = curve.Value(curve.FirstParameter())
        last = curve.Value(curve.LastParameter())
        if abs(first.Z() - top) > 1e-5 or abs(last.Z() - top) > 1e-5:
            continue
        point = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2.0)
        outer = abs(point.X()) > 0.40 * float(host["length"]) or abs(point.Y()) > 0.40 * float(host["width"])
        if not outer:
            continue
        candidates.append((first.Distance(last), -abs(point.X()), -abs(point.Y()), edge))
    if not candidates:
        raise ValueError("no clean outer top edge available for formal transition")
    # A short boundary edge avoids the split long edge produced where a supported
    # wall meets an L-shaped host. Dimensions remain generic and geometry-derived.
    return min(candidates, key=lambda item: item[:3])[3]


core._select_outer_top_edge = _select_clean_outer_top_edge


def build_planned_result(plan: PlannedSample) -> core.CleanFeatureResult:
    result = revision2.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["plan_revision"] = PLAN_REVISION
    parameters["transition_edge_selection"] = "shortest_outer_top_linear_edge_generic"
    return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
