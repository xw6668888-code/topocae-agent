from __future__ import annotations

from copy import deepcopy
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c import all_features as core
from p1_c import formal_generator_v3 as revision3
from p1_c.formal_plan_v4 import PLAN_REVISION, PlannedSample
from p1_c.geometry import cleanup_audit


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r4"
SUPPORTED_WALL_CLASSES = revision3.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision3.transformed_shape_and_audit


def _transition_candidates(shape: Any, host: dict[str, Any]) -> list[tuple[float, tuple[float, float, float], Any]]:
    top = float(host["top_z"])
    candidates: list[tuple[float, tuple[float, float, float], Any]] = []
    for edge in TopologyExplorer(shape).edges():
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line:
            continue
        first = curve.Value(curve.FirstParameter())
        last = curve.Value(curve.LastParameter())
        if abs(first.Z() - top) > 1e-5 or abs(last.Z() - top) > 1e-5:
            continue
        point = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2.0)
        midpoint = (float(point.X()), float(point.Y()), float(point.Z()))
        outer = abs(midpoint[0]) > 0.40 * float(host["length"]) or abs(midpoint[1]) > 0.40 * float(host["width"])
        if outer:
            candidates.append((float(first.Distance(last)), midpoint, edge))
    candidates.sort(key=lambda item: (item[0], -abs(item[1][0]), -abs(item[1][1])))
    return candidates


def _apply_clean_transition(
    shape: Any,
    tags: list[core.FaceTag],
    macro: str,
    instance_id: int,
    host: dict[str, Any],
    rng: Any,
):
    size = rng.uniform(1.2, 2.2)
    if macro == "fillet":
        builder_type = BRepFilletAPI_MakeFillet
        semantic_role = "fillet_blend_surface"
        parameter_key = "radius"
    elif macro == "chamfer":
        builder_type = BRepFilletAPI_MakeChamfer
        semantic_role = "chamfer_bevel_plane"
        parameter_key = "distance"
    else:
        raise KeyError(macro)
    failures: list[dict[str, Any]] = []
    candidates = _transition_candidates(shape, host)
    for rank, (length, midpoint, edge) in enumerate(candidates):
        builder = builder_type(shape)
        builder.Add(size, edge)
        builder.Build()
        if not builder.IsDone():
            failures.append({"rank": rank, "length": length, "reason": "builder_not_done"})
            continue
        new_shape = builder.Shape()
        audit = cleanup_audit(new_shape)
        if not audit["passed"]:
            failures.append(
                {
                    "rank": rank,
                    "length": length,
                    "reason": "cleanup_not_idempotent",
                    "removable_faces": audit["removable_same_domain_face_count"],
                    "removable_edges": audit["removable_same_domain_edge_count"],
                }
            )
            continue
        propagated = core._propagate_tags(builder, tags)
        created: list[core.FaceTag] = []
        for generated in builder.Generated(edge):
            if generated.ShapeType() == 4:
                core._append_tag(created, core.FaceTag(generated, semantic_role, macro, int(instance_id)))
        merged = core._merge_tags(propagated, created)
        feature_face_count = sum(tag.instance_id == instance_id for tag in merged)
        if feature_face_count != 1:
            failures.append(
                {"rank": rank, "length": length, "reason": f"feature_face_count_{feature_face_count}"}
            )
            continue
        record = {
            "macro_feature": macro,
            "instance_id": int(instance_id),
            "operation": "edge_transition",
            "center": list(midpoint),
            "parameters": {
                parameter_key: size,
                "candidate_rank": rank,
                "selected_edge_length": length,
                "selection_contract": "first_cleanup_idempotent_outer_top_linear_edge",
            },
            "feature_face_count": feature_face_count,
            "cleanup_after_operation": audit,
        }
        return new_shape, merged, record
    raise ValueError(f"{macro} has no cleanup-idempotent outer top edge: {failures}")


core._apply_transition = _apply_clean_transition


def build_planned_result(plan: PlannedSample) -> core.CleanFeatureResult:
    result = revision3.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["plan_revision"] = PLAN_REVISION
    parameters["transition_edge_selection"] = "first_cleanup_idempotent_outer_top_linear_edge"
    return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
