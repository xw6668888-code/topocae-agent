import math

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)


DEFAULT_TRANSITION_POLICY = {
    "min_edge_length": 6.0,
    "min_size": 0.8,
    "max_size": 3.0,
    "midpoint_clearance_factor": 5.0,
}


def merge_transition_policy(generation_context):
    policy = dict(DEFAULT_TRANSITION_POLICY)
    policy.update((generation_context or {}).get("transition_features", {}))
    return policy


def extract_transition_edge_candidates(
    stock,
    existing_params,
    *,
    min_edge_length,
    midpoint_clearance,
):
    topo = TopologyExplorer(stock)
    candidates = []
    for edge in topo.edges():
        connected_faces = list(topo.faces_from_edge(edge))
        if len(connected_faces) != 2:
            continue

        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()
        if curve_type not in {GeomAbs_Line, GeomAbs_Circle}:
            continue

        candidate = _edge_candidate(edge, curve, curve_type, connected_faces)
        if candidate["length"] < float(min_edge_length):
            continue
        if _is_near_existing_transition(candidate, existing_params, midpoint_clearance):
            continue

        candidates.append(candidate)
    return candidates


def compute_transition_size_limits(candidate, L, W, H, config):
    min_size = float(config.get("min_size", DEFAULT_TRANSITION_POLICY["min_size"]))
    requested_max = float(config.get("max_size", DEFAULT_TRANSITION_POLICY["max_size"]))
    model_scale_limit = min(float(H) * 0.055, min(float(L), float(W)) * 0.025, requested_max)

    if candidate.get("curve_type") == "circle":
        local_radius = float(candidate.get("curve_radius", 0.0))
        edge_limit = max(local_radius * 0.22, min_size)
    else:
        edge_limit = max(float(candidate.get("length", 0.0)) * 0.08, min_size)

    max_size = min(model_scale_limit, edge_limit)
    return min_size, max_size


def public_transition_parameters(*, inst, size_key, size_value, candidate):
    return {
        "inst": inst,
        size_key: size_value,
        "midpoint": list(candidate["midpoint"]),
        "axis": list(candidate["axis"]),
        "curve_type": candidate["curve_type"],
    }


def _edge_candidate(edge, curve, curve_type, connected_faces):
    mid_param = (curve.FirstParameter() + curve.LastParameter()) / 2.0
    mid_pnt = curve.Value(mid_param)
    midpoint = [mid_pnt.X(), mid_pnt.Y(), mid_pnt.Z()]
    surf_types = [BRepAdaptor_Surface(face).GetType() for face in connected_faces]

    if curve_type == GeomAbs_Line:
        axis_dir = curve.Line().Direction()
        length = abs(curve.LastParameter() - curve.FirstParameter())
        candidate_curve_type = "line"
        curve_radius = None
    else:
        circle = curve.Circle()
        axis_dir = circle.Axis().Direction()
        length = 2.0 * math.pi * circle.Radius()
        candidate_curve_type = "circle"
        curve_radius = circle.Radius()

    candidate = {
        "edge": edge,
        "midpoint": midpoint,
        "axis": [axis_dir.X(), axis_dir.Y(), axis_dir.Z()],
        "length": length,
        "curve_type": candidate_curve_type,
        "connected_surface_types": [str(surf_type) for surf_type in surf_types],
        "internal_edge_context": _classify_internal_context(candidate_curve_type, surf_types),
    }
    if curve_radius is not None:
        candidate["curve_radius"] = curve_radius
    return candidate


def _is_near_existing_transition(candidate, existing_params, midpoint_clearance):
    for params in existing_params:
        midpoint = params.get("midpoint")
        if midpoint is None:
            continue
        if math.dist(candidate["midpoint"], midpoint) < float(midpoint_clearance):
            return True
    return False


def _classify_internal_context(candidate_curve_type, surface_types):
    curved_surface_types = {
        GeomAbs_Cylinder,
        GeomAbs_Cone,
        GeomAbs_Torus,
        GeomAbs_SurfaceOfRevolution,
    }
    if candidate_curve_type == "circle":
        return "circular_feature_edge"
    if any(surf_type in curved_surface_types for surf_type in surface_types):
        return "curved_feature_edge"
    return "linear_model_edge"
