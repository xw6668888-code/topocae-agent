from __future__ import annotations

from utils.step_solid_validation import signed_solid_volume


def inspect_shape_quality(shape) -> dict:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Extend.TopologyUtils import TopologyExplorer

    topo = TopologyExplorer(shape)
    solid_count = len(list(topo.solids()))
    shell_count = len(list(topo.shells()))
    face_count = len(list(topo.faces()))
    edge_count = len(list(topo.edges()))
    signed_volume = signed_solid_volume(shape)
    topology_valid = BRepCheck_Analyzer(shape).IsValid()

    issues = []
    if solid_count == 0:
        issues.append("no_solid")
    if not topology_valid:
        issues.append("invalid_topology")
    if signed_volume is None or abs(signed_volume) <= 1e-9:
        issues.append("zero_solid_volume")

    return {
        "solid_count": solid_count,
        "shell_count": shell_count,
        "face_count": face_count,
        "edge_count": edge_count,
        "signed_volume": signed_volume,
        "topology_valid": topology_valid,
        "issues": issues,
        "acceptable": len(issues) == 0,
    }


def inspect_generation_candidate(shape, feature_count: int, require_nonempty_sample: bool) -> dict:
    report = inspect_shape_quality(shape)
    issues = list(report["issues"])
    if require_nonempty_sample and feature_count <= 0:
        issues.insert(0, "empty_sample")

    return {
        **report,
        "feature_count": int(feature_count),
        "require_nonempty_sample": bool(require_nonempty_sample),
        "issues": issues,
        "acceptable": len(issues) == 0,
    }
