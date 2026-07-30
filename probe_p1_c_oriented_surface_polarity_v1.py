from __future__ import annotations

"""Synthetic-only probe for rotation-invariant oriented surface polarity."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

from utils.p1_c_strict_run_paths import DATASET_ROOT, ROOT


OUTPUT = ROOT / "reports" / "p1_c_iterations_v1" / "iteration1_oriented_surface_polarity_probe.json"
CLASSES = ("pocket_circular", "cylinder_boss")


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _scale(vector: list[float], factor: float) -> list[float]:
    return [value * factor for value in vector]


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(_dot(vector, vector))
    return [value / length for value in vector] if length > 1e-12 else [0.0, 0.0, 0.0]


def _values(value: Any) -> list[float]:
    return [float(value.X()), float(value.Y()), float(value.Z())]


def _cylinder_oriented_radial_alignment(face: Any) -> float:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Cylinder:
        raise ValueError("polarity probe requires a cylindrical face")
    u = (float(surface.FirstUParameter()) + float(surface.LastUParameter())) / 2.0
    v = (float(surface.FirstVParameter()) + float(surface.LastVParameter())) / 2.0
    props = BRepLProp_SLProps(surface, u, v, 1, 1e-7)
    if not props.IsNormalDefined():
        raise RuntimeError("surface normal is undefined")
    normal = _values(props.Normal())
    if face.Orientation() == TopAbs_REVERSED:
        normal = _scale(normal, -1.0)
    point = _values(surface.Value(u, v))
    cylinder = surface.Cylinder()
    origin = _values(cylinder.Location())
    axis = _unit(_values(cylinder.Axis().Direction()))
    offset = _sub(point, origin)
    radial = _sub(offset, _scale(axis, _dot(offset, axis)))
    return max(-1.0, min(1.0, _dot(_unit(normal), _unit(radial))))


def main() -> int:
    values: dict[str, list[float]] = defaultdict(list)
    orientation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    graph_paths = sorted((DATASET_ROOT / "graphs").glob("*.json"))
    selected = []
    for path in graph_paths:
        graph = json.loads(path.read_text(encoding="utf-8"))
        sampling = graph.get("generation", {}).get("sampling_metadata", {})
        macro = str(sampling.get("primary_target", ""))
        if macro not in CLASSES or sampling.get("context") != "target_only":
            continue
        if sum(1 for item in selected if item[0] == macro) >= 40:
            continue
        selected.append((macro, path, graph))
        if all(sum(1 for item in selected if item[0] == name) >= 40 for name in CLASSES):
            break
    failures = []
    for macro, graph_path, graph in selected:
        step_path = DATASET_ROOT / "steps" / f"{graph['sample_id']}.step"
        shape = read_step_file(str(step_path), verbosity=False)
        faces = list(TopologyExplorer(shape).faces())
        if len(faces) != len(graph["nodes"]):
            failures.append({"sample_id": graph["sample_id"], "reason": "face_count_mismatch"})
            continue
        found = False
        for face, node in zip(faces, graph["nodes"]):
            if node.get("macro_feature") != macro or node.get("primitive_surface") != "cylinder":
                continue
            try:
                value = _cylinder_oriented_radial_alignment(face)
            except Exception as exc:
                failures.append({"sample_id": graph["sample_id"], "node": node["idx"], "reason": f"{type(exc).__name__}:{exc}"})
                continue
            values[macro].append(value)
            orientation_counts[macro][str(int(face.Orientation()))] += 1
            found = True
        if not found:
            failures.append({"sample_id": graph["sample_id"], "reason": "no_feature_cylinder"})
    summary = {
        macro: {
            "count": len(records),
            "mean": mean(records) if records else None,
            "std": pstdev(records) if len(records) > 1 else 0.0,
            "min": min(records) if records else None,
            "max": max(records) if records else None,
            "negative_fraction": sum(value < -0.5 for value in records) / max(len(records), 1),
            "positive_fraction": sum(value > 0.5 for value in records) / max(len(records), 1),
            "orientation_counts": dict(orientation_counts[macro]),
        }
        for macro, records in values.items()
    }
    result = {
        "schema_version": "topocae_p1_c_oriented_surface_polarity_probe_1",
        "feature": "rotation_invariant_oriented_normal_dot_cylinder_radial",
        "scope": "40 Synthetic target_only samples per class",
        "summary": summary,
        "failures": failures,
        "contracts": {"real_data_used": False, "holdout_used": False, "labels_used_for_probe_only": True},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary": summary, "failure_count": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
