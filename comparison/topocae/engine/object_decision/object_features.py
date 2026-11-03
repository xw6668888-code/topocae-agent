"""Inference-only, permutation-stable evidence for a complete feature object.

Face indices address graph arrays; they are never emitted as numeric features.
Only registered proposal mechanisms can become source features.  This module
does not inspect a sample name, truth object, candidate ID, or source path.
"""

from __future__ import annotations

import itertools
import math
from statistics import fmean


MECHANISMS = (
    "old_full", "pre_boundary", "initial_group", "raw_geometry",
    "bounded_geometry", "physical_split", "weak_relation_split",
    "axis_interval_split", "geometry_merge", "sparse_branch", "dense_branch",
)
NUMERIC_FEATURE_NAMES = (
    "face_count_log", "class_mean", "class_low_quantile", "class_min",
    "relation_applicable", "relation_present", "relation_coverage",
    "internal_mean", "internal_low_quantile", "internal_contradiction_fraction",
    "boundary_present", "boundary_high_fraction", "external_competition_fraction",
    "geometry_present", "axis_pair_parallel_fraction", "axis_separation_max",
    "axial_gap_max", "radius_difference_max", "host_contact_fraction",
    "mechanism_count_log", "cross_mechanism_agreement",
) + tuple(f"mechanism_{name}" for name in MECHANISMS)


def _probability(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be a finite probability")
    return number


def _quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    return ordered[low] * (1 - (position - low)) + ordered[math.ceil(position)] * (position - low)


def _vector(value):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 and all(math.isfinite(x) for x in result) else None


def _norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def _unit(vector):
    magnitude = _norm(vector)
    return tuple(x / magnitude for x in vector) if magnitude > 1e-12 else None


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _cylinder(node):
    if node.get("primitive_surface") != "cylinder":
        return None
    attributes = node.get("attributes", {})
    geometry = attributes.get("geometry_v2", {})
    specific = attributes.get("specific", {})
    axis = _vector(geometry.get("axis_direction", specific.get("axis_direction", specific.get("axis"))))
    location = _vector(geometry.get("axis_location", specific.get("axis_location")))
    radius = geometry.get("radius", specific.get("radius"))
    try:
        radius = float(radius)
    except (TypeError, ValueError):
        radius = None
    if radius is not None and (not math.isfinite(radius) or radius < 0):
        radius = None
    return {"axis": _unit(axis) if axis is not None else None,
            "location": location, "radius": radius, "node": node}


def _axis_interval(node, direction):
    geometry = node.get("attributes", {}).get("geometry_v2", {})
    lower = _vector(geometry.get("axis_aligned_bbox_min"))
    upper = _vector(geometry.get("axis_aligned_bbox_max"))
    if lower is None or upper is None or any(a > b for a, b in zip(lower, upper)):
        return None
    projections = [_dot(corner, direction) for corner in itertools.product(*zip(lower, upper))]
    return min(projections), max(projections)


def _relation_pairs(prediction, graph, node_count):
    edges = prediction.get("relation_edges")
    if edges is None:
        edges = graph.get("edges", []) if graph is not None else []
        probabilities = prediction.get("edge_probabilities")
    else:
        probabilities = prediction.get("relation_probabilities", prediction.get("edge_probabilities"))
    if probabilities is None:
        if edges:
            raise ValueError("Relation edges have no probability rows")
        return {}
    if len(edges) != len(probabilities):
        raise ValueError("Relation edge/probability length mismatch")
    pairs = {}
    for edge, value in zip(edges, probabilities):
        if len(edge) < 2:
            raise ValueError("Malformed relation edge")
        left, right = edge[:2]
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= node_count for x in (left, right)):
            raise ValueError("Invalid relation face index")
        if left == right:
            continue
        pairs.setdefault(tuple(sorted((left, right))), set()).add(_probability(value, "relation"))
    # Each unordered face pair has weight one, regardless of reciprocal or
    # parallel B-Rep rows.  Distinct row values are evidence within that pair.
    return {pair: fmean(values) for pair, values in pairs.items()}


def extract_object_features(candidate, prediction, graph=None, *, contradiction_threshold=0.2,
                            high_boundary_threshold=0.8):
    """Return numeric features and explicit relation availability diagnostics.

    ``candidate`` needs ``macro_feature``, ``face_indices`` and optional
    ``sources`` (registered mechanism names, not file/sample provenance).
    ``prediction`` uses the existing ``faces`` and relation arrays.  A single
    face has relation status ``not_applicable``; a multi-face object with no
    measured internal pair has status ``missing``; measured zero is ``observed``.
    """
    contradiction_threshold = _probability(contradiction_threshold, "contradiction_threshold")
    high_boundary_threshold = _probability(high_boundary_threshold, "high_boundary_threshold")
    macro = candidate["macro_feature"]
    faces = candidate["face_indices"]
    if not faces or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in faces) or len(faces) != len(set(faces)):
        raise ValueError("Candidate needs distinct nonnegative integer face indices")
    source_value = candidate.get("sources", ())
    if isinstance(source_value, str):
        source_value = (source_value,)
    # The pool also stores detailed R:stage:mechanism trace labels.  They are
    # useful for audit, but only the allowlisted general mechanism tokens may
    # affect prediction.  Unknown text (including paths/names) cannot leak.
    mechanisms = frozenset(value for value in source_value if value in MECHANISMS)
    face_rows = {row["idx"]: row for row in prediction["faces"]}
    if len(face_rows) != len(prediction["faces"]) or any(face not in face_rows for face in faces):
        raise ValueError("Candidate face is absent from prediction")
    if graph is not None and (len(graph["nodes"]) != len(face_rows) or any(i not in face_rows for i in range(len(face_rows)))):
        raise ValueError("Prediction/graph face count mismatch")
    class_values = [_probability(face_rows[face]["macro_probabilities"][macro], "face class") for face in faces]
    members = set(faces)
    relations = _relation_pairs(prediction, graph, len(face_rows))
    internal = [value for (a, b), value in relations.items() if a in members and b in members]
    outside = [(a if a not in members else b, value) for (a, b), value in relations.items()
               if (a in members) != (b in members)]
    possible_internal = len(faces) * (len(faces) - 1) // 2
    status = "not_applicable" if not possible_internal else "observed" if internal else "missing"
    competing = [value for other, value in outside
                 if _probability(face_rows[other]["macro_probabilities"].get(macro, 0), "external class") >= 0.5]
    host_contact = 0.0
    if graph is not None:
        touched = set()
        for a, b in graph.get("edges", ()):
            if (a in members) != (b in members):
                touched.add(a if a in members else b)
        host_contact = len(touched) / len(members)

    geo_values = {"axis_pair_parallel_fraction": 0.0, "axis_separation_max": 0.0,
                  "axial_gap_max": 0.0, "radius_difference_max": 0.0}
    geometry_present = 0.0
    if graph is not None:
        scale = graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_characteristic_length")
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 0.0
        if math.isfinite(scale) and scale > 0:
            cylinders = [_cylinder(graph["nodes"][i]) for i in faces]
            cylinder_pairs = [(a, b) for a, b in itertools.combinations(cylinders, 2)
                              if a is not None and b is not None and a["axis"] is not None and b["axis"] is not None]
            parallel = 0
            for a, b in cylinder_pairs:
                if abs(_dot(a["axis"], b["axis"])) < 1 - 1e-4:
                    continue
                parallel += 1
                if a["location"] is not None and b["location"] is not None:
                    delta = tuple(y - x for x, y in zip(a["location"], b["location"]))
                    # A cylinder centroid can lie off-axis: use the actual
                    # surface axis location, never the face center of mass.
                    separation = _norm(_cross(delta, a["axis"])) / scale
                    geo_values["axis_separation_max"] = max(geo_values["axis_separation_max"], math.log1p(separation))
                    geometry_present = 1.0
                ia = _axis_interval(a["node"], a["axis"])
                ib = _axis_interval(b["node"], a["axis"])
                if ia is not None and ib is not None:
                    gap = max(0.0, ia[0] - ib[1], ib[0] - ia[1]) / scale
                    geo_values["axial_gap_max"] = max(geo_values["axial_gap_max"], math.log1p(gap))
                    geometry_present = 1.0
                if a["radius"] is not None and b["radius"] is not None:
                    geo_values["radius_difference_max"] = max(
                        geo_values["radius_difference_max"], math.log1p(abs(a["radius"] - b["radius"]) / scale))
                    geometry_present = 1.0
            geo_values["axis_pair_parallel_fraction"] = parallel / len(cylinder_pairs) if cylinder_pairs else 0.0

    numeric = {"face_count_log": math.log1p(len(faces)), "class_mean": fmean(class_values),
               "class_low_quantile": _quantile(class_values, 0.1), "class_min": min(class_values),
               "relation_applicable": float(possible_internal > 0), "relation_present": float(bool(internal)),
               "relation_coverage": len(internal) / possible_internal if possible_internal else 0.0,
               "internal_mean": fmean(internal) if internal else 0.0,
               "internal_low_quantile": _quantile(internal, 0.1) if internal else 0.0,
               "internal_contradiction_fraction": sum(v < contradiction_threshold for v in internal) / len(internal) if internal else 0.0,
               "boundary_present": float(bool(outside)),
               "boundary_high_fraction": sum(v >= high_boundary_threshold for _, v in outside) / len(outside) if outside else 0.0,
               "external_competition_fraction": sum(v >= high_boundary_threshold for v in competing) / len(competing) if competing else 0.0,
               "geometry_present": geometry_present, "host_contact_fraction": host_contact,
               "mechanism_count_log": math.log1p(len(mechanisms)),
               "cross_mechanism_agreement": float(len(mechanisms) > 1), **geo_values,
               **{f"mechanism_{name}": float(name in mechanisms) for name in MECHANISMS}}
    assert set(numeric) == set(NUMERIC_FEATURE_NAMES)
    return {"numeric": numeric, "vector": tuple(numeric[name] for name in NUMERIC_FEATURE_NAMES),
            "internal_relation_status": status, "internal_observed_pairs": len(internal),
            "internal_missing_pairs": possible_internal - len(internal),
            "boundary_observed_pairs": len(outside)}


def deterministic_score(features):
    """Small signed A2 starting score; calibrate only on development validation."""
    f = features["numeric"]
    score = 2.0 * f["class_mean"] + f["class_low_quantile"] - 1.5
    if f["relation_present"]:
        score += 0.5 * (f["internal_mean"] - 0.5) - 1.2 * f["internal_contradiction_fraction"]
    score += 0.2 * f["cross_mechanism_agreement"]
    score -= 0.35 * f["external_competition_fraction"]
    if f["geometry_present"]:
        score -= 0.4 * f["axis_separation_max"] + 0.25 * f["axial_gap_max"]
    return score


__all__ = ["MECHANISMS", "NUMERIC_FEATURE_NAMES", "extract_object_features", "deterministic_score"]
