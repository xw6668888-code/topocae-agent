from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepTools import breptools
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Extend.TopologyUtils import TopologyExplorer

import core.AAGExtractor as aag


GRAPH_V2_SCHEMA = "topocae_graph_v2_relative_1"
EPSILON = 1e-12
MAX_STABLE_RATIO = 1.0e4


def extract_graph_v2(
    shape: Any,
    *,
    face_labels: list[dict[str, Any]] | None = None,
    primitive_surface_index: dict[str, Any] | None = None,
    build_patches: bool = True,
) -> dict[str, Any]:
    faces = list(TopologyExplorer(shape).faces())
    graph = aag.extract_aag_graph(
        shape,
        face_infos=None,
        primitive_surface_index=primitive_surface_index,
        include_label_fields=False,
    )
    if len(graph["nodes"]) != len(faces):
        raise RuntimeError("face traversal changed while constructing Graph v2")
    labels_provided = face_labels is not None
    labels = face_labels or [_default_stock_label() for _ in faces]
    if len(labels) != len(faces):
        raise ValueError("face_labels must have exactly one record per B-Rep face")
    _inject_exact_labels(graph, labels)
    _replace_edges_with_identity_safe_adjacency(graph, shape, faces)
    part_metadata = _part_metadata(shape)
    _annotate_nodes_v2(graph, faces, part_metadata)
    _annotate_edges_v2(graph, part_metadata)
    graph["schema_version"] = GRAPH_V2_SCHEMA
    graph["part_metadata_v2"] = part_metadata
    graph["graph_v2_protocol"] = {
        "face_identity": "TopoDS_Shape.IsSame",
        "recognition_scale": "rotation_invariant_farthest_point_diameter",
        "raw_physical_values_preserved": True,
        "labels_provided": labels_provided,
        "canonical_patches_non_destructive": bool(build_patches),
        "stable_ratio_cap": MAX_STABLE_RATIO,
    }
    if build_patches:
        patch_data = build_canonical_patches(graph, respect_labels=labels_provided)
        graph.update(patch_data)
    _assert_graph_v2_finite(graph)
    return graph


def _default_stock_label() -> dict[str, Any]:
    return {
        "semantic_label": 0,
        "instance_id": 0,
        "macro_feature": "stock",
        "macro_feature_id": 0,
    }


def _inject_exact_labels(graph: dict[str, Any], labels: list[dict[str, Any]]) -> None:
    instances: dict[int, dict[str, Any]] = {}
    for node, label in zip(graph["nodes"], labels):
        node["semantic_label"] = int(label.get("semantic_label", 0))
        node["instance_id"] = int(label.get("instance_id", 0))
        node["macro_feature"] = str(label.get("macro_feature", "stock"))
        node["macro_feature_id"] = int(label.get("macro_feature_id", 0))
        if "semantic_role" in label:
            node["semantic_role"] = str(label["semantic_role"])
        instance_id = node["instance_id"]
        if instance_id <= 0:
            continue
        record = instances.setdefault(instance_id, {
            "instance_id": instance_id,
            "macro_feature": node["macro_feature"],
            "macro_feature_id": node["macro_feature_id"],
            "node_indices": [],
        })
        if record["macro_feature"] != node["macro_feature"]:
            raise ValueError("one instance_id cannot contain multiple macro features")
        record["node_indices"].append(int(node["idx"]))
    graph["feature_sets"] = [instances[key] for key in sorted(instances)]


def _replace_edges_with_identity_safe_adjacency(graph: dict[str, Any], shape: Any, faces: list[Any]) -> None:
    topology = TopologyExplorer(shape)
    edges: list[list[int]] = []
    edge_attributes: list[dict[str, Any]] = []
    for edge in topology.edges():
        connected = list(topology.faces_from_edge(edge))
        if len(connected) != 2:
            continue
        source_idx = _identity_face_index(connected[0], faces)
        target_idx = _identity_face_index(connected[1], faces)
        if source_idx is None or target_idx is None or source_idx == target_idx:
            continue
        edges.append([source_idx, target_idx])
        edge_attributes.append(aag._build_directed_edge_attribute(
            edge, source_idx, target_idx, graph["nodes"], shape, connected[0], connected[1]
        ))
        edges.append([target_idx, source_idx])
        edge_attributes.append(aag._build_directed_edge_attribute(
            edge, target_idx, source_idx, graph["nodes"], shape, connected[1], connected[0]
        ))
    graph["edges"] = edges
    graph["edge_attributes"] = edge_attributes
    _reset_neighborhood_descriptors(graph["nodes"])
    aag._annotate_neighborhood_descriptors(graph["nodes"], edges)


def _identity_face_index(target: Any, faces: list[Any]) -> int | None:
    for index, face in enumerate(faces):
        if target.IsSame(face):
            return index
    return None


def _reset_neighborhood_descriptors(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        descriptors = node["attributes"].setdefault("shape_descriptors", {})
        for key in (
            "neighbor_count",
            "neighbor_area_mean",
            "neighbor_area_max",
            "neighbor_projected_aspect_mean",
            "neighbor_long_extent_mean",
        ):
            descriptors[key] = 0.0


def _part_metadata(shape: Any) -> dict[str, Any]:
    bbox = _shape_bbox(shape)
    sample_points = _sample_shape_points(shape)
    characteristic_length = _farthest_point_diameter(sample_points)
    if characteristic_length <= EPSILON:
        characteristic_length = max(float(bbox["diagonal"]), 1.0)
    surface_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, surface_props)
    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, volume_props)
    return {
        "raw_physical": {
            "total_surface_area": _finite(surface_props.Mass()),
            "solid_volume": _finite(volume_props.Mass()),
            "axis_aligned_bbox_min": bbox["min"],
            "axis_aligned_bbox_max": bbox["max"],
            "axis_aligned_bbox_extents": bbox["extents"],
            "axis_aligned_bbox_diagonal": bbox["diagonal"],
        },
        "normalization": {
            "part_total_area": _finite(surface_props.Mass()),
            "part_bbox_diagonal": _finite(bbox["diagonal"]),
            "part_characteristic_length": _finite(characteristic_length),
            "part_characteristic_area": _finite(characteristic_length * characteristic_length),
            "sample_point_count": len(sample_points),
        },
    }


def _shape_bbox(shape: Any) -> dict[str, Any]:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    extents = [max(0.0, xmax - xmin), max(0.0, ymax - ymin), max(0.0, zmax - zmin)]
    return {
        "min": [_finite(xmin), _finite(ymin), _finite(zmin)],
        "max": [_finite(xmax), _finite(ymax), _finite(zmax)],
        "extents": [_finite(value) for value in extents],
        "diagonal": _finite(math.sqrt(sum(value * value for value in extents))),
    }


def _sample_shape_points(shape: Any, samples_per_edge: int = 5, limit: int = 768) -> list[list[float]]:
    points: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for edge in TopologyExplorer(shape).edges():
        try:
            curve = BRepAdaptor_Curve(edge)
            first, last = float(curve.FirstParameter()), float(curve.LastParameter())
            for sample_index in range(samples_per_edge):
                fraction = sample_index / max(samples_per_edge - 1, 1)
                point = curve.Value(first + (last - first) * fraction)
                value = [float(point.X()), float(point.Y()), float(point.Z())]
                key = tuple(round(component, 10) for component in value)
                if key not in seen:
                    seen.add(key)
                    points.append(value)
        except Exception:
            continue
    if len(points) > limit:
        stride = len(points) / float(limit)
        points = [points[min(int(index * stride), len(points) - 1)] for index in range(limit)]
    return points


def _farthest_point_diameter(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    anchor = points[0]
    first = max(points, key=lambda point: _distance_sq(point, anchor))
    second = max(points, key=lambda point: _distance_sq(point, first))
    return math.sqrt(_distance_sq(first, second))


def _distance_sq(first: Iterable[float], second: Iterable[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(first, second))


def _annotate_nodes_v2(graph: dict[str, Any], faces: list[Any], part_metadata: dict[str, Any]) -> None:
    normalization = part_metadata["normalization"]
    total_area = max(_finite(normalization["part_total_area"]), EPSILON)
    bbox_diagonal = max(_finite(normalization["part_bbox_diagonal"]), EPSILON)
    scale = max(_finite(normalization["part_characteristic_length"]), EPSILON)
    scale_sq = scale * scale
    for node, face in zip(graph["nodes"], faces):
        attrs = node["attributes"]
        physical = attrs.get("physical", {})
        specific = attrs.get("specific", {})
        descriptors = attrs.get("shape_descriptors", {})
        area = max(0.0, _finite(physical.get("area", 0.0)))
        radius = max(0.0, _finite(specific.get("radius", 0.0)))
        major_radius = max(0.0, _finite(specific.get("major_radius", 0.0)))
        minor_radius = max(0.0, _finite(specific.get("minor_radius", 0.0)))
        neighbor_area_mean = max(0.0, _finite(descriptors.get("neighbor_area_mean", 0.0)))
        bbox_extents = sorted([
            max(0.0, _finite(descriptors.get("bbox_dx", 0.0))),
            max(0.0, _finite(descriptors.get("bbox_dy", 0.0))),
            max(0.0, _finite(descriptors.get("bbox_dz", 0.0))),
        ], reverse=True)
        projected_long = max(0.0, _finite(descriptors.get("projected_long_extent", 0.0)))
        projected_short = max(0.0, _finite(descriptors.get("projected_short_extent", 0.0)))
        uv_metric_long, uv_metric_short = _physical_uv_extents(face, radius, descriptors)
        pca_extents = _rotation_invariant_pca_extents(face)
        loop_stats = _face_loop_stats(face)
        geometry = _face_geometry_v2(face)
        recognition = {
            "face_area_over_part_total_area": _stable_ratio(area, total_area),
            "face_area_over_part_bbox_diagonal_sq": _stable_ratio(area, bbox_diagonal * bbox_diagonal),
            "face_area_over_part_scale_sq": _stable_ratio(area, scale_sq),
            "radius_over_part_bbox_diagonal": _stable_ratio(radius, bbox_diagonal),
            "radius_over_part_scale": _stable_ratio(radius, scale),
            "major_radius_over_part_scale": _stable_ratio(major_radius, scale),
            "minor_radius_over_part_scale": _stable_ratio(minor_radius, scale),
            "bbox_long_over_part_bbox_diagonal": _stable_ratio(bbox_extents[0], bbox_diagonal),
            "bbox_middle_over_part_bbox_diagonal": _stable_ratio(bbox_extents[1], bbox_diagonal),
            "bbox_short_over_part_bbox_diagonal": _stable_ratio(bbox_extents[2], bbox_diagonal),
            "projected_long_over_part_scale": _stable_ratio(projected_long, scale),
            "projected_short_over_part_scale": _stable_ratio(projected_short, scale),
            "uv_metric_long_over_part_scale": _stable_ratio(uv_metric_long, scale),
            "uv_metric_short_over_part_scale": _stable_ratio(uv_metric_short, scale),
            "pca_long_over_part_scale": _stable_ratio(pca_extents[0], scale),
            "pca_middle_over_part_scale": _stable_ratio(pca_extents[1], scale),
            "pca_short_over_part_scale": _stable_ratio(pca_extents[2], scale),
            "face_area_over_neighbor_area_mean": _stable_ratio(area, neighbor_area_mean),
            "neighbor_area_mean_over_part_total_area": _stable_ratio(neighbor_area_mean, total_area),
            "local_extent_long_middle_ratio": _stable_ratio(pca_extents[0], pca_extents[1]),
            "local_extent_middle_short_ratio": _stable_ratio(pca_extents[1], pca_extents[2]),
            "projected_aspect_ratio": _stable_nonnegative(descriptors.get("projected_aspect_ratio", 0.0)),
            "projected_area_fill_ratio": _stable_nonnegative(descriptors.get("projected_area_fill_ratio", 0.0)),
            "uv_aspect_ratio": _stable_ratio(uv_metric_long, uv_metric_short),
            "vertex_count": _stable_nonnegative(descriptors.get("vertex_count", 0.0)),
            "edge_count": _stable_nonnegative(descriptors.get("edge_count", 0.0)),
            "neighbor_count": _stable_nonnegative(descriptors.get("neighbor_count", 0.0)),
            "wire_count": float(loop_stats["wire_count"]),
            "outer_loop_count": float(loop_stats["outer_loop_count"]),
            "inner_loop_count": float(loop_stats["inner_loop_count"]),
            "outer_loop_edge_count": float(loop_stats["outer_loop_edge_count"]),
            "inner_loop_edge_count": float(loop_stats["inner_loop_edge_count"]),
        }
        recognition.update({
            f"log1p_{key}": _stable_log1p(value)
            for key, value in list(recognition.items())
            if key.endswith("ratio") or "_over_" in key
        })
        attrs["raw_physical_v2"] = {
            "area": area,
            "radius": radius,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "axis_aligned_bbox_extents": bbox_extents,
            "projected_extents": [projected_long, projected_short],
            "uv_metric_extents": [uv_metric_long, uv_metric_short],
            "pca_extents": pca_extents,
            "neighbor_area_mean": neighbor_area_mean,
        }
        attrs["topology_v2"] = loop_stats
        attrs["geometry_v2"] = geometry
        attrs["recognition_features_v2"] = recognition


def _physical_uv_extents(face: Any, radius: float, descriptors: dict[str, Any]) -> tuple[float, float]:
    try:
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        u_span, v_span = abs(float(umax - umin)), abs(float(vmax - vmin))
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder and radius > EPSILON:
            extents = sorted([u_span * radius, v_span], reverse=True)
        elif surface.GetType() == GeomAbs_Plane:
            extents = sorted([u_span, v_span], reverse=True)
        else:
            extents = sorted([
                max(0.0, _finite(descriptors.get("projected_long_extent", 0.0))),
                max(0.0, _finite(descriptors.get("projected_short_extent", 0.0))),
            ], reverse=True)
        return extents[0], extents[1]
    except Exception:
        return 0.0, 0.0


def _rotation_invariant_pca_extents(face: Any) -> list[float]:
    points = _sample_shape_points(face, samples_per_edge=7, limit=256)
    if len(points) < 2:
        return [0.0, 0.0, 0.0]
    mean = [sum(point[axis] for point in points) / len(points) for axis in range(3)]
    covariance = [[0.0] * 3 for _ in range(3)]
    for point in points:
        centered = [point[axis] - mean[axis] for axis in range(3)]
        for row in range(3):
            for column in range(3):
                covariance[row][column] += centered[row] * centered[column] / len(points)
    eigenvalues = _jacobi_eigenvalues_symmetric_3x3(covariance)
    return [2.0 * math.sqrt(max(0.0, value)) for value in sorted(eigenvalues, reverse=True)]


def _jacobi_eigenvalues_symmetric_3x3(matrix: list[list[float]]) -> list[float]:
    values = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    for _ in range(32):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        if abs(values[p][q]) <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * values[p][q], values[q][q] - values[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        app, aqq, apq = values[p][p], values[q][q], values[p][q]
        values[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        values[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        values[p][q] = values[q][p] = 0.0
        for index in range(3):
            if index in (p, q):
                continue
            aip, aiq = values[index][p], values[index][q]
            values[index][p] = values[p][index] = cosine * aip - sine * aiq
            values[index][q] = values[q][index] = sine * aip + cosine * aiq
    return [_finite(values[index][index]) for index in range(3)]


def _face_loop_stats(face: Any) -> dict[str, int]:
    wires = list(TopologyExplorer(face).wires())
    outer_wire = None
    try:
        outer_wire = breptools.OuterWire(face)
    except Exception:
        pass
    outer_count = 0
    inner_count = 0
    outer_edge_count = 0
    inner_edge_count = 0
    for wire in wires:
        edge_count = len(list(TopologyExplorer(wire).edges()))
        if outer_wire is not None and wire.IsSame(outer_wire):
            outer_count += 1
            outer_edge_count += edge_count
        else:
            inner_count += 1
            inner_edge_count += edge_count
    if wires and outer_count == 0:
        outer_count = 1
        inner_count = max(0, len(wires) - 1)
    return {
        "wire_count": len(wires),
        "outer_loop_count": outer_count,
        "inner_loop_count": inner_count,
        "outer_loop_edge_count": outer_edge_count,
        "inner_loop_edge_count": inner_edge_count,
    }


def _face_geometry_v2(face: Any) -> dict[str, Any]:
    surface = BRepAdaptor_Surface(face)
    geometry: dict[str, Any] = {"surface_type_id": int(surface.GetType())}
    if surface.GetType() == GeomAbs_Plane:
        plane = surface.Plane()
        geometry["surface_family"] = "plane"
        geometry["location"] = _point_values(plane.Location())
        geometry["normal"] = _direction_values(plane.Axis().Direction())
    elif surface.GetType() == GeomAbs_Cylinder:
        cylinder = surface.Cylinder()
        geometry["surface_family"] = "cylinder"
        geometry["axis_location"] = _point_values(cylinder.Location())
        geometry["axis_direction"] = _direction_values(cylinder.Axis().Direction())
        geometry["radius"] = _finite(cylinder.Radius())
    else:
        geometry["surface_family"] = "other"
    box = _shape_bbox(face)
    geometry["axis_aligned_bbox_min"] = box["min"]
    geometry["axis_aligned_bbox_max"] = box["max"]
    return geometry


def _point_values(point: Any) -> list[float]:
    return [_finite(point.X()), _finite(point.Y()), _finite(point.Z())]


def _direction_values(direction: Any) -> list[float]:
    return [_finite(direction.X()), _finite(direction.Y()), _finite(direction.Z())]


def _annotate_edges_v2(graph: dict[str, Any], part_metadata: dict[str, Any]) -> None:
    normalization = part_metadata["normalization"]
    bbox_diagonal = max(_finite(normalization["part_bbox_diagonal"]), EPSILON)
    scale = max(_finite(normalization["part_characteristic_length"]), EPSILON)
    for edge_attr in graph["edge_attributes"]:
        length = max(0.0, _finite(edge_attr.get("length", 0.0)))
        radius = max(0.0, _finite(edge_attr.get("circle_radius", 0.0)))
        wall_role = str(edge_attr.get("cylindrical_wall_role_hint", "unknown"))
        convexity = str(edge_attr.get("convexity", "unknown"))
        inner_evidence = 1.0 if wall_role == "inner_hole_wall" else (0.5 if convexity == "concave" else 0.0)
        outer_evidence = 1.0 if wall_role == "outer_cylinder_wall" else (0.5 if convexity == "convex" else 0.0)
        ambiguous_evidence = 1.0 if wall_role in {"unknown", "ambiguous"} else 0.0
        recognition = {
            "edge_length_over_part_bbox_diagonal": _stable_ratio(length, bbox_diagonal),
            "edge_length_over_part_scale": _stable_ratio(length, scale),
            "circle_radius_over_part_bbox_diagonal": _stable_ratio(radius, bbox_diagonal),
            "circle_radius_over_part_scale": _stable_ratio(radius, scale),
            "dihedral_angle_over_pi": _stable_ratio(edge_attr.get("dihedral_angle", 0.0), math.pi),
            "material_side_inner_evidence": inner_evidence,
            "material_side_outer_evidence": outer_evidence,
            "material_side_ambiguous_evidence": ambiguous_evidence,
        }
        recognition["log1p_edge_length_over_part_scale"] = _stable_log1p(recognition["edge_length_over_part_scale"])
        recognition["log1p_circle_radius_over_part_scale"] = _stable_log1p(recognition["circle_radius_over_part_scale"])
        edge_attr["raw_physical_v2"] = {"length": length, "circle_radius": radius}
        edge_attr["material_side_evidence"] = (
            "inner" if inner_evidence > outer_evidence and inner_evidence > 0.0
            else "outer" if outer_evidence > inner_evidence and outer_evidence > 0.0
            else "ambiguous"
        )
        edge_attr["recognition_features_v2"] = recognition


def build_canonical_patches(
    graph: dict[str, Any],
    *,
    respect_labels: bool = True,
    angular_tolerance_degrees: float = 2.0,
    relative_distance_tolerance: float = 1e-6,
    relative_radius_tolerance: float = 1e-6,
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    parent = list(range(len(nodes)))
    scale = max(_finite(graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_characteristic_length", 1.0)), EPSILON)
    angular_tolerance = math.radians(float(angular_tolerance_degrees))
    for (source, target), edge_attr in _unique_undirected_edges(graph):
        if _patch_merge_allowed(
            nodes[source],
            nodes[target],
            edge_attr,
            scale=scale,
            respect_labels=respect_labels,
            angular_tolerance=angular_tolerance,
            relative_distance_tolerance=relative_distance_tolerance,
            relative_radius_tolerance=relative_radius_tolerance,
        ):
            _union(parent, source, target)
    groups: dict[int, list[int]] = defaultdict(list)
    for node_index in range(len(nodes)):
        groups[_find(parent, node_index)].append(node_index)
    ordered_groups = sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])
    face_to_patch: dict[str, int] = {}
    patches: list[dict[str, Any]] = []
    total_area = max(_finite(graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_total_area", 0.0)), EPSILON)
    for patch_id, face_indices in enumerate(ordered_groups):
        patch_nodes = [nodes[index] for index in face_indices]
        area = sum(_finite(node["attributes"].get("physical", {}).get("area", 0.0)) for node in patch_nodes)
        primitive_ids = sorted({int(node.get("primitive_surface_id", 0)) for node in patch_nodes})
        instance_ids = sorted({int(node.get("instance_id", 0)) for node in patch_nodes})
        macro_features = sorted({str(node.get("macro_feature", "stock")) for node in patch_nodes})
        patch = {
            "patch_id": patch_id,
            "face_indices": face_indices,
            "face_count": len(face_indices),
            "primitive_surface_ids": primitive_ids,
            "primitive_surface": str(patch_nodes[0].get("primitive_surface", "unknown_surface")),
            "total_area": area,
            "area_over_part_total_area": _stable_ratio(area, total_area),
            "instance_ids": instance_ids,
            "macro_features": macro_features,
            "surface_signature": _patch_surface_signature(patch_nodes, scale),
        }
        patches.append(patch)
        for face_index in face_indices:
            face_to_patch[str(face_index)] = patch_id
    patch_edges = _aggregate_patch_edges(graph, face_to_patch)
    return {
        "canonical_patches": patches,
        "face_to_canonical_patch": face_to_patch,
        "canonical_patch_edges": patch_edges,
        "canonical_patch_protocol": {
            "non_destructive": True,
            "original_face_ids_preserved": True,
            "merge_only_adjacent_tangent_compatible_surfaces": True,
            "respect_instance_and_macro_labels": bool(respect_labels),
            "angular_tolerance_degrees": float(angular_tolerance_degrees),
            "relative_distance_tolerance": float(relative_distance_tolerance),
            "relative_radius_tolerance": float(relative_radius_tolerance),
        },
    }


def _unique_undirected_edges(graph: dict[str, Any]) -> list[tuple[tuple[int, int], dict[str, Any]]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for edge, attr in zip(graph.get("edges", []), graph.get("edge_attributes", [])):
        if len(edge) != 2:
            continue
        pair = tuple(sorted((int(edge[0]), int(edge[1]))))
        result.setdefault(pair, attr)
    return sorted(result.items())


def _patch_merge_allowed(
    first: dict[str, Any],
    second: dict[str, Any],
    edge_attr: dict[str, Any],
    *,
    scale: float,
    respect_labels: bool,
    angular_tolerance: float,
    relative_distance_tolerance: float,
    relative_radius_tolerance: float,
) -> bool:
    if int(first.get("primitive_surface_id", 0)) != int(second.get("primitive_surface_id", 0)):
        return False
    angle = abs(_finite(edge_attr.get("dihedral_angle", math.pi)))
    convexity = str(edge_attr.get("convexity", "unknown"))
    if convexity != "smooth" and angle > angular_tolerance:
        return False
    if respect_labels:
        if int(first.get("instance_id", 0)) != int(second.get("instance_id", 0)):
            return False
        if str(first.get("macro_feature", "stock")) != str(second.get("macro_feature", "stock")):
            return False
    first_geometry = first.get("attributes", {}).get("geometry_v2", {})
    second_geometry = second.get("attributes", {}).get("geometry_v2", {})
    family = first_geometry.get("surface_family")
    if family != second_geometry.get("surface_family"):
        return False
    if family == "plane":
        first_normal = _unit(first_geometry.get("normal", [0.0, 0.0, 0.0]))
        second_normal = _unit(second_geometry.get("normal", [0.0, 0.0, 0.0]))
        if _parallel_angle(first_normal, second_normal) > angular_tolerance:
            return False
        delta = _subtract(second_geometry.get("location", [0.0, 0.0, 0.0]), first_geometry.get("location", [0.0, 0.0, 0.0]))
        return abs(_dot(delta, first_normal)) / scale <= relative_distance_tolerance
    if family == "cylinder":
        first_axis = _unit(first_geometry.get("axis_direction", [0.0, 0.0, 0.0]))
        second_axis = _unit(second_geometry.get("axis_direction", [0.0, 0.0, 0.0]))
        if _parallel_angle(first_axis, second_axis) > angular_tolerance:
            return False
        first_radius = _finite(first_geometry.get("radius", 0.0))
        second_radius = _finite(second_geometry.get("radius", 0.0))
        if abs(first_radius - second_radius) / scale > relative_radius_tolerance:
            return False
        delta = _subtract(second_geometry.get("axis_location", [0.0, 0.0, 0.0]), first_geometry.get("axis_location", [0.0, 0.0, 0.0]))
        transverse = _subtract(delta, _multiply(first_axis, _dot(delta, first_axis)))
        return _norm(transverse) / scale <= relative_distance_tolerance
    return False


def _patch_surface_signature(nodes: list[dict[str, Any]], scale: float) -> dict[str, Any]:
    first = nodes[0]
    geometry = first.get("attributes", {}).get("geometry_v2", {})
    signature = {"surface_family": geometry.get("surface_family", "other")}
    if geometry.get("surface_family") == "cylinder":
        signature["radius_over_part_scale"] = _stable_ratio(geometry.get("radius", 0.0), scale)
    return signature


def _aggregate_patch_edges(graph: dict[str, Any], face_to_patch: dict[str, int]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for (source, target), attr in _unique_undirected_edges(graph):
        source_patch = int(face_to_patch[str(source)])
        target_patch = int(face_to_patch[str(target)])
        if source_patch == target_patch:
            continue
        key = (
            min(source_patch, target_patch),
            max(source_patch, target_patch),
            str(attr.get("curve_type", "unknown")),
            str(attr.get("convexity", "unknown")),
        )
        record = groups.setdefault(key, {
            "source_patch": key[0],
            "target_patch": key[1],
            "curve_type": key[2],
            "convexity": key[3],
            "total_length_over_part_scale": 0.0,
            "brep_edge_count": 0,
        })
        record["total_length_over_part_scale"] += _finite(
            attr.get("recognition_features_v2", {}).get("edge_length_over_part_scale", 0.0)
        )
        record["brep_edge_count"] += 1
    return [groups[key] for key in sorted(groups)]


def canonical_graph_signature(graph: dict[str, Any], ndigits: int = 6) -> dict[str, Any]:
    patches = graph.get("canonical_patches") or []
    patch_tokens = []
    for patch in patches:
        radius = _finite(patch.get("surface_signature", {}).get("radius_over_part_scale", 0.0))
        patch_tokens.append((
            tuple(patch.get("primitive_surface_ids", [])),
            round(_finite(patch.get("area_over_part_total_area", 0.0)), ndigits),
            round(radius, ndigits),
        ))
    edge_tokens = []
    patch_lookup = {int(patch["patch_id"]): patch for patch in patches}
    for edge in graph.get("canonical_patch_edges", []):
        source_patch = patch_lookup[int(edge["source_patch"])]
        target_patch = patch_lookup[int(edge["target_patch"])]
        endpoint_tokens = sorted((
            (tuple(source_patch["primitive_surface_ids"]), round(_finite(source_patch["area_over_part_total_area"]), ndigits)),
            (tuple(target_patch["primitive_surface_ids"]), round(_finite(target_patch["area_over_part_total_area"]), ndigits)),
        ))
        edge_tokens.append((
            tuple(endpoint_tokens),
            str(edge.get("curve_type", "unknown")),
            str(edge.get("convexity", "unknown")),
            round(_finite(edge.get("total_length_over_part_scale", 0.0)), ndigits),
        ))
    return {
        "patch_count": len(patches),
        "patch_tokens": sorted(patch_tokens),
        "edge_tokens": sorted(edge_tokens),
    }


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], first: int, second: int) -> None:
    first_root, second_root = _find(parent, first), _find(parent, second)
    if first_root != second_root:
        parent[max(first_root, second_root)] = min(first_root, second_root)


def _parallel_angle(first: list[float], second: list[float]) -> float:
    cosine = min(1.0, max(-1.0, abs(_dot(first, second))))
    return math.acos(cosine)


def _unit(vector: Iterable[float]) -> list[float]:
    values = [_finite(value) for value in vector]
    length = _norm(values)
    return [value / length for value in values] if length > EPSILON else [0.0, 0.0, 0.0]


def _subtract(first: Iterable[float], second: Iterable[float]) -> list[float]:
    return [_finite(a) - _finite(b) for a, b in zip(first, second)]


def _multiply(vector: Iterable[float], scalar: float) -> list[float]:
    return [_finite(value) * _finite(scalar) for value in vector]


def _dot(first: Iterable[float], second: Iterable[float]) -> float:
    return sum(_finite(a) * _finite(b) for a, b in zip(first, second))


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _stable_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _finite(numerator)
    denominator_value = abs(_finite(denominator))
    if denominator_value <= EPSILON:
        return 0.0
    return max(-MAX_STABLE_RATIO, min(MAX_STABLE_RATIO, numerator_value / denominator_value))


def _stable_nonnegative(value: Any) -> float:
    return max(0.0, min(MAX_STABLE_RATIO, _finite(value)))


def _stable_log1p(value: Any) -> float:
    return math.log1p(max(0.0, min(MAX_STABLE_RATIO, _finite(value))))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _assert_graph_v2_finite(graph: dict[str, Any]) -> None:
    for node in graph.get("nodes", []):
        for value in node.get("attributes", {}).get("recognition_features_v2", {}).values():
            if not math.isfinite(float(value)):
                raise ValueError("Graph v2 node recognition feature contains NaN/Inf")
            if abs(float(value)) > MAX_STABLE_RATIO:
                raise ValueError("Graph v2 node recognition feature exceeds stable cap")
    for edge in graph.get("edge_attributes", []):
        for value in edge.get("recognition_features_v2", {}).values():
            if not math.isfinite(float(value)):
                raise ValueError("Graph v2 edge recognition feature contains NaN/Inf")
            if abs(float(value)) > MAX_STABLE_RATIO:
                raise ValueError("Graph v2 edge recognition feature exceeds stable cap")


__all__ = [
    "GRAPH_V2_SCHEMA",
    "MAX_STABLE_RATIO",
    "build_canonical_patches",
    "canonical_graph_signature",
    "extract_graph_v2",
]
