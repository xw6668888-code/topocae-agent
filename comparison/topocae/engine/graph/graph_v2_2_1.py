from __future__ import annotations

import math
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Extend.TopologyUtils import TopologyExplorer

import core.graph_v2 as base
from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME, oriented_cylindrical_radial_alignment


GRAPH_V2_SCHEMA = "topocae_graph_v2_2_1_true_brep_planar_rank_policy_1"
DESCRIPTOR_REVISION = "surface_central_moment_rms_exact_planar_rank_2_v1"
EPSILON = 1.0e-12
EIGENVALUE_RELATIVE_ZERO_TOLERANCE = 1.0e-12
EXACT_PLANAR_DEGENERATE_REASON = "exact_planar_surface_rank_2"


def extract_graph_v2_2_1(
    shape: Any,
    *,
    face_labels: list[dict[str, Any]] | None = None,
    primitive_surface_index: dict[str, Any] | None = None,
    build_patches: bool = True,
) -> dict[str, Any]:
    graph = base.extract_graph_v2(
        shape,
        face_labels=face_labels,
        primitive_surface_index=primitive_surface_index,
        build_patches=False,
    )
    faces = list(TopologyExplorer(shape).faces())
    part_metadata = _part_metadata(shape)
    base._annotate_nodes_v2(graph, faces, part_metadata)
    for node, face in zip(graph["nodes"], faces):
        extents, audit = surface_moment_rms_extents(face)
        attrs = node["attributes"]
        recognition = attrs["recognition_features_v2"]
        scale = max(float(part_metadata["normalization"]["part_characteristic_length"]), EPSILON)

        recognition["pca_long_over_part_scale"] = base._stable_ratio(extents[0], scale)
        recognition["pca_middle_over_part_scale"] = base._stable_ratio(extents[1], scale)
        recognition["local_extent_long_middle_ratio"] = base._stable_ratio(extents[0], extents[1])
        if audit.get("degenerate_reason") == EXACT_PLANAR_DEGENERATE_REASON:
            # Exact B-Rep planes are rank two by geometry, before any middle/short ratio is formed.
            recognition["pca_short_over_part_scale"] = 0.0
            recognition["local_extent_middle_short_ratio"] = 0.0
        else:
            recognition["pca_short_over_part_scale"] = base._stable_ratio(extents[2], scale)
            recognition["local_extent_middle_short_ratio"] = base._stable_ratio(extents[1], extents[2])
        for key in (
            "pca_long_over_part_scale",
            "pca_middle_over_part_scale",
            "pca_short_over_part_scale",
            "local_extent_long_middle_ratio",
            "local_extent_middle_short_ratio",
        ):
            recognition[f"log1p_{key}"] = base._stable_log1p(recognition[key])

        polarity = float(oriented_cylindrical_radial_alignment(face))
        if not math.isfinite(polarity) or not -1.0 <= polarity <= 1.0:
            raise RuntimeError("invalid true B-Rep cylindrical polarity")
        recognition[FEATURE_NAME] = polarity
        attrs["raw_physical_v2"]["pca_extents"] = extents
        attrs["surface_moment_descriptor_audit"] = audit

    base._annotate_edges_v2(graph, part_metadata)
    graph["schema_version"] = GRAPH_V2_SCHEMA
    graph["part_metadata_v2"] = part_metadata
    graph["graph_v2_protocol"] = {
        "revision": DESCRIPTOR_REVISION,
        "face_identity": "TopoDS_Shape.IsSame",
        "recognition_scale": "surface_central_moment_rms_diameter",
        "local_pca_replacement": "surface_central_inertia_covariance_eigenvalues",
        "exact_planar_rank_policy": EXACT_PLANAR_DEGENERATE_REASON,
        "exact_planar_policy_applied_before_middle_short_ratio": True,
        "non_planar_small_third_scale_forced_planar": False,
        "global_coordinate_deduplication_used": False,
        "sampling_traversal_truncation_used": False,
        "labels_used_by_descriptor": False,
        "node_or_edge_order_used_by_descriptor": False,
        "oriented_cylindrical_radial_alignment": "dot(oriented_face_normal,local_cylinder_radial)",
        "canonical_patches_non_destructive": bool(build_patches),
        "stable_ratio_cap": base.MAX_STABLE_RATIO,
    }
    if build_patches:
        graph.update(base.build_canonical_patches(graph, respect_labels=face_labels is not None))
    base._assert_graph_v2_finite(graph)
    return graph


def surface_moment_rms_extents(shape: Any) -> tuple[list[float], dict[str, Any]]:
    is_exact_plane = _is_exact_planar_face(shape)
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    mass = float(props.Mass())
    if not math.isfinite(mass) or mass <= EPSILON:
        return [0.0, 0.0, 0.0], {
            "revision": DESCRIPTOR_REVISION,
            "mass": 0.0 if not math.isfinite(mass) else mass,
            "surface_geometry_type": "plane" if is_exact_plane else "unknown_or_non_planar",
            "degenerate": True,
            "degenerate_reason": EXACT_PLANAR_DEGENERATE_REASON if is_exact_plane else "zero_surface_mass",
            "degenerate_policy": "exact_planar_rank_2" if is_exact_plane else "zero_extents",
            "finite": True,
        }
    inertia = props.MatrixOfInertia()
    matrix = [[float(inertia.Value(row + 1, column + 1)) for column in range(3)] for row in range(3)]
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise RuntimeError("non-finite OCC surface central inertia matrix")
    trace_half = 0.5 * sum(matrix[index][index] for index in range(3))
    covariance = [
        [((trace_half if row == column else 0.0) - matrix[row][column]) / mass for column in range(3)]
        for row in range(3)
    ]
    eigenvalues = sorted(base._jacobi_eigenvalues_symmetric_3x3(covariance), reverse=True)
    scale = max(1.0, max(abs(value) for value in eigenvalues))
    if any(value < -1.0e-10 * scale for value in eigenvalues):
        raise RuntimeError(f"surface covariance has materially negative eigenvalue: {eigenvalues}")
    zero_threshold = EIGENVALUE_RELATIVE_ZERO_TOLERANCE * scale
    clipped = [0.0 if abs(value) <= zero_threshold else max(0.0, value) for value in eigenvalues]
    if is_exact_plane:
        clipped[-1] = 0.0
    extents = [2.0 * math.sqrt(value) for value in clipped]
    if is_exact_plane:
        extents[-1] = 0.0
    if not all(math.isfinite(value) for value in extents):
        raise RuntimeError("non-finite surface moment RMS extent")
    return extents, {
        "revision": DESCRIPTOR_REVISION,
        "mass": mass,
        "surface_geometry_type": "plane" if is_exact_plane else "non_planar_or_non_face",
        "degenerate": bool(is_exact_plane or sum(value > 0.0 for value in clipped) < 3),
        "degenerate_reason": EXACT_PLANAR_DEGENERATE_REASON if is_exact_plane else None,
        "degenerate_policy": "exact_planar_rank_2" if is_exact_plane else None,
        "eigenvalue_relative_zero_tolerance": EIGENVALUE_RELATIVE_ZERO_TOLERANCE,
        "eigenvalue_zero_threshold": zero_threshold,
        "effective_rank": sum(value > 0.0 for value in clipped),
        "rank_deficient": sum(value > 0.0 for value in clipped) < 3,
        "covariance_eigenvalues": clipped,
        "finite": True,
    }


def _is_exact_planar_face(shape: Any) -> bool:
    try:
        return BRepAdaptor_Surface(shape, True).GetType() == GeomAbs_Plane
    except (RuntimeError, TypeError):
        return False


def _part_metadata(shape: Any) -> dict[str, Any]:
    bbox = base._shape_bbox(shape)
    extents, audit = surface_moment_rms_extents(shape)
    characteristic_length = math.sqrt(sum(value * value for value in extents))
    if characteristic_length <= EPSILON:
        characteristic_length = 1.0
    surface_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, surface_props)
    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, volume_props)
    return {
        "raw_physical": {
            "total_surface_area": base._finite(surface_props.Mass()),
            "solid_volume": base._finite(volume_props.Mass()),
            "axis_aligned_bbox_min": bbox["min"],
            "axis_aligned_bbox_max": bbox["max"],
            "axis_aligned_bbox_extents": bbox["extents"],
            "axis_aligned_bbox_diagonal": bbox["diagonal"],
            "surface_moment_rms_extents": extents,
        },
        "normalization": {
            "part_total_area": base._finite(surface_props.Mass()),
            "part_bbox_diagonal": base._finite(bbox["diagonal"]),
            "part_characteristic_length": characteristic_length,
            "part_characteristic_area": characteristic_length * characteristic_length,
            "scale_definition": "l2_norm_of_surface_central_moment_rms_extents",
        },
        "surface_moment_descriptor_audit": audit,
    }


__all__ = [
    "DESCRIPTOR_REVISION",
    "EXACT_PLANAR_DEGENERATE_REASON",
    "GRAPH_V2_SCHEMA",
    "extract_graph_v2_2_1",
    "surface_moment_rms_extents",
]
