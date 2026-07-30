from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import random
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import extract_graph_v2
from p1_c import all_features as audited_core
from p1_c import formal_generator_v4 as transition_revision  # activates audited transition revision
from p1_c.geometry import clean_fuse, cleanup_audit, face_area, face_center, shape_volume, topology_counts
from p1_c.pilot_h_c2 import exact_labels
from utils.ontology import load_layer_index


CANONICAL_FEATURES = (
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_v",
    "slot_o",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rectangular_boss",
    "rib",
    "fillet",
    "chamfer",
)

FEATURE_FAMILY = {
    **{name: "hole" for name in CANONICAL_FEATURES[:6]},
    **{name: "slot" for name in CANONICAL_FEATURES[6:10]},
    "pocket_rect": "pocket",
    "pocket_circular": "pocket",
    "cylinder_boss": "protrusion",
    "rectangular_boss": "protrusion",
    "rib": "stiffener",
    "fillet": "transition",
    "chamfer": "transition",
}

COMPARISON_METRIC = {
    "through_hole_round": "diameter",
    "blind_hole_round": "diameter",
    "stepped_hole_round": "diameter",
    "counterbore_hole": "diameter",
    "countersink_hole": "diameter",
    "obround_hole": "length",
    "slot_rect": "length",
    "slot_u": "length",
    "slot_v": "length",
    "slot_o": "length",
    "pocket_rect": "length",
    "pocket_circular": "diameter",
    "cylinder_boss": "diameter",
    "rectangular_boss": "length",
    "rib": "area",
    "fillet": "area",
    "chamfer": "area",
}

COMPARISON_SEMANTICS = {
    "through_hole_round": "cylindrical_hole_diameter",
    "blind_hole_round": "cylindrical_hole_diameter",
    "stepped_hole_round": "largest_step_diameter",
    "counterbore_hole": "small_through_hole_diameter",
    "countersink_hole": "small_through_hole_diameter",
    "obround_hole": "overall_length",
    "slot_rect": "finished_opening_width",
    "slot_u": "finished_opening_width",
    "slot_v": "finished_opening_width",
    "slot_o": "finished_opening_width",
    "pocket_rect": "planform_length",
    "pocket_circular": "planform_diameter",
    "cylinder_boss": "planform_diameter",
    "rectangular_boss": "planform_length",
    "rib": "rib_thickness",
    "fillet": "fillet_radius",
    "chamfer": "chamfer_leg_length",
}

COMPLETE_DIMENSION_KEYS = {
    "through_hole_round": ("diameter",),
    "blind_hole_round": ("diameter", "depth"),
    "stepped_hole_round": ("small_diameter", "large_diameter", "step_depth"),
    "counterbore_hole": ("small_diameter", "large_diameter", "step_depth"),
    "countersink_hole": ("small_diameter", "large_diameter", "sink_depth"),
    "obround_hole": ("diameter", "center_distance", "overall_length"),
    "slot_rect": ("sweep_path_length", "opening_width", "profile_depth"),
    "slot_u": ("sweep_path_length", "opening_width", "profile_depth"),
    "slot_v": ("sweep_path_length", "opening_width", "profile_depth"),
    "slot_o": ("sweep_path_length", "opening_width", "profile_depth"),
    "pocket_rect": ("length", "width", "depth"),
    "pocket_circular": ("diameter", "depth"),
    "cylinder_boss": ("diameter", "height"),
    "rectangular_boss": ("length", "width", "height"),
    "rib": ("length", "thickness", "height"),
    "fillet": ("radius",),
    "chamfer": ("chamfer_leg_length",),
}


def build_multi_instance_result(
    feature_type: str,
    instance_count: int,
    seed: int,
    *,
    tie: bool = False,
) -> audited_core.CleanFeatureResult:
    if feature_type not in CANONICAL_FEATURES:
        raise KeyError(feature_type)
    if not 2 <= int(instance_count) <= 4:
        raise ValueError("instance_count must be between 2 and 4")
    if feature_type == "rib":
        return _build_supported_multi_rib(int(instance_count), int(seed), tie=tie)
    return _build_repeated_general_feature(feature_type, int(instance_count), int(seed), tie=tie)


def _build_repeated_general_feature(
    feature_type: str,
    instance_count: int,
    seed: int,
    *,
    tie: bool,
) -> audited_core.CleanFeatureResult:
    host_rng = random.Random(seed ^ 0x51A17)
    length = host_rng.uniform(128.0, 142.0)
    width = host_rng.uniform(96.0, 108.0)
    height = host_rng.uniform(28.0, 36.0)
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, height).Shape()
    host = {
        "family": "p3_a1r_large_plate",
        "construction_method": "audited_single_box_host",
        "length": length,
        "width": width,
        "height": height,
        "top_z": height,
        "origin": [-length / 2.0, -width / 2.0, 0.0],
    }
    host_cleanup = cleanup_audit(shape)
    if not host_cleanup["passed"]:
        raise ValueError("multi-instance host is not cleanup-idempotent")
    centers = _placement_centers(instance_count, length, width, symmetric=tie)
    tags: list[audited_core.FaceTag] = []
    records: list[dict[str, Any]] = []
    shared_feature_seed = seed ^ 0x71E if tie else None
    for offset, center in enumerate(centers, start=1):
        feature_seed = shared_feature_seed if shared_feature_seed is not None else seed + 1009 * offset
        feature_rng = random.Random(feature_seed)
        shape, tags, record = audited_core._apply_feature(
            shape,
            tags,
            macro=feature_type,
            instance_id=offset,
            center=center,
            host=host,
            rng=feature_rng,
        )
        records.append(record)
    cleanup = cleanup_audit(shape)
    parameters = {
        "schema_version": "topocae_p3_a1r_multi_instance_geometry/1",
        "generator_revision": "p3_a1r_repeated_audited_feature_kernel_r1",
        "audited_kernel": {
            "feature_boolean_and_labeling": audited_core.GENERATOR_REVISION,
            "transition_revision": transition_revision.GENERATOR_REVISION,
        },
        "seed": seed,
        "primary_target": feature_type,
        "secondary_features": [feature_type] * (instance_count - 1),
        "context": "same_canonical_multi_instance",
        "host_family": host["family"],
        "host_metadata": host,
        "host_cleanup_before_features": host_cleanup,
        "features": records,
        "instance_ids": list(range(1, instance_count + 1)),
        "tie_design": bool(tie),
        "construction_contract": {
            "cleanup_after_each_boolean": True,
            "final_labeling_after_cleanup": True,
            "feature_intersection_allowed": False,
            "same_canonical_feature_only": True,
        },
    }
    result = audited_core.CleanFeatureResult(
        shape,
        tuple(audited_core._filter_final_tags(shape, tags)),
        parameters,
        cleanup,
    )
    _validate_multi_result(result, feature_type, instance_count)
    return result


def _placement_centers(count: int, length: float, width: float, *, symmetric: bool) -> list[tuple[float, float]]:
    if symmetric:
        if count == 2:
            return [(-0.24 * length, 0.0), (0.24 * length, 0.0)]
        if count == 3:
            radius_x, radius_y = 0.22 * length, 0.22 * width
            return [
                (radius_x, 0.0),
                (-0.5 * radius_x, math.sqrt(3.0) * 0.5 * radius_y),
                (-0.5 * radius_x, -math.sqrt(3.0) * 0.5 * radius_y),
            ]
        return [
            (-0.21 * length, -0.18 * width),
            (-0.21 * length, 0.18 * width),
            (0.21 * length, -0.18 * width),
            (0.21 * length, 0.18 * width),
        ]
    layouts = {
        2: [(-0.26, -0.12), (0.18, 0.17)],
        3: [(-0.28, -0.18), (0.22, -0.10), (-0.05, 0.25)],
        4: [(-0.29, -0.20), (0.19, -0.23), (-0.17, 0.23), (0.27, 0.14)],
    }
    return [(x * length, y * width) for x, y in layouts[count]]


def _build_supported_multi_rib(instance_count: int, seed: int, *, tie: bool) -> audited_core.CleanFeatureResult:
    rng = random.Random(seed ^ 0xA18B)
    length = rng.uniform(126.0, 140.0)
    width = rng.uniform(82.0, 94.0)
    base_thickness = rng.uniform(5.5, 7.0)
    wall_thickness = rng.uniform(5.0, 6.5)
    wall_height = rng.uniform(31.0, 37.0)

    base = BRepPrimAPI_MakeBox(
        gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, width, base_thickness
    ).Shape()
    left_wall = BRepPrimAPI_MakeBox(
        gp_Pnt(-length / 2.0, -width / 2.0, 0.0), length, wall_thickness, wall_height
    ).Shape()
    right_wall = BRepPrimAPI_MakeBox(
        gp_Pnt(-length / 2.0, width / 2.0 - wall_thickness, 0.0), length, wall_thickness, wall_height
    ).Shape()
    _, host = clean_fuse(base, left_wall)
    _, host = clean_fuse(host, right_wall)
    host_cleanup = cleanup_audit(host)
    if not host_cleanup["passed"]:
        raise ValueError("multi-rib U-support host is not cleanup-idempotent")

    x_ratios = {
        2: (-0.24, 0.24),
        3: (-0.29, 0.0, 0.27),
        4: (-0.33, -0.11, 0.13, 0.34),
    }[instance_count]
    shape = host
    tags: list[audited_core.FaceTag] = []
    records = []
    shared_dimensions = (rng.uniform(2.2, 2.8), rng.uniform(18.0, 23.0)) if tie else None
    for instance_id, x_ratio in enumerate(x_ratios, start=1):
        if shared_dimensions:
            thickness, rib_height = shared_dimensions
        else:
            local = random.Random(seed + 2039 * instance_id)
            thickness = local.uniform(2.0, 3.5)
            rib_height = local.uniform(16.0, 26.0)
        x_center = x_ratio * length
        rib_tool = BRepPrimAPI_MakeBox(
            gp_Pnt(
                x_center - thickness / 2.0,
                -width / 2.0,
                base_thickness - 0.35,
            ),
            thickness,
            width,
            rib_height + 0.35,
        ).Shape()
        operation, new_shape = clean_fuse(shape, rib_tool)
        propagated = audited_core._propagate_tags(operation, tags)
        top_z = base_thickness + rib_height

        def rib_role(face: Any, expected_top: float = top_z) -> str:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Plane:
                normal = surface.Plane().Axis().Direction()
                if abs(normal.Z()) > 0.9 and face_center(face)[2] > expected_top - 0.2:
                    return "rib_top_plane"
            return "rib_side_plane"

        created = audited_core._tags_from_tool(
            operation,
            rib_tool,
            role=rib_role,
            macro_feature="rib",
            instance_id=instance_id,
        )
        tags = audited_core._merge_tags(propagated, created)
        shape = new_shape
        operation_cleanup = cleanup_audit(shape)
        if not operation_cleanup["passed"]:
            raise ValueError(f"rib {instance_id} result is not cleanup-idempotent")
        records.append({
            "macro_feature": "rib",
            "instance_id": instance_id,
            "operation": "supported_thin_wall_fuse",
            "center": [x_center, 0.0],
            "parameters": {
                "length": width - 2.0 * wall_thickness,
                "thickness": thickness,
                "height": rib_height,
                "positive_support_contact_count": 3,
                "support_regions": ["base_plate", "left_wall", "right_wall"],
                "span_thickness_ratio": (width - 2.0 * wall_thickness) / thickness,
                "height_thickness_ratio": rib_height / thickness,
            },
            "feature_face_count": sum(tag.instance_id == instance_id for tag in tags),
            "cleanup_after_operation": operation_cleanup,
        })
    cleanup = cleanup_audit(shape)
    parameters = {
        "schema_version": "topocae_p3_a1r_multi_instance_geometry/1",
        "generator_revision": "p3_a1r_supported_multi_rib_wrapper_r1",
        "audited_kernel": {
            "boolean_and_label_propagation": audited_core.GENERATOR_REVISION,
            "engineering_contract_source": "P1-B supported rib/web",
        },
        "seed": seed,
        "primary_target": "rib",
        "secondary_features": ["rib"] * (instance_count - 1),
        "context": "same_canonical_multi_instance",
        "host_family": "u_channel_three_support",
        "host_metadata": {
            "length": length,
            "width": width,
            "base_thickness": base_thickness,
            "wall_thickness": wall_thickness,
            "wall_height": wall_height,
        },
        "host_cleanup_before_features": host_cleanup,
        "features": records,
        "instance_ids": list(range(1, instance_count + 1)),
        "tie_design": bool(tie),
        "construction_contract": {
            "same_canonical_feature_only": True,
            "each_rib_contacts_three_support_regions": True,
            "feature_intersection_allowed": False,
        },
    }
    result = audited_core.CleanFeatureResult(
        shape,
        tuple(audited_core._filter_final_tags(shape, tags)),
        parameters,
        cleanup,
    )
    _validate_multi_result(result, "rib", instance_count)
    return result


def _validate_multi_result(result: audited_core.CleanFeatureResult, feature_type: str, instance_count: int) -> None:
    counts = topology_counts(result.shape)
    if not result.cleanup["passed"]:
        raise ValueError("final multi-instance result is not cleanup-idempotent")
    if counts["solid_count"] != 1 or not BRepCheck_Analyzer(result.shape).IsValid() or shape_volume(result.shape) <= 0.0:
        raise ValueError(f"invalid multi-instance solid: {counts}")
    final_faces = list(TopologyExplorer(result.shape).faces())
    for tag in result.face_tags:
        if not any(tag.face.IsSame(face) for face in final_faces):
            raise ValueError("multi-instance truth references a non-final face")
    owner_counts = Counter((tag.instance_id, tag.macro_feature) for tag in result.face_tags)
    expected = set(range(1, instance_count + 1))
    actual = {tag.instance_id for tag in result.face_tags}
    if actual != expected:
        raise ValueError(f"feature instances incomplete: {actual} != {expected}")
    if any(owner_counts[(instance_id, feature_type)] <= 0 for instance_id in expected):
        raise ValueError("one or more feature instances has no final labeled face")


def build_graph_with_metric_evidence(
    result: audited_core.CleanFeatureResult,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = exact_labels(result, repo_root)
    primitive_index = load_layer_index(repo_root / "CAE_feature_ontology.yaml", "primitive_surface")
    graph = extract_graph_v2(result.shape, face_labels=labels, primitive_surface_index=primitive_index)
    records = {int(item["instance_id"]): item for item in result.parameters["features"]}
    for feature_set in graph.get("feature_sets", []):
        instance_id = int(feature_set["instance_id"])
        record = records[instance_id]
        parameters = _normalize_parameters(feature_set["macro_feature"], record.get("parameters", {}))
        indices = [int(value) for value in feature_set.get("node_indices", [])]
        areas = [float(graph["nodes"][index]["attributes"]["physical"]["area"]) for index in indices]
        centers = [graph["nodes"][index]["attributes"]["physical"]["center_of_mass"] for index in indices]
        total_area = sum(areas)
        centroid = [
            sum(float(center[axis]) * area for center, area in zip(centers, areas, strict=True)) / total_area
            for axis in range(3)
        ]
        parameters.update({"area": total_area, "centroid": centroid})
        feature_set["geometry_parameters"] = parameters
        feature_type = str(feature_set["macro_feature"])
        feature_set["descriptor_contract"] = {
            "comparison_metric": COMPARISON_METRIC[feature_type],
            "comparison_semantics": COMPARISON_SEMANTICS[feature_type],
            "scope": "same_type_instance_ordering_only",
            "is_complete_feature_description": False,
            "complete_dimension_keys": list(COMPLETE_DIMENSION_KEYS[feature_type]),
        }
        feature_set["semantic_roles"] = sorted({
            str(graph["nodes"][index].get("semantic_role", "")) for index in indices
        })
        feature_set["metric_evidence"] = {
            "area": total_area,
            "centroid": centroid,
            "source_node_indices": indices,
        }
    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(result.shape, volume_props)
    center = volume_props.CentreOfMass()
    graph["part_reference"] = {
        "kind": "solid_centroid",
        "centroid": [float(center.X()), float(center.Y()), float(center.Z())],
        "solid_volume": float(volume_props.Mass()),
        "axis_aligned_bbox_min": deepcopy(graph["part_metadata_v2"]["raw_physical"]["axis_aligned_bbox_min"]),
        "axis_aligned_bbox_max": deepcopy(graph["part_metadata_v2"]["raw_physical"]["axis_aligned_bbox_max"]),
    }
    graph["generation"] = {
        "stage": "P3-A1R multi-instance pilot",
        "parameters": deepcopy(result.parameters),
        "cleanup_audit": deepcopy(result.cleanup),
    }
    return graph, labels


def _normalize_parameters(feature_type: str, raw: dict[str, Any]) -> dict[str, Any]:
    result = {key: deepcopy(value) for key, value in raw.items() if isinstance(value, (str, int, float, bool, list))}
    radius = raw.get("radius")
    if radius is None and raw.get("large_radius") is not None:
        radius = raw["large_radius"]
        result["small_radius"] = float(raw.get("small_radius", 0.0))
    if radius is not None:
        result["radius"] = float(radius)
        result["diameter"] = 2.0 * float(radius)
    if raw.get("small_radius") is not None:
        result["small_diameter"] = 2.0 * float(raw["small_radius"])
    if raw.get("large_radius") is not None:
        result["large_diameter"] = 2.0 * float(raw["large_radius"])
    if feature_type in {"counterbore_hole", "countersink_hole"}:
        result["diameter"] = float(result["small_diameter"])
    if feature_type == "obround_hole":
        radius_value = float(raw["radius"])
        result["overall_length"] = float(raw["center_distance"]) + 2.0 * radius_value
        result["length"] = result["overall_length"]
        result["width"] = 2.0 * radius_value
    if feature_type == "chamfer" and raw.get("distance") is not None:
        result["chamfer_leg_length"] = float(raw["distance"])
        result["length"] = result["chamfer_leg_length"]
    if raw.get("size_x") is not None:
        result["length"] = float(raw["size_x"])
        result["width"] = float(raw["size_y"])
    for key in ("length", "width", "depth", "height", "thickness"):
        if raw.get(key) is not None:
            result[key] = float(raw[key])
    return result


def metric_table(graph: dict[str, Any], feature_type: str) -> list[dict[str, Any]]:
    metric = COMPARISON_METRIC[feature_type]
    reference = graph["part_reference"]["centroid"]
    rows = []
    for feature_set in sorted(graph["feature_sets"], key=lambda item: int(item["instance_id"])):
        params = feature_set["geometry_parameters"]
        centroid = params["centroid"]
        distance = math.dist([float(value) for value in centroid], [float(value) for value in reference])
        rows.append({
            "instance_id": int(feature_set["instance_id"]),
            "comparison_metric": metric,
            "comparison_semantics": COMPARISON_SEMANTICS[feature_type],
            "descriptor_scope": "same_type_instance_ordering_only",
            "is_complete_feature_description": False,
            "comparison_value": float(params[metric]),
            "nearest_reference": "part_centroid",
            "nearest_distance": distance,
            "centroid": [float(value) for value in centroid],
        })
    return rows
