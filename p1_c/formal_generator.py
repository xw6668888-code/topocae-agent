from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import random
from typing import Any, Iterable

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c import all_features as core
from p1_c import all_features_sweep_v2 as sweep_generator  # activates continuous slot kernel
from p1_c.formal_plan_v3 import PlannedSample
from p1_c.geometry import cleanup_audit, face_area, topology_counts
from p1_c.supported_wall_families import build_supported_wall_sample


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r1"
SUPPORTED_WALL_CLASSES = {"rib", "web"}


def build_planned_result(plan: PlannedSample) -> core.CleanFeatureResult:
    macros = (plan.primary_target, *plan.secondary_features)
    supported_walls = [macro for macro in macros if macro in SUPPORTED_WALL_CLASSES]
    if len(supported_walls) > 1:
        raise ValueError(f"one formal sample cannot contain two supported-wall classes: {macros}")
    if not supported_walls:
        result = sweep_generator.build_feature_sample(
            plan.primary_target,
            plan.seed,
            secondary_features=plan.secondary_features,
            host_family=plan.host_family,
        )
        parameters = deepcopy(result.parameters)
        parameters.update(
            {
                "generator_revision": GENERATOR_REVISION,
                "plan_revision": "p1_c_clean_context_latin_plan_r3",
                "formal_host_family": plan.host_family,
                "formal_host_family_kind": plan.host_family_kind,
            }
        )
        return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)
    return _build_supported_context(plan, supported_walls[0])


def transformed_shape_and_audit(
    result: core.CleanFeatureResult,
    *,
    scale: float,
    rotation_axis: Iterable[float],
    rotation_angle_radians: float,
) -> tuple[Any, dict[str, Any]]:
    source_shape = result.shape
    source_faces = list(TopologyExplorer(source_shape).faces())
    scaled = source_shape
    if abs(float(scale) - 1.0) > 1e-12:
        scale_transform = gp_Trsf()
        scale_transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(scale))
        scaled = BRepBuilderAPI_Transform(source_shape, scale_transform, True).Shape()
    axis_values = tuple(float(value) for value in rotation_axis)
    transformed = scaled
    if abs(float(rotation_angle_radians)) > 1e-12:
        direction = gp_Dir(*axis_values)
        rotation = gp_Trsf()
        rotation.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), direction), float(rotation_angle_radians))
        transformed = BRepBuilderAPI_Transform(scaled, rotation, True).Shape()
    transformed_faces = list(TopologyExplorer(transformed).faces())
    if len(source_faces) != len(transformed_faces):
        raise ValueError("rotation/scale changed face count")
    area_factor = float(scale) ** 2
    maximum_relative_area_error = 0.0
    for source_face, transformed_face in zip(source_faces, transformed_faces, strict=True):
        expected_area = face_area(source_face) * area_factor
        actual_area = face_area(transformed_face)
        relative_error = abs(expected_area - actual_area) / max(abs(expected_area), abs(actual_area), 1e-12)
        maximum_relative_area_error = max(maximum_relative_area_error, relative_error)
    audit = cleanup_audit(transformed)
    checks = {
        "face_count_preserved": len(source_faces) == len(transformed_faces),
        "face_order_area_correspondence": maximum_relative_area_error < 1e-8,
        "cleanup_idempotent_after_transform": bool(audit["passed"]),
    }
    return transformed, {
        "passed": all(checks.values()),
        "checks": checks,
        "uniform_scale": float(scale),
        "rotation_axis": list(axis_values),
        "rotation_angle_radians": float(rotation_angle_radians),
        "maximum_relative_area_error": maximum_relative_area_error,
        "cleanup_audit": audit,
    }


def _build_supported_context(plan: PlannedSample, wall_macro: str) -> core.CleanFeatureResult:
    raw = build_supported_wall_sample(wall_macro, plan.host_family, "plain", plan.seed)
    wall_instance_id = (plan.primary_target, *plan.secondary_features).index(wall_macro) + 1
    tags: list[core.FaceTag] = []
    for tag in raw.face_tags:
        if tag.macro_feature == wall_macro:
            tags.append(core.FaceTag(tag.face, tag.semantic_role, wall_macro, wall_instance_id))
    if not tags:
        raise ValueError("supported-wall seed geometry has no target faces")
    host = _supported_host_parameters(raw, plan.host_family)
    indexed_general = [
        (macro, instance_id)
        for instance_id, macro in enumerate((plan.primary_target, *plan.secondary_features), 1)
        if macro != wall_macro
    ]
    centers = _supported_feature_centers(indexed_general, host, plan.host_family)
    execution = sorted(indexed_general, key=lambda item: (item[0] in core.TRANSITION_CLASSES, item[1]))
    rng = random.Random(plan.seed ^ 0x5A17C0DE)
    shape = raw.shape
    records: dict[int, dict[str, Any]] = {
        wall_instance_id: {
            "macro_feature": wall_macro,
            "instance_id": wall_instance_id,
            "operation": "supported_thin_wall_fuse",
            "center": [0.0, 0.0],
            "parameters": {
                "source_engineering_scenario": raw.parameters.get("source_engineering_scenario"),
                "support_contract": deepcopy(raw.parameters.get("source_parameters", {})),
                "structural_host_faces_are_stock": True,
            },
            "feature_face_count": len(tags),
            "cleanup_after_operation": deepcopy(raw.cleanup),
        }
    }
    for macro, instance_id in execution:
        shape, tags, record = core._apply_feature(
            shape,
            tags,
            macro=macro,
            instance_id=instance_id,
            center=centers[instance_id],
            host=host,
            rng=rng,
        )
        records[instance_id] = record
    cleanup = cleanup_audit(shape)
    parameters = {
        "schema_version": "topocae_p1_c_clean_feature_parameters_2",
        "generator_revision": GENERATOR_REVISION,
        "plan_revision": "p1_c_clean_context_latin_plan_r3",
        "seed": int(plan.seed),
        "primary_target": plan.primary_target,
        "secondary_features": list(plan.secondary_features),
        "context": plan.context,
        "host_family": plan.host_family,
        "formal_host_family": plan.host_family,
        "formal_host_family_kind": plan.host_family_kind,
        "host_metadata": host,
        "host_cleanup_before_features": deepcopy(raw.parameters["host_cleanup_before_features"]),
        "features": [records[index] for index in sorted(records)],
        "instance_ids": {
            record["macro_feature"]: int(record["instance_id"])
            for record in (records[index] for index in sorted(records))
        },
        "supported_wall": {
            "macro_feature": wall_macro,
            "instance_id": wall_instance_id,
            "source_parameters": deepcopy(raw.parameters),
            "structural_cavity_faces_reclassified_as_stock": plan.host_family in {"rect_cavity", "circular_cavity"},
        },
        "construction_contract": {
            "cleanup_after_each_boolean": True,
            "final_labeling_after_cleanup": True,
            "feature_intersection_allowed": False,
            "supported_wall_connects_two_support_regions": True,
            "structural_host_faces_are_not_feature_truth": True,
        },
    }
    result = core.CleanFeatureResult(shape, tuple(core._filter_final_tags(shape, tags)), parameters, cleanup)
    core._validate_result(result)
    _validate_supported_formal_result(result)
    return result


def _supported_host_parameters(
    raw: core.CleanFeatureResult,
    host_family: str,
) -> dict[str, Any]:
    source = raw.parameters["source_parameters"]
    metadata = deepcopy(source["host_metadata"])
    if host_family == "l_corner":
        length = float(metadata["length"])
        width = float(metadata["width"])
        top_z = float(metadata["base_thickness"])
        x_min = -length / 2.0 + float(metadata.get("coordinate_alignment_x", 0.0))
        x_max = x_min + length
    elif host_family == "two_supports":
        length = float(metadata["length"])
        width = float(metadata["depth"])
        top_z = float(metadata["base_thickness"])
        x_min, x_max = -length / 2.0, length / 2.0
    elif host_family in {"rect_cavity", "circular_cavity"}:
        length = float(metadata["length"])
        width = float(metadata["width"])
        top_z = float(metadata["total_height"])
        x_min, x_max = -length / 2.0, length / 2.0
    else:
        raise KeyError(host_family)
    normalized = {
        **metadata,
        "family": host_family,
        "length": length,
        "width": width,
        "height": top_z,
        "top_z": top_z,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": -width / 2.0,
        "y_max": width / 2.0,
        "construction_method": source["construction_method"],
    }
    return normalized


def _supported_feature_centers(
    indexed_general: list[tuple[str, int]],
    host: dict[str, Any],
    host_family: str,
) -> dict[int, tuple[float, float]]:
    if not indexed_general:
        return {}
    length = float(host["length"])
    width = float(host["width"])
    x_min = float(host["x_min"])
    x_max = float(host["x_max"])
    slot_items = [item for item in indexed_general if item[0] in sweep_generator.SLOT_PROFILE_BY_MACRO]
    non_slot_items = [item for item in indexed_general if item[0] not in sweep_generator.SLOT_PROFILE_BY_MACRO]
    centers: dict[int, tuple[float, float]] = {}
    if host_family == "l_corner":
        base_x = (x_min + 0.63 * length, x_min + 0.82 * length)
        slot_x = base_x
        y_values = (-0.16 * width, 0.16 * width)
    elif host_family == "two_supports":
        base_x = (-0.20 * length, 0.20 * length)
        slot_x = (0.0, 0.0)
        y_values = (-0.31 * width, 0.31 * width)
    else:
        base_x = (-0.35 * length, 0.35 * length)
        slot_x = base_x
        y_values = (-0.30 * width, 0.30 * width)
    for position, (_macro, instance_id) in enumerate(slot_items):
        centers[instance_id] = (slot_x[position % len(slot_x)], y_values[position % len(y_values)])
    for position, (_macro, instance_id) in enumerate(non_slot_items):
        if slot_items:
            y_value = y_values[(position + len(slot_items)) % len(y_values)]
        else:
            y_value = y_values[position % len(y_values)]
        centers[instance_id] = (base_x[position % len(base_x)], y_value)
    for instance_id, (x_value, y_value) in centers.items():
        if not (x_min + 0.08 * length < x_value < x_max - 0.08 * length):
            raise ValueError(f"supported feature x placement is too close to stock edge: {instance_id}")
        if abs(y_value) > 0.42 * width:
            raise ValueError(f"supported feature y placement is too close to stock edge: {instance_id}")
    return centers


def _validate_supported_formal_result(result: core.CleanFeatureResult) -> None:
    params = result.parameters
    if not result.cleanup["passed"]:
        raise ValueError("formal supported-wall result is not cleanup-idempotent")
    if not params["host_cleanup_before_features"]["passed"]:
        raise ValueError("formal supported-wall base host is not clean")
    expected_instances = set(range(1, len(params["secondary_features"]) + 2))
    actual_instances = {tag.instance_id for tag in result.face_tags}
    if actual_instances != expected_instances:
        raise ValueError(f"formal supported-wall instances differ: {actual_instances} != {expected_instances}")
    owner_counts = Counter((tag.instance_id, tag.macro_feature) for tag in result.face_tags)
    for instance_id, macro in enumerate((params["primary_target"], *params["secondary_features"]), 1):
        if owner_counts[(instance_id, macro)] <= 0:
            raise ValueError(f"formal supported-wall feature disappeared: {instance_id}/{macro}")
    support = params["supported_wall"]["source_parameters"]["source_parameters"]
    if int(support["positive_support_contact_count"]) < 2:
        raise ValueError("formal supported wall lost its two-support contract")
    if float(support["span_thickness_ratio"]) < 5.0 or float(support["height_thickness_ratio"]) < 2.0:
        raise ValueError("formal supported wall is not reinforcement-like")


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
