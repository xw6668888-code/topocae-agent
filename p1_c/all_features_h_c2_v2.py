"""H-C2 generator bound to strict, full-height rib/web hosts."""

from __future__ import annotations

from copy import deepcopy

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from p1_c import all_features as core
from p1_c import all_features_h_c2 as general_revision
from p1_c.geometry import clean_cut, cleanup_audit
from p1_c.supported_wall_families_v3 import build_supported_wall_sample


GENERATOR_REVISION = "p1_c_all_features_h_c2_r3_strict_full_height_hosts"
STRICT_H_C2_HOST_BY_MACRO = {
    "rib": "t_junction",
    "web": "two_supports_buttressed",
}


def build_feature_sample(
    primary: str,
    seed: int,
    *,
    secondary_features: tuple[str, ...] = (),
    host_family: str | None = None,
) -> core.CleanFeatureResult:
    if primary not in STRICT_H_C2_HOST_BY_MACRO:
        result = general_revision.build_feature_sample(
            primary,
            int(seed),
            secondary_features=secondary_features,
            host_family=host_family,
        )
        parameters = deepcopy(result.parameters)
        parameters["generator_revision"] = GENERATOR_REVISION
        return core.CleanFeatureResult(
            result.shape,
            result.face_tags,
            parameters,
            result.cleanup,
        )

    selected_host = host_family or STRICT_H_C2_HOST_BY_MACRO[primary]
    if selected_host != STRICT_H_C2_HOST_BY_MACRO[primary]:
        raise ValueError(f"unexpected strict H-C2 host: {primary}/{selected_host}")
    if secondary_features not in {(), ("through_hole_round",)}:
        raise ValueError("strict H-C2 rib/web permits only one independent through-hole")

    target = build_supported_wall_sample(primary, selected_host, "plain", int(seed))
    parameters = deepcopy(target.parameters)
    parameters.update(
        {
            "generator_revision": GENERATOR_REVISION,
            "primary_target": primary,
            "secondary_features": [],
            "context": "target_only",
            "strict_h_c2_host_family": selected_host,
        }
    )
    target = core.CleanFeatureResult(
        target.shape,
        target.face_tags,
        parameters,
        target.cleanup,
    )
    core._validate_result(target)
    if not secondary_features:
        return target

    raw = target.parameters["source_parameters"]
    host = raw["host_metadata"]
    base_thickness = float(host["base_thickness"])
    radius = min(2.6, base_thickness * 0.34)
    if primary == "rib":
        cx = float(host["length"]) * 0.32
        cy = float(host["width"]) * 0.24
        clearance = min(
            abs(cx) - float(raw["span"]) * 0.5,
            abs(cy) - float(raw["thickness"]) * 0.5,
        )
    else:
        cx = 0.0
        cy = float(host["depth"]) * 0.32
        clearance = abs(cy) - float(raw["thickness"]) * 0.5
    if clearance <= radius + 2.0:
        raise ValueError("independent strict H-C2 host hole does not clear the supported wall")
    tool = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(cx, cy, -1.0), gp_Dir(0.0, 0.0, 1.0)),
        radius,
        base_thickness + 2.0,
    ).Shape()
    operation, shape = clean_cut(target.shape, tool)
    propagated = core._propagate_tags(operation, target.face_tags)
    hole_tags = core._tags_from_tool(
        operation,
        tool,
        role=lambda _face: "through_hole_cylinder",
        macro_feature="through_hole_round",
        instance_id=2,
    )
    tags = tuple(core._filter_final_tags(shape, core._merge_tags(propagated, hole_tags)))
    cleanup = cleanup_audit(shape)
    features = [deepcopy(target.parameters["features"][0])]
    features.append(
        {
            "macro_feature": "through_hole_round",
            "instance_id": 2,
            "operation": "non_intersecting_host_through_hole_cut",
            "center": [cx, cy],
            "parameters": {
                "radius": radius,
                "through_base_only": True,
                "target_clearance": clearance,
            },
            "feature_face_count": sum(tag.instance_id == 2 for tag in tags),
            "cleanup_after_operation": cleanup,
        }
    )
    parameters = deepcopy(target.parameters)
    parameters.update(
        {
            "generator_revision": GENERATOR_REVISION,
            "secondary_features": ["through_hole_round"],
            "context": "pair",
            "features": features,
            "instance_ids": {primary: 1, "through_hole_round": 2},
            "composition_clearance": clearance,
            "composition_contract": (
                "host hole is spatially disjoint from the strict supported thin wall"
            ),
        }
    )
    result = core.CleanFeatureResult(shape, tags, parameters, cleanup)
    core._validate_result(result)
    return result


CleanFeatureResult = core.CleanFeatureResult
FaceTag = core.FaceTag
H_C2_SECONDARY_BY_PRIMARY = core.H_C2_SECONDARY_BY_PRIMARY
HOST_FAMILIES = core.HOST_FAMILIES
MACRO_CLASSES = core.MACRO_CLASSES


__all__ = [
    "CleanFeatureResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "H_C2_SECONDARY_BY_PRIMARY",
    "HOST_FAMILIES",
    "MACRO_CLASSES",
    "STRICT_H_C2_HOST_BY_MACRO",
    "build_feature_sample",
]
