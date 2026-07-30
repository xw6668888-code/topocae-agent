"""Canonical H-C2 entrypoint with explicit slot-kernel dependency and contracts."""

from __future__ import annotations

from copy import deepcopy

from p1_c import all_features as core
from p1_c import all_features_h_c2_v2 as strict_wall_revision
from p1_c import all_features_sweep_v2 as sweep_revision
from p1_c.strict_generation_contract import assert_generation_parameters


GENERATOR_REVISION = "p1_c_all_features_h_c2_r4_canonical_fail_closed"
CANONICAL_ENTRYPOINT = "p1_c.all_features_h_c2_v3.build_feature_sample"
SLOT_KERNEL_REVISION = sweep_revision.GENERATOR_REVISION


def _activate_authoritative_slot_kernel() -> None:
    # Do this on every call, not only as an import side effect. A later import can
    # no longer silently restore the legacy planform-extrusion tool.
    core._feature_tool = sweep_revision._feature_tool


def build_feature_sample(
    primary: str,
    seed: int,
    *,
    secondary_features: tuple[str, ...] = (),
    host_family: str | None = None,
) -> core.CleanFeatureResult:
    if primary in {"rib", "web"}:
        result = strict_wall_revision.build_feature_sample(
            primary,
            int(seed),
            secondary_features=secondary_features,
            host_family=host_family,
        )
        component_revision = strict_wall_revision.GENERATOR_REVISION
    else:
        _activate_authoritative_slot_kernel()
        result = sweep_revision.build_feature_sample(
            primary,
            int(seed),
            secondary_features=secondary_features,
            host_family=host_family,
        )
        component_revision = sweep_revision.GENERATOR_REVISION

    parameters = deepcopy(result.parameters)
    parameters.update(
        {
            "generator_revision": GENERATOR_REVISION,
            "canonical_generation_entrypoint": CANONICAL_ENTRYPOINT,
            "component_generator_revision": component_revision,
            "authoritative_slot_kernel_revision": SLOT_KERNEL_REVISION,
            "legacy_planform_slot_tool_permitted": False,
        }
    )
    contract = assert_generation_parameters(parameters)
    parameters["strict_generation_contract_audit"] = contract
    return core.CleanFeatureResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


CleanFeatureResult = core.CleanFeatureResult
FaceTag = core.FaceTag
H_C2_SECONDARY_BY_PRIMARY = core.H_C2_SECONDARY_BY_PRIMARY
HOST_FAMILIES = core.HOST_FAMILIES
MACRO_CLASSES = core.MACRO_CLASSES


__all__ = [
    "CANONICAL_ENTRYPOINT",
    "CleanFeatureResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "H_C2_SECONDARY_BY_PRIMARY",
    "HOST_FAMILIES",
    "MACRO_CLASSES",
    "SLOT_KERNEL_REVISION",
    "build_feature_sample",
]
