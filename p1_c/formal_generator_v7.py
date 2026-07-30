"""Canonical fail-closed generator for the regenerated formal P1-C dataset."""

from __future__ import annotations

from copy import deepcopy

from p1_c import all_features as feature_core
from p1_c import all_features_sweep_v2 as sweep_revision
from p1_c import formal_generator as base_protocol
from p1_c import formal_generator_v4 as transition_protocol
from p1_c import formal_generator_v5 as transition_revision
from p1_c import formal_generator_v6 as revision6
from p1_c.formal_plan_v6 import PLAN_REVISION, PlannedSample
from p1_c import rib_web_v5
from p1_c import supported_wall_families_v3 as supported_wall_revision
from p1_c.strict_generation_contract import assert_generation_parameters


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r7_canonical_fail_closed"
CANONICAL_ENTRYPOINT = "p1_c.formal_generator_v7.build_planned_result"
SLOT_KERNEL_REVISION = sweep_revision.GENERATOR_REVISION
SUPPORTED_WALL_ADAPTER_REVISION = supported_wall_revision.GENERATOR_REVISION
RIB_WEB_GENERATOR_REVISION = rib_web_v5.GENERATOR_REVISION
SUPPORTED_WALL_CLASSES = revision6.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision6.transformed_shape_and_audit


def _activate_authoritative_dependencies() -> None:
    """Rebind final dependencies on every call, not only through imports."""

    feature_core._feature_tool = sweep_revision._feature_tool
    base_protocol.build_supported_wall_sample = (
        supported_wall_revision.build_supported_wall_sample
    )
    base_protocol._supported_host_parameters = (
        revision6._strict_supported_host_parameters
    )
    base_protocol._supported_feature_centers = (
        revision6._strict_supported_feature_centers
    )
    transition_protocol._transition_candidates = (
        transition_revision._all_geometric_outer_linear_edges
    )


def build_planned_result(plan: PlannedSample) -> feature_core.CleanFeatureResult:
    _activate_authoritative_dependencies()
    result = revision6.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters.update(
        {
            "generator_revision": GENERATOR_REVISION,
            "plan_revision": PLAN_REVISION,
            "canonical_generation_entrypoint": CANONICAL_ENTRYPOINT,
            "authoritative_slot_kernel_revision": SLOT_KERNEL_REVISION,
            "supported_wall_adapter_revision": SUPPORTED_WALL_ADAPTER_REVISION,
            "rib_web_generator_revision": RIB_WEB_GENERATOR_REVISION,
            "legacy_planform_slot_tool_permitted": False,
            "legacy_semantic_relabel_permitted": False,
        }
    )
    contract = assert_generation_parameters(parameters)
    parameters["strict_generation_contract_audit"] = contract
    return feature_core.CleanFeatureResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


__all__ = [
    "CANONICAL_ENTRYPOINT",
    "GENERATOR_REVISION",
    "RIB_WEB_GENERATOR_REVISION",
    "SLOT_KERNEL_REVISION",
    "SUPPORTED_WALL_ADAPTER_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
