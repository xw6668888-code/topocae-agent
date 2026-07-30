from __future__ import annotations

from copy import deepcopy

from p1_c import all_features as core
from p1_c import formal_generator as revision1
from p1_c.formal_plan_v4 import PLAN_REVISION, PlannedSample


GENERATOR_REVISION = "p1_c_clean_context_formal_generator_r2"
SUPPORTED_WALL_CLASSES = revision1.SUPPORTED_WALL_CLASSES
transformed_shape_and_audit = revision1.transformed_shape_and_audit


def build_planned_result(plan: PlannedSample) -> core.CleanFeatureResult:
    result = revision1.build_planned_result(plan)
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["plan_revision"] = PLAN_REVISION
    return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


__all__ = [
    "GENERATOR_REVISION",
    "SUPPORTED_WALL_CLASSES",
    "build_planned_result",
    "transformed_shape_and_audit",
]
