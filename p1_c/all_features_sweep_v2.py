from __future__ import annotations

from typing import Any

from p1_c import all_features as core
from p1_c import all_features_h_c2 as h_c2
from p1_c.swept_slots import SLOT_PROFILE_BY_MACRO, build_swept_slot_tool


GENERATOR_REVISION = "p1_c_all_features_clean_swept_slot_r2"
_ORIGINAL_FEATURE_TOOL = core._feature_tool


def _feature_tool(macro: str, center: tuple[float, float], host: dict[str, Any], rng: Any):
    if macro not in SLOT_PROFILE_BY_MACRO:
        return _ORIGINAL_FEATURE_TOOL(macro, center, host, rng)
    swept = build_swept_slot_tool(macro, center, host, rng)
    return swept.tool, "cut", swept.role_for_face, swept.parameters


core._feature_tool = _feature_tool


def build_feature_sample(
    primary: str,
    seed: int,
    *,
    secondary_features: tuple[str, ...] = (),
    host_family: str | None = None,
):
    result = h_c2.build_feature_sample(
        primary,
        seed,
        secondary_features=secondary_features,
        host_family=host_family,
    )
    parameters = dict(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    return core.CleanFeatureResult(result.shape, result.face_tags, parameters, result.cleanup)


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
    "build_feature_sample",
]
