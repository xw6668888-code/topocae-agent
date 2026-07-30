"""Strict supported-wall adapter bound to the human-approved r5 geometry."""

from __future__ import annotations

from copy import deepcopy

from p1_c import supported_wall_families_v2 as protocol
from p1_c import rib_web_v5 as generator


GENERATOR_REVISION = "p1_c_supported_wall_r3_full_height_strict_concept_hosts"
RIB_WEB_GENERATOR_REVISION = generator.GENERATOR_REVISION
SUPPORTED_WALL_HOST_FAMILIES = protocol.SUPPORTED_WALL_HOST_FAMILIES
HOST_FAMILIES_BY_MACRO = protocol.HOST_FAMILIES_BY_MACRO


protocol.build_rib_web_sample = generator.build_rib_web_sample
protocol.RIB_WEB_GENERATOR_REVISION = generator.GENERATOR_REVISION


def build_supported_wall_sample(
    primary: str,
    host_family: str,
    variant: str,
    seed: int,
):
    result = protocol.build_supported_wall_sample(primary, host_family, variant, int(seed))
    parameters = deepcopy(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    parameters["rib_web_generator_revision"] = RIB_WEB_GENERATOR_REVISION
    parameters["concept_contract"] = {
        **deepcopy(parameters["concept_contract"]),
        "web_full_height_connection_required": True,
        "web_exposed_end_faces_forbidden": True,
    }
    source = parameters["source_parameters"]
    if source["macro_feature"] == "web":
        if int(source["exposed_feature_end_face_count"]) != 0:
            raise ValueError("strict adapter received a web with exposed end faces")
        if not source["full_height_two_support_connection"]:
            raise ValueError("strict adapter received a partially connected web")
    return protocol.CleanFeatureResult(
        result.shape,
        result.face_tags,
        parameters,
        result.cleanup,
    )


__all__ = [
    "GENERATOR_REVISION",
    "HOST_FAMILIES_BY_MACRO",
    "RIB_WEB_GENERATOR_REVISION",
    "SUPPORTED_WALL_HOST_FAMILIES",
    "build_supported_wall_sample",
]
