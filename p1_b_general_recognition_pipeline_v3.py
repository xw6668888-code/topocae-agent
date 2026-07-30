"""Official P1-B pipeline entrypoint using topology verifier v3."""

from __future__ import annotations

import utils.p1_b_topology_verifier_v3
from p1_b_general_recognition_pipeline import run_p1_b_general_recognition as _run


PIPELINE_SCHEMA_VERSION = "topocae_p1_b_general_recognition_pipeline_3"


def run_p1_b_general_recognition(*args, **kwargs):
    result = _run(*args, **kwargs)
    result["schema_version"] = PIPELINE_SCHEMA_VERSION
    result["pipeline_manifest"]["topology_verifier_schema"] = "topocae_p1_b_topology_verifier_3"
    return result


__all__ = ["PIPELINE_SCHEMA_VERSION", "run_p1_b_general_recognition"]
