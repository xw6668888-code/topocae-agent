"""Coordinate-aligned revision of the P1-C clean rib/web generator."""

from __future__ import annotations

from typing import Any

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec

from p1_c import rib_web as core


GENERATOR_REVISION = "p1_c_clean_rib_web_r2"
_ORIGINAL_L_CORNER_HOST = core._l_corner_host


def _transform(shape: Any, dx: float) -> Any:
    transformation = gp_Trsf()
    transformation.SetTranslation(gp_Vec(float(dx), 0.0, 0.0))
    return BRepBuilderAPI_Transform(shape, transformation, True).Shape()


def _aligned_l_corner_host(rng: Any):
    host, host_tags, support_regions, metadata, span, z0 = _ORIGINAL_L_CORNER_HOST(rng)
    inner_wall_x = -float(metadata["length"]) / 2.0 + float(metadata["wall_thickness"])
    desired_inner_wall_x = -float(span) / 2.0
    shift_x = desired_inner_wall_x - inner_wall_x
    aligned_host = _transform(host, shift_x)
    aligned_supports = tuple((name, _transform(region, shift_x)) for name, region in support_regions)
    aligned_metadata = dict(metadata)
    aligned_metadata["coordinate_alignment_x"] = shift_x
    aligned_metadata["generator_revision"] = GENERATOR_REVISION
    return aligned_host, host_tags, aligned_supports, aligned_metadata, span, z0


core._l_corner_host = _aligned_l_corner_host


def build_rib_web_sample(scenario_key: str, seed: int):
    result = core.build_rib_web_sample(scenario_key, seed)
    parameters = dict(result.parameters)
    parameters["generator_revision"] = GENERATOR_REVISION
    return core.CleanRibWebResult(result.shape, result.face_tags, parameters, result.cleanup)


CleanRibWebResult = core.CleanRibWebResult
FaceTag = core.FaceTag
SCENARIO_SPECS = core.SCENARIO_SPECS
scenario_keys = core.scenario_keys


__all__ = [
    "CleanRibWebResult",
    "FaceTag",
    "GENERATOR_REVISION",
    "SCENARIO_SPECS",
    "build_rib_web_sample",
    "scenario_keys",
]
