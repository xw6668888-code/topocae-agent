from __future__ import annotations

import random

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.TopologyUtils import TopologyExplorer

import features.p1_b_rib_web as core
from features.p1_b_rib_web import FaceTag, P1BRibWebResult, SCENARIO_SPECS


def scenario_keys() -> list[str]:
    return core.scenario_keys()


def build_rib_web_sample(scenario_key: str, seed: int) -> P1BRibWebResult:
    if scenario_key not in core.SCENARIO_BY_KEY:
        raise KeyError(f"unknown P1-B rib/web scenario: {scenario_key}")
    _, _macro_key, context_kind, variant = core.SCENARIO_BY_KEY[scenario_key]
    rng = random.Random(int(seed))
    if context_kind == "l_corner":
        result = core._build_l_corner_rib(scenario_key, variant, rng, int(seed))
    elif context_kind in {"rect_cavity", "circular_cavity"}:
        result = core._build_cavity_rib(scenario_key, context_kind, variant, rng, int(seed))
    else:
        result = core._build_two_support_web(scenario_key, variant, rng, int(seed))
    _validate_engineering_result(result)
    return result


def _validate_engineering_result(result: P1BRibWebResult) -> None:
    params = result.parameters
    if not BRepCheck_Analyzer(result.shape).IsValid():
        raise ValueError("P1-B rib/web result is not a valid B-Rep")
    if len(list(TopologyExplorer(result.shape).solids())) != 1:
        raise ValueError("P1-B rib/web result must be one solid")
    if core._volume(result.shape) <= 0.0:
        raise ValueError("P1-B rib/web result must have positive volume")
    if params["positive_support_contact_count"] < 2:
        raise ValueError("thin wall does not contact at least two support regions")
    if params["span_thickness_ratio"] < 5.0 or params["height_thickness_ratio"] < 2.0:
        raise ValueError("thin-wall dimensional ratio is not reinforcement-like")
    if params["feature_face_count"] < 2:
        raise ValueError("thin-wall exposed faces were not preserved")
    feature_tags = [
        tag for tag in result.face_tags
        if tag.macro_key == params["macro_key"] and tag.instance_id == params["feature_instance_id"]
    ]
    if not feature_tags or {tag.instance_id for tag in feature_tags} != {params["feature_instance_id"]}:
        raise ValueError("thin-wall faces do not share one instance")
    if params["context_kind"] in {"rect_cavity", "circular_cavity"} and params["pocket_face_count"] < 2:
        raise ValueError("cavity/pocket labels were not preserved independently")
    if params["has_hole"] and params["hole_face_count"] < 1:
        raise ValueError("hole variant did not preserve an independent hole wall")


__all__ = [
    "FaceTag",
    "P1BRibWebResult",
    "SCENARIO_SPECS",
    "build_rib_web_sample",
    "scenario_keys",
]
