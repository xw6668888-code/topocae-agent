"""Pilot planner refinements that preserve v1 logic while tightening record identity."""

from __future__ import annotations

import copy

from .constants import SEED
from .io import canonical_sha256
from .planner import generate_plans_for_source, select_pilot_sources


def _strip_filter_tolerances(plan: dict) -> None:
    """Keep stable selection tolerance in ground truth, not as an unnatural query value."""

    result = plan["intent_result"]
    if result.get("status") == "VALID":
        for operation in result["intent"]["operations"]:
            for condition in operation.get("target", {}).get("filters", {}).values():
                if isinstance(condition, dict) and isinstance(condition.get("value"), dict):
                    condition["value"].pop("tolerance", None)
    for condition in plan["verbalization"].get("filters", {}).values():
        if isinstance(condition, dict) and isinstance(condition.get("value"), dict):
            condition["value"].pop("tolerance", None)


def _source_aware_fingerprint(plan: dict) -> str:
    result = copy.deepcopy(plan["intent_result"])
    if result.get("status") == "VALID":
        result["intent"]["source_language"] = "__LANGUAGE__"
    source = plan["source_record"]
    return canonical_sha256(
        {
            "source_graph_sha256": str(source["graph_sha256"]).lower(),
            "plan_ordinal": plan["plan_ordinal"],
            "level": plan["level"],
            "intent_result": result,
            "ground_truth": plan["ground_truth"],
            "verbalization": plan["verbalization"],
        }
    )


def generate_pilot_plans(crosswalk: dict, *, seed: int = SEED) -> list[dict]:
    plans: list[dict] = []
    for selected_source in select_pilot_sources(seed=seed):
        source_plans = generate_plans_for_source(selected_source, crosswalk, seed=seed)
        for plan in source_plans:
            _strip_filter_tolerances(plan)
            fingerprint = _source_aware_fingerprint(plan)
            plan["canonical_semantics_sha256"] = fingerprint
            plan["canonical_record_id"] = f"CQG.{fingerprint[:24]}"
            plans.append(plan)
    ids = [plan["canonical_record_id"] for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("Source-aware canonical record IDs are not unique")
    return plans

