from __future__ import annotations

from typing import Any

from . import planner as base_planner
from .constants import SEED
from .formal_planner import (
    _attach_identity,
    generate_formal_plans_for_source as generate_v1,
)


def generate_formal_plans_for_source(
    selected_source: dict[str, Any], crosswalk: dict[str, Any], *, seed: int = SEED
) -> list[dict[str, Any]]:
    plans = generate_v1(selected_source, crosswalk, seed=seed)
    for index, plan in enumerate(plans):
        if plan["status"] != "NEEDS_CLARIFICATION":
            continue
        feature = plan["verbalization"]["feature_type"]
        result = base_planner._clarification_result(
            code="M_REFERENCE",
            field_path="/operations/0/target/filters/nearest_to",
            description="A nearest-target request requires an explicit spatial reference.",
        )
        replacement = {
            "level": "L4",
            "status": "NEEDS_CLARIFICATION",
            "intent_result": result,
            "ground_truth": base_planner._non_executable_ground(
                "NEEDS_CLARIFICATION",
                "Nearest is not uniquely defined because the request omits the spatial reference.",
            ),
            "verbalization": {
                "utterance_kind": "missing_reference",
                "feature_type": feature,
                "face_role": "feature_region",
                "quantifier": "nearest",
                "operations": [{"operation_type": "local_mesh_size"}],
            },
        }
        plans[index] = _attach_identity(
            replacement,
            selected_source,
            int(plan["plan_ordinal"]),
            seed=seed,
        )
    return plans

