from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from . import planner as base
from .constants import DATASET_SPLIT_MAP, SEED, SOURCE_MANIFEST, assert_source_path
from .io import canonical_sha256, read_jsonl
from .planner_v2 import _strip_filter_tolerances


FINAL_RECORD_VERSION = "cae-query-ground-record/1.0.0-h3.01"
FINAL_DATASET_VERSION = "cae-query-ground/1.0.0-h3.01"
FORMAL_SOURCE_QUOTAS = {"train": 2400, "val": 300, "development_test": 300}
FORMAL_CLARIFICATION_QUOTAS = {"train": 160, "val": 20, "development_test": 20}
FORMAL_INVALID_QUOTAS = {"train": 80, "val": 10, "development_test": 10}


def select_formal_sources(*, seed: int = SEED) -> list[dict[str, Any]]:
    selected = base.select_pilot_sources(
        read_jsonl(SOURCE_MANIFEST), seed=seed, split_quotas=FORMAL_SOURCE_QUOTAS
    )
    ranks: Counter[str] = Counter()
    for item in selected:
        source_split = item["record"]["split"]
        item["formal_rank_in_split"] = ranks[source_split]
        ranks[source_split] += 1
    if dict(ranks) != FORMAL_SOURCE_QUOTAS:
        raise AssertionError(f"Formal source quotas differ: {dict(ranks)}")
    return selected


def _quantity(dimension: str, value: float, unit: str) -> dict[str, Any]:
    return {"dimension": dimension, "value": float(value), "unit": unit}


def _vary_valid_operations(plan: dict[str, Any], token: int) -> None:
    displacement = (0.05, 0.1, 0.2, 0.25, 0.5)
    pressure = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.2)
    local_mesh = (0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
    global_mesh = (2.0, 3.0, 4.0, 5.0, 6.0)
    directions = (
        {"kind": "global_cartesian", "vector": [1.0, 0.0, 0.0]},
        {"kind": "global_cartesian", "vector": [0.0, -1.0, 0.0]},
        {"kind": "global_cartesian", "vector": [0.0, 0.0, 1.0]},
    )
    operations = plan["intent_result"]["intent"]["operations"]
    for index, operation in enumerate(operations):
        choice = token + index * 17
        kind = operation["operation_type"]
        parameters = operation.get("parameters", {})
        if kind == "displacement":
            parameters["components"][0]["value"] = _quantity(
                "displacement", displacement[choice % len(displacement)], "mm"
            )
        elif kind == "pressure":
            parameters["magnitude"] = _quantity(
                "pressure", pressure[choice % len(pressure)], "MPa"
            )
            parameters["direction"] = {
                "kind": "surface_normal" if choice % 2 == 0 else "opposite_surface_normal"
            }
        elif kind == "surface_traction":
            parameters["magnitude"] = _quantity(
                "pressure", pressure[(choice + 3) % len(pressure)], "MPa"
            )
            parameters["direction"] = copy.deepcopy(directions[choice % len(directions)])
        elif kind == "local_mesh_size":
            parameters["size"] = _quantity(
                "length", local_mesh[choice % len(local_mesh)], "mm"
            )
        elif kind == "global_mesh_size":
            parameters["size"] = _quantity(
                "length", global_mesh[choice % len(global_mesh)], "mm"
            )
        elif kind == "material_assignment":
            modulus = (70.0, 110.0, 200.0, 210.0)[choice % 4]
            poisson = (0.28, 0.3, 0.33)[choice % 3]
            parameters["material_name"] = "CQG-LinearElastic"
            parameters["elastic"] = {
                "youngs_modulus": _quantity("pressure", modulus, "GPa"),
                "poisson_ratio": poisson,
            }
    plan["verbalization"]["operations"] = copy.deepcopy(operations)


def _plan_identity(plan: dict[str, Any]) -> str:
    result = copy.deepcopy(plan["intent_result"])
    if result.get("status") == "VALID":
        result["intent"]["source_language"] = "__LANGUAGE__"
    return canonical_sha256({
        "level": plan["level"],
        "intent_result": result,
        "ground_truth": plan["ground_truth"],
        "verbalization": plan["verbalization"],
    })


def _attach_identity(
    plan: dict[str, Any], selected_source: dict[str, Any], ordinal: int, *, seed: int
) -> dict[str, Any]:
    record = selected_source["record"]
    plan["source_record"] = copy.deepcopy(record)
    plan["focus_feature"] = selected_source["focus_feature"]
    plan["plan_ordinal"] = ordinal
    _strip_filter_tolerances(plan)
    semantic = _plan_identity(plan)
    fingerprint = canonical_sha256({
        "source_graph_sha256": str(record["graph_sha256"]).lower(),
        "source_sample_id": record["sample_id"],
        "semantic": semantic,
        "seed": seed,
    })
    plan["canonical_semantics_sha256"] = fingerprint
    plan["canonical_record_id"] = f"CQG1.{fingerprint[:24]}"
    return plan


def generate_formal_plans_for_source(
    selected_source: dict[str, Any], crosswalk: dict[str, Any], *, seed: int = SEED
) -> list[dict[str, Any]]:
    record = selected_source["record"]
    graph, objects = base.load_feature_objects(record, crosswalk)
    groups = base._groups(objects, selected_source["focus_feature"])
    origin = int(base._stable_score(seed, record["sample_id"])[:8], 16)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    makers = (
        lambda offset: base._plan_l1(groups, offset),
        lambda offset: base._plan_l2(groups, offset),
        lambda offset: base._plan_l3(groups, offset),
        lambda offset: base._plan_l4(groups, graph, offset),
        lambda offset: base._plan_l5_multi(groups, offset),
    )
    for delta in range(24):
        for maker_index, maker in enumerate(makers):
            plan = maker(origin + delta * 11 + maker_index)
            if plan["status"] != "VALID":
                continue
            _vary_valid_operations(plan, origin + delta * 31 + maker_index)
            _strip_filter_tolerances(plan)
            identity = _plan_identity(plan)
            if identity not in seen:
                seen.add(identity)
                candidates.append(plan)
            if len(candidates) == 7:
                break
        if len(candidates) == 7:
            break
    if len(candidates) < 7:
        raise ValueError(f"Unable to generate seven distinct valid plans for {record['sample_id']}")

    rank = int(selected_source["formal_rank_in_split"])
    source_split = record["split"]
    if rank < FORMAL_CLARIFICATION_QUOTAS[source_split]:
        clarification = base._plan_l5_clarification(groups)
        if clarification["verbalization"]["utterance_kind"] == "missing_direction":
            clarification["verbalization"]["operations"][0]["magnitude"] = (
                0.5,
                0.75,
                1.0,
                1.25,
            )[rank % 4]
        candidates[-2] = clarification
    if rank < FORMAL_INVALID_QUOTAS[source_split]:
        invalid = base._plan_l5_invalid(groups)
        invalid["verbalization"]["operations"][0]["size"] = (
            0.25,
            0.5,
            0.75,
            1.0,
        )[rank % 4]
        candidates[-1] = invalid
    return [
        _attach_identity(copy.deepcopy(plan), selected_source, ordinal, seed=seed)
        for ordinal, plan in enumerate(candidates, start=1)
    ]


def generate_formal_plans(crosswalk: dict[str, Any], *, seed: int = SEED) -> list[dict[str, Any]]:
    plans = [
        plan
        for source in select_formal_sources(seed=seed)
        for plan in generate_formal_plans_for_source(source, crosswalk, seed=seed)
    ]
    ids = [plan["canonical_record_id"] for plan in plans]
    if len(plans) < 20000 or len(ids) != len(set(ids)):
        raise ValueError("Formal plan count or canonical ID uniqueness failed")
    return plans


def source_ref_for_plan(plan: dict[str, Any]) -> dict[str, Any]:
    record = plan["source_record"]
    return {
        "canonical_record_id": plan["canonical_record_id"],
        "canonical_semantics_sha256": plan["canonical_semantics_sha256"],
        "sample_id": record["sample_id"],
        "graph_path": str(assert_source_path(record["graph_path"])),
        "graph_sha256": str(record["graph_sha256"]).lower(),
        "geometry_path": str(assert_source_path(record["geometry_path"])),
        "geometry_format": record["geometry_format"],
        "base_geometry_family_id": record["base_geometry_family_id"],
        "source_split": record["split"],
        "dataset_split": DATASET_SPLIT_MAP[record["split"]],
        "derivation_kind": record["derivation_kind"],
        "exact_graph_signature": str(record["exact_graph_signature"]).lower(),
        "near_graph_signature": str(record["near_graph_signature"]).lower(),
    }


def make_formal_record(
    plan: dict[str, Any], *, record_kind: str, query: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(plan["intent_result"])
    if result.get("status") == "VALID":
        result["intent"]["source_language"] = query["language"]
    return {
        "record_schema_version": FINAL_RECORD_VERSION,
        "record_kind": record_kind,
        "source_ref": source_ref_for_plan(plan),
        "query": copy.deepcopy(query),
        "intent_result": result,
        "ground_truth": copy.deepcopy(plan["ground_truth"]),
    }

