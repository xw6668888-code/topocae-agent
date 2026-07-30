from __future__ import annotations

from typing import Any

from . import formal_pipeline as base
from .constants import ROOT, SEED
from .formal_planner import generate_formal_plans
from .formal_renderer_v3 import render_formal_all


_ORIGINAL_ASSET_SOURCES = base._asset_sources


def _deterministic_proof(
    plans: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    renderings: list[dict[str, Any]],
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    first_plan_hash = base._hash_sequence(plans)
    first_record_hash = base._hash_sequence(canonical)
    first_rendering_hash = base._hash_sequence(renderings)
    print("starting deterministic plan regeneration", flush=True)
    repeat_plans = generate_formal_plans(crosswalk, seed=SEED)
    repeat_plan_hash = base._hash_sequence(repeat_plans)
    repeat_pairs = [render_formal_all(plan) for plan in repeat_plans]
    repeat_query_hash = base._hash_sequence(
        [pair[0] for pair in repeat_pairs]
        + [query for pair in repeat_pairs for query in pair[1]]
    )
    first_query_hash = base._hash_sequence(
        [record["query"] for record in canonical]
        + [record["query"] for record in renderings]
    )
    if first_plan_hash != repeat_plan_hash or first_query_hash != repeat_query_hash:
        raise RuntimeError("Deterministic formal regeneration failed")
    return {
        "automatic_pass": True,
        "plan_sha256": first_plan_hash,
        "repeat_plan_sha256": repeat_plan_hash,
        "canonical_record_sha256": first_record_hash,
        "rendering_record_sha256": first_rendering_hash,
        "query_sha256": first_query_hash,
        "repeat_query_sha256": repeat_query_hash,
        "seed": SEED,
    }


def _asset_sources():
    assets = _ORIGINAL_ASSET_SOURCES()
    assets.extend([
        (ROOT / "build_cae_query_grounding_dataset_formal_final.py", "executed_formal_entrypoint"),
        (ROOT / "utils" / "cae_query_ground" / "formal_pipeline_final.py", "executed_pipeline_module"),
    ])
    return assets


def run_formal(*, force: bool):
    base._deterministic_proof = _deterministic_proof
    base._asset_sources = _asset_sources
    return base.run_formal(force=force)

