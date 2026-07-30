"""Full formal pipeline with the released formal-scale manifest contract."""

from __future__ import annotations

from . import formal_pipeline as base
from .formal_contract_release import build_final_schemas
from .formal_pipeline_release_v2 import _asset_sources, _deterministic_proof
from .formal_planner_parallel_release import generate_formal_plans_parallel


_base_dataset_manifest = base.dataset_manifest


def _formal_dataset_manifest(**kwargs):
    manifest = _base_dataset_manifest(**kwargs)
    manifest["scope_guards"]["formal_scale_generated"] = True
    return manifest


def run_formal(*, force: bool):
    base.generate_formal_plans = generate_formal_plans_parallel
    base._deterministic_proof = _deterministic_proof
    base._asset_sources = _asset_sources
    base.build_final_schemas = build_final_schemas
    base.dataset_manifest = _formal_dataset_manifest
    return base.run_formal(force=force)

