from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .constants import SEED
from .formal_planner import generate_formal_plans_for_source, select_formal_sources


_CROSSWALK: dict[str, Any] | None = None
_SEED = SEED


def _initialize(crosswalk: dict[str, Any], seed: int) -> None:
    global _CROSSWALK, _SEED
    _CROSSWALK = crosswalk
    _SEED = seed


def _worker(selected_source: dict[str, Any]) -> list[dict[str, Any]]:
    if _CROSSWALK is None:
        raise RuntimeError("Formal planner worker was not initialized")
    return generate_formal_plans_for_source(selected_source, _CROSSWALK, seed=_SEED)


def generate_formal_plans_parallel(
    crosswalk: dict[str, Any], *, seed: int = SEED
) -> list[dict[str, Any]]:
    sources = select_formal_sources(seed=seed)
    workers = min(12, max(1, os.cpu_count() or 1))
    plans: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(crosswalk, seed),
    ) as executor:
        for source_plans in executor.map(_worker, sources, chunksize=4):
            plans.extend(source_plans)
    ids = [plan["canonical_record_id"] for plan in plans]
    if len(plans) < 20000 or len(ids) != len(set(ids)):
        raise ValueError("Parallel formal planning count or ID uniqueness failed")
    return plans

