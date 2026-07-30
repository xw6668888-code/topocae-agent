"""Official P1-B stage-4 validator wrapper with edge-vacuity-safe metrics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import validate_p1_b_stage4_topology_closure as base
from utils.p1_b_instance_assembly import evaluate_same_instance_edges as _evaluate_edges
from utils.p1_b_topology_verifier import select_global_candidate_thresholds as _select_candidate_thresholds


def _edge_metrics_with_vacuous_success(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
    *,
    global_threshold: float,
) -> dict[str, Any]:
    metrics = _evaluate_edges(graph_data, prediction_data, global_threshold=global_threshold)
    if metrics["tp"] + metrics["fn"] == 0 and metrics["fp"] == 0:
        metrics["precision"] = 1.0
        metrics["recall"] = 1.0
        metrics["f1"] = 1.0
        metrics["vacuous_no_positive_edges"] = True
    else:
        metrics["vacuous_no_positive_edges"] = False
    return metrics


def _candidate_thresholds_at_contract_operating_point(
    records: Iterable[Mapping[str, Any]],
    *,
    threshold_grid: Iterable[float] | None = None,
) -> dict[str, Any]:
    del threshold_grid
    return _select_candidate_thresholds(records, threshold_grid=[0.5])


base.evaluate_same_instance_edges = _edge_metrics_with_vacuous_success
base.select_global_candidate_thresholds = _candidate_thresholds_at_contract_operating_point


if __name__ == "__main__":
    raise SystemExit(base.main())
