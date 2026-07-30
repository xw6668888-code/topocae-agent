from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

import torch

from .common import ROOT, candidate_area, read_json, write_csv, write_json
from .layered import _instrument_pipeline
from .oracle import (
    AREA_TRUTH_PATH,
    CURRENT_PIPELINE,
    ORACLE_ROOT,
    _aggregate_oracles,
    _candidate_pool,
    _macro_family,
    _oracle_sample,
    _reconcile_sample,
    _report,
    _reviewed_exact_by_sample,
    _write_human_gate,
)
from .runtime import get_spec, load_candidate_model


def _supplement_reviewed_exact(
    reconciliation: list[dict],
    sample_context: dict[str, dict],
    reviewed: dict[str, list[dict]],
) -> list[dict]:
    existing = {
        (record["sample_key"], tuple(sorted(record["face_indices"])))
        for record in reconciliation
        if record["status"] == "auto_unique" and record.get("face_indices")
    }
    supplemental = []
    for sample_key, records in reviewed.items():
        graph = sample_context[sample_key]["graph"]
        for record in records:
            signature = (sample_key, tuple(sorted(record["face_indices"])))
            if signature in existing:
                continue
            area = candidate_area(graph, record["face_indices"])
            supplemental.append({
                "sample_key": sample_key,
                "truth_index": None,
                "truth_id": sample_key + "::frozen_" + str(record["truth_id"]),
                "macro_feature": record["macro_feature"],
                "family": _macro_family(record["macro_feature"]),
                "truth_area": area,
                "area": area,
                "truth_face_count": len(record["face_indices"]),
                "status": "frozen_reviewed_exact",
                "accepted_candidate_id": "reviewed_" + str(record["truth_id"]),
                "face_indices": list(record["face_indices"]),
                "candidate_count": 1,
                "candidates": [],
            })
            existing.add(signature)
    return supplemental


def _gate_summary(reconciliation: list[dict], supplemental: list[dict]) -> tuple[bool, dict]:
    statuses = Counter(str(record["status"]) for record in reconciliation)
    totals = Counter(str(record["family"]) for record in reconciliation)
    reliable = Counter(
        str(record["family"]) for record in reconciliation if record["status"] == "auto_unique"
    )
    reliable.update(str(record["family"]) for record in supplemental)
    blocking = [
        family for family in ("hole", "slot", "rib_web", "transition")
        if totals[family] >= 3 and reliable[family] == 0
    ]
    reliable_total = statuses["auto_unique"] + len(supplemental)
    required = bool(blocking or reliable_total < 20)
    return required, {
        "status_counts": dict(statuses),
        "family_total": dict(totals),
        "family_reliable": dict(reliable),
        "blocking_families": blocking,
        "minimum_reliable_required": 20,
        "reliable_total": reliable_total,
        "supplemental_frozen_reviewed_exact_count": len(supplemental),
        "reason": (
            "major family has no reliable object or reliable subset is too small"
            if required else
            "existing frozen reviewed exact face sets supply all-family coverage; no repeat human review is required"
        ),
    }


def main() -> int:
    truth_payload = read_json(AREA_TRUTH_PATH)
    truths_by_sample = truth_payload["samples"]
    reviewed = _reviewed_exact_by_sample()
    device = torch.device("cpu")
    candidate_model = load_candidate_model(get_spec(CURRENT_PIPELINE), device)
    sample_context: dict[str, dict] = {}
    reconciliation: list[dict] = []
    for sample_key, truths in truths_by_sample.items():
        graph = read_json(ROOT / ("reports/p1_b_real_dev_v1/graphs_v2/" + sample_key + ".json"))
        prediction = read_json(ROOT / ("reports/p1_b_real_dev_v1/predictions/" + sample_key + ".json"))
        pipeline, _ = _instrument_pipeline(CURRENT_PIPELINE, graph, prediction, candidate_model, device)
        reconciliation.extend(
            _reconcile_sample(
                sample_key,
                truths,
                graph,
                _candidate_pool(graph, prediction, pipeline),
                reviewed.get(sample_key, []),
            )
        )
        sample_context[sample_key] = {"graph": graph, "prediction": prediction, "pipeline": pipeline}

    for record in reconciliation:
        record["area"] = float(record["truth_area"])
    supplemental = _supplement_reviewed_exact(reconciliation, sample_context, reviewed)
    gate_required, summary = _gate_summary(reconciliation, supplemental)
    truth_reconciliation = {
        "schema_version": "topocae_p1_d_lite_truth_reconciliation_2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_source": str(AREA_TRUTH_PATH.resolve()),
        "absolute_area_tolerance": 5.0,
        "relative_area_tolerance": 0.01,
        "summary": summary,
        "human_gate_required": gate_required,
        "records": reconciliation,
        "supplemental_frozen_reviewed_exact": supplemental,
        "contracts": {
            "ambiguous_truth_not_auto_selected": True,
            "unresolved_truth_not_fabricated": True,
            "truth_used_only_offline": True,
            "existing_review_reused_before_requesting_new_review": True,
        },
    }
    write_json(ORACLE_ROOT / "truth_reconciliation.json", truth_reconciliation)
    if gate_required:
        blocking_records = [
            record for record in reconciliation
            if record["status"] != "auto_unique" and record["family"] in set(summary["blocking_families"])
        ]
        _write_human_gate(blocking_records, summary)
        print("H-Lite-1 required")
        return 2

    reliable_by_sample: dict[str, list[dict]] = defaultdict(list)
    for record in reconciliation:
        if record["status"] == "auto_unique":
            reliable_by_sample[record["sample_key"]].append(record)
    for record in supplemental:
        reliable_by_sample[record["sample_key"]].append(record)

    predictions_by_oracle: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for sample_key, reliable in reliable_by_sample.items():
        context = sample_context[sample_key]
        result = _oracle_sample(
            context["graph"], context["prediction"], context["pipeline"],
            reliable, candidate_model, device,
        )
        for oracle, predictions in result.items():
            predictions_by_oracle[oracle][sample_key] = predictions

    oracle_metrics, per_class = _aggregate_oracles(reliable_by_sample, predictions_by_oracle)
    ranking = sorted(
        (name for name in oracle_metrics if name != "O0_current_full"),
        key=lambda name: oracle_metrics[name]["object_f1_gain_vs_O0"], reverse=True,
    )
    metrics = {
        "schema_version": "topocae_p1_d_lite_oracle_metrics_2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_pipeline": "P1-B frozen",
        "evaluation_scope": {
            "truth_status": "auto_unique plus non-duplicated frozen reviewed exact face sets",
            "prediction_scope": "objects intersecting reliable faces or area-compatible with reliable truth of the same macro",
            "truth_count": sum(len(records) for records in reliable_by_sample.values()),
            "sample_count": len(reliable_by_sample),
        },
        "oracles": oracle_metrics,
        "ranking": ranking,
        "primary_bottleneck": ranking[0],
        "secondary_bottleneck": ranking[1],
        "human_gate_triggered": False,
        "contracts": {
            "oracle_truth_not_available_to_formal_inference": True,
            "unreliable_truth_excluded": True,
            "historical_outputs_modified": False,
            "truth_area_field_normalized_for_matching": True,
        },
    }
    write_json(ORACLE_ROOT / "metrics.json", metrics)
    write_csv(ORACLE_ROOT / "per_class.csv", per_class)
    (ORACLE_ROOT / "report.zh-CN.md").write_text(_report(metrics, summary), encoding="utf-8")
    print(ORACLE_ROOT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

