from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_final(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("best_by_final_survival_recall") or {})


def metric_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    best = best_final(report)
    failure_counts = best.get("failure_stage_counts", {})
    return {
        "name": name,
        "proposal_recall": float(best.get("proposal_stage", {}).get("recall", 0.0)),
        "classified_recall": float(best.get("classified_stage", {}).get("recall", 0.0)),
        "accepted_recall": float(best.get("accepted_stage", {}).get("recall", 0.0)),
        "final_survival_recall": float(best.get("final_survival_stage", {}).get("recall", 0.0)),
        "no_compatible_proposal": int(failure_counts.get("no_area_compatible_proposal", 0)),
        "classifier_wrong_or_rejected": int(failure_counts.get("proposal_exists_classifier_wrong_or_rejected", 0)),
        "accepted_but_final_changes": int(failure_counts.get("classifier_accepts_but_final_changes_or_removes", 0)),
        "survives_final": int(failure_counts.get("survives_final_prediction", 0)),
    }


def parse_named_report_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report argument must use name=path format, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"report argument must use name=path format, got {value!r}")
    return name, Path(raw_path)


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Candidate Classifier Comparison",
        "",
        "| Version | Proposal recall | Classified recall | Accepted recall | Final-survival recall | No proposal | Classifier wrong/rejected | Final changes | Survives final |",
        "|---------|-----------------|-------------------|-----------------|-----------------------|-------------|---------------------------|---------------|----------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | "
            f"{row['proposal_recall']:.4f} | "
            f"{row['classified_recall']:.4f} | "
            f"{row['accepted_recall']:.4f} | "
            f"{row['final_survival_recall']:.4f} | "
            f"{row['no_compatible_proposal']} | "
            f"{row['classifier_wrong_or_rejected']} | "
            f"{row['accepted_but_final_changes']} | "
            f"{row['survives_final']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    v513 = by_name.get("v5.13")
    if v513 is None:
        raise ValueError("v5.13 baseline row is required for decision")

    candidates = [
        row for row in rows
        if row["name"].startswith("v5.17")
    ]
    promotable = [
        row for row in candidates
        if row["accepted_recall"] > v513["accepted_recall"]
        and row["classifier_wrong_or_rejected"] <= v513["classifier_wrong_or_rejected"]
    ]
    if promotable:
        best = max(promotable, key=lambda row: (row["accepted_recall"], row["final_survival_recall"]))
        recommendation = f"promote_{best['name']}"
        reason = (
            f"{best['name']} beats v5.13 accepted recall "
            f"({best['accepted_recall']:.4f} > {v513['accepted_recall']:.4f}) "
            f"without increasing classifier wrong/rejected "
            f"({best['classifier_wrong_or_rejected']} <= {v513['classifier_wrong_or_rejected']})."
        )
    elif any(row["accepted_recall"] > by_name.get("v5.14", v513)["accepted_recall"] for row in candidates):
        recommendation = "keep_v5_13_but_record_v5_17_partial_gain"
        reason = "At least one v5.17 variant improves over v5.14 but does not beat the v5.13 current-best classifier."
    else:
        recommendation = "keep_v5_13_and_move_to_proposal_recall_repairs"
        reason = "v5.17 role-aware features do not improve real S3 accepted recall enough; remaining work should target no-compatible-proposal failures."

    lines = [
        "# v5.17 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate classifier recall reports.")
    parser.add_argument("--report", action="append", required=True, help="Named report in name=path format.")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--decision-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_reports = [parse_named_report_arg(item) for item in args.report]
    rows = [metric_row(name, load_json(path)) for name, path in named_reports]
    write_comparison(args.output_md, rows)
    write_decision(args.decision_md, rows)
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
