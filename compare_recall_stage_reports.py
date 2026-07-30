from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_stage_metrics(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    proposal = best.get("proposal_stage") or {}
    classified = best.get("classified_stage") or {}
    accepted = best.get("accepted_stage") or {}
    final = best.get("final_survival_stage") or {}
    return {
        "truth_count": int(best.get("truth_count", report.get("truth_count", 0))),
        "proposal_recall": float(proposal.get("recall", 0.0)),
        "proposal_matched": int(proposal.get("matched_truth_count", 0)),
        "classified_recall": float(classified.get("recall", 0.0)),
        "classified_matched": int(classified.get("matched_truth_count", 0)),
        "accepted_recall": float(accepted.get("recall", 0.0)),
        "accepted_matched": int(accepted.get("matched_truth_count", 0)),
        "final_survival_recall": float(final.get("recall", 0.0)),
        "final_survival_matched": int(final.get("matched_truth_count", 0)),
    }


def compare_stage_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_metrics = extract_stage_metrics(old)
    new_metrics = extract_stage_metrics(new)
    delta: dict[str, Any] = {}
    for key, old_value in old_metrics.items():
        if key == "truth_count":
            delta[key] = int(new_metrics[key]) - int(old_value)
        elif key.endswith("_matched"):
            delta[key] = int(new_metrics[key]) - int(old_value)
        else:
            delta[key] = float(new_metrics[key]) - float(old_value)
    return {"old": old_metrics, "new": new_metrics, "delta": delta}


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# v5.24 vs v5.23 Recall Stage Comparison",
        "",
        "| Metric | v5.23 | v5.24 | Delta |",
        "|--------|-------|-------|-------|",
    ]
    keys = [
        "proposal_recall",
        "classified_recall",
        "accepted_recall",
        "final_survival_recall",
        "proposal_matched",
        "classified_matched",
        "accepted_matched",
        "final_survival_matched",
    ]
    for key in keys:
        old_value = comparison["old"][key]
        new_value = comparison["new"][key]
        delta = comparison["delta"][key]
        if isinstance(old_value, float):
            lines.append(f"| {key} | {old_value:.4f} | {new_value:.4f} | {delta:+.4f} |")
        else:
            lines.append(f"| {key} | {old_value} | {new_value} | {delta:+d} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare object proposal recall stages between two reports.")
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--new-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_stage_reports(load_json(args.old_json), load_json(args.new_json))
    write_markdown(args.output_md, comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
