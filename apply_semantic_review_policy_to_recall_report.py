"""Apply user-approved semantic review policy as an evaluation-only recall metric."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MACRO_ALIASES = {
    "through_hole": "through_hole_round",
    "blind_hole": "blind_hole_round",
    "stepped_hole": "stepped_hole_round",
    "cylindrical_boss": "cylinder_boss",
    "rectangle_boss": "rectangular_boss",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_key_macro(macro: str) -> str:
    token = str(macro or "").strip()
    return MACRO_ALIASES.get(token, token)


def rounded_area(value: Any) -> float:
    return round(float(value), 4)


def accepted_policy_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, float]]:
    keys: set[tuple[str, str, float]] = set()
    for row in rows:
        if str(row.get("user_decision", "")).strip() != "accept_family_match":
            continue
        keys.add(
            (
                str(row.get("sample", "")).strip(),
                normalize_key_macro(str(row.get("truth_macro", "")).strip()),
                rounded_area(row.get("truth_area", "0")),
            )
        )
    return keys


def areas_match(a: float, b: float, area_tolerance: float) -> bool:
    return abs(a - b) <= area_tolerance


def apply_semantic_policy(report: dict[str, Any], review_rows: list[dict[str, str]], area_tolerance: float) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    strict_stage = best.get("final_survival_stage") or {}
    truth_count = int(best.get("truth_count", report.get("truth_count", 0)))
    strict_matched = int(strict_stage.get("matched_truth_count", 0))
    accepted_keys = accepted_policy_keys(review_rows)
    recovered: list[dict[str, Any]] = []
    for sample in best.get("samples", []):
        sample_name = str(sample.get("sample", ""))
        for failure in sample.get("truth_failures", []):
            failure_macro = normalize_key_macro(str(failure.get("macro", "")))
            failure_area = float(failure.get("area", 0.0))
            for key_sample, key_macro, key_area in accepted_keys:
                if key_sample == sample_name and key_macro == failure_macro and areas_match(key_area, failure_area, area_tolerance):
                    recovered.append(
                        {
                            "sample": sample_name,
                            "macro": failure_macro,
                            "area": failure_area,
                            "matched_policy_key": {"sample": key_sample, "macro": key_macro, "area": key_area},
                        }
                    )
                    break
    policy_recovered = len(recovered)
    semantic_matched = strict_matched + policy_recovered
    return {
        "schema": "semantic_review_policy_recall_v1",
        "truth_count": truth_count,
        "strict_matched_truth_count": strict_matched,
        "strict_recall": strict_matched / truth_count if truth_count else 0.0,
        "policy_recovered_count": policy_recovered,
        "semantic_policy_matched_truth_count": semantic_matched,
        "semantic_policy_recall": semantic_matched / truth_count if truth_count else 0.0,
        "accepted_policy_key_count": len(accepted_keys),
        "recovered_rows": recovered,
        "note": "Evaluation-only metric. This does not change model predictions or strict macro labels.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# v5.21 Semantic Policy Recall",
        "",
        "> This is an evaluation-only CAE hole-family metric. It does not change prediction labels.",
        "",
        "## Summary",
        "",
        f"- truth_count: `{payload['truth_count']}`",
        f"- strict_matched_truth_count: `{payload['strict_matched_truth_count']}`",
        f"- strict_recall: `{payload['strict_recall']:.4f}`",
        f"- policy_recovered_count: `{payload['policy_recovered_count']}`",
        f"- semantic_policy_matched_truth_count: `{payload['semantic_policy_matched_truth_count']}`",
        f"- semantic_policy_recall: `{payload['semantic_policy_recall']:.4f}`",
        "",
        "## Recovered Rows",
        "",
        "| Sample | Macro | Area |",
        "|--------|-------|------|",
    ]
    for row in payload["recovered_rows"]:
        lines.append(f"| {row['sample']} | {row['macro']} | {row['area']:.4f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply user-approved semantic review policy as an evaluation-only recall metric.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--area-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_json(args.report_json)
    review_rows = read_review_rows(args.review_csv)
    payload = apply_semantic_policy(report, review_rows, args.area_tolerance)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
