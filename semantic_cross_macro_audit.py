"""v5.19 Cross-Macro Semantic Audit.

Audit cross-macro area-compatible S3 failures, classify them into CAE-semantic
review categories, and simulate evaluation policies before changing code.

Diagnostic-only — does not modify model, proposer, classifier, or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HOLE_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
}
TRANSITION_MACROS = {"fillet", "chamfer"}
RECTANGULAR_RECESS_MACROS = {"slot_rect", "pocket_rect"}
RIB_CONTEXT_MACROS = {"rib", "web", "rectangular_boss", "cylinder_boss", "slot_v", "slot_rect"}
POLICY_CATEGORIES = {
    "strict_current": set(),
    "safe_hole_composite": {"hole_subfeature_vs_composite_hole"},
    "safe_hole_and_rectangular_recess": {
        "hole_subfeature_vs_composite_hole",
        "rectangular_recess_slot_vs_pocket",
    },
    "review_all_cross_macro_compatible": {
        "hole_subfeature_vs_composite_hole",
        "rectangular_recess_slot_vs_pocket",
        "thinwall_rib_vs_boss_or_web",
        "transition_vs_nontransition_conflict",
        "circular_recess_or_hole_conflict",
        "unknown_cross_macro_conflict",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def macro_pair_key(left: str, right: str) -> str:
    return "|".join(sorted([str(left), str(right)]))


def classify_review_category(truth_macro: str, candidate_macro: str) -> str:
    pair = {truth_macro, candidate_macro}
    if truth_macro in HOLE_MACROS and candidate_macro in HOLE_MACROS and truth_macro != candidate_macro:
        return "hole_subfeature_vs_composite_hole"
    if pair == RECTANGULAR_RECESS_MACROS:
        return "rectangular_recess_slot_vs_pocket"
    if "pocket_circular" in pair and bool(pair & HOLE_MACROS):
        return "circular_recess_or_hole_conflict"
    if bool(pair & {"rib", "web"}) and bool(pair & RIB_CONTEXT_MACROS - {"rib", "web"}):
        return "thinwall_rib_vs_boss_or_web"
    if (truth_macro in TRANSITION_MACROS) != (candidate_macro in TRANSITION_MACROS):
        return "transition_vs_nontransition_conflict"
    return "unknown_cross_macro_conflict"


def best_cross_candidate(miss: dict[str, Any]) -> dict[str, Any] | None:
    candidates = miss.get("nearest_cross_macro", [])
    if not candidates:
        return None
    return candidates[0]


def summarize_cross_macro_misses(misses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for miss in misses:
        if miss.get("miss_cause") != "cross_macro_area_compatible":
            continue
        candidate = best_cross_candidate(miss)
        if not candidate:
            continue
        truth_macro = str(miss.get("macro", ""))
        candidate_macro = str(candidate.get("proposal_macro", ""))
        category = classify_review_category(truth_macro, candidate_macro)
        pair = macro_pair_key(truth_macro, candidate_macro)
        rows.append(
            {
                "sample": str(miss.get("sample", "")),
                "truth_macro": truth_macro,
                "candidate_macro": candidate_macro,
                "pair": pair,
                "review_category": category,
                "truth_area": float(miss.get("area", 0.0) or 0.0),
                "candidate_area": float(candidate.get("area", 0.0) or 0.0),
                "relative_error": float(candidate.get("relative_error", 0.0) or 0.0),
                "truth_face_count": int(miss.get("face_count", 0) or 0),
                "candidate_face_count": int(candidate.get("face_count", 0) or 0),
                "candidate_source": str(candidate.get("source", "")),
                "candidate_action": str(candidate.get("action", "")),
                "candidate_confidence": candidate.get("confidence"),
                "candidate_faces": list(candidate.get("face_indices", [])),
            }
        )
    summary = {
        "cross_macro_compatible_count": len(rows),
        "by_pair": dict(Counter(row["pair"] for row in rows).most_common()),
        "by_category": dict(Counter(row["review_category"] for row in rows).most_common()),
        "by_truth_macro": dict(Counter(row["truth_macro"] for row in rows).most_common()),
        "by_candidate_macro": dict(Counter(row["candidate_macro"] for row in rows).most_common()),
        "by_part": dict(Counter(row["sample"] for row in rows).most_common()),
    }
    return rows, summary


def simulate_policy(policy: str, rows: list[dict[str, Any]], baseline_miss_count: int) -> dict[str, Any]:
    if policy not in POLICY_CATEGORIES:
        raise ValueError(f"Unknown policy {policy!r}")
    allowed = POLICY_CATEGORIES[policy]
    recovered = [row for row in rows if row["review_category"] in allowed]
    return {
        "policy": policy,
        "baseline_miss_count": int(baseline_miss_count),
        "recovered_count": len(recovered),
        "remaining_miss_count": int(baseline_miss_count) - len(recovered),
        "recovered_by_category": dict(Counter(row["review_category"] for row in recovered).most_common()),
        "recovered_by_pair": dict(Counter(row["pair"] for row in recovered).most_common()),
    }


def build_audit(atlas: dict[str, Any]) -> dict[str, Any]:
    misses = list(atlas.get("misses", []))
    rows, summary = summarize_cross_macro_misses(misses)
    baseline_miss_count = int(atlas.get("summary", {}).get("miss_count", len(misses)))
    simulations = {
        policy: simulate_policy(policy, rows, baseline_miss_count)
        for policy in POLICY_CATEGORIES
    }
    return {
        "schema": "semantic_cross_macro_audit_v1",
        "baseline_miss_count": baseline_miss_count,
        "summary": summary,
        "rows": rows,
        "policy_simulations": simulations,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "truth_macro",
        "candidate_macro",
        "pair",
        "review_category",
        "truth_area",
        "candidate_area",
        "relative_error",
        "truth_face_count",
        "candidate_face_count",
        "candidate_source",
        "candidate_action",
        "candidate_confidence",
        "candidate_faces",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            written = dict(row)
            written["candidate_faces"] = " ".join(str(item) for item in row.get("candidate_faces", []))
            writer.writerow(written)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    lines = [
        "# v5.19 Cross-Macro Semantic Audit",
        "",
        "## Summary",
        "",
        f"- baseline_miss_count: `{audit['baseline_miss_count']}`",
        f"- cross_macro_compatible_count: `{summary['cross_macro_compatible_count']}`",
        "",
        "## Review Categories",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for category, count in summary["by_category"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Macro Pairs", "", "| Pair | Count |", "|------|-------|"])
    for pair, count in summary["by_pair"].items():
        lines.append(f"| {pair} | {count} |")
    lines.extend(["", "## Review Rows", "", "| Sample | Truth | Candidate | Category | Truth Area | Candidate Area | Rel Error | Faces |", "|--------|-------|-----------|----------|------------|----------------|-----------|-------|"])
    for row in audit["rows"]:
        lines.append(
            f"| {row['sample']} | {row['truth_macro']} | {row['candidate_macro']} | "
            f"{row['review_category']} | {row['truth_area']:.2f} | {row['candidate_area']:.2f} | "
            f"{row['relative_error']:.4f} | {' '.join(str(item) for item in row['candidate_faces'])} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_policy_simulation(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# v5.19 Semantic Policy Simulation",
        "",
        "| Policy | Baseline Misses | Recovered | Remaining Misses |",
        "|--------|-----------------|-----------|------------------|",
    ]
    for result in audit["policy_simulations"].values():
        lines.append(
            f"| {result['policy']} | {result['baseline_miss_count']} | "
            f"{result['recovered_count']} | {result['remaining_miss_count']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def decide_next_step(audit: dict[str, Any]) -> tuple[str, str]:
    summary = audit["summary"]
    category_counts = summary.get("by_category", {})
    cross_count = int(summary.get("cross_macro_compatible_count", 0))
    if cross_count <= 0:
        return "return_to_proposal_generation_repair", "No cross-macro compatible misses remain to audit."
    hole_count = int(category_counts.get("hole_subfeature_vs_composite_hole", 0))
    recess_count = int(category_counts.get("rectangular_recess_slot_vs_pocket", 0))
    transition_count = int(category_counts.get("transition_vs_nontransition_conflict", 0))
    rib_count = int(category_counts.get("thinwall_rib_vs_boss_or_web", 0))
    if hole_count + recess_count >= max(transition_count, rib_count) and (hole_count + recess_count) / cross_count >= 0.40:
        return (
            "define_safe_semantic_evaluation_policy",
            "Most cross-macro compatible misses are hole-composite or rectangular recess granularity conflicts. Define an explicit evaluation policy before changing proposer code.",
        )
    if transition_count >= max(hole_count + recess_count, rib_count) and transition_count / cross_count >= 0.30:
        return (
            "inspect_transition_truth_and_face_roles",
            "Transition conflicts dominate. Review fillet/chamfer truth rows and face roles before changing transition proposers.",
        )
    if rib_count >= max(hole_count + recess_count, transition_count) and rib_count / cross_count >= 0.30:
        return (
            "inspect_rib_boss_semantic_boundary",
            "Rib/boss conflicts dominate. Review whether the truth should mark these as rib, rectangular_boss, or context/background.",
        )
    return (
        "manual_semantic_review_before_code_changes",
        "Cross-macro compatible misses are heterogeneous. Manual semantic review should pick which policy is valid for the paper.",
    )


def write_decision(path: Path, audit: dict[str, Any]) -> None:
    recommendation, reason = decide_next_step(audit)
    lines = [
        "# v5.19 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Next Action",
        "",
        "If the recommendation is `define_safe_semantic_evaluation_policy`, the next implementation should add a separate evaluation-only semantic matching policy and compare strict vs policy-aware metrics. It must not change the inference output labels.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-macro compatible proposal misses.")
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(load_json(args.atlas))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "cross_macro_semantic_audit.json", audit)
    write_csv(args.output_dir / "cross_macro_semantic_audit.csv", audit["rows"])
    write_markdown(args.output_dir / "cross_macro_semantic_audit.md", audit)
    write_policy_simulation(args.output_dir / "semantic_policy_simulation.md", audit)
    write_decision(args.output_dir / "v5_19_decision.md", audit)
    print(json.dumps({"summary": audit["summary"], "decision": decide_next_step(audit)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
