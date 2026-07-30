from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.colored_step_rib_truth import (
    apply_colored_rib_truth_to_template,
    build_colored_truth_from_step,
)


STEP_SUFFIXES = (".stp", ".step")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_colored_truth_by_sample(
    *,
    template: dict[str, Any],
    colored_step_dir: Path,
    graph_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    colored_truth_by_sample: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for sample in template.get("samples", []):
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            continue
        step_path = find_step_path(colored_step_dir, sample_id)
        graph_path = resolve_graph_path(sample, graph_dir)
        if step_path is None:
            missing.append({"sample_id": sample_id, "missing": "colored_step"})
            continue
        if graph_path is None or not graph_path.exists():
            missing.append({"sample_id": sample_id, "missing": "graph", "graph_path": str(graph_path or "")})
            continue
        graph_data = load_json(graph_path)
        truth = build_colored_truth_from_step(step_path, graph_data)
        truth["colored_step_path"] = str(step_path)
        truth["graph_path"] = str(graph_path)
        colored_truth_by_sample[sample_id] = truth
    return colored_truth_by_sample, missing


def find_step_path(colored_step_dir: Path, sample_id: str) -> Path | None:
    for suffix in STEP_SUFFIXES:
        candidate = colored_step_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def resolve_graph_path(sample: dict[str, Any], graph_dir: Path | None) -> Path | None:
    if graph_dir is not None:
        candidate = graph_dir / f"{sample.get('sample_id')}.json"
        if candidate.exists():
            return candidate
    graph_path = sample.get("graph_path")
    if graph_path:
        return Path(graph_path)
    return None


def write_markdown_report(path: Path, diagnostics: dict[str, Any], missing: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v5.8-B Colored Rib Truth Report",
        "",
        "Rule: `Medium Steel` is background; every other face color is treated as a rib/web instance. Different non-background colors become different rib instance ids within each sample.",
        "",
        "## Summary",
        "",
        "| Sample | Candidates | Appended | True filled | False filled | Null | Colored rib faces | Overrides | Face count match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sample in diagnostics.get("samples", []):
        lines.append(
            "| {sample_id} | {candidate_count} | {appended} | {filled_true} | {filled_false} | {left_null} | {rib_count} | {override_count} | {match} |".format(
                sample_id=sample.get("sample_id", ""),
                candidate_count=sample.get("candidate_count", 0),
                appended=sample.get("appended_colored_rib_candidates", 0),
                filled_true=sample.get("filled_true", 0),
                filled_false=sample.get("filled_false", 0),
                left_null=sample.get("left_null", 0),
                rib_count=len(sample.get("colored_rib_faces", [])),
                override_count=sample.get("colored_override_count", ""),
                match=sample.get("face_count_matches_graph", ""),
            )
        )
    if missing:
        lines.extend(["", "## Missing Inputs", ""])
        for item in missing:
            lines.append(f"- `{item.get('sample_id')}`: missing `{item.get('missing')}` {item.get('graph_path', '')}")
    problem_samples = [
        sample
        for sample in diagnostics.get("samples", [])
        if sample.get("left_null", 0)
        or sample.get("unmatched_colored_faces")
        or sample.get("ambiguous_colored_faces")
        or sample.get("face_count_matches_graph") is False
    ]
    if problem_samples:
        lines.extend(["", "## Items To Inspect", ""])
        for sample in problem_samples:
            lines.append(
                f"- `{sample.get('sample_id')}`: null={sample.get('left_null', 0)}, "
                f"face_count_match={sample.get('face_count_matches_graph')}, "
                f"unmatched={len(sample.get('unmatched_colored_faces', []))}, "
                f"ambiguous={len(sample.get('ambiguous_colored_faces', []))}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply colored STEP rib face truth to a rib face review template.")
    parser.add_argument("--template-json", type=Path, required=True)
    parser.add_argument("--colored-step-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = load_json(args.template_json)
    colored_truth_by_sample, missing = build_colored_truth_by_sample(
        template=template,
        colored_step_dir=args.colored_step_dir,
        graph_dir=args.graph_dir,
    )
    reviewed, diagnostics = apply_colored_rib_truth_to_template(template, colored_truth_by_sample)
    diagnostics["missing_inputs"] = missing
    diagnostics["colored_step_dir"] = str(args.colored_step_dir)
    write_json(args.output_json, reviewed)
    write_json(args.diagnostics_json, diagnostics)
    write_markdown_report(args.output_md, diagnostics, missing)
    print(f"Reviewed template written: {args.output_json}")
    print(f"Diagnostics written: {args.diagnostics_json}")
    print(f"Markdown report written: {args.output_md}")


if __name__ == "__main__":
    main()


