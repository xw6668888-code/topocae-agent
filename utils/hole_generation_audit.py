from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


HOLE_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
}


def audit_hole_generation(graphs_dir: Path) -> dict:
    graph_paths = sorted(Path(graphs_dir).glob("*.json"))
    macro_counts: dict[str, int] = {}
    axis_counts: dict[str, int] = {}
    node_counts: list[int] = []
    internal_entry_risks: list[dict] = []

    for graph_path in graph_paths:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        stock_type = graph.get("stock_profile", {}).get("stock_type", "unknown")
        for feature_set in graph.get("feature_sets", []):
            macro = feature_set.get("macro_feature")
            if macro not in HOLE_MACROS:
                continue
            raw_parameters = feature_set.get("raw_parameters", {})
            geometry_parameters = feature_set.get("geometry_parameters", {})
            axis = str(raw_parameters.get("axis", geometry_parameters.get("axis", "z")))
            node_count = len(feature_set.get("node_indices", []))

            macro_counts[macro] = macro_counts.get(macro, 0) + 1
            axis_counts[axis] = axis_counts.get(axis, 0) + 1
            node_counts.append(node_count)
            if _is_internal_entry_risk(stock_type, axis):
                internal_entry_risks.append({
                    "graph": graph_path.name,
                    "stock_type": stock_type,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                    "axis": axis,
                    "node_count": node_count,
                })

    return {
        "graphs_dir": str(graphs_dir),
        "graph_count": len(graph_paths),
        "hole_total": sum(macro_counts.values()),
        "macro_counts": dict(sorted(macro_counts.items())),
        "axis_counts": dict(sorted(axis_counts.items())),
        "internal_entry_risk_count": len(internal_entry_risks),
        "internal_entry_risks": internal_entry_risks,
        "hole_node_count_summary": _node_count_summary(node_counts),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _is_internal_entry_risk(stock_type: str, axis: str) -> bool:
    return stock_type in {"cylinder", "composite"} and axis in {"x", "y"}


def _node_count_summary(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hole generation axis distribution and internal-entry risk.")
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_hole_generation(args.graphs_dir)
    write_json(args.output_path, report)
    print(f"Hole generation audit written: {args.output_path}")
    print(f"hole_total={report['hole_total']}; axis_counts={report['axis_counts']}; internal_entry_risk_count={report['internal_entry_risk_count']}")


if __name__ == "__main__":
    main()
