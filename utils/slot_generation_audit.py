from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


SLOT_MACROS = {"slot_u", "slot_v", "slot_o"}


def audit_slot_generation(graphs_dir: Path) -> dict:
    graph_paths = sorted(Path(graphs_dir).glob("*.json"))
    slot_counts: dict[str, int] = {}
    path_kind_counts: dict[str, int] = {}
    slot_node_counts: list[int] = []
    empty_slot_feature_sets = []
    examples: list[dict] = []

    for graph_path in graph_paths:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        for feature_set in graph.get("feature_sets", []):
            macro = feature_set.get("macro_feature")
            if macro not in SLOT_MACROS:
                continue
            raw_parameters = feature_set.get("raw_parameters", {})
            geometry_parameters = feature_set.get("geometry_parameters", {})
            path_kind = raw_parameters.get("path_curve_kind", geometry_parameters.get("path_curve_kind", "unknown"))
            node_count = len(feature_set.get("node_indices", []))

            slot_counts[macro] = slot_counts.get(macro, 0) + 1
            path_kind_counts[path_kind] = path_kind_counts.get(path_kind, 0) + 1
            slot_node_counts.append(node_count)
            if node_count == 0:
                empty_slot_feature_sets.append({
                    "graph": graph_path.name,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                })
            if len(examples) < 40:
                examples.append({
                    "graph": graph_path.name,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                    "path_curve_kind": path_kind,
                    "node_count": node_count,
                })

    return {
        "graphs_dir": str(graphs_dir),
        "graph_count": len(graph_paths),
        "slot_total": sum(slot_counts.values()),
        "slot_counts": dict(sorted(slot_counts.items())),
        "path_kind_counts": dict(sorted(path_kind_counts.items())),
        "empty_slot_feature_set_count": len(empty_slot_feature_sets),
        "empty_slot_feature_sets": empty_slot_feature_sets,
        "slot_node_count_summary": _node_count_summary(slot_node_counts),
        "examples": examples,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _node_count_summary(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit swept slot generation in exported graph JSON files.")
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_slot_generation(args.graphs_dir)
    write_json(args.output_path, report)
    print(f"Slot generation audit written: {args.output_path}")
    print(f"slot_total={report['slot_total']}; path_kind_counts={report['path_kind_counts']}; empty_slot_feature_set_count={report['empty_slot_feature_set_count']}")


if __name__ == "__main__":
    main()
