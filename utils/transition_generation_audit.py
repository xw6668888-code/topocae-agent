from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


TRANSITION_MACROS = {"fillet", "chamfer"}
FORBIDDEN_CONTEXT_KEYS = {
    "transition_context",
    "edge_context",
    "source_context",
    "internal_edge_context",
}


def audit_transition_generation(graphs_dir: Path, max_size_ratio: float = 0.08) -> dict:
    graph_paths = sorted(Path(graphs_dir).glob("*.json"))
    macro_counts: dict[str, int] = {}
    curve_type_counts: dict[str, int] = {}
    node_counts: list[int] = []
    transition_sizes: list[float] = []
    empty_feature_sets: list[dict] = []
    oversized_transitions: list[dict] = []
    forbidden_context_keys: list[dict] = []
    examples: list[dict] = []

    for graph_path in graph_paths:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        size_limit = _transition_size_limit(graph.get("stock_profile", {}), max_size_ratio)
        for feature_set in graph.get("feature_sets", []):
            macro = feature_set.get("macro_feature")
            if macro not in TRANSITION_MACROS:
                continue

            raw_parameters = feature_set.get("raw_parameters", {})
            geometry_parameters = feature_set.get("geometry_parameters", {})
            node_count = len(feature_set.get("node_indices", []))
            curve_type = str(raw_parameters.get("curve_type", geometry_parameters.get("curve_type", "unknown")))
            size_value = _transition_size_value(macro, raw_parameters, geometry_parameters)

            macro_counts[macro] = macro_counts.get(macro, 0) + 1
            curve_type_counts[curve_type] = curve_type_counts.get(curve_type, 0) + 1
            node_counts.append(node_count)
            if size_value is not None:
                transition_sizes.append(size_value)
                if size_value > size_limit:
                    oversized_transitions.append({
                        "graph": graph_path.name,
                        "macro_feature": macro,
                        "instance_id": feature_set.get("instance_id"),
                        "size": size_value,
                        "limit": size_limit,
                    })
            if node_count == 0:
                empty_feature_sets.append({
                    "graph": graph_path.name,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                })

            for key in sorted(FORBIDDEN_CONTEXT_KEYS.intersection(raw_parameters.keys())):
                forbidden_context_keys.append({
                    "graph": graph_path.name,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                    "key": key,
                })

            if len(examples) < 40:
                examples.append({
                    "graph": graph_path.name,
                    "macro_feature": macro,
                    "instance_id": feature_set.get("instance_id"),
                    "node_count": node_count,
                    "curve_type": curve_type,
                    "size": size_value,
                })

    return {
        "graphs_dir": str(graphs_dir),
        "graph_count": len(graph_paths),
        "transition_total": sum(macro_counts.values()),
        "macro_counts": dict(sorted(macro_counts.items())),
        "curve_type_counts": dict(sorted(curve_type_counts.items())),
        "empty_transition_feature_set_count": len(empty_feature_sets),
        "empty_transition_feature_sets": empty_feature_sets,
        "oversized_transition_count": len(oversized_transitions),
        "oversized_transitions": oversized_transitions,
        "forbidden_context_key_count": len(forbidden_context_keys),
        "forbidden_context_keys": forbidden_context_keys,
        "transition_node_count_summary": _summary(node_counts),
        "transition_size_summary": _summary(transition_sizes),
        "examples": examples,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _transition_size_value(macro: str, raw_parameters: dict, geometry_parameters: dict) -> float | None:
    key = "radius" if macro == "fillet" else "distance"
    value = raw_parameters.get(key, geometry_parameters.get(key))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transition_size_limit(stock_profile: dict, max_size_ratio: float) -> float:
    dimensions = stock_profile.get("dimensions", {}) if isinstance(stock_profile, dict) else {}
    values = []
    for key in ("L", "W", "H", "radius"):
        try:
            value = float(dimensions.get(key))
        except (TypeError, ValueError):
            continue
        if key == "radius":
            value *= 2.0
        if value > 0.0:
            values.append(value)
    if not values:
        return float("inf")
    return min(values) * float(max_size_ratio)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit fillet/chamfer generation in exported graph JSON files.")
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-size-ratio", type=float, default=0.08)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_transition_generation(args.graphs_dir, max_size_ratio=args.max_size_ratio)
    write_json(args.output_path, report)
    print(f"Transition generation audit written: {args.output_path}")
    print(
        "transition_total="
        f"{report['transition_total']}; macro_counts={report['macro_counts']}; "
        f"empty={report['empty_transition_feature_set_count']}; "
        f"oversized={report['oversized_transition_count']}; "
        f"forbidden_context={report['forbidden_context_key_count']}"
    )


if __name__ == "__main__":
    main()
