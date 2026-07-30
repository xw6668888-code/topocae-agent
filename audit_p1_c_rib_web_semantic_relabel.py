from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GRAPH_ROOT = ROOT / "output_generation_p1_c_clean_context_7200" / "graphs"
RIB_HOSTS = {"l_corner", "rect_cavity", "circular_cavity"}
WEB_HOSTS = {"two_supports"}


def main() -> int:
    graph_paths = sorted(GRAPH_ROOT.glob("*.json"))
    wall_sample_count = 0
    wall_instance_count: Counter[str] = Counter()
    compatible_count: Counter[str] = Counter()
    incompatible_count: Counter[str] = Counter()
    semantic_relabel_count: Counter[str] = Counter()
    host_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    source_scenario_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    source_macro_to_target: dict[str, Counter[str]] = defaultdict(Counter)
    split_by_incompatible: Counter[str] = Counter()
    examples = []
    for graph_path in graph_paths:
        graph = _read_json(graph_path)
        feature_sets = [
            record for record in graph.get("feature_sets", [])
            if str(record.get("macro_feature")) in {"rib", "web"}
        ]
        if not feature_sets:
            continue
        wall_sample_count += 1
        if len(feature_sets) != 1:
            raise AssertionError(f"expected one supported wall per sample: {graph_path}")
        target = str(feature_sets[0]["macro_feature"])
        parameters = graph.get("generation", {}).get("parameters", {})
        host = str(parameters.get("host_family", "unknown"))
        split = str(parameters.get("sampling_metadata", {}).get("split", "unknown"))
        supported = parameters.get("supported_wall", {}).get("source_parameters", {})
        relabel = supported.get("semantic_relabel", {})
        source_parameters = supported.get("source_parameters", {})
        source_scenario = str(
            supported.get("source_engineering_scenario")
            or source_parameters.get("scenario_key")
            or "unknown"
        )
        source_macro = str(
            relabel.get("source")
            or source_parameters.get("macro_feature")
            or ("web" if source_scenario.startswith("web_") else "rib" if source_scenario.startswith("rib_") else "unknown")
        )
        relabel_target = str(relabel.get("target") or target)
        compatible = host in (RIB_HOSTS if target == "rib" else WEB_HOSTS)
        wall_instance_count[target] += 1
        host_by_label[target][host] += 1
        source_scenario_by_label[source_scenario][target] += 1
        source_macro_to_target[source_macro][target] += 1
        if source_macro != target or relabel_target != target:
            semantic_relabel_count[target] += 1
        if compatible:
            compatible_count[target] += 1
        else:
            incompatible_count[target] += 1
            split_by_incompatible[split] += 1
            if len(examples) < 24:
                examples.append({
                    "sample_id": graph.get("sample_id"),
                    "target_label": target,
                    "host_family": host,
                    "source_macro": source_macro,
                    "source_scenario": source_scenario,
                    "split": split,
                    "graph_path": str(graph_path.resolve()),
                })
    ambiguous_scenarios = {
        scenario: dict(counts)
        for scenario, counts in sorted(source_scenario_by_label.items())
        if len([count for count in counts.values() if count > 0]) > 1
    }
    result = {
        "schema_version": "topocae_p1_c_rib_web_semantic_relabel_audit_1",
        "graph_count": len(graph_paths),
        "supported_wall_sample_count": wall_sample_count,
        "wall_instance_count": dict(wall_instance_count),
        "goal_concept_compatible_count": dict(compatible_count),
        "goal_concept_incompatible_count": dict(incompatible_count),
        "source_macro_relabel_count": dict(semantic_relabel_count),
        "host_by_target_label": {key: dict(value) for key, value in sorted(host_by_label.items())},
        "source_macro_to_target_label": {key: dict(value) for key, value in sorted(source_macro_to_target.items())},
        "ambiguous_source_scenarios_labeled_as_both_classes": ambiguous_scenarios,
        "incompatible_split_distribution": dict(split_by_incompatible),
        "incompatible_examples": examples,
        "automatic_pass": sum(incompatible_count.values()) == 0 and sum(semantic_relabel_count.values()) == 0,
        "contracts": {
            "step_geometry_read": False,
            "graph_generation_metadata_only": True,
            "real_dev_used": False,
            "holdout_used": False,
            "test_part_s3_accessed": False,
        },
    }
    destination = ROOT / "reports" / "p1_c_iterations_v1" / "rib_web_semantic_relabel_audit.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "automatic_pass": result["automatic_pass"],
        "wall_instance_count": result["wall_instance_count"],
        "compatible": result["goal_concept_compatible_count"],
        "incompatible": result["goal_concept_incompatible_count"],
        "relabel": result["source_macro_relabel_count"],
        "ambiguous_scenario_count": len(ambiguous_scenarios),
    }, ensure_ascii=False))
    return 0 if result["automatic_pass"] else 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
