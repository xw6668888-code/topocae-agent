from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.topology_rules import validate_feature_topology


def audit_graph_topology(graph_data: dict, graph_path: str | None = None) -> dict:
    invalid_feature_sets = []
    feature_sets = list(graph_data.get("feature_sets", []))

    for feature_set in feature_sets:
        macro_feature = feature_set.get("macro_feature", "unknown_feature")
        face_indices = [int(idx) for idx in feature_set.get("node_indices", [])]
        validation = validate_feature_topology(macro_feature, face_indices, graph_data)
        if validation["valid"]:
            continue
        invalid_feature_sets.append({
            "instance_id": feature_set.get("instance_id"),
            "macro_feature": macro_feature,
            "normalized_family": validation["normalized_family"],
            "face_indices": face_indices,
            "face_count": len(face_indices),
            "reasons": validation["reasons"],
            "evidence": validation["evidence"],
        })

    return {
        "sample_id": Path(graph_path).stem if graph_path else graph_data.get("sample_id"),
        "graph_path": graph_path,
        "feature_set_count": len(feature_sets),
        "invalid_feature_set_count": len(invalid_feature_sets),
        "invalid_feature_sets": invalid_feature_sets,
    }


def aggregate_topology_audits(sample_reports: list[dict]) -> dict:
    invalid_by_macro = Counter()
    reason_counts = Counter()
    invalid_samples = []

    for report in sample_reports:
        if int(report.get("invalid_feature_set_count", 0)) > 0:
            invalid_samples.append(report)
        for invalid in report.get("invalid_feature_sets", []):
            invalid_by_macro[invalid.get("macro_feature", "unknown_feature")] += 1
            for reason in invalid.get("reasons", []):
                reason_counts[reason] += 1

    feature_set_count = sum(int(report.get("feature_set_count", 0)) for report in sample_reports)
    invalid_feature_set_count = sum(int(report.get("invalid_feature_set_count", 0)) for report in sample_reports)
    return {
        "sample_count": len(sample_reports),
        "feature_set_count": feature_set_count,
        "invalid_feature_set_count": invalid_feature_set_count,
        "invalid_rate": invalid_feature_set_count / max(feature_set_count, 1),
        "invalid_by_macro": dict(sorted(invalid_by_macro.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "invalid_sample_count": len(invalid_samples),
        "invalid_samples": invalid_samples,
    }


def collect_graph_files(graphs_dir: Path) -> list[Path]:
    return sorted(path for path in graphs_dir.rglob("*.json") if path.is_file())


def audit_graph_directory(graphs_dir: Path, output_path: Path | None = None) -> dict:
    sample_reports = [
        audit_graph_topology(load_json(graph_path), graph_path=str(graph_path.resolve()))
        for graph_path in collect_graph_files(graphs_dir)
    ]
    aggregate = aggregate_topology_audits(sample_reports)
    if output_path is not None:
        write_json(output_path, aggregate)
    return aggregate


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated graph feature_sets with B-Rep topology legality rules.")
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_graph_directory(args.graphs_dir, args.output_path)
    print(f"topology rule audit written: {args.output_path}")


if __name__ == "__main__":
    main()
