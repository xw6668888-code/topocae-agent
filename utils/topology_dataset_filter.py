from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def filter_topology_valid_samples(source_root: Path, target_root: Path) -> dict:
    source_root = Path(source_root)
    target_root = Path(target_root)
    audit_path = source_root / "reports" / "topology_rule_audit.json"
    audit = load_json(audit_path)
    invalid_graph_paths = {
        Path(sample["graph_path"]).resolve()
        for sample in audit.get("invalid_samples", [])
        if sample.get("graph_path")
    }

    prepare_target_root(target_root)
    graph_files = sorted((source_root / "graphs").glob("*.json"))
    accepted = []
    rejected = []
    for graph_path in graph_files:
        if graph_path.resolve() in invalid_graph_paths:
            rejected.append(graph_path.stem)
            continue
        copy_sample(source_root, target_root, graph_path)
        accepted.append(graph_path.stem)

    copy_tree_if_exists(source_root / "label_maps", target_root / "label_maps")
    copy_dataset_meta_if_exists(source_root, target_root)
    report = {
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "source_sample_count": len(graph_files),
        "accepted_sample_count": len(accepted),
        "rejected_sample_count": len(rejected),
        "accepted_samples": accepted,
        "rejected_samples": rejected,
        "topology_audit_path": str(audit_path.resolve()),
    }
    write_json(target_root / "reports" / "topology_filter_report.json", report)
    return report


def prepare_target_root(target_root: Path) -> None:
    for subdir in ("graphs", "steps", "reports", "label_maps", "training"):
        path = target_root / subdir
        if path.exists():
            shutil.rmtree(path)
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "graphs").mkdir(parents=True, exist_ok=True)
    (target_root / "steps").mkdir(parents=True, exist_ok=True)
    (target_root / "reports").mkdir(parents=True, exist_ok=True)


def copy_sample(source_root: Path, target_root: Path, graph_path: Path) -> None:
    shutil.copy2(graph_path, target_root / "graphs" / graph_path.name)
    step_path = source_root / "steps" / f"{graph_path.stem}.step"
    if step_path.exists():
        shutil.copy2(step_path, target_root / "steps" / step_path.name)


def copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)


def copy_dataset_meta_if_exists(source_root: Path, target_root: Path) -> None:
    meta_path = source_root / "dataset_meta.json"
    if not meta_path.exists():
        return
    meta = load_json(meta_path)
    meta["topology_filtered_from"] = str(source_root.resolve())
    write_json(target_root / "dataset_meta.json", meta)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy only topology-valid generated samples into a clean dataset root.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = filter_topology_valid_samples(args.source_root, args.target_root)
    print(
        "topology filtered dataset written: "
        f"{args.target_root} accepted={report['accepted_sample_count']} rejected={report['rejected_sample_count']}"
    )


if __name__ == "__main__":
    main()
