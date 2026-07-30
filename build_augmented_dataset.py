from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.topology_rule_audit import audit_graph_directory
from utils.training_export import NODE_FEATURE_MODE_ROTATION_INVARIANT, export_training_package


LABEL_MAP_FILES = [
    "primitive_surface_label_map.json",
    "macro_feature_label_map.json",
    "face_semantic_subtype_label_map.json",
    "idealization_tag_label_map.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an augmented v4 dataset by merging a full clean distribution with focused hard-case samples."
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--hardcase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-prefix", default="full")
    parser.add_argument("--hardcase-prefix", default="hardcase")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--node-feature-mode", default=NODE_FEATURE_MODE_ROTATION_INVARIANT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild output-dir. The path name must contain 'v4_8' or 'augmented'.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_same_label_maps(full_root: Path, hardcase_root: Path) -> None:
    mismatches = []
    for filename in LABEL_MAP_FILES:
        full_path = full_root / "label_maps" / filename
        hardcase_path = hardcase_root / "label_maps" / filename
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        if not hardcase_path.exists():
            raise FileNotFoundError(hardcase_path)
        if file_sha256(full_path) != file_sha256(hardcase_path):
            mismatches.append(filename)
    if mismatches:
        raise ValueError(f"Label map files differ and cannot be merged safely: {mismatches}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to rebuild it.")
        lowered = output_dir.name.lower()
        if "v4_8" not in lowered and "augmented" not in lowered:
            raise ValueError(f"Refusing to overwrite a non-v4.8/non-augmented directory: {output_dir}")
        shutil.rmtree(output_dir)
    for subdir in ["graphs", "steps", "label_maps", "reports", "manifests", "splits", "training"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def copy_label_maps(source_root: Path, output_dir: Path) -> None:
    for filename in LABEL_MAP_FILES:
        shutil.copy2(source_root / "label_maps" / filename, output_dir / "label_maps" / filename)


def collect_graph_files(dataset_root: Path) -> list[Path]:
    graphs_dir = dataset_root / "graphs"
    files = sorted(graphs_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No graph files found under: {graphs_dir}")
    return files


def rewrite_step_validation_path(graph_data: dict[str, Any], step_path: Path) -> None:
    quality = graph_data.setdefault("generation_quality", {})
    step_validation = quality.setdefault("step_write_validation", {})
    step_validation["step_path"] = str(step_path.resolve())


def copy_dataset_samples(
    source_root: Path,
    output_dir: Path,
    *,
    prefix: str,
    source_role: str,
) -> list[dict[str, Any]]:
    records = []
    for index, graph_path in enumerate(collect_graph_files(source_root)):
        source_sample_id = graph_path.stem
        new_sample_id = f"{prefix}_{index:04d}"
        source_step = source_root / "steps" / f"{source_sample_id}.step"
        dest_graph = output_dir / "graphs" / f"{new_sample_id}.json"
        dest_step = output_dir / "steps" / f"{new_sample_id}.step"

        graph_data = load_json(graph_path)
        graph_data["merged_dataset_source"] = {
            "source_role": source_role,
            "source_root": str(source_root.resolve()),
            "source_sample_id": source_sample_id,
            "merged_sample_id": new_sample_id,
        }
        if source_step.exists():
            shutil.copy2(source_step, dest_step)
            rewrite_step_validation_path(graph_data, dest_step)
        write_json(dest_graph, graph_data)

        macro_counts = Counter(
            feature_set.get("macro_feature", "unknown")
            for feature_set in graph_data.get("feature_sets", [])
        )
        records.append(
            {
                "sample_id": new_sample_id,
                "source_role": source_role,
                "source_sample_id": source_sample_id,
                "source_graph_path": str(graph_path.resolve()),
                "source_step_path": str(source_step.resolve()) if source_step.exists() else None,
                "merged_graph_path": str(dest_graph.resolve()),
                "merged_step_path": str(dest_step.resolve()) if dest_step.exists() else None,
                "node_count": len(graph_data.get("nodes", [])),
                "edge_count": len(graph_data.get("edges", [])),
                "feature_set_count": len(graph_data.get("feature_sets", [])),
                "macro_counts": dict(sorted(macro_counts.items())),
                "stock_type": graph_data.get("stock_profile", {}).get("stock_type"),
            }
        )
    return records


def aggregate_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        counter.update(record.get("macro_counts", {}))
    return dict(sorted(counter.items()))


def write_augmented_meta(
    output_dir: Path,
    *,
    full_root: Path,
    hardcase_root: Path,
    full_records: list[dict[str, Any]],
    hardcase_records: list[dict[str, Any]],
    seed: int,
    node_feature_mode: str,
) -> None:
    full_meta = load_json(full_root / "dataset_meta.json")
    hardcase_meta = load_json(hardcase_root / "dataset_meta.json")
    records = full_records + hardcase_records
    full_counts = aggregate_counts(full_records)
    hardcase_counts = aggregate_counts(hardcase_records)
    combined_counts = aggregate_counts(records)
    target_macros = sorted(set(full_counts) | set(hardcase_counts))

    dataset_meta = {
        "dataset_name": "cae_feature_dataset_v4_8_augmented_full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_dir.resolve()),
        "config_path": "build_augmented_dataset.py",
        "num_samples_requested": len(records),
        "actual_samples_written": len(records),
        "random_seed": seed,
        "source_datasets": {
            "full": {
                "root": str(full_root.resolve()),
                "sample_count": len(full_records),
                "source_random_seed": full_meta.get("random_seed"),
                "achieved_instance_counts": full_counts,
            },
            "hardcase": {
                "root": str(hardcase_root.resolve()),
                "sample_count": len(hardcase_records),
                "source_random_seed": hardcase_meta.get("random_seed"),
                "achieved_instance_counts": hardcase_counts,
            },
        },
        "coverage_summary": {
            "enabled": True,
            "ablation_mode": "augmented_full_plus_focused_hardcase",
            "actual_samples_written": len(records),
            "target_macro_features": target_macros,
            "achieved_instance_counts": combined_counts,
            "source_mix": {
                "full_samples": len(full_records),
                "hardcase_samples": len(hardcase_records),
                "hardcase_fraction": len(hardcase_records) / max(len(records), 1),
            },
        },
        "macro_feature_ontology_path": str((Path(__file__).resolve().parent / "CAE_feature_ontology.yaml").resolve()),
        "semantic_subtype_ontology_path": str((Path(__file__).resolve().parent / "face_semantic_subtype_ontology.yaml").resolve()),
        "primitive_surface_label_map_path": str((output_dir / "label_maps" / "primitive_surface_label_map.json").resolve()),
        "macro_feature_label_map_path": str((output_dir / "label_maps" / "macro_feature_label_map.json").resolve()),
        "semantic_subtype_label_map_path": str((output_dir / "label_maps" / "face_semantic_subtype_label_map.json").resolve()),
        "idealization_tag_label_map_path": str((output_dir / "label_maps" / "idealization_tag_label_map.json").resolve()),
        "semantic_subtype_connected": True,
        "derived_cae_labels_connected": True,
        "graph_node_fields": full_meta.get("graph_node_fields", []),
        "feature_set_fields": full_meta.get("feature_set_fields", []),
        "exports": {
            "write_step": True,
            "write_graph": True,
            "build_training_package": True,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "llm_round_ndigits": 4,
            "gnn_round_ndigits": 6,
            "node_feature_mode": node_feature_mode,
        },
    }
    write_json(output_dir / "dataset_meta.json", dataset_meta)
    write_json(
        output_dir / "reports" / "v4_8_augmented_source_mix.json",
        {
            "full_sample_count": len(full_records),
            "hardcase_sample_count": len(hardcase_records),
            "total_sample_count": len(records),
            "full_macro_counts": full_counts,
            "hardcase_macro_counts": hardcase_counts,
            "combined_macro_counts": combined_counts,
        },
    )
    write_json(
        output_dir / "manifests" / "augmented_sample_index.json",
        {
            "records": records,
        },
    )


def build_augmented_dataset(args: argparse.Namespace) -> dict[str, Any]:
    full_root = args.full_root.resolve()
    hardcase_root = args.hardcase_root.resolve()
    output_dir = args.output_dir.resolve()

    assert_same_label_maps(full_root, hardcase_root)
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    copy_label_maps(full_root, output_dir)

    full_records = copy_dataset_samples(
        full_root,
        output_dir,
        prefix=args.full_prefix,
        source_role="full_distribution",
    )
    hardcase_records = copy_dataset_samples(
        hardcase_root,
        output_dir,
        prefix=args.hardcase_prefix,
        source_role="focused_hardcase",
    )
    write_augmented_meta(
        output_dir,
        full_root=full_root,
        hardcase_root=hardcase_root,
        full_records=full_records,
        hardcase_records=hardcase_records,
        seed=args.seed,
        node_feature_mode=args.node_feature_mode,
    )
    audit_graph_directory(output_dir / "graphs", output_dir / "reports" / "topology_rule_audit.json")

    split_total = args.train_ratio + args.val_ratio + args.test_ratio
    export_training_package(
        output_dir=str(output_dir),
        split_ratios={
            "train": args.train_ratio / split_total,
            "val": args.val_ratio / split_total,
            "test": args.test_ratio / split_total,
        },
        seed=args.seed,
        node_feature_mode=args.node_feature_mode,
    )

    return {
        "output_dir": str(output_dir),
        "full_samples": len(full_records),
        "hardcase_samples": len(hardcase_records),
        "total_samples": len(full_records) + len(hardcase_records),
        "combined_macro_counts": aggregate_counts(full_records + hardcase_records),
    }


def main() -> None:
    args = parse_args()
    summary = build_augmented_dataset(args)
    print("augmented_dataset_complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
