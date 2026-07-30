from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict, deque
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

NODE_LABEL_FIELDS = [
    "macro_feature",
    "macro_feature_id",
    "semantic_label",
    "semantic_subtype",
    "semantic_subtype_id",
    "derived_macro_features",
    "derived_macro_feature_ids",
    "idealization_tags",
    "idealization_tag_ids",
    "instance_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-extract geometry and edge attributes from existing STEP files while preserving "
            "the labels and feature_sets from an existing generated graph dataset."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("CAE_feature_ontology.yaml"))
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--node-feature-mode", default=NODE_FEATURE_MODE_ROTATION_INVARIANT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--export-training-package", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild output-dir. The path name must contain 'looprole' or 'upgraded'.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to rebuild it.")
        lowered = output_dir.name.lower()
        if "looprole" not in lowered and "upgraded" not in lowered:
            raise ValueError(f"Refusing to overwrite non-looprole/non-upgraded directory: {output_dir}")
        shutil.rmtree(output_dir)
    for subdir in ["graphs", "steps", "label_maps", "reports", "manifests", "splits", "training"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def edge_counter(edges: list[list[int]]) -> Counter[tuple[int, int]]:
    return Counter((int(edge[0]), int(edge[1])) for edge in edges)


def align_refreshed_edge_attributes(
    old_edges: list[list[int]],
    refreshed_edges: list[list[int]],
    refreshed_edge_attributes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(old_edges) != len(refreshed_edges):
        raise ValueError(f"edge count mismatch: old={len(old_edges)} refreshed={len(refreshed_edges)}")
    if len(refreshed_edge_attributes) != len(refreshed_edges):
        raise ValueError(
            "refreshed edge_attributes count mismatch: "
            f"edges={len(refreshed_edges)} attrs={len(refreshed_edge_attributes)}"
        )
    if edge_counter(old_edges) != edge_counter(refreshed_edges):
        raise ValueError("edge topology mismatch between old and refreshed graph")

    by_edge: dict[tuple[int, int], deque[dict[str, Any]]] = defaultdict(deque)
    for edge, attr in zip(refreshed_edges, refreshed_edge_attributes):
        by_edge[(int(edge[0]), int(edge[1]))].append(dict(attr))
    return [by_edge[(int(edge[0]), int(edge[1]))].popleft() for edge in old_edges]


def merge_refreshed_graph(
    old_graph: dict[str, Any],
    refreshed_graph: dict[str, Any],
    step_path: str,
) -> dict[str, Any]:
    old_nodes = old_graph.get("nodes", [])
    refreshed_nodes = refreshed_graph.get("nodes", [])
    if len(old_nodes) != len(refreshed_nodes):
        raise ValueError(f"node count mismatch: old={len(old_nodes)} refreshed={len(refreshed_nodes)}")

    merged_nodes = []
    for old_node, refreshed_node in zip(old_nodes, refreshed_nodes):
        merged_node = dict(refreshed_node)
        for field in NODE_LABEL_FIELDS:
            if field in old_node:
                merged_node[field] = old_node[field]
        merged_nodes.append(merged_node)

    old_edges = [list(edge) for edge in old_graph.get("edges", [])]
    refreshed_edges = [list(edge) for edge in refreshed_graph.get("edges", [])]
    refreshed_edge_attributes = list(refreshed_graph.get("edge_attributes", []))
    edge_attributes = align_refreshed_edge_attributes(old_edges, refreshed_edges, refreshed_edge_attributes)

    merged = dict(old_graph)
    merged["nodes"] = merged_nodes
    merged["edges"] = old_edges
    merged["edge_attributes"] = edge_attributes
    merged["source_step_path"] = step_path
    merged["loop_role_upgrade"] = {
        "status": "merged",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "refreshed_node_count": len(refreshed_nodes),
        "refreshed_edge_count": len(refreshed_edges),
    }
    return merged


def collect_graph_files(source_root: Path, limit: int | None = None) -> list[Path]:
    graph_files = sorted((source_root / "graphs").glob("*.json"))
    if limit is not None:
        graph_files = graph_files[:limit]
    if not graph_files:
        raise RuntimeError(f"No graph JSON files found under: {source_root / 'graphs'}")
    return graph_files


def find_step_path(source_root: Path, sample_id: str) -> Path:
    for suffix in [".step", ".stp"]:
        path = source_root / "steps" / f"{sample_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(source_root / "steps" / f"{sample_id}.step")


def refresh_graph_from_step(step_path: Path, primitive_surface_index: dict[str, int]) -> dict[str, Any]:
    from OCC.Extend.DataExchange import read_step_file
    from core.AAGExtractor import extract_aag_graph

    shape = read_step_file(str(step_path))
    return extract_aag_graph(
        shape,
        face_infos=None,
        primitive_surface_index=primitive_surface_index,
        include_label_fields=False,
    )


def copy_label_maps(source_root: Path, output_dir: Path) -> None:
    for filename in LABEL_MAP_FILES:
        source = source_root / "label_maps" / filename
        if source.exists():
            shutil.copy2(source, output_dir / "label_maps" / filename)


def copy_step(step_path: Path, output_dir: Path) -> Path:
    dest = output_dir / "steps" / step_path.name
    shutil.copy2(step_path, dest)
    return dest


def write_dataset_meta(
    source_root: Path,
    output_dir: Path,
    *,
    sample_count: int,
    seed: int,
    node_feature_mode: str,
) -> None:
    source_meta_path = source_root / "dataset_meta.json"
    meta = load_json(source_meta_path) if source_meta_path.exists() else {}
    meta.update(
        {
            "dataset_name": f"{meta.get('dataset_name', source_root.name)}_looprole_edgeattr",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_dir.resolve()),
            "source_dataset_root": str(source_root.resolve()),
            "actual_samples_written": sample_count,
            "random_seed": seed,
            "edge_attribute_upgrade": {
                "enabled": True,
                "method": "reextract_geometry_from_step_preserve_labels",
                "preserved_fields": NODE_LABEL_FIELDS,
            },
            "exports": {
                **meta.get("exports", {}),
                "build_training_package": True,
                "node_feature_mode": node_feature_mode,
            },
        }
    )
    write_json(output_dir / "dataset_meta.json", meta)


def upgrade_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from utils.ontology import load_layer_index

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    copy_label_maps(source_root, output_dir)

    primitive_surface_index = load_layer_index(args.ontology, "primitive_surface")
    graph_files = collect_graph_files(source_root, args.limit)
    records = []
    failures = []

    for index, graph_path in enumerate(graph_files, start=1):
        sample_id = graph_path.stem
        try:
            old_graph = load_json(graph_path)
            source_step = find_step_path(source_root, sample_id)
            dest_step = copy_step(source_step, output_dir)
            refreshed_graph = refresh_graph_from_step(source_step, primitive_surface_index)
            merged_graph = merge_refreshed_graph(old_graph, refreshed_graph, str(dest_step.resolve()))
            dest_graph = output_dir / "graphs" / graph_path.name
            write_json(dest_graph, merged_graph)
            records.append(
                {
                    "sample_id": sample_id,
                    "source_graph_path": str(graph_path.resolve()),
                    "source_step_path": str(source_step.resolve()),
                    "graph_path": str(dest_graph.resolve()),
                    "step_path": str(dest_step.resolve()),
                    "node_count": len(merged_graph.get("nodes", [])),
                    "edge_count": len(merged_graph.get("edges", [])),
                }
            )
        except Exception as exc:  # pragma: no cover - batch diagnostics are written to manifest.
            failures.append({"sample_id": sample_id, "error": str(exc)})
        if index == 1 or index % 100 == 0:
            print(f"upgraded {index}/{len(graph_files)} samples")

    write_json(output_dir / "manifests" / "looprole_upgrade_manifest.json", {"records": records, "failures": failures})
    write_dataset_meta(
        source_root,
        output_dir,
        sample_count=len(records),
        seed=args.seed,
        node_feature_mode=args.node_feature_mode,
    )
    audit_graph_directory(output_dir / "graphs", output_dir / "reports" / "topology_rule_audit.json")

    if args.export_training_package:
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

    return {"output_dir": str(output_dir), "sample_count": len(records), "failure_count": len(failures)}


def main() -> None:
    args = parse_args()
    summary = upgrade_dataset(args)
    print("looprole_upgrade_complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
