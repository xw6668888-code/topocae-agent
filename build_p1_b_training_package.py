from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.training_export_v2_1 import (
    GRAPH_V2_EDGE_FEATURE_SCHEMA,
    GRAPH_V2_NODE_FEATURE_SCHEMA,
    build_graph_v2_edge_feature_vector,
    build_graph_v2_node_feature_vector,
    export_graph_v2_training_record,
    fit_training_quantile_transform,
)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3"
OUTPUT = DATASET / "training_p1_b_v1"
MACRO_CLASSES = [
    "stock",
    "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole",
    "countersink_hole", "obround_hole", "slot_u", "slot_rect", "pocket_rect",
    "pocket_circular", "cylinder_boss", "rib", "web", "rectangular_boss", "fillet",
    "chamfer", "slot_v", "slot_o",
]


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"training package already exists and is non-empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_records = {
        split: _read_jsonl(DATASET / "splits" / f"{split}.jsonl")
        for split in ("train", "val", "development_test")
    }
    print("fitting Graph v2 clipping bounds on train only", flush=True)
    train_node_vectors = []
    train_edge_vectors = []
    for position, manifest in enumerate(split_records["train"]):
        graph = _load_graph(manifest)
        train_node_vectors.extend(build_graph_v2_node_feature_vector(node) for node in graph.get("nodes", []))
        train_edge_vectors.extend(build_graph_v2_edge_feature_vector(edge) for edge in graph.get("edge_attributes", []))
        if (position + 1) % 500 == 0:
            print(f"  fit read {position + 1}/{len(split_records['train'])}", flush=True)
    transform = fit_training_quantile_transform(
        train_node_vectors,
        train_edge_vectors,
        split_name="train",
        lower_quantile=0.005,
        upper_quantile=0.995,
        max_abs=32.0,
    )
    node_clipped = np.asarray(
        [[max(low, min(high, value)) for value, low, high in zip(row, transform.node_lower, transform.node_upper)] for row in train_node_vectors],
        dtype=np.float32,
    )
    edge_clipped = np.asarray(
        [[max(low, min(high, value)) for value, low, high in zip(row, transform.edge_lower, transform.edge_upper)] for row in train_edge_vectors],
        dtype=np.float32,
    )
    node_mean = node_clipped.mean(axis=0)
    node_std = node_clipped.std(axis=0)
    node_std[node_std < 1e-6] = 1.0
    edge_mean = edge_clipped.mean(axis=0)
    edge_std = edge_clipped.std(axis=0)
    edge_std[edge_std < 1e-6] = 1.0
    del train_node_vectors, train_edge_vectors, node_clipped, edge_clipped

    macro_to_id = {macro: index for index, macro in enumerate(MACRO_CLASSES)}
    output_counts = {}
    finite_audit = {"value_count": 0, "max_abs_normalized": 0.0, "all_finite": True}
    for split, manifests in split_records.items():
        packaged = []
        macro_face_counts = Counter()
        same_edge_counts = Counter()
        for position, manifest in enumerate(manifests):
            graph = _load_graph(manifest)
            exported = export_graph_v2_training_record(graph, quantile_transform=transform)
            node_features = (np.asarray(exported["node_features"], dtype=np.float32) - node_mean) / node_std
            edge_features = (np.asarray(exported["edge_features"], dtype=np.float32) - edge_mean) / edge_std
            finite_audit["value_count"] += int(node_features.size + edge_features.size)
            if node_features.size:
                finite_audit["max_abs_normalized"] = max(finite_audit["max_abs_normalized"], float(np.abs(node_features).max()))
                finite_audit["all_finite"] &= bool(np.isfinite(node_features).all())
            if edge_features.size:
                finite_audit["max_abs_normalized"] = max(finite_audit["max_abs_normalized"], float(np.abs(edge_features).max()))
                finite_audit["all_finite"] &= bool(np.isfinite(edge_features).all())
            nodes = graph.get("nodes", [])
            macro_ids = [macro_to_id[str(node.get("macro_feature", "stock"))] for node in nodes]
            semantic_ids = [int(node.get("semantic_label", 0) or 0) for node in nodes]
            primitive_ids = [int(node.get("primitive_surface_id", 0) or 0) for node in nodes]
            instance_ids = [int(node.get("instance_id", 0) or 0) for node in nodes]
            edge_labels, edge_mask = _same_instance_labels(graph.get("edges", []), instance_ids)
            macro_face_counts.update(macro_ids)
            same_edge_counts.update(
                "positive" if label == 1 else "negative"
                for label, mask in zip(edge_labels, edge_mask)
                if mask > 0
            )
            packaged.append({
                "sample_id": manifest["sample_id"],
                "source_graph_path": manifest["graph_path"],
                "base_geometry_family_id": manifest["base_geometry_family_id"],
                "derivation_kind": manifest["derivation_kind"],
                "node_features": torch.from_numpy(node_features),
                "edge_index": torch.tensor(graph.get("edges", []), dtype=torch.long).t().contiguous() if graph.get("edges") else torch.empty((2, 0), dtype=torch.long),
                "edge_features": torch.from_numpy(edge_features),
                "primitive_surface_ids": torch.tensor(primitive_ids, dtype=torch.long),
                "macro_targets": torch.tensor(macro_ids, dtype=torch.long),
                "semantic_targets": torch.tensor(semantic_ids, dtype=torch.long),
                "instance_ids": torch.tensor(instance_ids, dtype=torch.long),
                "edge_same_instance_targets": torch.tensor(edge_labels, dtype=torch.float32),
                "edge_supervision_mask": torch.tensor(edge_mask, dtype=torch.bool),
                "feature_sets": graph.get("feature_sets", []),
                "face_to_canonical_patch": graph.get("face_to_canonical_patch", {}),
            })
            if (position + 1) % 500 == 0:
                print(f"  package {split} {position + 1}/{len(manifests)}", flush=True)
        path = OUTPUT / f"{split}.pt"
        torch.save(packaged, path)
        output_counts[split] = {
            "sample_count": len(packaged),
            "macro_face_counts": {MACRO_CLASSES[index]: macro_face_counts[index] for index in range(len(MACRO_CLASSES))},
            "same_instance_edge_counts": dict(same_edge_counts),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    metadata = {
        "schema_version": "topocae_p1_b_training_package_1",
        "source_dataset": str(DATASET.resolve()),
        "source_dataset_meta_sha256": _sha256(DATASET / "dataset_meta.json"),
        "macro_classes": MACRO_CLASSES,
        "macro_to_compact_id": macro_to_id,
        "semantic_class_count": 63,
        "primitive_surface_class_count": 11,
        "node_feature_schema": GRAPH_V2_NODE_FEATURE_SCHEMA,
        "edge_feature_schema": GRAPH_V2_EDGE_FEATURE_SCHEMA,
        "quantile_transform": transform.to_dict(),
        "normalization": {
            "fitted_split": "train",
            "node_mean": node_mean.tolist(),
            "node_std": node_std.tolist(),
            "edge_mean": edge_mean.tolist(),
            "edge_std": edge_std.tolist(),
        },
        "splits": output_counts,
        "finite_audit": finite_audit,
        "contracts": {
            "quantiles_fit_train_only": True,
            "normalization_fit_train_only": True,
            "grouped_source_splits_preserved": True,
            "real_data_used": False,
        },
    }
    metadata_path = OUTPUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_b_training_package_manifest_1",
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in [OUTPUT / "train.pt", OUTPUT / "val.pt", OUTPUT / "development_test.pt", metadata_path]
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    automatic_pass = finite_audit["all_finite"] and transform.fitted_split == "train" and sum(value["sample_count"] for value in output_counts.values()) == 5000
    print(json.dumps({
        "automatic_pass": automatic_pass,
        "split_counts": {key: value["sample_count"] for key, value in output_counts.items()},
        "finite_audit": finite_audit,
        "output": str(OUTPUT),
    }, ensure_ascii=False), flush=True)
    return 0 if automatic_pass else 1


def _load_graph(manifest: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(manifest["graph_path"]).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _same_instance_labels(edges: list[list[int]], instance_ids: list[int]) -> tuple[list[float], list[bool]]:
    labels, mask = [], []
    for edge in edges:
        left, right = int(edge[0]), int(edge[1])
        left_instance, right_instance = instance_ids[left], instance_ids[right]
        if left_instance == 0 and right_instance == 0:
            labels.append(-1.0)
            mask.append(False)
        else:
            labels.append(float(left_instance > 0 and left_instance == right_instance))
            mask.append(True)
    return labels, mask


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
