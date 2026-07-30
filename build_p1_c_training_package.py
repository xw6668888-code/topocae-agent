from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
DATASET = ROOT / "output_generation_p1_c_clean_context_7200"
PACKAGE_ROOT = DATASET / "training_packages"
MACRO_CLASSES = [
    "stock",
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rib",
    "web",
    "rectangular_boss",
    "fillet",
    "chamfer",
    "slot_v",
    "slot_o",
]
VARIANTS = ("clean_7200", "clean_5000", "single_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = PACKAGE_ROOT / args.variant
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"training package already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split_records = _load_variant_manifests(args.variant)
    print(f"P1-C {args.variant}: fitting Graph v2 quantiles on train only", flush=True)
    train_node_vectors: list[list[float]] = []
    train_edge_vectors: list[list[float]] = []
    truth_stripping_records: list[dict[str, Any]] = []
    for position, manifest in enumerate(split_records["train"]):
        graph = _load_graph(manifest)
        train_node_vectors.extend(build_graph_v2_node_feature_vector(node) for node in graph.get("nodes", []))
        train_edge_vectors.extend(build_graph_v2_edge_feature_vector(edge) for edge in graph.get("edge_attributes", []))
        if position < 36:
            truth_stripping_records.append(_truth_stripping_audit(graph, manifest["sample_id"]))
        if (position + 1) % 500 == 0:
            print(f"  quantile read {position + 1}/{len(split_records['train'])}", flush=True)
    transform = fit_training_quantile_transform(
        train_node_vectors,
        train_edge_vectors,
        split_name="train",
        lower_quantile=0.005,
        upper_quantile=0.995,
        max_abs=32.0,
    )
    node_clipped = np.asarray(
        [
            [max(low, min(high, value)) for value, low, high in zip(row, transform.node_lower, transform.node_upper)]
            for row in train_node_vectors
        ],
        dtype=np.float32,
    )
    edge_clipped = np.asarray(
        [
            [max(low, min(high, value)) for value, low, high in zip(row, transform.edge_lower, transform.edge_upper)]
            for row in train_edge_vectors
        ],
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
    output_counts: dict[str, Any] = {}
    finite_audit = {"value_count": 0, "max_abs_normalized": 0.0, "all_finite": True}
    balance_audit = {
        "sample_node_weight_sum_min": math.inf,
        "sample_node_weight_sum_max": 0.0,
        "sample_edge_weight_sum_min": math.inf,
        "sample_edge_weight_sum_max": 0.0,
    }
    for split, manifests in split_records.items():
        packaged: list[dict[str, Any]] = []
        macro_face_counts: Counter[int] = Counter()
        same_edge_counts: Counter[str] = Counter()
        for position, manifest in enumerate(manifests):
            graph = _load_graph(manifest)
            exported = export_graph_v2_training_record(graph, quantile_transform=transform)
            node_features = (np.asarray(exported["node_features"], dtype=np.float32) - node_mean) / node_std
            edge_features = (np.asarray(exported["edge_features"], dtype=np.float32) - edge_mean) / edge_std
            finite_audit["value_count"] += int(node_features.size + edge_features.size)
            if node_features.size:
                finite_audit["max_abs_normalized"] = max(
                    finite_audit["max_abs_normalized"], float(np.abs(node_features).max())
                )
                finite_audit["all_finite"] &= bool(np.isfinite(node_features).all())
            if edge_features.size:
                finite_audit["max_abs_normalized"] = max(
                    finite_audit["max_abs_normalized"], float(np.abs(edge_features).max())
                )
                finite_audit["all_finite"] &= bool(np.isfinite(edge_features).all())
            nodes = graph.get("nodes", [])
            macro_ids = [macro_to_id[str(node.get("macro_feature", "stock"))] for node in nodes]
            semantic_ids = [int(node.get("semantic_label", 0) or 0) for node in nodes]
            primitive_ids = [int(node.get("primitive_surface_id", 0) or 0) for node in nodes]
            instance_ids = [int(node.get("instance_id", 0) or 0) for node in nodes]
            edge_labels, edge_mask = _same_instance_labels(graph.get("edges", []), instance_ids)
            node_weights = _instance_balanced_node_weights(instance_ids)
            edge_weights = _sample_balanced_edge_weights(edge_labels, edge_mask)
            balance_audit["sample_node_weight_sum_min"] = min(balance_audit["sample_node_weight_sum_min"], sum(node_weights))
            balance_audit["sample_node_weight_sum_max"] = max(balance_audit["sample_node_weight_sum_max"], sum(node_weights))
            supervised_edge_sum = sum(weight for weight, mask in zip(edge_weights, edge_mask) if mask)
            balance_audit["sample_edge_weight_sum_min"] = min(balance_audit["sample_edge_weight_sum_min"], supervised_edge_sum)
            balance_audit["sample_edge_weight_sum_max"] = max(balance_audit["sample_edge_weight_sum_max"], supervised_edge_sum)
            macro_face_counts.update(macro_ids)
            same_edge_counts.update(
                "positive" if label == 1 else "negative"
                for label, mask in zip(edge_labels, edge_mask)
                if mask
            )
            packaged.append(
                {
                    "sample_id": manifest["sample_id"],
                    "source_graph_path": manifest["graph_path"],
                    "base_geometry_family_id": manifest["base_geometry_family_id"],
                    "primary_target": manifest["primary_target"],
                    "context": manifest["context"],
                    "host_family": manifest["host_family"],
                    "node_features": torch.from_numpy(node_features),
                    "edge_index": torch.tensor(graph.get("edges", []), dtype=torch.long).t().contiguous()
                    if graph.get("edges")
                    else torch.empty((2, 0), dtype=torch.long),
                    "edge_features": torch.from_numpy(edge_features),
                    "primitive_surface_ids": torch.tensor(primitive_ids, dtype=torch.long),
                    "macro_targets": torch.tensor(macro_ids, dtype=torch.long),
                    "semantic_targets": torch.tensor(semantic_ids, dtype=torch.long),
                    "instance_ids": torch.tensor(instance_ids, dtype=torch.long),
                    "edge_same_instance_targets": torch.tensor(edge_labels, dtype=torch.float32),
                    "edge_supervision_mask": torch.tensor(edge_mask, dtype=torch.bool),
                    "node_instance_balanced_weights": torch.tensor(node_weights, dtype=torch.float32),
                    "edge_sample_balanced_weights": torch.tensor(edge_weights, dtype=torch.float32),
                    "feature_sets": graph.get("feature_sets", []),
                    "face_to_canonical_patch": graph.get("face_to_canonical_patch", {}),
                }
            )
            if (position + 1) % 500 == 0:
                print(f"  package {split} {position + 1}/{len(manifests)}", flush=True)
        path = output / f"{split}.pt"
        torch.save(packaged, path)
        output_counts[split] = {
            "sample_count": len(packaged),
            "macro_face_counts": {
                MACRO_CLASSES[index]: macro_face_counts[index] for index in range(len(MACRO_CLASSES))
            },
            "same_instance_edge_counts": dict(same_edge_counts),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    truth_stripping = {
        "sample_count": len(truth_stripping_records),
        "pass_count": sum(item["passed"] for item in truth_stripping_records),
        "passed": all(item["passed"] for item in truth_stripping_records),
        "records": truth_stripping_records,
    }
    prohibited_tokens = ("primary", "context", "host", "seed", "sample", "path", "generator")
    schema_input_safe = not any(
        token in feature.lower()
        for feature in (*GRAPH_V2_NODE_FEATURE_SCHEMA, *GRAPH_V2_EDGE_FEATURE_SCHEMA)
        for token in prohibited_tokens
    )
    metadata = {
        "schema_version": "topocae_p1_c_training_package_1",
        "variant": args.variant,
        "source_dataset": str(DATASET.resolve()),
        "source_dataset_meta_sha256": _sha256(DATASET / "dataset_meta.json"),
        "macro_classes": MACRO_CLASSES,
        "macro_to_compact_id": macro_to_id,
        "macro_class_count": len(MACRO_CLASSES),
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
        "balance_audit": balance_audit,
        "truth_stripping_audit": truth_stripping,
        "schema_input_safe": schema_input_safe,
        "contracts": {
            "quantiles_fit_train_only": True,
            "normalization_fit_train_only": True,
            "grouped_source_splits_preserved": True,
            "sample_instance_balanced_weights_available": True,
            "sampling_metadata_is_model_input": False,
            "real_data_used": False,
            "holdout_used": False,
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [output / "train.pt", output / "val.pt", output / "development_test.pt", metadata_path]
    manifest = {
        "schema_version": "topocae_p1_c_training_package_manifest_1",
        "variant": args.variant,
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    expected_counts = {
        "clean_7200": {"train": 5760, "val": 720, "development_test": 720},
        "clean_5000": {"train": 4000, "val": 500, "development_test": 500},
        "single_only": {"train": 3456, "val": 432, "development_test": 432},
    }[args.variant]
    automatic_pass = (
        finite_audit["all_finite"]
        and truth_stripping["passed"]
        and schema_input_safe
        and transform.fitted_split == "train"
        and {split: payload["sample_count"] for split, payload in output_counts.items()} == expected_counts
        and abs(balance_audit["sample_node_weight_sum_min"] - 1.0) < 1e-5
        and abs(balance_audit["sample_node_weight_sum_max"] - 1.0) < 1e-5
    )
    print(
        json.dumps(
            {
                "automatic_pass": automatic_pass,
                "variant": args.variant,
                "split_counts": {key: value["sample_count"] for key, value in output_counts.items()},
                "finite_audit": finite_audit,
                "truth_stripping": {key: truth_stripping[key] for key in ("sample_count", "pass_count", "passed")},
                "schema_input_safe": schema_input_safe,
                "output": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if automatic_pass else 1


def _load_variant_manifests(variant: str) -> dict[str, list[dict[str, Any]]]:
    if variant == "clean_7200":
        root = DATASET / "splits"
    elif variant == "clean_5000":
        root = DATASET / "splits_matched_5000"
    else:
        root = DATASET / "splits"
    records = {
        split: _read_jsonl(root / f"{split}.jsonl")
        for split in ("train", "val", "development_test")
    }
    if variant == "single_only":
        records = {
            split: [record for record in values if record["context"] == "target_only"]
            for split, values in records.items()
        }
    return records


def _truth_stripping_audit(graph: dict[str, Any], sample_id: str) -> dict[str, Any]:
    reference = export_graph_v2_training_record(graph)
    stripped = deepcopy(graph)
    stripped.pop("feature_sets", None)
    stripped.pop("generation", None)
    stripped.pop("p1_c_automatic_audit", None)
    stripped.pop("sample_id", None)
    stripped.pop("base_geometry_family_id", None)
    stripped.pop("parametric_family_id", None)
    stripped.pop("geometry_signatures", None)
    for node in stripped.get("nodes", []):
        for key in ("semantic_label", "instance_id", "macro_feature", "macro_feature_id", "semantic_role"):
            node.pop(key, None)
    candidate = export_graph_v2_training_record(stripped)
    passed = reference["node_features"] == candidate["node_features"] and reference["edge_features"] == candidate["edge_features"]
    return {"sample_id": sample_id, "passed": passed}


def _instance_balanced_node_weights(instance_ids: list[int]) -> list[float]:
    groups: dict[int, list[int]] = {}
    for index, instance_id in enumerate(instance_ids):
        groups.setdefault(int(instance_id), []).append(index)
    group_weight = 1.0 / max(len(groups), 1)
    weights = [0.0] * len(instance_ids)
    for indices in groups.values():
        per_face = group_weight / len(indices)
        for index in indices:
            weights[index] = per_face
    return weights


def _sample_balanced_edge_weights(labels: list[float], mask: list[bool]) -> list[float]:
    weights = [0.0] * len(labels)
    groups = {
        0: [index for index, (label, active) in enumerate(zip(labels, mask)) if active and label <= 0.0],
        1: [index for index, (label, active) in enumerate(zip(labels, mask)) if active and label > 0.0],
    }
    active_groups = [indices for indices in groups.values() if indices]
    for indices in active_groups:
        per_edge = 1.0 / len(active_groups) / len(indices)
        for index in indices:
            weights[index] = per_edge
    return weights


def _load_graph(manifest: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(manifest["graph_path"]).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _same_instance_labels(edges: list[list[int]], instance_ids: list[int]) -> tuple[list[float], list[bool]]:
    labels: list[float] = []
    mask: list[bool] = []
    for left, right in edges:
        left_instance = instance_ids[int(left)]
        right_instance = instance_ids[int(right)]
        if left_instance == 0 and right_instance == 0:
            labels.append(-1.0)
            mask.append(False)
        else:
            labels.append(float(left_instance > 0 and left_instance == right_instance))
            mask.append(True)
    return labels, mask


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
