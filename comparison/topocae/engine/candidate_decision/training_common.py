from __future__ import annotations

import math
import random
from typing import Any, Mapping

import numpy as np
import torch
from models.gnn_baseline import EdgeAwareMultiHeadGNN

from .common import MACRO_CLASSES, RAW_TO_CONTIGUOUS


NODE_DIM = 39
EDGE_DIM = 20
PRIMITIVE_COUNT = 11
SEMANTIC_COUNT = 64
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.08
EPOCHS = 56
BATCH_SIZE = 192
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001
PATIENCE = 14
LABEL_SMOOTHING = 0.01
GRADIENT_CLIP_NORM = 5.0
LOSS_WEIGHTS = {"macro": 1.0, "semantic": 0.55, "same_instance": 0.65}


def build_model() -> EdgeAwareMultiHeadGNN:
    return EdgeAwareMultiHeadGNN(
        input_dim=NODE_DIM,
        edge_feature_dim=EDGE_DIM,
        num_primitive_surfaces=PRIMITIVE_COUNT,
        num_macro_classes=len(MACRO_CLASSES),
        num_semantic_classes=SEMANTIC_COUNT,
        derived_target_dim=0,
        idealization_target_dim=0,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        primitive_embedding_dim=16,
        dropout=DROPOUT,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remap_macro(raw: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(raw, -1)
    for source, target in RAW_TO_CONTIGUOUS.items():
        output[raw == source] = target
    if bool((output < 0).any()):
        unknown = sorted(set(int(value) for value in raw[output < 0].tolist()))
        raise ValueError(f"unknown macro IDs: {unknown}")
    return output


def adapt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    node = record["node_features"].float()
    edge = record["edge_attr"].float()
    if node.ndim != 2 or node.shape[1] != NODE_DIM:
        raise ValueError(f"{record.get('sample_id')}: node shape {tuple(node.shape)}")
    if edge.ndim != 2 or edge.shape[1] != EDGE_DIM:
        raise ValueError(f"{record.get('sample_id')}: edge shape {tuple(edge.shape)}")
    return {
        "sample_id": str(record["sample_id"]),
        "source_graph_path": str(record["source_graph_path"]),
        "source_step_path": str(record.get("source_step_path", "")),
        "source_brep_path": str(record.get("source_brep_path", "")),
        "base_geometry_family_id": str(record["geometry_family_id"]),
        "geometry_family_id": str(record["geometry_family_id"]),
        "derivation_kind": "identity",
        "stock_type": str(record["stock_type"]),
        "split": str(record["split"]),
        "node_features": node,
        "primitive_surface_ids": node[:, :PRIMITIVE_COUNT].argmax(dim=1).long(),
        "macro_targets": remap_macro(record["macro_feature_ids"].long()),
        "semantic_targets": record["face_class_labels"].long(),
        "instance_ids": record["instance_ids"].long(),
        "edge_index": record["edge_index"].long(),
        "edge_features": edge,
        "edge_same_instance_targets": record["edge_same_instance_labels"].float(),
        "edge_supervision_mask": record["edge_supervision_mask"].bool(),
        "feature_sets": list(record.get("feature_sets", [])),
        "face_to_canonical_patch": dict(record.get("face_to_canonical_patch", {})),
        "graph_structure_sha256": str(record["graph_structure_sha256"]),
        "semantic_graph_sha256": str(record["semantic_graph_sha256"]),
    }


def finite_or_raise(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_or_raise(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_or_raise(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value at {label}: {value}")


__all__ = [
    "BATCH_SIZE", "DROPOUT", "EDGE_DIM", "EPOCHS", "GRADIENT_CLIP_NORM",
    "HIDDEN_DIM", "LABEL_SMOOTHING", "LEARNING_RATE", "LOSS_WEIGHTS",
    "NODE_DIM", "NUM_LAYERS", "PATIENCE", "PRIMITIVE_COUNT", "SEMANTIC_COUNT",
    "WEIGHT_DECAY", "adapt_record", "build_model", "finite_or_raise", "set_seed",
]
