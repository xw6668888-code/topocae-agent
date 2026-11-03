"""Frozen decoding and exact-instance metrics for MFTReNet-SI-D0."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


def _components(nodes: list[int], adjacency: np.ndarray) -> list[list[int]]:
    remaining = set(nodes)
    result: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = []
        stack = [start]
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = sorted(
                candidate
                for candidate in remaining
                if adjacency[current, candidate] or adjacency[candidate, current]
            )
            for candidate in neighbours:
                remaining.remove(candidate)
                stack.append(candidate)
        result.append(sorted(component))
    return sorted(result)


def decode_instances(
    class_logits: np.ndarray,
    pair_logits: np.ndarray,
    *,
    threshold: float,
    native_to_step: np.ndarray | None = None,
) -> list[dict]:
    """Decode one part without truth, filenames, or sidecar labels.

    Every non-stock face is retained. Pair links are permitted only between
    faces with the same predicted class and components are deterministic.
    """
    scores = np.asarray(class_logits, dtype=np.float64)
    pairs = np.asarray(pair_logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 19:
        raise ValueError(f"Expected Nx19 class logits, got {scores.shape}")
    count = scores.shape[0]
    if pairs.shape != (count, count):
        raise ValueError(f"Expected {count}x{count} pair logits, got {pairs.shape}")
    if not np.isfinite(scores).all() or not np.isfinite(pairs).all():
        raise ValueError("Decoder received nonfinite model output")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Instance threshold must be strictly between zero and one")
    mapping = np.arange(count, dtype=np.int64) if native_to_step is None else np.asarray(
        native_to_step, dtype=np.int64
    )
    if sorted(mapping.tolist()) != list(range(count)):
        raise ValueError("native_to_step must be a zero-based bijection")

    labels = scores.argmax(axis=1)
    probability = 1.0 / (1.0 + np.exp(-np.clip(pairs, -60.0, 60.0)))
    linked = probability >= float(threshold)
    instances = []
    for class_id in range(1, 19):
        nodes = np.flatnonzero(labels == class_id).tolist()
        for component in _components(nodes, linked):
            instances.append(
                {
                    "class_id": class_id,
                    "faces": sorted(int(mapping[index]) for index in component),
                }
            )
    return sorted(instances, key=lambda item: (item["class_id"], item["faces"]))


def truth_instances(
    macro_step: Iterable[int], instance_step: Iterable[int]
) -> list[dict]:
    macro = np.asarray(list(macro_step), dtype=np.int64)
    instance = np.asarray(list(instance_step), dtype=np.int64)
    if macro.shape != instance.shape or macro.ndim != 1:
        raise ValueError("Truth macro and instance arrays must be aligned vectors")
    result = []
    for instance_id in sorted(set(instance.tolist()) - {0}):
        faces = np.flatnonzero(instance == instance_id).tolist()
        classes = set(macro[faces].tolist())
        if len(classes) != 1 or next(iter(classes)) not in range(1, 19):
            raise ValueError(f"Instance {instance_id} does not have one valid class")
        result.append({"class_id": int(macro[faces[0]]), "faces": faces})
    return sorted(result, key=lambda item: (item["class_id"], item["faces"]))


def exact_counts(predicted: list[dict], truth: list[dict]) -> np.ndarray:
    counts = np.zeros((18, 3), dtype=np.int64)
    for class_id in range(1, 19):
        predicted_sets = Counter(
            tuple(item["faces"]) for item in predicted if item["class_id"] == class_id
        )
        truth_sets = Counter(
            tuple(item["faces"]) for item in truth if item["class_id"] == class_id
        )
        true_positive = sum((predicted_sets & truth_sets).values())
        counts[class_id - 1] = (
            true_positive,
            sum(predicted_sets.values()) - true_positive,
            sum(truth_sets.values()) - true_positive,
        )
    return counts


def summarize_counts(counts: np.ndarray) -> dict:
    value = np.asarray(counts, dtype=np.int64)
    if value.shape != (18, 3) or (value < 0).any():
        raise ValueError("Exact-instance counts must have shape 18x3 and be nonnegative")

    def prf(row: np.ndarray) -> dict:
        true_positive, false_positive, false_negative = map(int, row)
        return {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0,
            "recall": true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0,
            "f1": (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else 0.0,
        }

    per_class = {str(index + 1): prf(row) for index, row in enumerate(value)}
    return {
        "per_class": per_class,
        "micro": prf(value.sum(axis=0)),
        "macro_f1_fixed_18": float(np.mean([row["f1"] for row in per_class.values()])),
    }


def choose_threshold(metrics: dict[str, dict]) -> float:
    """Apply the preregistered D0-validation-only threshold tie-break."""
    expected = {str(value) for value in THRESHOLD_GRID}
    if set(metrics) != expected:
        raise ValueError(f"Threshold metrics must exactly cover {sorted(expected)}")
    ranked = sorted(
        THRESHOLD_GRID,
        key=lambda threshold: (
            -float(metrics[str(threshold)]["macro_f1_fixed_18"]),
            -float(metrics[str(threshold)]["micro"]["f1"]),
            abs(threshold - 0.5),
            threshold,
        ),
    )
    return ranked[0]
