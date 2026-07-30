from __future__ import annotations

import math
from typing import Any, Iterable

DEFAULT_EDGE_THRESHOLDS = tuple(round(0.30 + 0.05 * idx, 2) for idx in range(13))

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - depends on optional runtime dependency
    average_precision_score = None
    roc_auc_score = None


def compute_edge_same_instance_metrics(
    scores: Any,
    labels: Any,
    mask: Any | None = None,
    *,
    threshold: float = 0.5,
    thresholds: Iterable[float] | None = None,
    scores_are_logits: bool = True,
) -> dict[str, Any]:
    """Compute supervised same-instance edge metrics from logits or probabilities."""
    flat_scores = _flatten_numeric(scores)
    flat_labels = _flatten_numeric(labels)
    flat_mask = [1.0] * len(flat_scores) if mask is None else _flatten_numeric(mask)
    if len(flat_scores) != len(flat_labels) or len(flat_scores) != len(flat_mask):
        raise ValueError(
            "scores, labels, and mask must contain the same number of edge entries "
            f"(got {len(flat_scores)}, {len(flat_labels)}, {len(flat_mask)})"
        )

    probabilities: list[float] = []
    binary_labels: list[int] = []
    for score, label, mask_value in zip(flat_scores, flat_labels, flat_mask):
        if mask_value <= 0.5 or label < 0:
            continue
        probabilities.append(_sigmoid(score) if scores_are_logits else float(score))
        binary_labels.append(1 if label >= 0.5 else 0)

    threshold_values = _threshold_values(thresholds)
    main_metrics = _metrics_at_threshold(probabilities, binary_labels, threshold)
    auc_metrics = _compute_auc_metrics(probabilities, binary_labels)
    threshold_scan = [
        {"threshold": scan_threshold, **_metrics_at_threshold(probabilities, binary_labels, scan_threshold)}
        for scan_threshold in threshold_values
    ]
    best = _best_threshold_row(threshold_scan) if probabilities else None

    return {
        "threshold": float(threshold),
        **main_metrics,
        **auc_metrics,
        "threshold_scan": threshold_scan,
        "best_f1_threshold": best["threshold"] if best is not None else None,
        "best_f1": best["f1"] if best is not None else 0.0,
    }


def _threshold_values(thresholds: Iterable[float] | None) -> list[float]:
    if thresholds is None:
        return list(DEFAULT_EDGE_THRESHOLDS)
    return [round(float(value), 6) for value in thresholds]


def _flatten_numeric(values: Any) -> list[float]:
    if values is None:
        return []
    if _is_tensor_like(values):
        tensor = values.detach().cpu().reshape(-1)
        return [float(value) for value in tensor.tolist()]
    if _is_array_like(values):
        return _flatten_numeric(values.tolist())
    if isinstance(values, (str, bytes)):
        raise TypeError("edge metric inputs must be numeric, not strings")
    if isinstance(values, Iterable):
        flattened: list[float] = []
        for item in values:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return [float(values)]


def _is_tensor_like(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("detach", "cpu", "reshape", "tolist"))


def _is_array_like(value: Any) -> bool:
    return (
        hasattr(value, "tolist")
        and hasattr(value, "reshape")
        and not _is_tensor_like(value)
        and not isinstance(value, (str, bytes))
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics_at_threshold(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for probability, label in zip(probabilities, labels):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = len(labels)
    positive_count = sum(labels)
    negative_count = total - positive_count
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "valid_edge_count": total,
        "positive_edge_count": positive_count,
        "negative_edge_count": negative_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _safe_divide(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_auc_metrics(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    if not labels:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "auc_available": False,
            "auc_reason": "no_valid_edges",
        }
    if len(set(labels)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "auc_available": False,
            "auc_reason": "single_class",
        }
    if roc_auc_score is None or average_precision_score is None:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "auc_available": False,
            "auc_reason": "sklearn_unavailable",
        }
    try:
        return {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "auc_available": True,
            "auc_reason": None,
        }
    except Exception as exc:  # pragma: no cover - defensive guard around optional dependency
        return {
            "roc_auc": None,
            "pr_auc": None,
            "auc_available": False,
            "auc_reason": f"sklearn_error: {exc}",
        }


def _best_threshold_row(threshold_scan: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not threshold_scan:
        return None
    return max(threshold_scan, key=lambda row: (row["f1"], -row["threshold"]))
