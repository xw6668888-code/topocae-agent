from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

import evaluate_paper_v1_recognition as base


def fast_bootstrap_object_metrics(
    sample_reports: list[dict], *, replicates: int, confidence: float, seed: int
) -> dict:
    computed = [report for report in sample_reports if report["status"] == "computed"]
    if not computed:
        return {"status": "not_computable", "replicates": 0}
    tp = np.asarray([report["tp"] for report in computed], dtype=np.float64)
    fp = np.asarray([report["fp"] for report in computed], dtype=np.float64)
    fn = np.asarray([report["fn"] for report in computed], dtype=np.float64)
    exact = np.asarray([report["exact"] for report in computed], dtype=np.float64)
    truth = np.asarray([report["truth_count"] for report in computed], dtype=np.float64)
    iou = np.asarray([report["matched_iou_sum"] for report in computed], dtype=np.float64)
    part_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    part_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    part_f1 = np.divide(
        2.0 * part_precision * part_recall,
        part_precision + part_recall,
        out=np.zeros_like(tp),
        where=(part_precision + part_recall) != 0,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(computed), size=(replicates, len(computed)))
    tp_sum = tp[indices].sum(axis=1)
    fp_sum = fp[indices].sum(axis=1)
    fn_sum = fn[indices].sum(axis=1)
    truth_sum = truth[indices].sum(axis=1)
    precision = np.divide(tp_sum, tp_sum + fp_sum, out=np.zeros_like(tp_sum), where=(tp_sum + fp_sum) != 0)
    recall = np.divide(tp_sum, tp_sum + fn_sum, out=np.zeros_like(tp_sum), where=(tp_sum + fn_sum) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp_sum), where=(precision + recall) != 0)
    values = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_instance_ratio": np.divide(
            exact[indices].sum(axis=1), truth_sum, out=np.zeros_like(truth_sum), where=truth_sum != 0
        ),
        "instance_iou": np.divide(
            iou[indices].sum(axis=1), truth_sum, out=np.zeros_like(truth_sum), where=truth_sum != 0
        ),
        "mean_part_object_f1": part_f1[indices].mean(axis=1),
    }
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "computed",
        "replicates": replicates,
        "confidence": confidence,
        "seed": seed,
        "aggregation_unit": "part",
        "implementation": "numpy_vectorized_equivalent_resampling",
        "intervals": {
            name: {
                "low": float(np.quantile(items, alpha)),
                "high": float(np.quantile(items, 1.0 - alpha)),
            }
            for name, items in sorted(values.items())
        },
    }


if __name__ == "__main__":
    base.bootstrap_object_metrics = fast_bootstrap_object_metrics
    base.main()
