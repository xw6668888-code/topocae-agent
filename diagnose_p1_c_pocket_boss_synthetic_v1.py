from __future__ import annotations

"""Read-only Synthetic-Val diagnosis of circular pocket/boss confusion."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from utils.p1_c_strict_run_paths import ROOT, strict_run_dir


OUTPUT = ROOT / "reports" / "p1_c_iterations_v1" / "iteration1_pocket_boss_synthetic_diagnosis.json"
CLASSES = ("pocket_circular", "cylinder_boss")
FEATURES = (
    "inner_loop_face_fraction",
    "outer_loop_face_fraction",
    "boundary_circle_edge_fraction",
    "internal_concave_edge_fraction",
    "internal_convex_edge_fraction",
    "boundary_material_inner_mean",
    "boundary_material_outer_mean",
    "boundary_material_ambiguous_mean",
    "internal_convexity_fraction_concave",
    "internal_convexity_fraction_convex",
    "boundary_convexity_fraction_concave",
    "boundary_convexity_fraction_convex",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0] if ordered else 0.0,
        "median": ordered[len(ordered) // 2] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


def main() -> int:
    seed_rows = []
    for seed in (20260718, 20260719, 20260720):
        run = strict_run_dir("clean_7200", seed)
        gnn = _read(run / "metrics.json")["evaluation"]["val"]["face"]["per_class"]
        pipeline = _read(run / "synthetic_pipeline_v1" / "metrics.json")["splits"]["val"]["topology_verifier_global_decode"]["per_class"]
        candidate = _read(run / "candidate_model_v1" / "metrics.json")["evaluation"]["val"]["macro"]
        macro_classes = _read(run / "candidate_model_v1" / "config.json")["macro_classes"]
        seed_rows.append({
            "seed": seed,
            "face": {name: gnn[name] for name in CLASSES},
            "candidate": {name: candidate["per_class"][str(macro_classes.index(name))] for name in CLASSES},
            "object": {name: pipeline[name] for name in CLASSES},
        })

    run = strict_run_dir("clean_7200", 20260718)
    package = torch.load(run / "candidate_training_v1" / "val.pt", map_location="cpu", weights_only=False)
    predictions = torch.load(run / "candidate_model_v1" / "predictions.pt", map_location="cpu", weights_only=False)["val"]
    metadata = _read(run / "candidate_training_v1" / "metadata.json")
    schema = list(metadata["engineered_feature_schema"])
    feature_index = {name: schema.index(name) for name in FEATURES}
    macro_classes = list(metadata["macro_classes"])
    by_key = {
        (str(record["sample_id"]), str(record["candidate_id"])): record
        for record in predictions
    }
    feature_values: dict[str, dict[str, list[float]]] = {
        name: {feature: [] for feature in FEATURES} for name in CLASSES
    }
    confusion: dict[str, Counter[str]] = {name: Counter() for name in CLASSES}
    exact_counts = Counter()
    for record in package:
        macro = str(record.get("matched_truth_macro"))
        if macro not in CLASSES or float(record.get("matched_truth_iou", 0.0)) < 0.999999:
            continue
        if int(float(record["validity_target"].item())) != 1:
            continue
        exact_counts[macro] += 1
        vector = record["engineered_features"].float().tolist()
        for feature, index in feature_index.items():
            feature_values[macro][feature].append(float(vector[index]))
        predicted = by_key.get((str(record["sample_id"]), str(record["candidate_id"])))
        if predicted:
            probabilities = list(predicted["macro_probabilities"])
            confusion[macro][macro_classes[max(range(len(probabilities)), key=probabilities.__getitem__)]] += 1
    summaries = {
        macro: {feature: _summary(values) for feature, values in feature_values[macro].items()}
        for macro in CLASSES
    }
    separation = {}
    for feature in FEATURES:
        left = feature_values[CLASSES[0]][feature]
        right = feature_values[CLASSES[1]][feature]
        spread = math.sqrt((pstdev(left) ** 2 + pstdev(right) ** 2) / 2.0) if len(left) > 1 and len(right) > 1 else 0.0
        separation[feature] = {
            "mean_delta_pocket_minus_boss": mean(left) - mean(right),
            "pooled_std": spread,
            "absolute_standardized_separation": abs(mean(left) - mean(right)) / spread if spread > 1e-12 else None,
        }
    result = {
        "schema_version": "topocae_p1_c_iteration1_pocket_boss_synthetic_diagnosis_1",
        "evidence_scope": "Synthetic-Val only",
        "three_seed_metrics": seed_rows,
        "exact_candidate_counts": dict(exact_counts),
        "exact_candidate_macro_confusion": {name: dict(counts) for name, counts in confusion.items()},
        "engineered_feature_summaries": summaries,
        "engineered_feature_separation": separation,
        "contracts": {
            "real_data_used": False,
            "holdout_used": False,
            "sample_names_used_as_features": False,
            "diagnostic_is_read_only": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "confusion": result["exact_candidate_macro_confusion"], "separation": separation}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
