from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable

import joblib
import numpy as np
import torch

from audit_paper_v1_provenance import (
    REPO_ROOT,
    assert_allowed_path,
    load_json,
    load_split_ids,
    resolve_repo_path,
    stable_hash,
    write_json,
)
from generate_area_truth_review import build_review as build_area_truth_review
from prediction_to_feature_objects import build_feature_objects
from run_gnn_inference import build_model_from_checkpoint, predict_one_graph
from utils.candidate_classifier_postprocess import load_candidate_classifier_bundle
from utils.rib_boundary_classifier_postprocess import load_rib_boundary_classifier_bundle
from utils.rib_candidate_proposer_postprocess import load_rib_candidate_proposer_bundle


DEFAULT_DATASET_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "datasets.json"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "models.json"
DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "evaluation.json"
DEFAULT_EXCLUSIONS = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "quality_exclusions.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "paper_v1_clean_candidate" / "selected_thresholds.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "paper_v1_p1" / "evaluations"
SPLIT_NAMES = ("train", "val", "test")
VARIANT_NAMES = ("clean_core", "clean_rib")
OUTPUT_LOCATION_KEYS = {
    "source_prediction_path",
    "prediction_path",
    "feature_objects_path",
    "output_path",
    "output_root",
    "run_id",
    "generated_at",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and evaluate the Phase 1 paper recognition baselines."
    )
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--quality-exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--selected-thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="repeat_1")
    parser.add_argument("--variants", nargs="+", choices=VARIANT_NAMES, default=list(VARIANT_NAMES))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--compare-run-a", type=Path, default=None)
    parser.add_argument("--compare-run-b", type=Path, default=None)
    return parser.parse_args()


def load_effective_splits(dataset_config: dict, exclusion_path: Path) -> dict[str, list[str]]:
    splits = load_split_ids(dataset_config)
    exclusions = load_json(exclusion_path)["exclusions"]
    return {
        split_name: [
            sample_id
            for sample_id in sample_ids
            if sample_id not in exclusions.get(split_name, {})
        ]
        for split_name, sample_ids in splits.items()
    }


def collect_graph_files(root: Path, dataset_config: dict) -> list[Path]:
    root = assert_allowed_path(root, dataset_config)
    paths = []
    for path in sorted(root.glob("*.json")):
        path = assert_allowed_path(path, dataset_config)
        try:
            payload = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("nodes"), list) and isinstance(payload.get("edges"), list):
            paths.append(path)
    return paths


def dataset_graphs(
    dataset_config: dict,
    effective_splits: dict[str, list[str]],
) -> dict[str, list[Path]]:
    formal = dataset_config["formal_synthetic"]
    synthetic_root = assert_allowed_path(resolve_repo_path(formal["graph_root"]), dataset_config)
    datasets = {
        "synthetic_test": [
            assert_allowed_path(synthetic_root / f"{sample_id}.json", dataset_config)
            for sample_id in effective_splits["test"]
        ]
    }
    for name, descriptor in dataset_config["real_dev"].items():
        datasets[f"real_{name}"] = collect_graph_files(
            resolve_repo_path(descriptor["graph_root"]), dataset_config
        )
    return datasets


def load_models(
    dataset_config: dict,
    model_config: dict,
    thresholds: dict,
) -> dict[str, Any]:
    checkpoint_path = assert_allowed_path(
        resolve_repo_path(model_config["frozen_gnn"]["path"]), dataset_config
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cpu")
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    clean = model_config["clean_components"]
    candidate_path = assert_allowed_path(resolve_repo_path(clean["candidate_classifier"]), dataset_config)
    proposer_path = assert_allowed_path(resolve_repo_path(clean["rib_candidate_proposer"]), dataset_config)
    boundary_path = assert_allowed_path(resolve_repo_path(clean["rib_boundary_classifier"]), dataset_config)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "model": model,
        "candidate": load_candidate_classifier_bundle(candidate_path),
        "candidate_path": candidate_path,
        "proposer": load_rib_candidate_proposer_bundle(proposer_path),
        "proposer_path": proposer_path,
        "boundary": load_rib_boundary_classifier_bundle(boundary_path),
        "boundary_path": boundary_path,
        "proposer_threshold": float(
            thresholds["rib_candidate_proposer"]["selected_threshold"]
        ),
        "boundary_threshold": float(
            thresholds["rib_boundary_classifier"]["selected_threshold"]
        ),
    }


def predict_dataset(
    *,
    variant: str,
    graph_paths: list[Path],
    output_dir: Path,
    models: dict[str, Any],
    dataset_config: dict,
) -> dict:
    prediction_dir = output_dir / "predictions"
    feature_dir = output_dir / "feature_objects"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for index, graph_path in enumerate(graph_paths, start=1):
            graph_path = assert_allowed_path(graph_path, dataset_config)
            graph = load_json(graph_path)
            prediction = predict_one_graph(
                model=models["model"],
                graph_data=graph,
                graph_path=graph_path,
                checkpoint_config=models["checkpoint"]["config"],
                ndigits=6,
                device=models["device"],
                topology_postprocess=True,
                same_instance_threshold=0.5,
                candidate_classifier_bundle=models["candidate"],
                candidate_classifier_min_confidence=0.5,
                rib_candidate_proposer_bundle=models["proposer"] if variant == "clean_rib" else None,
                rib_candidate_proposer_min_confidence=models["proposer_threshold"],
                rib_boundary_classifier_bundle=models["boundary"] if variant == "clean_rib" else None,
                rib_boundary_classifier_min_confidence=models["boundary_threshold"],
            )
            objects = build_feature_objects(prediction, graph, ndigits=6)
            prediction_path = prediction_dir / f"{graph_path.stem}.json"
            feature_path = feature_dir / f"{graph_path.stem}.json"
            write_json(prediction_path, prediction)
            write_json(feature_path, objects)
            manifest.append(
                {
                    "sample_id": graph_path.stem,
                    "graph_path": str(graph_path.resolve()),
                    "prediction_path": str(prediction_path.resolve()),
                    "feature_objects_path": str(feature_path.resolve()),
                }
            )
            if index % 250 == 0 or index == len(graph_paths):
                print(f"evaluation_progress={variant}:{output_dir.name}:{index}/{len(graph_paths)}")
    payload = {
        "variant": variant,
        "checkpoint": str(models["checkpoint_path"].resolve()),
        "candidate_classifier": str(models["candidate_path"].resolve()),
        "rib_candidate_proposer": str(models["proposer_path"].resolve()) if variant == "clean_rib" else None,
        "rib_boundary_classifier": str(models["boundary_path"].resolve()) if variant == "clean_rib" else None,
        "rib_candidate_proposer_threshold": models["proposer_threshold"] if variant == "clean_rib" else None,
        "rib_boundary_classifier_threshold": models["boundary_threshold"] if variant == "clean_rib" else None,
        "sample_count": len(manifest),
        "samples": manifest,
    }
    write_json(output_dir / "manifest.json", payload)
    return payload


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2.0 * precision * recall, precision + recall)


def classification_report(pairs: list[tuple[int, int]], id_to_name: dict[str, str]) -> dict:
    if not pairs:
        return {
            "status": "not_computable",
            "coverage": 0.0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
        }
    class_ids = sorted({label for pair in pairs for label in pair if label >= 0})
    per_class = {}
    f1_values = []
    correct = sum(predicted == truth for predicted, truth in pairs)
    for class_id in class_ids:
        tp = sum(predicted == class_id and truth == class_id for predicted, truth in pairs)
        fp = sum(predicted == class_id and truth != class_id for predicted, truth in pairs)
        fn = sum(predicted != class_id and truth == class_id for predicted, truth in pairs)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = f1_score(precision, recall)
        f1_values.append(f1)
        per_class[id_to_name.get(str(class_id), f"id_{class_id}")] = {
            "id": class_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "status": "computed",
        "coverage": 1.0,
        "accuracy": correct / len(pairs),
        "macro_f1": sum(f1_values) / len(f1_values),
        "active_class_count": len(class_ids),
        "per_class": per_class,
    }


def graph_has_truth(graph: dict) -> bool:
    nodes = graph.get("nodes", [])
    return bool(nodes) and all("macro_feature_id" in node and "instance_id" in node for node in nodes)


def face_and_semantic_pairs(prediction: dict, graph: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not graph_has_truth(graph):
        return [], []
    pred_faces = {int(face["idx"]): face for face in prediction.get("faces", [])}
    macro_pairs = []
    semantic_pairs = []
    for node in graph.get("nodes", []):
        face_idx = int(node["idx"])
        if face_idx not in pred_faces:
            continue
        face = pred_faces[face_idx]
        macro_pairs.append((int(face.get("macro_feature_id", -1)), int(node.get("macro_feature_id", -1))))
        semantic_pairs.append(
            (
                int(face.get("semantic_subtype_id", -1)),
                int(node.get("semantic_label", node.get("semantic_subtype_id", -1))),
            )
        )
    return macro_pairs, semantic_pairs


def edge_counts(prediction: dict, graph: dict, threshold: float = 0.5) -> dict:
    if not graph_has_truth(graph):
        return {"tp": 0, "fp": 0, "fn": 0, "count": 0, "status": "not_computable"}
    instance_by_face = {int(node["idx"]): node.get("instance_id") for node in graph.get("nodes", [])}
    tp = fp = fn = 0
    for edge in prediction.get("edge_same_instance", []) or []:
        source = int(edge.get("source_face", -1))
        target = int(edge.get("target_face", -1))
        left = instance_by_face.get(source)
        right = instance_by_face.get(target)
        truth = left is not None and right is not None and int(left) > 0 and int(left) == int(right)
        predicted = float(edge.get("same_instance_probability", 0.0)) >= threshold
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "count": tp + fp + fn, "status": "computed"}


def truth_objects(graph: dict) -> list[dict]:
    if not graph_has_truth(graph):
        return []
    return [
        {
            "macro_feature": feature.get("macro_feature"),
            "face_indices": sorted({int(index) for index in feature.get("node_indices", [])}),
            "instance_id": feature.get("instance_id"),
        }
        for feature in graph.get("feature_sets", []) or []
        if feature.get("node_indices")
    ]


def predicted_objects(feature_doc: dict) -> list[dict]:
    return [
        {
            "macro_feature": feature.get("macro_feature"),
            "face_indices": sorted({int(index) for index in feature.get("face_indices", [])}),
            "feature_object_id": feature.get("feature_object_id"),
        }
        for feature in feature_doc.get("feature_objects", []) or []
        if feature.get("face_indices")
    ]


def set_iou(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    return safe_divide(len(left_set & right_set), len(left_set | right_set))


def object_match_report(feature_doc: dict, graph: dict, minimum_iou: float) -> dict:
    truths = truth_objects(graph)
    predictions = predicted_objects(feature_doc)
    if not graph_has_truth(graph):
        return {
            "status": "not_computable",
            "truth_count": 0,
            "prediction_count": len(predictions),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact": 0,
            "matched_iou_sum": 0.0,
            "per_class_counts": {},
            "matches": [],
        }
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if prediction["macro_feature"] != truth["macro_feature"]:
                continue
            iou = set_iou(prediction["face_indices"], truth["face_indices"])
            if iou >= minimum_iou:
                candidates.append((iou, prediction_index, truth_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches = []
    for iou, prediction_index, truth_index in candidates:
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "truth_index": truth_index,
                "macro_feature": truths[truth_index]["macro_feature"],
                "iou": iou,
                "exact": iou == 1.0,
            }
        )
    per_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for match in matches:
        per_class_counts[match["macro_feature"]]["tp"] += 1
    for index, prediction in enumerate(predictions):
        if index not in used_predictions:
            per_class_counts[str(prediction["macro_feature"])]["fp"] += 1
    for index, truth in enumerate(truths):
        if index not in used_truths:
            per_class_counts[str(truth["macro_feature"])]["fn"] += 1
    return {
        "status": "computed",
        "truth_count": len(truths),
        "prediction_count": len(predictions),
        "tp": len(matches),
        "fp": len(predictions) - len(matches),
        "fn": len(truths) - len(matches),
        "exact": sum(bool(match["exact"]) for match in matches),
        "matched_iou_sum": sum(float(match["iou"]) for match in matches),
        "per_class_counts": {key: value for key, value in sorted(per_class_counts.items())},
        "matches": matches,
    }


def aggregate_edge(sample_reports: list[dict]) -> dict:
    computed = [report for report in sample_reports if report["status"] == "computed"]
    if not computed:
        return {"status": "not_computable", "coverage": 0.0, "precision": None, "recall": None, "f1": None}
    tp = sum(report["tp"] for report in computed)
    fp = sum(report["fp"] for report in computed)
    fn = sum(report["fn"] for report in computed)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "status": "computed",
        "coverage": len(computed) / len(sample_reports),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def object_metrics_from_reports(sample_reports: list[dict]) -> dict:
    computed = [report for report in sample_reports if report["status"] == "computed"]
    if not computed:
        return {
            "status": "not_computable",
            "coverage": 0.0,
            "precision": None,
            "recall": None,
            "f1": None,
            "exact_instance_ratio": None,
            "instance_iou": None,
            "per_class": {},
        }
    tp = sum(report["tp"] for report in computed)
    fp = sum(report["fp"] for report in computed)
    fn = sum(report["fn"] for report in computed)
    truth_count = sum(report["truth_count"] for report in computed)
    exact = sum(report["exact"] for report in computed)
    iou_sum = sum(report["matched_iou_sum"] for report in computed)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    for report in computed:
        for class_name, counts in report["per_class_counts"].items():
            class_counts[class_name].update(counts)
    per_class = {}
    for class_name, counts in sorted(class_counts.items()):
        class_precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        class_recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        per_class[class_name] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": class_precision,
            "recall": class_recall,
            "f1": f1_score(class_precision, class_recall),
        }
    return {
        "status": "computed",
        "coverage": len(computed) / len(sample_reports),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "exact_instance_ratio": safe_divide(exact, truth_count),
        "instance_iou": safe_divide(iou_sum, truth_count),
        "truth_count": truth_count,
        "prediction_count": tp + fp,
        "per_class": per_class,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def bootstrap_object_metrics(
    sample_reports: list[dict], *, replicates: int, confidence: float, seed: int
) -> dict:
    computed = [report for report in sample_reports if report["status"] == "computed"]
    if not computed:
        return {"status": "not_computable", "replicates": 0}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = [computed[rng.randrange(len(computed))] for _ in range(len(computed))]
        metrics = object_metrics_from_reports(sampled)
        for name in ("precision", "recall", "f1", "exact_instance_ratio", "instance_iou"):
            values[name].append(float(metrics[name]))
        part_f1_values = []
        for report in sampled:
            precision = safe_divide(report["tp"], report["tp"] + report["fp"])
            recall = safe_divide(report["tp"], report["tp"] + report["fn"])
            part_f1_values.append(f1_score(precision, recall))
        values["mean_part_object_f1"].append(sum(part_f1_values) / len(part_f1_values))
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "computed",
        "replicates": replicates,
        "confidence": confidence,
        "seed": seed,
        "aggregation_unit": "part",
        "intervals": {
            name: {"low": percentile(items, alpha), "high": percentile(items, 1.0 - alpha)}
            for name, items in sorted(values.items())
        },
    }


def evaluate_labeled_dataset(
    *,
    graph_paths: list[Path],
    prediction_dir: Path,
    feature_dir: Path,
    checkpoint_config: dict,
    evaluation_config: dict,
) -> dict:
    macro_pairs: list[tuple[int, int]] = []
    semantic_pairs: list[tuple[int, int]] = []
    edge_reports = []
    object_reports = []
    sample_reports = []
    for graph_path in graph_paths:
        graph = load_json(graph_path)
        prediction = load_json(prediction_dir / f"{graph_path.stem}.json")
        feature_doc = load_json(feature_dir / f"{graph_path.stem}.json")
        sample_macro_pairs, sample_semantic_pairs = face_and_semantic_pairs(prediction, graph)
        macro_pairs.extend(sample_macro_pairs)
        semantic_pairs.extend(sample_semantic_pairs)
        edge_report = edge_counts(prediction, graph)
        object_report = object_match_report(
            feature_doc, graph, float(evaluation_config["object_match_iou"])
        )
        edge_reports.append(edge_report)
        object_reports.append(object_report)
        sample_reports.append(
            {
                "sample_id": graph_path.stem,
                "face_truth_available": bool(sample_macro_pairs),
                "face_count": len(sample_macro_pairs),
                "edge": edge_report,
                "object": object_report,
            }
        )
    labels = checkpoint_config["label_spaces"]
    object_metrics = object_metrics_from_reports(object_reports)
    bootstrap = evaluation_config["bootstrap"]
    return {
        "sample_count": len(graph_paths),
        "face": classification_report(macro_pairs, labels["macro_feature_id_to_key"]),
        "semantic": classification_report(semantic_pairs, labels["semantic_subtype_id_to_key"]),
        "edge": aggregate_edge(edge_reports),
        "object": object_metrics,
        "bootstrap": bootstrap_object_metrics(
            object_reports,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]),
        ),
        "query": {
            "status": "not_computable",
            "coverage": 0.0,
            "reason": "P1 geometry-recognition evaluation has no query dataset; query metrics begin after Phase 3.",
        },
        "samples": sample_reports,
    }


def object_signature(feature: dict) -> tuple:
    return (
        str(feature.get("macro_feature")),
        tuple(sorted(int(index) for index in feature.get("face_indices", []))),
    )


def rotation_consistency(
    *,
    prediction_dir: Path,
    feature_dir: Path,
    groups: dict[str, list[str]],
) -> dict:
    group_reports = []
    class_numerator = class_denominator = 0
    role_numerator = role_denominator = 0
    object_scores = []
    for group_id, sample_ids in sorted(groups.items()):
        predictions = [load_json(prediction_dir / f"{sample_id}.json") for sample_id in sample_ids]
        feature_docs = [load_json(feature_dir / f"{sample_id}.json") for sample_id in sample_ids]
        face_maps = [
            {int(face["idx"]): face for face in prediction.get("faces", [])}
            for prediction in predictions
        ]
        union_faces = set().union(*(set(face_map) for face_map in face_maps))
        class_consistent = 0
        role_consistent = 0
        for face_idx in union_faces:
            faces = [face_map.get(face_idx) for face_map in face_maps]
            if all(face is not None for face in faces):
                if len({face.get("macro_feature") for face in faces}) == 1:
                    class_consistent += 1
                if len({face.get("semantic_subtype") for face in faces}) == 1:
                    role_consistent += 1
        class_numerator += class_consistent
        class_denominator += len(union_faces)
        role_numerator += role_consistent
        role_denominator += len(union_faces)
        reference = {object_signature(feature) for feature in feature_docs[0].get("feature_objects", [])}
        pair_scores = []
        for feature_doc in feature_docs[1:]:
            current = {object_signature(feature) for feature in feature_doc.get("feature_objects", [])}
            pair_scores.append(safe_divide(2 * len(reference & current), len(reference) + len(current)))
        group_object_score = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
        object_scores.append(group_object_score)
        group_reports.append(
            {
                "group_id": group_id,
                "sample_ids": sample_ids,
                "class_consistency": safe_divide(class_consistent, len(union_faces)),
                "face_role_consistency": safe_divide(role_consistent, len(union_faces)),
                "object_correspondence_consistency": group_object_score,
                "face_correspondence_method": "stable graph face idx within the existing traced rotation family",
            }
        )
    return {
        "status": "computed" if group_reports else "not_computable",
        "coverage": 1.0 if group_reports else 0.0,
        "group_count": len(group_reports),
        "class_consistency": safe_divide(class_numerator, class_denominator),
        "object_correspondence_consistency": sum(object_scores) / len(object_scores) if object_scores else None,
        "face_role_consistency": safe_divide(role_numerator, role_denominator),
        "groups": group_reports,
    }


def prediction_risk_summary(feature_dir: Path) -> dict:
    object_count = 0
    low_confidence = 0
    macro_counts: Counter[str] = Counter()
    for path in sorted(feature_dir.glob("*.json")):
        document = load_json(path)
        for feature in document.get("feature_objects", []) or []:
            object_count += 1
            macro_counts[str(feature.get("macro_feature", "unknown"))] += 1
            confidence = float(feature.get("confidence", {}).get("macro_confidence_mean", 0.0) or 0.0)
            if confidence < 0.6:
                low_confidence += 1
    return {
        "prediction_object_count": object_count,
        "low_confidence_object_count": low_confidence,
        "low_confidence_threshold": 0.6,
        "predicted_macro_distribution": dict(sorted(macro_counts.items())),
    }


def evaluate_variant(
    *,
    variant: str,
    run_root: Path,
    graph_sets: dict[str, list[Path]],
    dataset_config: dict,
    evaluation_config: dict,
    models: dict[str, Any],
) -> dict:
    variant_root = run_root / variant
    metrics = {}
    for dataset_name, graph_paths in graph_sets.items():
        output_dir = variant_root / dataset_name
        predict_dataset(
            variant=variant,
            graph_paths=graph_paths,
            output_dir=output_dir,
            models=models,
            dataset_config=dataset_config,
        )
        dataset_metrics = evaluate_labeled_dataset(
            graph_paths=graph_paths,
            prediction_dir=output_dir / "predictions",
            feature_dir=output_dir / "feature_objects",
            checkpoint_config=models["checkpoint"]["config"],
            evaluation_config=evaluation_config,
        )
        dataset_metrics["prediction_risk"] = prediction_risk_summary(output_dir / "feature_objects")
        if dataset_name == "real_test_part_s2":
            truth_path = assert_allowed_path(
                resolve_repo_path(dataset_config["real_dev"]["test_part_s2"]["truth"]), dataset_config
            )
            area_review = build_area_truth_review(
                output_dir / "feature_objects", truth_path, False, 5.0, 0.01
            )
            dataset_metrics["area_truth"] = area_review
            write_json(output_dir / "area_truth_review.json", area_review)
        metrics[dataset_name] = dataset_metrics
    rotation_source = variant_root / "real_test_part_s"
    rotation = rotation_consistency(
        prediction_dir=rotation_source / "predictions",
        feature_dir=rotation_source / "feature_objects",
        groups=dataset_config["rotation_dev"]["groups"],
    )
    metrics["rotation_dev"] = rotation
    return metrics


def flatten_metric_rows(metrics: dict) -> list[dict]:
    rows = []
    for variant, variant_metrics in metrics.items():
        for dataset_name, dataset in variant_metrics.items():
            if dataset_name == "rotation_dev":
                for name in ("class_consistency", "object_correspondence_consistency", "face_role_consistency"):
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset_name,
                            "level": "rotation",
                            "metric": name,
                            "value": dataset.get(name),
                            "ci_low": None,
                            "ci_high": None,
                            "coverage": dataset.get("coverage"),
                            "status": dataset.get("status"),
                        }
                    )
                continue
            for level, names in {
                "face": ("accuracy", "macro_f1"),
                "semantic": ("macro_f1",),
                "edge": ("precision", "recall", "f1"),
                "object": ("precision", "recall", "f1", "exact_instance_ratio", "instance_iou"),
            }.items():
                section = dataset.get(level, {})
                intervals = dataset.get("bootstrap", {}).get("intervals", {}) if level == "object" else {}
                for name in names:
                    interval = intervals.get(name, {})
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset_name,
                            "level": level,
                            "metric": name,
                            "value": section.get(name),
                            "ci_low": interval.get("low"),
                            "ci_high": interval.get("high"),
                            "coverage": section.get("coverage"),
                            "status": section.get("status"),
                        }
                    )
            if "area_truth" in dataset:
                area = dataset["area_truth"]
                for name in ("precision", "recall"):
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset_name,
                            "level": "area_truth",
                            "metric": name,
                            "value": area.get(name),
                            "ci_low": None,
                            "ci_high": None,
                            "coverage": safe_divide(area.get("sample_count", 0), dataset.get("sample_count", 0)),
                            "status": "computed",
                        }
                    )
    return rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "dataset", "level", "metric", "value", "ci_low", "ci_high", "coverage", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: canonicalize(item)
            for key, item in sorted(value.items())
            if key not in OUTPUT_LOCATION_KEYS
        }
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def canonical_file_hash(path: Path) -> str:
    return stable_hash(canonicalize(load_json(path)))


def compare_runs(run_a: Path, run_b: Path, output_path: Path) -> dict:
    files_a = {
        path.relative_to(run_a).as_posix(): path
        for path in run_a.rglob("*.json")
        if path.name not in {"manifest.json", "metrics.json", "area_truth_review.json"}
    }
    files_b = {
        path.relative_to(run_b).as_posix(): path
        for path in run_b.rglob("*.json")
        if path.name not in {"manifest.json", "metrics.json", "area_truth_review.json"}
    }
    missing_in_a = sorted(set(files_b) - set(files_a))
    missing_in_b = sorted(set(files_a) - set(files_b))
    mismatches = []
    for relative_path in sorted(set(files_a) & set(files_b)):
        hash_a = canonical_file_hash(files_a[relative_path])
        hash_b = canonical_file_hash(files_b[relative_path])
        if hash_a != hash_b:
            mismatches.append({"path": relative_path, "hash_a": hash_a, "hash_b": hash_b})
    metrics_a = canonicalize(load_json(run_a / "metrics.json"))
    metrics_b = canonicalize(load_json(run_b / "metrics.json"))
    payload = {
        "schema": "topocae_paper_v1_repeatability_v1",
        "run_a": str(run_a.resolve()),
        "run_b": str(run_b.resolve()),
        "compared_prediction_and_object_file_count": len(set(files_a) & set(files_b)),
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "prediction_or_object_mismatches": mismatches,
        "metrics_exact_match": metrics_a == metrics_b,
        "status": (
            "pass"
            if not missing_in_a and not missing_in_b and not mismatches and metrics_a == metrics_b
            else "fail"
        ),
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    if (args.compare_run_a is None) != (args.compare_run_b is None):
        raise ValueError("--compare-run-a and --compare-run-b must be supplied together")
    if args.compare_run_a is not None:
        result = compare_runs(
            args.compare_run_a.resolve(),
            args.compare_run_b.resolve(),
            REPO_ROOT / "reports" / "paper_v1_p1" / "repeatability.json",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] != "pass":
            raise SystemExit(2)
        return

    dataset_config = load_json(args.dataset_config.resolve())
    model_config = load_json(args.model_config.resolve())
    evaluation_config = load_json(args.evaluation_config.resolve())
    thresholds = load_json(args.selected_thresholds.resolve())
    effective_splits = load_effective_splits(dataset_config, args.quality_exclusions.resolve())
    graph_sets = dataset_graphs(dataset_config, effective_splits)
    models = load_models(dataset_config, model_config, thresholds)
    run_root = assert_allowed_path(args.output_root.resolve() / args.run_id, dataset_config)
    run_root.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for variant in args.variants:
        metrics[variant] = evaluate_variant(
            variant=variant,
            run_root=run_root,
            graph_sets=graph_sets,
            dataset_config=dataset_config,
            evaluation_config=evaluation_config,
            models=models,
        )
    payload = {
        "schema": "topocae_paper_v1_recognition_metrics_v1",
        "phase": "P1-A1",
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formal_synthetic_test_count": len(graph_sets["synthetic_test"]),
        "real_dev_counts": {
            name: len(paths) for name, paths in graph_sets.items() if name.startswith("real_")
        },
        "thresholds": {
            "candidate_classifier": 0.5,
            "rib_candidate_proposer": models["proposer_threshold"],
            "rib_boundary_classifier": models["boundary_threshold"],
        },
        "metrics": metrics,
    }
    write_json(run_root / "metrics.json", payload)
    rows = flatten_metric_rows(metrics)
    write_metrics_csv(run_root / "metrics.csv", rows)
    if args.publish:
        write_json(REPO_ROOT / "reports" / "paper_v1_p1" / "metrics.json", payload)
        write_metrics_csv(REPO_ROOT / "reports" / "paper_v1_p1" / "metrics.csv", rows)
    print(
        json.dumps(
            {
                "run_root": str(run_root.resolve()),
                "variants": args.variants,
                "synthetic_test_count": len(graph_sets["synthetic_test"]),
                "real_dev_count": sum(
                    len(paths) for name, paths in graph_sets.items() if name.startswith("real_")
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
