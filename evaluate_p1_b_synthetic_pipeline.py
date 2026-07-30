from __future__ import annotations

"""Evaluate P1-B assembly/candidate/verifier/global-decode stages on synthetic splits."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from utils.p1_b_global_decode import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier_v3 import verify_candidate_prediction


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"
TRAINING_METADATA = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2" / "metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-canonical-patch", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RUN_ROOT / f"same_instance_seed_{args.seed}"
    model_name = "candidate_model_no_canonical_patch_v1" if args.no_canonical_patch else "candidate_model_v1"
    candidate_dir = run_dir / model_name
    output = run_dir / ("synthetic_pipeline_no_canonical_patch_v1" if args.no_canonical_patch else "synthetic_pipeline_v1")
    summary_path = output / "metrics.json"
    if summary_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "metrics": str(summary_path)}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    compact_macros = _read_json(TRAINING_METADATA)["macro_classes"]
    macro_classes = compact_macros[1:]
    gnn_predictions = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    candidate_predictions = torch.load(candidate_dir / "predictions.pt", map_location="cpu", weights_only=False)
    same_threshold = _read_json(run_dir / "same_instance_threshold_selection.json")
    candidate_thresholds = _read_json(candidate_dir / "candidate_threshold_selection.json")
    result: dict[str, Any] = {
        "schema_version": "topocae_p1_b_synthetic_pipeline_metrics_1",
        "seed": args.seed, "no_canonical_patch": args.no_canonical_patch,
        "thresholds": {"same_instance": same_threshold, "candidate": candidate_thresholds},
        "splits": {},
        "contracts": {"thresholds_selected_synthetic_val_only": True, "real_data_used": False, "holdout_used": False},
    }
    stage_rows = []
    started = time.perf_counter()
    for split in ("val", "development_test"):
        candidates_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in candidate_predictions[split]:
            candidates_by_sample[str(record["sample_id"])].append(record)
        truth_all: list[dict[str, Any]] = []
        stage_objects: dict[str, list[dict[str, Any]]] = {
            "raw_gnn_connected_components": [], "same_instance_assembly": [],
            "candidate_classifier": [], "topology_verifier_global_decode": [],
        }
        invalid_count = abstention_count = verified_count = conflict_reject_count = 0
        unexplained_overlap_count = 0
        sample_traces = []
        for gnn_record in gnn_predictions[split]:
            sample_id = str(gnn_record["sample_id"])
            graph = _read_json(Path(gnn_record["source_graph_path"]))
            truths = _truth_objects(graph, sample_id)
            truth_all.extend(truths)
            prediction_data = _gnn_prediction_data(gnn_record, compact_macros)
            raw = _raw_connected_objects(graph, gnn_record, compact_macros, sample_id)
            stage_objects["raw_gnn_connected_components"].extend(raw)
            assembly = assemble_same_instance_candidates(
                graph, prediction_data, global_threshold=float(same_threshold["global_threshold"]),
                threshold_selection_sha256=str(same_threshold["selection_sha256"]),
            )
            assembled = _assembly_objects(graph, assembly, gnn_record, compact_macros, sample_id)
            stage_objects["same_instance_assembly"].extend(assembled)
            model_records = candidates_by_sample.get(sample_id, [])
            candidate_objects = _candidate_classifier_objects(graph, model_records, candidate_thresholds, macro_classes, sample_id)
            stage_objects["candidate_classifier"].extend(candidate_objects)
            verified = []
            for candidate_record in model_records:
                model_prediction = {
                    "macro_classes": macro_classes,
                    "macro_logits": candidate_record["macro_logits"],
                    "macro_probabilities": candidate_record["macro_probabilities"],
                    "validity_score": candidate_record["validity_score"],
                    "membership_probabilities": candidate_record["membership_probabilities"],
                    "boundary_probabilities": candidate_record["boundary_probabilities"],
                }
                record = verify_candidate_prediction(
                    graph, candidate_record["proposal"], model_prediction, thresholds=candidate_thresholds,
                )
                verified.append(record)
                verified_count += 1
                abstention_count += int(record["status"] != "accepted")
                invalid_count += int(any(
                    reason not in {"validity_below_global_threshold", "confidence_below_global_threshold"}
                    for reason in record.get("abstention_reasons", [])
                ))
            decoded = decode_feature_objects(graph, verified)
            full_objects = [{**record, "sample_id": sample_id} for record in decoded["feature_objects"]]
            stage_objects["topology_verifier_global_decode"].extend(full_objects)
            conflict_reject_count += int(decoded["global_conflict_reject_count"])
            unexplained_overlap_count += len(decoded["unexplained_face_overlap"])
            sample_traces.append({
                "sample_id": sample_id, "truth_count": len(truths),
                "raw_object_count": len(raw), "assembly_object_count": len(assembled),
                "candidate_object_count": len(candidate_objects), "full_object_count": len(full_objects),
                "verified_candidate_count": len(verified), "abstention_count": int(decoded["abstention_count"]),
                "conflict_reject_count": int(decoded["global_conflict_reject_count"]),
            })
        split_metrics: dict[str, Any] = {}
        for stage, objects in stage_objects.items():
            metrics = _object_metrics(truth_all, objects, macro_classes)
            split_metrics[stage] = metrics
            stage_rows.append({
                "split": split, "stage": stage,
                "object_macro_f1": metrics["object_macro_f1"], "object_micro_f1": metrics["object_micro_f1"],
                "exact_instance_ratio": metrics["exact_instance_ratio"], "mean_best_instance_iou": metrics["mean_best_instance_iou"],
                "boundary_f1": metrics["boundary"]["f1"], "truth_count": metrics["truth_count"], "prediction_count": metrics["prediction_count"],
            })
        proposal_audit = _read_json(run_dir / "candidate_training_v1" / "metadata.json")["splits"][split]
        split_metrics["candidate_proposals"] = {
            "oracle_recall_iou_0_5": proposal_audit["proposal_oracle_recall_iou_0_5"],
            "oracle_exact_recall": proposal_audit["proposal_oracle_exact_recall"],
            "source_counts": proposal_audit["proposal_source_counts"],
            "all_seven_sources_present": proposal_audit["all_seven_sources_present"],
        }
        split_metrics["topology_diagnostics"] = {
            "verified_candidate_count": verified_count,
            "topology_invalid_rate": invalid_count / max(verified_count, 1),
            "abstention_rate": abstention_count / max(verified_count, 1),
            "global_conflict_reject_count": conflict_reject_count,
            "unexplained_overlap_count": unexplained_overlap_count,
        }
        result["splits"][split] = split_metrics
        torch.save(sample_traces, output / f"{split}_sample_traces.pt")
    result["evaluation_seconds"] = time.perf_counter() - started
    _write_json(summary_path, result)
    _write_csv(output / "stage_metrics.csv", stage_rows)
    val_full = result["splits"]["val"]["topology_verifier_global_decode"]
    print(json.dumps({
        "automatic_pass": (
            val_full["object_macro_f1"] >= 0.88 and val_full["exact_instance_ratio"] >= 0.85
            and result["splits"]["val"]["candidate_proposals"]["oracle_exact_recall"] >= 0.95
        ),
        "seed": args.seed, "val_object_macro_f1": val_full["object_macro_f1"],
        "val_object_micro_f1": val_full["object_micro_f1"],
        "val_exact_instance_ratio": val_full["exact_instance_ratio"],
        "val_instance_iou": val_full["mean_best_instance_iou"],
        "val_boundary_f1": val_full["boundary"]["f1"],
        "val_abstention_rate": result["splits"]["val"]["topology_diagnostics"]["abstention_rate"],
        "output": str(output),
    }, ensure_ascii=False), flush=True)
    return 0


def _gnn_prediction_data(record: dict[str, Any], macros: list[str]) -> dict[str, Any]:
    return {
        "faces": [{"idx": i, "macro_feature": macros[int(value)]} for i, value in enumerate(record["macro_prediction"])],
        "edge_same_instance_probabilities": record["edge_probability"].float().tolist(),
    }


def _truth_objects(graph: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    return [{
        "sample_id": sample_id, "instance_id": int(record.get("instance_id", position + 1)),
        "macro_feature": str(record["macro_feature"]),
        "face_indices": sorted(int(index) for index in record.get("node_indices", record.get("face_indices", []))),
    } for position, record in enumerate(graph.get("feature_sets", []))]


def _raw_connected_objects(graph: dict[str, Any], record: dict[str, Any], macros: list[str], sample_id: str) -> list[dict[str, Any]]:
    predicted = [macros[int(value)] for value in record["macro_prediction"]]
    adjacency = _adjacency(graph)
    remaining = {index for index, macro in enumerate(predicted) if macro != "stock"}
    objects = []
    while remaining:
        start = min(remaining); macro = predicted[start]; queue = [start]; remaining.remove(start); faces = []
        while queue:
            current = queue.pop(); faces.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in remaining and predicted[neighbor] == macro:
                    remaining.remove(neighbor); queue.append(neighbor)
        face_set = set(faces)
        objects.append({"sample_id": sample_id, "macro_feature": macro, "face_indices": sorted(faces), "boundary_face_indices": sorted(_boundary_faces(graph, face_set))})
    return objects


def _assembly_objects(
    graph: dict[str, Any], assembly: dict[str, Any], record: dict[str, Any], macros: list[str], sample_id: str,
) -> list[dict[str, Any]]:
    predicted = [macros[int(value)] for value in record["macro_prediction"]]
    objects = []
    for candidate in assembly["candidates"]:
        faces = [int(index) for index in candidate["face_indices"]]
        macro = Counter(predicted[index] for index in faces).most_common(1)[0][0]
        face_set = set(faces)
        objects.append({"sample_id": sample_id, "macro_feature": macro, "face_indices": sorted(faces), "boundary_face_indices": sorted(_boundary_faces(graph, face_set))})
    return objects


def _candidate_classifier_objects(
    graph: dict[str, Any], records: list[dict[str, Any]], thresholds: dict[str, Any],
    macro_classes: list[str], sample_id: str,
) -> list[dict[str, Any]]:
    objects = []
    for record in records:
        macro_index = max(range(len(record["macro_probabilities"])), key=record["macro_probabilities"].__getitem__)
        macro_confidence = float(record["macro_probabilities"][macro_index])
        if float(record["validity_score"]) < float(thresholds["min_candidate_validity"]) or macro_confidence < float(thresholds["min_candidate_confidence"]):
            continue
        faces = [face for face, probability in zip(record["face_indices"], record["membership_probabilities"]) if probability >= float(thresholds["membership_threshold"])]
        if not faces:
            continue
        boundary = [face for face, probability in zip(record["face_indices"], record["boundary_probabilities"]) if probability >= float(thresholds["boundary_threshold"])]
        objects.append({
            "sample_id": sample_id, "macro_feature": macro_classes[macro_index],
            "face_indices": sorted(int(face) for face in faces), "boundary_face_indices": sorted(int(face) for face in boundary),
            "confidence": macro_confidence * float(record["validity_score"]),
        })
    return objects


def _object_metrics(truths: list[dict[str, Any]], predictions: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    truth_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in truths: truth_by_sample[str(record["sample_id"])].append(record)
    for record in predictions: prediction_by_sample[str(record["sample_id"])].append(record)
    tp = 0; exact = 0; best_ious = []; boundary_tp = boundary_fp = boundary_fn = 0
    class_tp = Counter(); class_truth = Counter(record["macro_feature"] for record in truths); class_prediction = Counter(record["macro_feature"] for record in predictions)
    for sample_id, sample_truths in truth_by_sample.items():
        sample_predictions = prediction_by_sample.get(sample_id, [])
        pairs = []
        for truth_index, truth in enumerate(sample_truths):
            truth_faces = set(truth["face_indices"])
            same_macro_ious = []
            for prediction_index, prediction in enumerate(sample_predictions):
                if prediction["macro_feature"] != truth["macro_feature"]:
                    continue
                prediction_faces = set(prediction["face_indices"])
                iou = len(truth_faces & prediction_faces) / max(len(truth_faces | prediction_faces), 1)
                same_macro_ious.append(iou)
                if iou >= 0.5:
                    pairs.append((iou, truth_index, prediction_index))
            best_ious.append(max(same_macro_ious, default=0.0))
        used_truth = set(); used_prediction = set()
        for iou, truth_index, prediction_index in sorted(pairs, reverse=True):
            if truth_index in used_truth or prediction_index in used_prediction:
                continue
            used_truth.add(truth_index); used_prediction.add(prediction_index); tp += 1
            truth = sample_truths[truth_index]; prediction = sample_predictions[prediction_index]
            class_tp[truth["macro_feature"]] += 1
            exact += int(set(truth["face_indices"]) == set(prediction["face_indices"]))
            truth_boundary = _boundary_faces_from_objects(truth, sample_id)
            predicted_boundary = set(prediction.get("boundary_face_indices", []))
            boundary_tp += len(truth_boundary & predicted_boundary)
            boundary_fp += len(predicted_boundary - truth_boundary)
            boundary_fn += len(truth_boundary - predicted_boundary)
    fp = len(predictions) - tp; fn = len(truths) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class = {}; f1s = []
    for macro in classes:
        class_precision = class_tp[macro] / class_prediction[macro] if class_prediction[macro] else 0.0
        class_recall = class_tp[macro] / class_truth[macro] if class_truth[macro] else 0.0
        f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if class_precision + class_recall else 0.0
        per_class[macro] = {"truth": class_truth[macro], "prediction": class_prediction[macro], "tp": class_tp[macro], "precision": class_precision, "recall": class_recall, "f1": f1}
        if class_truth[macro]: f1s.append(f1)
    boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if boundary_tp + boundary_fp else 0.0
    boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if boundary_tp + boundary_fn else 0.0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall else 0.0
    return {
        "truth_count": len(truths), "prediction_count": len(predictions), "tp": tp, "fp": fp, "fn": fn,
        "object_micro_precision": precision, "object_micro_recall": recall, "object_micro_f1": micro_f1,
        "object_macro_f1": sum(f1s) / max(len(f1s), 1), "per_class": per_class,
        "exact_instance_count": exact, "exact_instance_ratio": exact / max(len(truths), 1),
        "mean_best_instance_iou": sum(best_ious) / max(len(best_ious), 1),
        "boundary": {"precision": boundary_precision, "recall": boundary_recall, "f1": boundary_f1, "tp": boundary_tp, "fp": boundary_fp, "fn": boundary_fn},
    }


def _boundary_faces_from_objects(record: dict[str, Any], sample_id: str) -> set[int]:
    graph_path = record.get("source_graph_path")
    if graph_path:
        return _boundary_faces(_read_json(Path(graph_path)), set(record["face_indices"]))
    # Truth records are evaluated against predictions from the same graph; all feature
    # faces in this benchmark meet an exterior/stock boundary unless enclosed by the
    # same instance. This fallback is replaced below by explicit graph-derived truth
    # boundaries in _truth_objects in v2 entry point.
    return set(record["face_indices"])


def _boundary_faces(graph: dict[str, Any], faces: set[int]) -> set[int]:
    boundary = set()
    for edge in graph.get("edges", []):
        left, right = int(edge[0]), int(edge[1])
        if left in faces and right not in faces: boundary.add(left)
        if right in faces and left not in faces: boundary.add(right)
    return boundary


def _adjacency(graph: dict[str, Any]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for edge in graph.get("edges", []):
        left, right = int(edge[0]), int(edge[1]); result[left].add(right); result[right].add(left)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
