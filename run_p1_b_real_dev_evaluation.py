from __future__ import annotations

"""Run the frozen P1-B pipeline on the allowed Real-Dev collections."""

import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from core.graph_v2_1 import extract_graph_v2
from models.gnn_baseline import EdgeAwareMultiHeadGNN
from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from p1_b_general_recognition_pipeline_v4 import run_p1_b_general_recognition
from utils.ontology import load_layer_index
from utils.step_solid_validation import read_step_shape
from utils.training_export_v2 import GraphV2QuantileTransform
from utils.training_export_v2_1 import export_graph_v2_training_record


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p1_b_real_dev_v1"
RUN = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718"
TRAINING_PACKAGE = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2"
AREA_TRUTH = OUTPUT / "area_truth.json"
HUMAN_REVIEW = ROOT / "reports" / "paper_v1_p1_a2" / "human_review_package_v2" / "human_review_real_dev_10.csv"
ALLOWED_DIRS = [ROOT / "test_part_s", ROOT / "test_part_s1", ROOT / "test_part_s2"]
ABSOLUTE_AREA_TOLERANCE = 5.0
RELATIVE_AREA_TOLERANCE = 0.01
REVIEWED_FACE_TRUTH = {
    "test_part_s/part21": [
        {"truth_id": "reviewed_countersink", "macro_feature": "countersink_hole", "face_indices": [11, 12]},
    ],
    "test_part_s2/part2": [
        {"truth_id": "reviewed_slot_rect", "macro_feature": "slot_rect", "face_indices": [17, 18, 19]},
    ],
    "test_part_s2/part4": [
        {"truth_id": "reviewed_rib_a", "macro_feature": "rib", "face_indices": [10, 11, 12, 14, 15]},
        {"truth_id": "reviewed_rib_b", "macro_feature": "rib", "face_indices": [16, 17, 18, 20, 21]},
        {"truth_id": "reviewed_through_a", "macro_feature": "through_hole_round", "face_indices": [13]},
        {"truth_id": "reviewed_through_b", "macro_feature": "through_hole_round", "face_indices": [19]},
    ],
}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    graph_dir = OUTPUT / "graphs_v2"; prediction_dir = OUTPUT / "predictions"; object_dir = OUTPUT / "feature_objects"
    graph_dir.mkdir(exist_ok=True); prediction_dir.mkdir(exist_ok=True); object_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_metadata = _read_json(TRAINING_PACKAGE / "metadata.json")
    gnn, gnn_config = _load_gnn(device)
    candidate_model = _load_candidate_model(device)
    same_thresholds = _read_json(RUN / "same_instance_threshold_selection.json")
    candidate_thresholds = _read_json(RUN / "candidate_model_v1" / "candidate_threshold_selection.json")
    global_selection = _read_json(RUN / "global_decode_v2_selection.json")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    step_paths = _allowed_real_steps()
    outputs: dict[str, dict[str, Any]] = {}
    timings = []
    for position, step_path in enumerate(step_paths):
        dataset = step_path.parent.name; sample_id = step_path.stem; sample_key = f"{dataset}/{sample_id}"
        started = time.perf_counter()
        shape = read_step_shape(step_path)
        graph = extract_graph_v2(shape, primitive_surface_index=primitive_index)
        graph.update({
            "sample_id": sample_id, "dataset": dataset, "source_step_path": str(step_path.resolve()),
            "feature_sets": [], "graph_mode": "real_inference_unlabeled",
        })
        graph_path = graph_dir / dataset / f"{sample_id}.json"; graph_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(graph_path, graph)
        face_prediction = _predict_gnn(graph, gnn, gnn_config, training_metadata, device)
        face_prediction.update({"sample_id": sample_id, "dataset": dataset, "sample_key": sample_key})
        pipeline = run_p1_b_general_recognition(
            graph, face_prediction, candidate_model,
            same_instance_thresholds=same_thresholds,
            candidate_thresholds=candidate_thresholds,
            global_decode_selection=global_selection,
            device=str(device),
        )
        objects = pipeline["decoded_feature_objects"]["feature_objects"]
        for object_record in objects:
            object_record["area"] = sum(_face_area(graph, index) for index in object_record["face_indices"])
            object_record["face_count"] = len(object_record["face_indices"])
        prediction_path = prediction_dir / dataset / f"{sample_id}.json"; prediction_path.parent.mkdir(parents=True, exist_ok=True)
        object_path = object_dir / dataset / f"{sample_id}.json"; object_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(prediction_path, {
            "schema_version": "topocae_p1_b_real_face_prediction_1", "sample_key": sample_key,
            "source_step_path": str(step_path.resolve()), "faces": face_prediction["faces"],
            "edge_same_instance_probabilities": face_prediction["edge_same_instance_probabilities"],
            "contracts": {"truth_used": False, "real_specific_thresholds": False},
        })
        _write_json(object_path, {
            "schema_version": "topocae_p1_b_real_feature_objects_1", "sample_key": sample_key,
            "source_step_path": str(step_path.resolve()), "feature_objects": objects,
            "pipeline_manifest": pipeline["pipeline_manifest"],
            "diagnostics": {
                "proposal_count": pipeline["candidate_proposals"]["proposal_count"],
                "verified_candidate_count": len(pipeline["verified_candidates"]),
                "abstention_count": pipeline["decoded_feature_objects"]["abstention_count"],
                "global_conflict_reject_count": pipeline["decoded_feature_objects"]["global_conflict_reject_count"],
            },
        })
        elapsed = time.perf_counter() - started
        timings.append({"sample_key": sample_key, "seconds": elapsed, "face_count": len(graph["nodes"]), "object_count": len(objects)})
        outputs[sample_key] = {
            "graph": graph, "face_prediction": face_prediction, "pipeline": pipeline,
            "objects": objects, "step_path": str(step_path.resolve()),
            "graph_path": str(graph_path.resolve()), "prediction_path": str(prediction_path.resolve()),
            "object_path": str(object_path.resolve()),
        }
        print(f"real {position + 1}/{len(step_paths)} {sample_key}: faces={len(graph['nodes'])} objects={len(objects)} seconds={elapsed:.2f}", flush=True)

    truth = _read_json(AREA_TRUTH)
    area_metrics = _area_metrics(truth["samples"], outputs)
    face_set_metrics = _face_set_metrics(outputs)
    human_review = _human_review_comparison(outputs, area_metrics["samples"])
    failure_atlas = _failure_atlas(truth["samples"], outputs, area_metrics, face_set_metrics)
    historical_precision = 0.982301
    historical_recall = 0.982456
    historical_f1 = 2 * historical_precision * historical_recall / (historical_precision + historical_recall)
    comparison = [
        {
            "pipeline": "A0_frozen_raw_gnn", "precision": 0.472, "recall": 0.614, "f1": 0.534,
            "scope_note": "preflight value on Feature_Comparison_Table_New; frozen historical evidence",
        },
        {
            "pipeline": "historical_rule_pipeline_development_upper_bound", "precision": historical_precision,
            "recall": historical_recall, "f1": historical_f1,
            "scope_note": "historical Clean-Core test_part_s2 development upper bound; not independent generalization",
        },
        {
            "pipeline": "p1_b_full_general_pipeline", "precision": area_metrics["precision"],
            "recall": area_metrics["recall"], "f1": area_metrics["f1"],
            "scope_note": "all 12 workbook samples except user-excluded test_part_s2/part10; frozen Synthetic-Val thresholds",
        },
    ]
    result = {
        "schema_version": "topocae_p1_b_real_dev_report_1",
        "evaluated_step_count": len(step_paths), "area_truth_sample_count": area_metrics["sample_count"],
        "human_review_sample_count": len(human_review),
        "area_macro": area_metrics, "reviewed_face_sets": face_set_metrics,
        "comparison": comparison,
        "failure_atlas_summary": failure_atlas["summary"],
        "timing": {"samples": timings, "total_seconds": sum(row["seconds"] for row in timings), "mean_seconds": sum(row["seconds"] for row in timings) / max(len(timings), 1)},
        "frozen_selections": {
            "same_instance_sha256": same_thresholds["selection_sha256"],
            "candidate_sha256": candidate_thresholds["selection_sha256"],
            "global_decode_sha256": global_selection["selection_sha256"],
        },
        "contracts": {
            "real_dev_used_for_training": False, "real_dev_used_for_threshold_or_model_selection": False,
            "part_specific_thresholds": False, "llm_step_recognition": False,
            "excluded_test_part_s2_part10": True, "forbidden_collection_used": False,
        },
    }
    _write_json(OUTPUT / "metrics.json", result)
    _write_json(OUTPUT / "failure_atlas.json", failure_atlas)
    _write_csv(OUTPUT / "pipeline_comparison.csv", comparison)
    _write_csv(OUTPUT / "human_review_comparison.csv", human_review)
    _write_csv(OUTPUT / "failure_atlas.csv", failure_atlas["records"])
    _write_csv(OUTPUT / "timing.csv", timings)
    (OUTPUT / "report.zh-CN.md").write_text(_report_markdown(result), encoding="utf-8")
    manifest_files = [
        OUTPUT / "area_truth.json", OUTPUT / "metrics.json", OUTPUT / "failure_atlas.json",
        OUTPUT / "pipeline_comparison.csv", OUTPUT / "human_review_comparison.csv",
        OUTPUT / "failure_atlas.csv", OUTPUT / "timing.csv", OUTPUT / "report.zh-CN.md",
    ]
    _write_json(OUTPUT / "manifest.json", {
        "schema_version": "topocae_p1_b_real_dev_manifest_1",
        "files": [{"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in manifest_files],
        "source_human_review_sha256": _sha256(HUMAN_REVIEW),
        "source_area_truth_sha256": _sha256(AREA_TRUTH),
    })
    print(json.dumps({
        "real_dev_complete": True, "evaluated_steps": len(step_paths),
        "area_macro": {key: area_metrics[key] for key in ("precision", "recall", "f1", "truth_count", "prediction_count", "matched_count")},
        "reviewed_face_sets": face_set_metrics,
        "development_target_ge_0_75": area_metrics["f1"] >= 0.75,
        "output": str(OUTPUT),
    }, ensure_ascii=False), flush=True)
    return 0


def _load_gnn(device: torch.device) -> tuple[EdgeAwareMultiHeadGNN, dict[str, Any]]:
    checkpoint = torch.load(RUN / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    dims = config["model"]["dimensions"]
    model = EdgeAwareMultiHeadGNN(
        input_dim=dims["node"], edge_feature_dim=dims["edge"],
        num_primitive_surfaces=dims["primitive"], num_macro_classes=dims["macro"],
        num_semantic_classes=dims["semantic"], derived_target_dim=0, idealization_target_dim=0,
        hidden_dim=config["model"]["hidden_dim"], num_layers=config["model"]["num_layers"],
        primitive_embedding_dim=16, dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model, config


def _load_candidate_model(device: torch.device) -> CandidateTopologyClassifier:
    checkpoint = torch.load(RUN / "candidate_model_v1" / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    dimensions = CandidateModelDimensions(**config["dimensions"])
    model = CandidateTopologyClassifier(dimensions, macro_classes=config["macro_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model


def _predict_gnn(
    graph: dict[str, Any], model: EdgeAwareMultiHeadGNN, config: dict[str, Any],
    metadata: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    transform = GraphV2QuantileTransform.from_dict(metadata["quantile_transform"])
    exported = export_graph_v2_training_record(graph, quantile_transform=transform)
    normalization = metadata["normalization"]
    nodes = (np.asarray(exported["node_features"], dtype=np.float32) - np.asarray(normalization["node_mean"], dtype=np.float32)) / np.asarray(normalization["node_std"], dtype=np.float32)
    edges = (np.asarray(exported["edge_features"], dtype=np.float32) - np.asarray(normalization["edge_mean"], dtype=np.float32)) / np.asarray(normalization["edge_std"], dtype=np.float32)
    edge_index = torch.tensor(graph["edges"], dtype=torch.long, device=device).t().contiguous()
    primitive = torch.tensor([int(node.get("primitive_surface_id", 0) or 0) for node in graph["nodes"]], dtype=torch.long, device=device)
    primitive = torch.where(primitive < model.primitive_embedding.num_embeddings, primitive, torch.zeros_like(primitive))
    with torch.no_grad():
        output = model(
            torch.tensor(nodes, dtype=torch.float32, device=device), edge_index, primitive,
            edge_features=torch.tensor(edges, dtype=torch.float32, device=device),
        )
        macro_probability = torch.softmax(output["macro_logits"], dim=1)
        semantic_probability = torch.softmax(output["semantic_logits"], dim=1)
        edge_probability = torch.sigmoid(output["edge_same_instance_logits"])
    macro_names = _read_json(TRAINING_PACKAGE / "metadata.json")["macro_classes"]
    faces = []
    for index in range(len(graph["nodes"])):
        macro_id = int(macro_probability[index].argmax())
        semantic_id = int(semantic_probability[index].argmax())
        faces.append({
            "idx": index, "macro_feature": macro_names[macro_id], "macro_class_id": macro_id,
            "macro_confidence": float(macro_probability[index, macro_id]),
            "semantic_label": semantic_id, "semantic_confidence": float(semantic_probability[index, semantic_id]),
            "embedding": output["node_embeddings"][index].detach().cpu().tolist(),
        })
    return {
        "faces": faces, "node_embeddings": output["node_embeddings"].detach().cpu().tolist(),
        "edge_same_instance_probabilities": edge_probability.detach().cpu().tolist(),
        "contracts": {"primitive_oov_maps_to_unknown_embedding": True, "truth_used": False},
    }


def _allowed_real_steps() -> list[Path]:
    result = []
    for directory in ALLOWED_DIRS:
        for path in sorted(directory.glob("*.stp")):
            if directory.name == "test_part_s2" and path.stem == "part10":
                continue
            result.append(path.resolve())
    return result


def _area_metrics(truth_by_sample: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    samples = {}; total_truth = total_prediction = total_match = 0
    macro_truth = Counter(); macro_prediction = Counter(); macro_match = Counter()
    for sample_key, truths in truth_by_sample.items():
        if sample_key not in outputs:
            continue
        predictions = outputs[sample_key]["objects"]
        matches = []
        candidates = []
        for truth_index, truth in enumerate(truths):
            for prediction_index, prediction in enumerate(predictions):
                if prediction["macro_feature"] != truth["macro_feature"]:
                    continue
                absolute = abs(float(prediction["area"]) - float(truth["area"]))
                relative = absolute / max(abs(float(truth["area"])), 1e-9)
                if absolute <= ABSOLUTE_AREA_TOLERANCE or relative <= RELATIVE_AREA_TOLERANCE:
                    candidates.append((relative, absolute, truth_index, prediction_index))
        used_truth = set(); used_prediction = set()
        for relative, absolute, truth_index, prediction_index in sorted(candidates):
            if truth_index in used_truth or prediction_index in used_prediction:
                continue
            used_truth.add(truth_index); used_prediction.add(prediction_index)
            truth = truths[truth_index]; prediction = predictions[prediction_index]
            matches.append({
                "truth_index": truth_index, "prediction_index": prediction_index,
                "macro_feature": truth["macro_feature"], "truth_area": truth["area"],
                "prediction_area": prediction["area"], "absolute_error": absolute, "relative_error": relative,
                "prediction_face_indices": prediction["face_indices"],
            })
            macro_match[truth["macro_feature"]] += 1
        false_positives = [
            {"prediction_index": index, "macro_feature": record["macro_feature"], "area": record["area"], "face_indices": record["face_indices"]}
            for index, record in enumerate(predictions) if index not in used_prediction
        ]
        missed = [
            {"truth_index": index, **record} for index, record in enumerate(truths) if index not in used_truth
        ]
        total_truth += len(truths); total_prediction += len(predictions); total_match += len(matches)
        macro_truth.update(record["macro_feature"] for record in truths)
        macro_prediction.update(record["macro_feature"] for record in predictions)
        samples[sample_key] = {
            "truth_count": len(truths), "prediction_count": len(predictions), "matched_count": len(matches),
            "precision": len(matches) / max(len(predictions), 1), "recall": len(matches) / max(len(truths), 1),
            "matches": matches, "false_positives": false_positives, "missed_truth": missed,
        }
    precision = total_match / max(total_prediction, 1); recall = total_match / max(total_truth, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class = {}
    for macro in sorted(macro_truth):
        p = macro_match[macro] / macro_prediction[macro] if macro_prediction[macro] else 0.0
        r = macro_match[macro] / macro_truth[macro] if macro_truth[macro] else 0.0
        per_class[macro] = {"truth": macro_truth[macro], "prediction": macro_prediction[macro], "matched": macro_match[macro], "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return {
        "absolute_area_tolerance": ABSOLUTE_AREA_TOLERANCE, "relative_area_tolerance": RELATIVE_AREA_TOLERANCE,
        "sample_count": len(samples), "truth_count": total_truth, "prediction_count": total_prediction,
        "matched_count": total_match, "precision": precision, "recall": recall, "f1": f1,
        "per_class": per_class, "samples": samples,
    }


def _face_set_metrics(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth_count = matched = exact = 0; ious = []; records = []
    for sample_key, truths in REVIEWED_FACE_TRUTH.items():
        if sample_key not in outputs:
            continue
        objects = outputs[sample_key]["objects"]
        for truth in truths:
            truth_count += 1; truth_faces = set(truth["face_indices"])
            candidates = [record for record in objects if record["macro_feature"] == truth["macro_feature"]]
            scored = [(len(truth_faces & set(record["face_indices"])) / max(len(truth_faces | set(record["face_indices"])), 1), record) for record in candidates]
            best_iou, best = max(scored, key=lambda item: item[0], default=(0.0, None))
            matched += int(best_iou >= 0.5); exact += int(best_iou == 1.0); ious.append(best_iou)
            records.append({
                "sample_key": sample_key, "truth_id": truth["truth_id"], "macro_feature": truth["macro_feature"],
                "truth_face_indices": truth["face_indices"], "best_iou": best_iou,
                "prediction_face_indices": best["face_indices"] if best else [],
            })
    return {
        "truth_count": truth_count, "matched_iou_0_5": matched,
        "recall_iou_0_5": matched / max(truth_count, 1), "exact_instance_ratio": exact / max(truth_count, 1),
        "mean_instance_iou": sum(ious) / max(len(ious), 1), "records": records,
        "scope_note": "only face sets explicitly recoverable from user review notes; evaluation-only annotations",
    }


def _human_review_comparison(outputs: dict[str, dict[str, Any]], area_samples: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    with HUMAN_REVIEW.open("r", encoding="utf-8-sig", newline="") as handle:
        for prior in csv.DictReader(handle):
            sample_key = f"{prior['dataset']}/{prior['sample_id']}"
            if sample_key not in outputs:
                continue
            objects = outputs[sample_key]["objects"]
            summary = " | ".join(
                f"{record['macro_feature']} faces={record['face_indices']} area={record['area']:.4f} conf={record.get('confidence', 0.0):.4f}"
                for record in objects
            )
            area = area_samples.get(sample_key, {})
            rows.append({
                "review_id": prior["review_id"], "sample_key": sample_key,
                "prior_review_result": prior.get("review_result", ""), "prior_error_type": prior.get("error_type", ""),
                "prior_review_note": prior.get("review_note", ""),
                "new_pipeline_object_count": len(objects), "new_area_truth_count": area.get("truth_count", ""),
                "new_area_matched_count": area.get("matched_count", ""), "new_area_precision": area.get("precision", ""),
                "new_area_recall": area.get("recall", ""),
                "new_review_status": "NOT_REVIEWED_AUTOMATED_DIAGNOSTIC",
                "new_prediction_summary": summary,
            })
    return rows


def _failure_atlas(
    truth_by_sample: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]],
    area_metrics: dict[str, Any], face_set_metrics: dict[str, Any],
) -> dict[str, Any]:
    records = []
    for sample_key, sample in area_metrics["samples"].items():
        predictions = outputs[sample_key]["objects"]
        for miss in sample["missed_truth"]:
            truth_area = float(miss["area"]); macro = miss["macro_feature"]
            same = [record for record in predictions if record["macro_feature"] == macro]
            cross_compatible = [record for record in predictions if _area_compatible(truth_area, float(record["area"]))]
            if not same:
                category = "macro_role_confusion" if cross_compatible else "proposal_miss"
            else:
                nearest = min(same, key=lambda record: abs(float(record["area"]) - truth_area))
                ratio = float(nearest["area"]) / max(truth_area, 1e-9)
                category = "fragment" if ratio < 0.8 else ("merge" if ratio > 1.2 else "boundary_error")
            records.append({
                "sample_key": sample_key, "failure_category": category, "record_kind": "missed_truth",
                "macro_feature": macro, "truth_area": truth_area, "prediction_area": "", "face_indices": "",
                "detail": "area+macro truth not matched under frozen 5 mm / 1% protocol",
            })
        for false_positive in sample["false_positives"]:
            records.append({
                "sample_key": sample_key, "failure_category": "macro_role_confusion", "record_kind": "false_positive",
                "macro_feature": false_positive["macro_feature"], "truth_area": "", "prediction_area": false_positive["area"],
                "face_indices": json.dumps(false_positive["face_indices"]), "detail": "unmatched predicted object",
            })
    for sample_key, output in outputs.items():
        verified = output["pipeline"]["verified_candidates"]
        for record in verified:
            if record["status"] == "accepted":
                continue
            reasons = record.get("abstention_reasons", [])
            category = "invalid_topology" if any(reason not in {
                "validity_below_global_threshold", "confidence_below_global_threshold",
                "face_candidate_consensus_corrected_confidence_below_global_threshold",
            } for reason in reasons) else "abstention"
            records.append({
                "sample_key": sample_key, "failure_category": category, "record_kind": "candidate_abstention",
                "macro_feature": record.get("masked_top1", "unknown"), "truth_area": "", "prediction_area": "",
                "face_indices": json.dumps(record.get("original_face_indices", [])), "detail": "|".join(reasons),
            })
    for record in face_set_metrics["records"]:
        if record["best_iou"] < 1.0:
            records.append({
                "sample_key": record["sample_key"], "failure_category": "boundary_error", "record_kind": "reviewed_face_set",
                "macro_feature": record["macro_feature"], "truth_area": "", "prediction_area": "",
                "face_indices": json.dumps(record["prediction_face_indices"]),
                "detail": f"reviewed face-set best IoU={record['best_iou']:.6f}",
            })
    summary = {
        "record_count": len(records), "by_category": dict(Counter(record["failure_category"] for record in records)),
        "by_macro": dict(Counter(record["macro_feature"] for record in records)),
        "required_categories": ["proposal_miss", "macro_role_confusion", "fragment", "merge", "boundary_error", "invalid_topology", "abstention"],
    }
    return {"schema_version": "topocae_p1_b_real_failure_atlas_1", "summary": summary, "records": records}


def _area_compatible(truth: float, prediction: float) -> bool:
    absolute = abs(truth - prediction)
    return absolute <= ABSOLUTE_AREA_TOLERANCE or absolute / max(abs(truth), 1e-9) <= RELATIVE_AREA_TOLERANCE


def _face_area(graph: dict[str, Any], index: int) -> float:
    node = graph["nodes"][int(index)]
    return float(node.get("attributes", {}).get("raw_physical_v2", {}).get("area", node.get("attributes", {}).get("physical", {}).get("area", 0.0)) or 0.0)


def _report_markdown(result: dict[str, Any]) -> str:
    area = result["area_macro"]; face = result["reviewed_face_sets"]
    lines = [
        "# P1-B Real-Dev 冻结诊断报告", "",
        "本报告只使用 Synthetic-Val 冻结后的模型与全局阈值。Real-Dev 未参与训练、early stopping、模型选择或规则修改。", "",
        "## 统一 area+macro", "",
        f"- 样本：{area['sample_count']}；truth={area['truth_count']}；prediction={area['prediction_count']}；matched={area['matched_count']}",
        f"- Precision={area['precision']:.6f}；Recall={area['recall']:.6f}；F1={area['f1']:.6f}",
        f"- 开发目标 >=0.75：{'PASS' if area['f1'] >= 0.75 else 'FAIL'}", "",
        "## 可恢复的人工 face-set 真值", "",
        f"- truth={face['truth_count']}；IoU>=0.5 recall={face['recall_iou_0_5']:.6f}；exact={face['exact_instance_ratio']:.6f}；mean IoU={face['mean_instance_iou']:.6f}", "",
        "## 对比说明", "",
    ]
    for row in result["comparison"]:
        lines.append(f"- {row['pipeline']}: P={row['precision']:.6f}, R={row['recall']:.6f}, F1={row['f1']:.6f}；{row['scope_note']}")
    lines.extend(["", "## Failure atlas", ""])
    for key, value in sorted(result["failure_atlas_summary"]["by_category"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "逐样本、逐类和逐失败记录见同目录 JSON/CSV；历史规则结果仅作为已开发数据上界，不能当未知真实件泛化结论。", ""])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
