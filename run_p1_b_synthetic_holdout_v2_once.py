from __future__ import annotations

"""Run exactly one frozen inference pass on Synthetic-Holdout-v2."""

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import torch

import evaluate_p1_b_synthetic_pipeline as metric_impl
from p1_b_general_recognition_pipeline_v4 import run_p1_b_general_recognition
import run_p1_b_real_dev_evaluation as frozen_inference
from utils.p1_b_candidate_proposer import evaluate_proposal_oracle_recall


ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "output_generation_p1_b_synthetic_holdout_v2"
REPORT = ROOT / "reports" / "p1_b_holdout_v2_once"
SELECTION = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"
SIDECAR = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json"
RUN = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718"
TRAINING_PACKAGE = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2"
STATE = REPORT / "formal_inference_state.json"


def main() -> int:
    selection_sha = _verify_frozen_inputs()
    script_sha = _sha256(Path(__file__).resolve())
    generation_manifest_path = HOLDOUT / "holdout_generation_manifest.json"
    generation_manifest = _read_json(generation_manifest_path)
    if int(generation_manifest.get("inference_pass_count", 0)) != 0:
        raise RuntimeError("Synthetic-Holdout-v2 formal inference has already completed")
    records = _read_json(HOLDOUT / "graph_step_manifest.json")["records"]
    if len(records) != 1080:
        raise RuntimeError("formal Holdout-v2 requires exactly 1080 generated samples")
    leakage = _read_json(HOLDOUT / "leakage_audit.json")
    if leakage.get("status") != "pass" or any(int(leakage[key]) for key in (
        "base_collision_with_training", "exact_collision_with_training", "near_collision_with_training",
        "holdout_internal_base_collision", "holdout_internal_exact_collision", "holdout_internal_near_collision",
    )):
        raise RuntimeError("Holdout-v2 leakage audit is not clean")
    REPORT.mkdir(parents=True, exist_ok=True)
    prediction_dir = REPORT / "sample_predictions"
    prediction_dir.mkdir(exist_ok=True)
    state = _start_or_resume_state(selection_sha, script_sha, len(records))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_metadata = _read_json(TRAINING_PACKAGE / "metadata.json")
    macro_names = list(training_metadata["macro_classes"])
    gnn, gnn_config = frozen_inference._load_gnn(device)
    candidate_model = frozen_inference._load_candidate_model(device)
    same_thresholds = _read_json(RUN / "same_instance_threshold_selection.json")
    candidate_thresholds = _read_json(RUN / "candidate_model_v1" / "candidate_threshold_selection.json")
    global_selection = _read_json(RUN / "global_decode_v2_selection.json")
    started = time.perf_counter()
    for position, manifest_record in enumerate(records):
        sample_id = str(manifest_record["sample_id"])
        prediction_path = prediction_dir / f"{sample_id}.json"
        if prediction_path.exists():
            existing = _read_json(prediction_path)
            if existing.get("execution_id") != state["execution_id"]:
                raise RuntimeError(f"prediction {sample_id} belongs to another execution")
            continue
        graph_path = Path(manifest_record["graph_path"])
        if _sha256(graph_path) != manifest_record["graph_sha256"]:
            raise RuntimeError(f"Holdout graph hash drift: {sample_id}")
        truth_graph = _read_json(graph_path)
        inference_graph = _strip_truth(truth_graph)
        _assert_truth_free(inference_graph)
        face_prediction = frozen_inference._predict_gnn(
            inference_graph, gnn, gnn_config, training_metadata, device
        )
        pipeline = run_p1_b_general_recognition(
            inference_graph, face_prediction, candidate_model,
            same_instance_thresholds=same_thresholds,
            candidate_thresholds=candidate_thresholds,
            global_decode_selection=global_selection,
            device=str(device),
        )
        truth_objects = _truth_objects(truth_graph, sample_id, graph_path)
        prediction_objects = [
            {**record, "sample_id": sample_id}
            for record in pipeline["decoded_feature_objects"]["feature_objects"]
        ]
        proposals = pipeline["candidate_proposals"]["proposals"]
        oracle_05 = evaluate_proposal_oracle_recall(proposals, truth_objects, iou_threshold=0.5)
        oracle_exact = evaluate_proposal_oracle_recall(proposals, truth_objects, iou_threshold=0.95)
        face_truth_macro = [macro_names.index(str(node["macro_feature"])) for node in truth_graph["nodes"]]
        face_prediction_macro = [int(record["macro_class_id"]) for record in face_prediction["faces"]]
        semantic_truth = [int(node.get("semantic_label", 0) or 0) for node in truth_graph["nodes"]]
        semantic_prediction = [int(record["semantic_label"]) for record in face_prediction["faces"]]
        edge_truth = _edge_truth(truth_graph)
        edge_probability = [float(value) for value in face_prediction["edge_same_instance_probabilities"]]
        edge_prediction = [int(value >= float(same_thresholds["global_threshold"])) for value in edge_probability]
        _write_json(prediction_path, {
            "schema_version": "topocae_p1_b_holdout_v2_sample_prediction_1",
            "execution_id": state["execution_id"],
            "sample_id": sample_id,
            "target_macro_feature": manifest_record["target_macro_feature"],
            "source_graph_path": str(graph_path.resolve()),
            "source_graph_sha256": manifest_record["graph_sha256"],
            "face_truth_macro": face_truth_macro,
            "face_prediction_macro": face_prediction_macro,
            "semantic_truth": semantic_truth,
            "semantic_prediction": semantic_prediction,
            "edge_truth": edge_truth,
            "edge_prediction": edge_prediction,
            "edge_probability": edge_probability,
            "truth_objects": truth_objects,
            "prediction_objects": prediction_objects,
            "proposal_oracle_iou_0_5": oracle_05,
            "proposal_oracle_iou_0_95": oracle_exact,
            "proposal_source_counts": pipeline["candidate_proposals"]["source_counts"],
            "diagnostics": {
                "proposal_count": pipeline["candidate_proposals"]["candidate_count"],
                "verified_candidate_count": len(pipeline["verified_candidates"]),
                "abstention_count": pipeline["decoded_feature_objects"]["abstention_count"],
                "global_conflict_reject_count": pipeline["decoded_feature_objects"]["global_conflict_reject_count"],
            },
            "contracts": {"truth_visible_to_inference": False, "thresholds_selected_synthetic_val_only": True},
        })
        if (position + 1) % 20 == 0 or position + 1 == len(records):
            print(f"holdout_v2_inference={position + 1}/{len(records)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    sample_outputs = [_read_json(prediction_dir / f"{record['sample_id']}.json") for record in records]
    metrics = _aggregate_metrics(sample_outputs, macro_names)
    review = _build_review_package(records, sample_outputs)
    result = {
        "schema_version": "topocae_p1_b_holdout_v2_formal_report_1",
        "phase": "P1-B",
        "dataset_name": "Synthetic-Holdout-v2",
        "formal_inference_pass_count": 1,
        "execution_id": state["execution_id"],
        "sample_count": len(records),
        "selection_sha256": selection_sha,
        "inference_script_sha256": script_sha,
        "metrics": metrics,
        "review_package": review,
        "timing_seconds": time.perf_counter() - started,
        "contracts": {
            "holdout_used_for_training": False,
            "holdout_used_for_threshold_or_model_selection": False,
            "results_used_to_modify_model_rule_or_threshold": False,
            "truth_visible_to_inference": False,
            "part_specific_thresholds": False,
            "llm_step_recognition": False,
            "test_part_s3_used": False,
            "second_formal_inference_forbidden": True,
        },
    }
    _write_json(REPORT / "metrics.json", result)
    _write_csv(REPORT / "per_class_object_metrics.csv", [
        {"macro_feature": macro, **values}
        for macro, values in metrics["object"]["per_class"].items()
    ])
    _write_csv(REPORT / "per_class_face_metrics.csv", [
        {"class_id": key, **value} for key, value in metrics["face_macro"]["per_class"].items()
    ])
    (REPORT / "report.zh-CN.md").write_text(_report_markdown(result), encoding="utf-8")
    final_state = {
        **state,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_sample_count": len(records),
        "metrics_sha256": _sha256(REPORT / "metrics.json"),
    }
    _write_json(STATE, final_state)
    generation_manifest["inference_pass_count"] = 1
    generation_manifest["formal_inference"] = {
        "execution_id": state["execution_id"],
        "selection_sha256": selection_sha,
        "script_sha256": script_sha,
        "metrics_sha256": final_state["metrics_sha256"],
        "completed_at_utc": final_state["completed_at_utc"],
    }
    _write_json(generation_manifest_path, generation_manifest)
    _write_freeze_manifest()
    print(json.dumps({
        "formal_inference_complete": True,
        "sample_count": len(records),
        "face_macro_f1": metrics["face_macro"]["macro_f1"],
        "semantic_macro_f1": metrics["semantic"]["macro_f1"],
        "same_instance_edge_f1": metrics["same_instance_edge"]["f1"],
        "proposal_oracle_recall_iou_0_5": metrics["candidate_proposals"]["oracle_recall_iou_0_5"],
        "object_macro_f1": metrics["object"]["object_macro_f1"],
        "exact_instance_ratio": metrics["object"]["exact_instance_ratio"],
        "boundary_f1": metrics["object"]["boundary"]["f1"],
        "output": str(REPORT),
    }, ensure_ascii=False), flush=True)
    return 0


def _start_or_resume_state(selection_sha: str, script_sha: str, sample_count: int) -> dict[str, Any]:
    if STATE.exists():
        state = _read_json(STATE)
        if state.get("status") == "completed":
            raise RuntimeError("formal Holdout-v2 inference already completed; a second pass is forbidden")
        if state.get("selection_sha256") != selection_sha or state.get("script_sha256") != script_sha:
            raise RuntimeError("cannot resume formal inference after frozen selection or script drift")
        return state
    state = {
        "schema_version": "topocae_p1_b_holdout_v2_inference_state_1",
        "status": "running",
        "formal_inference_pass": 1,
        "execution_id": f"P1B-HOLDOUT-V2-{int(time.time_ns())}",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": selection_sha,
        "script_sha256": script_sha,
        "expected_sample_count": sample_count,
        "resume_contract": "resume may read existing outputs but must never recompute an existing sample",
    }
    _write_json(STATE, state)
    return state


def _strip_truth(graph: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(graph)
    result["feature_sets"] = []
    result.pop("p1_b_holdout_provenance", None)
    for node in result.get("nodes", []):
        for key in ("semantic_label", "instance_id", "macro_feature", "macro_feature_id", "semantic_role"):
            node.pop(key, None)
    return result


def _assert_truth_free(graph: dict[str, Any]) -> None:
    if graph.get("feature_sets"):
        raise RuntimeError("truth feature_sets reached inference")
    forbidden = {"semantic_label", "instance_id", "macro_feature", "macro_feature_id", "semantic_role"}
    if any(forbidden & set(node) for node in graph.get("nodes", [])):
        raise RuntimeError("truth node labels reached inference")


def _truth_objects(graph: dict[str, Any], sample_id: str, graph_path: Path) -> list[dict[str, Any]]:
    return [{
        "sample_id": sample_id,
        "instance_id": int(record.get("instance_id", position + 1)),
        "macro_feature": str(record["macro_feature"]),
        "face_indices": sorted(int(index) for index in record.get("node_indices", record.get("face_indices", []))),
        "source_graph_path": str(graph_path.resolve()),
    } for position, record in enumerate(graph.get("feature_sets", []))]


def _edge_truth(graph: dict[str, Any]) -> list[int]:
    instances = [int(node.get("instance_id", 0) or 0) for node in graph["nodes"]]
    return [int(instances[int(left)] > 0 and instances[int(left)] == instances[int(right)]) for left, right in graph["edges"]]


def _aggregate_metrics(outputs: list[dict[str, Any]], macro_names: list[str]) -> dict[str, Any]:
    face_truth = [value for item in outputs for value in item["face_truth_macro"]]
    face_prediction = [value for item in outputs for value in item["face_prediction_macro"]]
    semantic_truth = [value for item in outputs for value in item["semantic_truth"]]
    semantic_prediction = [value for item in outputs for value in item["semantic_prediction"]]
    edge_truth = [value for item in outputs for value in item["edge_truth"]]
    edge_prediction = [value for item in outputs for value in item["edge_prediction"]]
    truths = [record for item in outputs for record in item["truth_objects"]]
    predictions = [record for item in outputs for record in item["prediction_objects"]]
    object_metrics = metric_impl._object_metrics(truths, predictions, macro_names[1:])
    proposal_truth = sum(item["proposal_oracle_iou_0_5"]["truth_instance_count"] for item in outputs)
    proposal_hit_05 = sum(item["proposal_oracle_iou_0_5"]["hit_count"] for item in outputs)
    proposal_hit_exact = sum(item["proposal_oracle_iou_0_95"]["hit_count"] for item in outputs)
    source_counts = Counter()
    for item in outputs:
        source_counts.update(item["proposal_source_counts"])
    required_sources = {
        "same_instance_strong_component", "coaxial_chain", "loop_guided_region",
        "concave_region_grow", "elongated_channel", "thin_wall", "parent_transition",
    }
    return {
        "face_macro": _classification_metrics(face_truth, face_prediction, labels=list(range(len(macro_names))), names=macro_names),
        "semantic": _classification_metrics(semantic_truth, semantic_prediction),
        "same_instance_edge": _binary_metrics(edge_truth, edge_prediction),
        "candidate_proposals": {
            "truth_instance_count": proposal_truth,
            "hit_iou_0_5": proposal_hit_05,
            "hit_iou_0_95": proposal_hit_exact,
            "oracle_recall_iou_0_5": proposal_hit_05 / max(proposal_truth, 1),
            "oracle_recall_iou_0_95": proposal_hit_exact / max(proposal_truth, 1),
            "source_counts": dict(sorted(source_counts.items())),
            "all_seven_sources_present": required_sources <= set(source_counts),
        },
        "object": object_metrics,
    }


def _classification_metrics(
    truth: list[int], prediction: list[int], labels: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    active = labels if labels is not None else sorted(set(truth))
    per_class: dict[str, Any] = {}
    f1s = []
    correct = sum(int(left == right) for left, right in zip(truth, prediction))
    for label in active:
        tp = sum(int(left == label and right == label) for left, right in zip(truth, prediction))
        fp = sum(int(left != label and right == label) for left, right in zip(truth, prediction))
        fn = sum(int(left == label and right != label) for left, right in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = names[label] if names is not None and label < len(names) else str(label)
        support = sum(int(value == label) for value in truth)
        per_class[key] = {"class_id": label, "support": support, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        if support:
            f1s.append(f1)
    return {"sample_count": len(truth), "accuracy": correct / max(len(truth), 1), "macro_f1": sum(f1s) / max(len(f1s), 1), "active_class_count": len(f1s), "per_class": per_class}


def _binary_metrics(truth: list[int], prediction: list[int]) -> dict[str, Any]:
    tp = sum(int(left == 1 and right == 1) for left, right in zip(truth, prediction))
    fp = sum(int(left == 0 and right == 1) for left, right in zip(truth, prediction))
    fn = sum(int(left == 1 and right == 0) for left, right in zip(truth, prediction))
    tn = sum(int(left == 0 and right == 0) for left, right in zip(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _build_review_package(manifest: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(record["target_macro_feature"], int(record["class_index"])): record for record in manifest}
    output_by_id = {record["sample_id"]: record for record in outputs}
    classes = []
    for record in manifest:
        macro = record["target_macro_feature"]
        if macro not in classes:
            classes.append(macro)
    selected = [by_key[(macro, 0)] for macro in classes]
    selected.extend([by_key[("slot_rect", 1)], by_key[("rib", 1)]])
    review_dir = REPORT / "human_review_stratified_20"
    review_dir.mkdir(exist_ok=True)
    rows = []
    for position, record in enumerate(selected, start=1):
        destination = review_dir / f"{position:02d}_{record['sample_id']}.step"
        shutil.copy2(record["step_path"], destination)
        output = output_by_id[record["sample_id"]]
        truths = [{"macro": item["macro_feature"], "faces": item["face_indices"]} for item in output["truth_objects"]]
        predictions = [{"macro": item["macro_feature"], "faces": item["face_indices"]} for item in output["prediction_objects"]]
        rows.append({
            "review_index": position,
            "sample_id": record["sample_id"],
            "stratum_macro_feature": record["target_macro_feature"],
            "step_file": destination.name,
            "truth_instance_summary": json.dumps(truths, ensure_ascii=False),
            "frozen_prediction_summary": json.dumps(predictions, ensure_ascii=False),
            "human_geometry_valid": "",
            "human_instance_boundary_correct": "",
            "human_notes": "",
        })
    csv_path = REPORT / "human_review_stratified_20.csv"
    _write_csv(csv_path, rows)
    instructions = [
        "# Synthetic-Holdout-v2 20 件分层审查包", "",
        "该包用于留档审查，不用于修改本次冻结模型、规则或阈值。", "",
        "- 20 个 STEP 覆盖全部 18 类；额外抽取 slot_rect 与 rib 各 1 件。",
        "- 在 UG/NX 中检查几何是否为单实体、目标特征概念是否成立、同一实例边界是否连续。",
        "- 如需人工留档，可填写 human_review_stratified_20.csv 的最后三列；填写结果不得触发本 Holdout 的返工。", "",
    ]
    (REPORT / "human_review_instructions.zh-CN.md").write_text("\n".join(instructions), encoding="utf-8")
    return {"step_count": len(selected), "class_count": len(classes), "all_18_classes_covered": len(classes) == 18, "slot_present": any("slot_" in item["target_macro_feature"] for item in selected), "rib_present": any(item["target_macro_feature"] == "rib" for item in selected), "web_present": any(item["target_macro_feature"] == "web" for item in selected), "csv_path": str(csv_path.resolve()), "step_directory": str(review_dir.resolve())}


def _verify_frozen_inputs() -> str:
    sidecar = _read_json(SIDECAR)
    selection_sha = _sha256(SELECTION)
    if sidecar.get("sha256") != selection_sha or not sidecar.get("verified_after_write"):
        raise RuntimeError("preholdout selection hash verification failed")
    selection = _read_json(SELECTION)
    for record in selection["frozen_files"]:
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {path}")
    return selection_sha


def _write_freeze_manifest() -> None:
    files = [
        REPORT / "metrics.json", REPORT / "report.zh-CN.md", REPORT / "per_class_object_metrics.csv",
        REPORT / "per_class_face_metrics.csv", REPORT / "human_review_stratified_20.csv",
        REPORT / "human_review_instructions.zh-CN.md", STATE,
        HOLDOUT / "holdout_generation_manifest.json", HOLDOUT / "leakage_audit.json", HOLDOUT / "quality_report.json",
    ]
    _write_json(REPORT / "freeze_manifest.json", {
        "schema_version": "topocae_p1_b_holdout_v2_freeze_manifest_1",
        "formal_inference_pass_count": 1,
        "files": [{"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in files],
        "sample_prediction_count": len(list((REPORT / "sample_predictions").glob("*.json"))),
        "review_step_count": len(list((REPORT / "human_review_stratified_20").glob("*.step"))),
    })


def _report_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    obj = metrics["object"]
    return "\n".join([
        "# P1-B Synthetic-Holdout-v2 一次性正式报告", "",
        f"- 样本：{result['sample_count']}（18 类各 60）",
        f"- 正式推理次数：{result['formal_inference_pass_count']}",
        f"- Face Macro-F1：{metrics['face_macro']['macro_f1']:.6f}",
        f"- Semantic Macro-F1：{metrics['semantic']['macro_f1']:.6f}",
        f"- Same-instance edge F1：{metrics['same_instance_edge']['f1']:.6f}",
        f"- Candidate proposal oracle recall (IoU>=0.5)：{metrics['candidate_proposals']['oracle_recall_iou_0_5']:.6f}",
        f"- Candidate proposal oracle recall (IoU>=0.95)：{metrics['candidate_proposals']['oracle_recall_iou_0_95']:.6f}",
        f"- Object Macro-F1：{obj['object_macro_f1']:.6f}",
        f"- Object Micro-F1：{obj['object_micro_f1']:.6f}",
        f"- Exact instance ratio：{obj['exact_instance_ratio']:.6f}",
        f"- Mean best instance IoU：{obj['mean_best_instance_iou']:.6f}",
        f"- Boundary F1：{obj['boundary']['f1']:.6f}", "",
        "本结果在 preholdout selection hash 写入、1080 件生成及 base/exact/near 零碰撞审计后，仅正式推理一次。",
        "Holdout-v2 未参与训练、early stopping、阈值或模型选择；无论指标高低，均不得回头修改本冻结版本。", "",
    ])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


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
