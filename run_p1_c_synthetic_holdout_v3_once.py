from __future__ import annotations

"""Exactly-once paired P1-B/P1-C inference on frozen Synthetic-Holdout-v3."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import evaluate_p1_b_synthetic_pipeline as object_metrics_impl
from models.gnn_baseline import EdgeAwareMultiHeadGNN
from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from p1_b_general_recognition_pipeline_v4 import run_p1_b_general_recognition
from p1_c_general_recognition_pipeline_v1 import run_p1_c_general_recognition
import run_p1_b_real_dev_evaluation as inference
import run_p1_b_synthetic_holdout_v2_once as metric_helpers
from utils.p1_b_candidate_proposer import evaluate_proposal_oracle_recall


ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "output_generation_p1_c_synthetic_holdout_v3"
REPORT = ROOT / "reports" / "p1_c_holdout_v3_once"
STATE = REPORT / "formal_inference_state.json"
P1_B_FREEZE = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"
P1_B_SIDECAR = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json"
P1_C_FREEZE = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.json"
P1_C_SIDECAR = ROOT / "reports" / "p1_c_freeze_v1" / "preholdout_selection.sha256.json"
P1_B_RUN = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718"
P1_B_PACKAGE = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2"
P1_C_GNN = ROOT / "reports" / "p1_c_strict_training_runs_v2" / "clean_7200_seed_20260718"
P1_C_CANDIDATE = ROOT / "reports" / "p1_c_stage6_iteration1_runs_v1" / "clean_7200_seed_20260718" / "candidate_model_v1"
P1_C_ITERATION2 = ROOT / "reports" / "p1_c_stage6_iteration2_runs_v1" / "clean_7200_seed_20260718"
P1_C_PACKAGE = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2" / "training_packages" / "clean_7200"
VARIANTS = ("P1-B-frozen", "P1-C-clean-7200-seed-20260718")
REQUIRED_PROPOSAL_SOURCES = {
    "same_instance_strong_component", "coaxial_cylinder_cone_chain", "loop_material_side_guided",
    "concave_boundary_region", "elongated_swept_channel", "thin_wall_rib_web", "parent_surface_transition",
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze(selection: Path, sidecar_path: Path) -> str:
    sidecar = read_json(sidecar_path)
    actual = digest(selection)
    if actual != str(sidecar["sha256"]).lower() or not sidecar["verified_after_write"]:
        raise RuntimeError(f"freeze sidecar mismatch: {selection}")
    payload = read_json(selection)
    for record in payload["frozen_files"]:
        path = Path(record["path"])
        if digest(path) != str(record["sha256"]).lower():
            raise RuntimeError(f"frozen input drift: {path}")
    return actual


def verify_inputs() -> dict[str, str]:
    p1_b_sha = verify_freeze(P1_B_FREEZE, P1_B_SIDECAR)
    p1_c_sha = verify_freeze(P1_C_FREEZE, P1_C_SIDECAR)
    generation_manifest = read_json(HOLDOUT / "holdout_generation_manifest.json")
    if int(generation_manifest["sample_count"]) != 1440:
        raise RuntimeError("Holdout-v3 sample count mismatch")
    if int(generation_manifest.get("inference_pass_count", 0)) != 0:
        raise RuntimeError("Holdout-v3 formal inference already completed")
    leakage = read_json(HOLDOUT / "reports" / "leakage_audit.json")
    quality = read_json(HOLDOUT / "reports" / "quality_report.json")
    if not leakage["all_zero"] or not quality["automatic_pass"]:
        raise RuntimeError("Holdout-v3 generation audit failed")
    declared = read_json(P1_C_FREEZE)["holdout_v3_declared_comparison_variants"]
    declared_ids = [record["id"] for record in declared]
    if declared_ids != ["P1-B-frozen", "P1-C-clean-7200-seed-20260718"]:
        raise RuntimeError(f"unexpected predeclared Holdout variants: {declared_ids}")
    return {
        "p1_b_selection_sha256": p1_b_sha,
        "p1_c_selection_sha256": p1_c_sha,
        "generation_manifest_sha256": digest(HOLDOUT / "holdout_generation_manifest.json"),
    }


def load_manifest() -> list[dict[str, Any]]:
    records = []
    with (HOLDOUT / "manifests" / "sample_manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != 1440:
        raise RuntimeError("Holdout-v3 manifest requires 1440 records")
    return records


def start_or_resume_state(hashes: dict[str, str], script_sha: str, sample_count: int) -> dict[str, Any]:
    if STATE.exists():
        state = read_json(STATE)
        if state["status"] == "completed":
            raise RuntimeError("Holdout-v3 formal inference already completed; second run forbidden")
        expected = {**hashes, "script_sha256": script_sha, "expected_sample_count": sample_count}
        differences = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
        if differences:
            raise RuntimeError(f"formal inference resume lock mismatch: {differences}")
        return state
    REPORT.mkdir(parents=True, exist_ok=False)
    (REPORT / "sample_predictions").mkdir()
    state = {
        "schema_version": "topocae_p1_c_holdout_v3_inference_state_1",
        "status": "running",
        "formal_inference_pass": 1,
        "execution_id": f"P1C-HOLDOUT-V3-{time.time_ns()}",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **hashes,
        "script_sha256": script_sha,
        "expected_sample_count": sample_count,
        "comparison_variants": list(VARIANTS),
        "resume_contract": "same script/hash may compute missing samples only; existing sample outputs are never recomputed",
        "contracts": {"test_part_s3_used": False, "holdout_used_for_selection": False},
    }
    write_json(STATE, state)
    return state


def load_gnn(checkpoint_path: Path, device: torch.device) -> tuple[EdgeAwareMultiHeadGNN, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dims = config["model"]["dimensions"]
    model = EdgeAwareMultiHeadGNN(
        input_dim=dims["node"], edge_feature_dim=dims["edge"],
        num_primitive_surfaces=dims["primitive"], num_macro_classes=dims["macro"],
        num_semantic_classes=dims["semantic"], derived_target_dim=0, idealization_target_dim=0,
        hidden_dim=config["model"]["hidden_dim"], num_layers=config["model"]["num_layers"],
        primitive_embedding_dim=16, dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def load_candidate(path: Path, device: torch.device) -> CandidateTopologyClassifier:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CandidateTopologyClassifier(
        CandidateModelDimensions(**config["dimensions"]), macro_classes=config["macro_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_gnn(graph, model, config, metadata, package: Path, device):
    previous = inference.TRAINING_PACKAGE
    inference.TRAINING_PACKAGE = package
    try:
        return inference._predict_gnn(graph, model, config, metadata, device)
    finally:
        inference.TRAINING_PACKAGE = previous


def variant_record(
    variant: str, truth_graph: dict[str, Any], inference_graph: dict[str, Any], face_prediction: dict[str, Any],
    pipeline: dict[str, Any], same_thresholds: dict[str, Any], macro_names: list[str], sample_id: str, graph_path: Path,
) -> dict[str, Any]:
    truth_objects = metric_helpers._truth_objects(truth_graph, sample_id, graph_path)
    prediction_objects = [{**record, "sample_id": sample_id} for record in pipeline["decoded_feature_objects"]["feature_objects"]]
    proposals = pipeline["candidate_proposals"]["proposals"]
    oracle_05 = evaluate_proposal_oracle_recall(proposals, truth_objects, iou_threshold=0.5)
    oracle_095 = evaluate_proposal_oracle_recall(proposals, truth_objects, iou_threshold=0.95)
    return {
        "variant": variant,
        "face_truth_macro": [macro_names.index(str(node["macro_feature"])) for node in truth_graph["nodes"]],
        "face_prediction_macro": [int(record["macro_class_id"]) for record in face_prediction["faces"]],
        "semantic_truth": [int(node.get("semantic_label", 0) or 0) for node in truth_graph["nodes"]],
        "semantic_prediction": [int(record["semantic_label"]) for record in face_prediction["faces"]],
        "edge_truth": metric_helpers._edge_truth(truth_graph),
        "edge_prediction": [
            int(float(value) >= float(same_thresholds["global_threshold"]))
            for value in face_prediction["edge_same_instance_probabilities"]
        ],
        "truth_objects": truth_objects,
        "prediction_objects": prediction_objects,
        "proposal_oracle_iou_0_5": oracle_05,
        "proposal_oracle_iou_0_95": oracle_095,
        "proposal_source_counts": pipeline["candidate_proposals"]["source_counts"],
        "diagnostics": {
            "proposal_count": pipeline["candidate_proposals"]["candidate_count"],
            "verified_candidate_count": len(pipeline["verified_candidates"]),
            "abstention_count": pipeline["decoded_feature_objects"]["abstention_count"],
            "global_conflict_reject_count": pipeline["decoded_feature_objects"]["global_conflict_reject_count"],
        },
        "pipeline_manifest": pipeline["pipeline_manifest"],
    }


def aggregate(outputs: list[dict[str, Any]], macro_names: list[str]) -> dict[str, Any]:
    face_truth = [value for item in outputs for value in item["face_truth_macro"]]
    face_prediction = [value for item in outputs for value in item["face_prediction_macro"]]
    semantic_truth = [value for item in outputs for value in item["semantic_truth"]]
    semantic_prediction = [value for item in outputs for value in item["semantic_prediction"]]
    edge_truth = [value for item in outputs for value in item["edge_truth"]]
    edge_prediction = [value for item in outputs for value in item["edge_prediction"]]
    truths = [record for item in outputs for record in item["truth_objects"]]
    predictions = [record for item in outputs for record in item["prediction_objects"]]
    objects = object_metrics_impl._object_metrics(truths, predictions, macro_names[1:])
    proposal_truth = sum(item["proposal_oracle_iou_0_5"]["truth_instance_count"] for item in outputs)
    hit_05 = sum(item["proposal_oracle_iou_0_5"]["hit_count"] for item in outputs)
    hit_095 = sum(item["proposal_oracle_iou_0_95"]["hit_count"] for item in outputs)
    source_counts = Counter()
    for item in outputs:
        source_counts.update(item["proposal_source_counts"])
    return {
        "face_macro": metric_helpers._classification_metrics(face_truth, face_prediction, labels=list(range(len(macro_names))), names=macro_names),
        "semantic": metric_helpers._classification_metrics(semantic_truth, semantic_prediction),
        "same_instance_edge": metric_helpers._binary_metrics(edge_truth, edge_prediction),
        "candidate_proposals": {
            "truth_instance_count": proposal_truth, "hit_iou_0_5": hit_05, "hit_iou_0_95": hit_095,
            "oracle_recall_iou_0_5": hit_05 / max(proposal_truth, 1),
            "oracle_recall_iou_0_95": hit_095 / max(proposal_truth, 1),
            "source_counts": dict(sorted(source_counts.items())),
            "all_seven_sources_present": REQUIRED_PROPOSAL_SOURCES <= set(source_counts),
        },
        "object": objects,
    }


def stratified(sample_outputs: list[dict[str, Any]], macro_names: list[str], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sample_outputs:
        groups[str(item[key])].append(item["variant_output"])
    result = {}
    for name, outputs in sorted(groups.items()):
        metrics = aggregate(outputs, macro_names)
        result[name] = {
            "sample_count": len(outputs),
            "face_macro_f1": metrics["face_macro"]["macro_f1"],
            "semantic_macro_f1": metrics["semantic"]["macro_f1"],
            "same_instance_edge_f1": metrics["same_instance_edge"]["f1"],
            "proposal_oracle_recall_iou_0_5": metrics["candidate_proposals"]["oracle_recall_iou_0_5"],
            "object_macro_f1": metrics["object"]["object_macro_f1"],
            "object_micro_f1": metrics["object"]["object_micro_f1"],
            "exact_instance_ratio": metrics["object"]["exact_instance_ratio"],
            "instance_iou": metrics["object"]["mean_best_instance_iou"],
            "boundary_f1": metrics["object"]["boundary"]["f1"],
        }
    return result


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "face_macro_f1": float(metrics["face_macro"]["macro_f1"]),
        "semantic_macro_f1": float(metrics["semantic"]["macro_f1"]),
        "same_instance_edge_f1": float(metrics["same_instance_edge"]["f1"]),
        "proposal_oracle_recall_iou_0_5": float(metrics["candidate_proposals"]["oracle_recall_iou_0_5"]),
        "object_macro_f1": float(metrics["object"]["object_macro_f1"]),
        "object_micro_f1": float(metrics["object"]["object_micro_f1"]),
        "exact_instance_ratio": float(metrics["object"]["exact_instance_ratio"]),
        "instance_iou": float(metrics["object"]["mean_best_instance_iou"]),
        "boundary_f1": float(metrics["object"]["boundary"]["f1"]),
    }


def bootstrap_deltas(sample_records: list[dict[str, Any]], macro_names: list[str], replicates: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(20260719)
    n = len(sample_records)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        for variant in VARIANTS:
            outputs = [sample_records[int(index)]["variants"][variant] for index in indices]
            scalars = scalar_metrics(aggregate(outputs, macro_names))
            for key, value in scalars.items():
                values[f"{variant}::{key}"].append(value)
    result = {}
    for key in scalar_metrics(aggregate([sample_records[0]["variants"][VARIANTS[0]]], macro_names)):
        left = np.asarray(values[f"{VARIANTS[0]}::{key}"])
        right = np.asarray(values[f"{VARIANTS[1]}::{key}"])
        delta = right - left
        result[key] = {
            "delta_p1_c_minus_p1_b": float(delta.mean()),
            "paired_bootstrap_95_ci": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
            "replicates": replicates,
            "seed": 20260719,
        }
    return result


def main() -> int:
    hashes = verify_inputs()
    records = load_manifest()
    script_sha = digest(Path(__file__).resolve())
    state = start_or_resume_state(hashes, script_sha, len(records))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p1_b_metadata = read_json(P1_B_PACKAGE / "metadata.json")
    p1_c_metadata = read_json(P1_C_PACKAGE / "metadata.json")
    if p1_b_metadata["macro_classes"] != p1_c_metadata["macro_classes"]:
        raise RuntimeError("P1-B/P1-C macro ontology mismatch")
    macro_names = list(p1_c_metadata["macro_classes"])
    p1_b_gnn, p1_b_gnn_config = load_gnn(P1_B_RUN / "best.pt", device)
    p1_b_candidate = load_candidate(P1_B_RUN / "candidate_model_v1" / "best.pt", device)
    p1_c_gnn, p1_c_gnn_config = load_gnn(P1_C_GNN / "best.pt", device)
    p1_c_candidate = load_candidate(P1_C_CANDIDATE / "best.pt", device)
    p1_b_same = read_json(P1_B_RUN / "same_instance_threshold_selection.json")
    p1_b_candidate_thresholds = read_json(P1_B_RUN / "candidate_model_v1" / "candidate_threshold_selection.json")
    p1_b_global = read_json(P1_B_RUN / "global_decode_v2_selection.json")
    p1_c_same = read_json(P1_C_GNN / "same_instance_threshold_selection.json")
    p1_c_candidate_thresholds = read_json(P1_C_ITERATION2 / "candidate_thresholds_iteration2.json")
    p1_c_global = read_json(P1_C_ITERATION2 / "global_candidate_decode_selection.json")
    prediction_dir = REPORT / "sample_predictions"
    started = time.perf_counter()
    for position, manifest_record in enumerate(records, 1):
        sample_id = str(manifest_record["sample_id"])
        output_path = prediction_dir / f"{sample_id}.json"
        if output_path.exists():
            existing = read_json(output_path)
            if existing["execution_id"] != state["execution_id"]:
                raise RuntimeError(f"existing output belongs to another execution: {sample_id}")
            if existing["source_graph_sha256"].lower() != str(manifest_record["graph_sha256"]).lower():
                raise RuntimeError(f"existing output source hash mismatch: {sample_id}")
            continue
        graph_path = Path(manifest_record["graph_path"])
        if digest(graph_path) != str(manifest_record["graph_sha256"]).lower():
            raise RuntimeError(f"Holdout-v3 graph hash drift: {sample_id}")
        truth_graph = read_json(graph_path)
        inference_graph = metric_helpers._strip_truth(truth_graph)
        inference_graph.pop("holdout_v3_provenance", None)
        metric_helpers._assert_truth_free(inference_graph)

        p1_b_face = predict_gnn(inference_graph, p1_b_gnn, p1_b_gnn_config, p1_b_metadata, P1_B_PACKAGE, device)
        p1_b_pipeline = run_p1_b_general_recognition(
            inference_graph, p1_b_face, p1_b_candidate,
            same_instance_thresholds=p1_b_same, candidate_thresholds=p1_b_candidate_thresholds,
            global_decode_selection=p1_b_global, device=str(device),
        )
        p1_c_face = predict_gnn(inference_graph, p1_c_gnn, p1_c_gnn_config, p1_c_metadata, P1_C_PACKAGE, device)
        p1_c_pipeline = run_p1_c_general_recognition(
            inference_graph, p1_c_face, p1_c_candidate,
            same_instance_thresholds=p1_c_same, candidate_thresholds=p1_c_candidate_thresholds,
            global_candidate_decode_selection=p1_c_global, device=str(device),
        )
        variants = {
            VARIANTS[0]: variant_record(VARIANTS[0], truth_graph, inference_graph, p1_b_face, p1_b_pipeline, p1_b_same, macro_names, sample_id, graph_path),
            VARIANTS[1]: variant_record(VARIANTS[1], truth_graph, inference_graph, p1_c_face, p1_c_pipeline, p1_c_same, macro_names, sample_id, graph_path),
        }
        write_json(output_path, {
            "schema_version": "topocae_p1_c_holdout_v3_paired_sample_prediction_1",
            "execution_id": state["execution_id"], "sample_id": sample_id,
            "primary_target": manifest_record["primary_target"], "context": manifest_record["context"],
            "holdout_host_template_id": manifest_record["holdout_host_template_id"],
            "source_graph_path": str(graph_path), "source_graph_sha256": manifest_record["graph_sha256"],
            "variants": variants,
            "contracts": {"truth_visible_to_inference": False, "same_job_paired_comparison": True, "test_part_s3_used": False},
        })
        if position % 20 == 0 or position == len(records):
            print(f"holdout_v3_paired_inference={position}/{len(records)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    samples = [read_json(prediction_dir / f"{record['sample_id']}.json") for record in records]
    metrics_by_variant = {}
    for variant in VARIANTS:
        outputs = [sample["variants"][variant] for sample in samples]
        metrics = aggregate(outputs, macro_names)
        variant_samples = [{
            "context": sample["context"], "primary_target": sample["primary_target"],
            "host_template": sample["holdout_host_template_id"], "variant_output": sample["variants"][variant],
        } for sample in samples]
        metrics_by_variant[variant] = {
            "overall": metrics,
            "stratified": {
                "context": stratified(variant_samples, macro_names, "context"),
                "primary_target": stratified(variant_samples, macro_names, "primary_target"),
                "host_template": stratified(variant_samples, macro_names, "host_template"),
            },
        }
    scalar_by_variant = {variant: scalar_metrics(metrics_by_variant[variant]["overall"]) for variant in VARIANTS}
    direct_deltas = {
        key: scalar_by_variant[VARIANTS[1]][key] - scalar_by_variant[VARIANTS[0]][key]
        for key in scalar_by_variant[VARIANTS[0]]
    }
    bootstrap = bootstrap_deltas(samples, macro_names, replicates=1000)
    result = {
        "schema_version": "topocae_p1_c_holdout_v3_formal_report_1",
        "dataset_name": "Synthetic-Holdout-v3", "formal_inference_pass_count": 1,
        "execution_id": state["execution_id"], "sample_count": len(samples),
        "comparison_variants": list(VARIANTS), "frozen_inputs": hashes,
        "metrics_by_variant": metrics_by_variant,
        "scalar_comparison": scalar_by_variant,
        "deltas_p1_c_minus_p1_b": direct_deltas,
        "paired_bootstrap": bootstrap,
        "review_package": {
            "path": str(HOLDOUT / "human_review_36"), "step_count": 36,
            "human_gate": False, "used_for_return_work": False,
        },
        "timing_seconds": time.perf_counter() - started,
        "contracts": {
            "holdout_used_for_training_or_selection": False,
            "results_used_to_modify_model_data_rule_or_threshold": False,
            "truth_visible_to_inference": False,
            "one_formal_job_with_both_variants": True,
            "part_or_class_specific_thresholds": False,
            "llm_step_recognition": False, "test_part_s3_used": False,
            "second_formal_inference_forbidden": True,
        },
    }
    metrics_path = REPORT / "metrics.json"
    write_json(metrics_path, result)
    write_csv(REPORT / "overall_comparison.csv", [
        {"variant": variant, **scalar_by_variant[variant]} for variant in VARIANTS
    ])
    class_rows = []
    for variant in VARIANTS:
        for macro, row in metrics_by_variant[variant]["overall"]["object"]["per_class"].items():
            class_rows.append({"variant": variant, "macro_feature": macro, **row})
    write_csv(REPORT / "per_class_object_metrics.csv", class_rows)
    stratified_rows = []
    for variant in VARIANTS:
        for dimension, groups in metrics_by_variant[variant]["stratified"].items():
            for group, values in groups.items():
                stratified_rows.append({"variant": variant, "dimension": dimension, "group": group, **values})
    write_csv(REPORT / "stratified_metrics.csv", stratified_rows)
    lines = [
        "# P1-C Synthetic-Holdout-v3 一次性正式报告", "",
        f"正式推理次数：1；样本数：{len(samples)}；P1-B 与 P1-C 在同一作业中配对完成。", "",
        "| 变体 | face F1 | semantic F1 | edge F1 | proposal oracle | object F1 | exact | IoU | boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = scalar_by_variant[variant]
        lines.append(
            f"| {variant} | {row['face_macro_f1']:.6f} | {row['semantic_macro_f1']:.6f} | "
            f"{row['same_instance_edge_f1']:.6f} | {row['proposal_oracle_recall_iou_0_5']:.6f} | "
            f"{row['object_macro_f1']:.6f} | {row['exact_instance_ratio']:.6f} | "
            f"{row['instance_iou']:.6f} | {row['boundary_f1']:.6f} |"
        )
    lines += ["", "P1-C−P1-B 的逐指标差值及 1000 次配对 bootstrap 95% CI 见 `metrics.json`。结果不得触发任何回改。", ""]
    (REPORT / "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    final_state = {
        **state, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_sample_count": len(samples), "metrics_sha256": digest(metrics_path),
    }
    generation_manifest_path = HOLDOUT / "holdout_generation_manifest.json"
    generation_manifest = read_json(generation_manifest_path)
    if int(generation_manifest.get("inference_pass_count", 0)) != 0:
        raise RuntimeError("Holdout-v3 inference counter changed during job")
    generation_manifest["inference_pass_count"] = 1
    generation_manifest["formal_inference"] = {
        "execution_id": state["execution_id"], "script_sha256": script_sha,
        "p1_b_selection_sha256": hashes["p1_b_selection_sha256"],
        "p1_c_selection_sha256": hashes["p1_c_selection_sha256"],
        "metrics_sha256": final_state["metrics_sha256"],
        "completed_at_utc": final_state["completed_at_utc"],
    }
    write_json(generation_manifest_path, generation_manifest)
    write_json(STATE, final_state)
    frozen_files = [
        metrics_path, REPORT / "overall_comparison.csv", REPORT / "per_class_object_metrics.csv",
        REPORT / "stratified_metrics.csv", REPORT / "report.zh-CN.md", STATE,
        generation_manifest_path, HOLDOUT / "reports" / "quality_report.json",
        HOLDOUT / "reports" / "leakage_audit.json", HOLDOUT / "human_review_36" / "review_package_manifest.json",
    ]
    write_json(REPORT / "freeze_manifest.json", {
        "schema_version": "topocae_p1_c_holdout_v3_freeze_manifest_1",
        "formal_inference_pass_count": 1,
        "files": [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for path in frozen_files],
        "sample_prediction_count": len(list(prediction_dir.glob("*.json"))),
        "review_step_count": len(list((HOLDOUT / "human_review_36").glob("*.step"))),
        "second_run_forbidden": True,
    })
    print(json.dumps({
        "formal_inference_complete": True, "sample_count": len(samples),
        "metrics": scalar_by_variant, "deltas_p1_c_minus_p1_b": direct_deltas,
        "output": str(REPORT),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
