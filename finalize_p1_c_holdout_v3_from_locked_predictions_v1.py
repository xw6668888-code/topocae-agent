from __future__ import annotations

"""Finalize Holdout-v3 from 1440 locked paired predictions without inference."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import evaluate_p1_b_synthetic_pipeline as object_metrics_impl
import run_p1_c_synthetic_holdout_v3_once as formal


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    return np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)


def _sample_stats(samples: list[dict[str, Any]], variant: str, macro_names: list[str]) -> dict[str, np.ndarray]:
    n = len(samples)
    face_k = len(macro_names)
    semantic_k = 1 + max(
        value for sample in samples for value in (
            sample["variants"][variant]["semantic_truth"] + sample["variants"][variant]["semantic_prediction"]
        )
    )
    object_classes = macro_names[1:]
    arrays = {
        "face_tp": np.zeros((n, face_k)), "face_fp": np.zeros((n, face_k)), "face_fn": np.zeros((n, face_k)),
        "face_support": np.zeros((n, face_k)),
        "semantic_tp": np.zeros((n, semantic_k)), "semantic_fp": np.zeros((n, semantic_k)),
        "semantic_fn": np.zeros((n, semantic_k)), "semantic_support": np.zeros((n, semantic_k)),
        "edge": np.zeros((n, 3)), "proposal": np.zeros((n, 2)),
        "object_tp": np.zeros((n, len(object_classes))), "object_fp": np.zeros((n, len(object_classes))),
        "object_fn": np.zeros((n, len(object_classes))), "object_support": np.zeros((n, len(object_classes))),
        "exact": np.zeros((n, 2)), "iou": np.zeros((n, 2)), "boundary": np.zeros((n, 3)),
    }
    for index, sample in enumerate(samples):
        output = sample["variants"][variant]
        for prefix, truth_key, prediction_key, classes in (
            ("face", "face_truth_macro", "face_prediction_macro", face_k),
            ("semantic", "semantic_truth", "semantic_prediction", semantic_k),
        ):
            truth = np.asarray(output[truth_key], dtype=int)
            prediction = np.asarray(output[prediction_key], dtype=int)
            for label in range(classes):
                arrays[f"{prefix}_tp"][index, label] = np.sum((truth == label) & (prediction == label))
                arrays[f"{prefix}_fp"][index, label] = np.sum((truth != label) & (prediction == label))
                arrays[f"{prefix}_fn"][index, label] = np.sum((truth == label) & (prediction != label))
                arrays[f"{prefix}_support"][index, label] = np.sum(truth == label)
        edge = formal.metric_helpers._binary_metrics(output["edge_truth"], output["edge_prediction"])
        arrays["edge"][index] = [edge["tp"], edge["fp"], edge["fn"]]
        arrays["proposal"][index] = [
            output["proposal_oracle_iou_0_5"]["hit_count"],
            output["proposal_oracle_iou_0_5"]["truth_instance_count"],
        ]
        obj = object_metrics_impl._object_metrics(output["truth_objects"], output["prediction_objects"], object_classes)
        for class_index, macro in enumerate(object_classes):
            row = obj["per_class"][macro]
            arrays["object_tp"][index, class_index] = row["tp"]
            arrays["object_fp"][index, class_index] = row["prediction"] - row["tp"]
            arrays["object_fn"][index, class_index] = row["truth"] - row["tp"]
            arrays["object_support"][index, class_index] = row["truth"]
        truth_count = obj["truth_count"]
        arrays["exact"][index] = [obj["exact_instance_count"], truth_count]
        arrays["iou"][index] = [obj["mean_best_instance_iou"] * truth_count, truth_count]
        arrays["boundary"][index] = [obj["boundary"]["tp"], obj["boundary"]["fp"], obj["boundary"]["fn"]]
    return arrays


def _bootstrap_scalars(weights: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for prefix, key in (("face_macro_f1", "face"), ("semantic_macro_f1", "semantic"), ("object_macro_f1", "object")):
        tp = weights @ arrays[f"{key}_tp"]
        fp = weights @ arrays[f"{key}_fp"]
        fn = weights @ arrays[f"{key}_fn"]
        support = weights @ arrays[f"{key}_support"]
        f1 = _f1(tp, fp, fn)
        result[prefix] = np.divide(
            np.sum(f1 * (support > 0), axis=1), np.sum(support > 0, axis=1),
            out=np.zeros(weights.shape[0]), where=np.sum(support > 0, axis=1) != 0,
        )
    edge = weights @ arrays["edge"]
    result["same_instance_edge_f1"] = _f1(edge[:, 0], edge[:, 1], edge[:, 2])
    proposal = weights @ arrays["proposal"]
    result["proposal_oracle_recall_iou_0_5"] = np.divide(
        proposal[:, 0], proposal[:, 1], out=np.zeros(weights.shape[0]), where=proposal[:, 1] != 0
    )
    object_tp = np.sum(weights @ arrays["object_tp"], axis=1)
    object_fp = np.sum(weights @ arrays["object_fp"], axis=1)
    object_fn = np.sum(weights @ arrays["object_fn"], axis=1)
    result["object_micro_f1"] = _f1(object_tp, object_fp, object_fn)
    for name in ("exact", "iou"):
        totals = weights @ arrays[name]
        result["exact_instance_ratio" if name == "exact" else "instance_iou"] = np.divide(
            totals[:, 0], totals[:, 1], out=np.zeros(weights.shape[0]), where=totals[:, 1] != 0
        )
    boundary = weights @ arrays["boundary"]
    result["boundary_f1"] = _f1(boundary[:, 0], boundary[:, 1], boundary[:, 2])
    return result


def optimized_bootstrap(samples: list[dict[str, Any]], macro_names: list[str], replicates: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(20260719)
    weights = rng.multinomial(len(samples), np.full(len(samples), 1.0 / len(samples)), size=replicates)
    stats = {variant: _sample_stats(samples, variant, macro_names) for variant in formal.VARIANTS}
    scalars = {variant: _bootstrap_scalars(weights, stats[variant]) for variant in formal.VARIANTS}
    result = {}
    for key in scalars[formal.VARIANTS[0]]:
        delta = scalars[formal.VARIANTS[1]][key] - scalars[formal.VARIANTS[0]][key]
        result[key] = {
            "delta_p1_c_minus_p1_b": float(np.mean(delta)),
            "paired_bootstrap_95_ci": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
            "replicates": replicates, "seed": 20260719,
            "implementation": "sample_sufficient_statistics_multinomial_weights_v1",
        }
    return result


def main() -> int:
    started = time.perf_counter()
    hashes = formal.verify_inputs()
    state = formal.read_json(formal.STATE)
    if state["status"] != "running" or int(state["formal_inference_pass"]) != 1:
        raise RuntimeError("formal inference state is not recoverable")
    if formal.digest(Path(formal.__file__).resolve()) != state["script_sha256"]:
        raise RuntimeError("original formal inference script drift")
    records = formal.load_manifest()
    paths = [formal.REPORT / "sample_predictions" / f"{row['sample_id']}.json" for row in records]
    if len(paths) != 1440 or any(not path.is_file() for path in paths):
        raise RuntimeError("locked paired prediction inventory is incomplete")
    samples = [formal.read_json(path) for path in paths]
    if any(sample["execution_id"] != state["execution_id"] for sample in samples):
        raise RuntimeError("paired predictions belong to different executions")
    if any(sample["source_graph_sha256"].lower() != str(record["graph_sha256"]).lower() for sample, record in zip(samples, records)):
        raise RuntimeError("paired prediction source hash mismatch")
    macro_names = list(formal.read_json(formal.P1_C_PACKAGE / "metadata.json")["macro_classes"])
    metrics_by_variant = {}
    for variant in formal.VARIANTS:
        outputs = [sample["variants"][variant] for sample in samples]
        overall = formal.aggregate(outputs, macro_names)
        variant_samples = [{
            "context": sample["context"], "primary_target": sample["primary_target"],
            "host_template": sample["holdout_host_template_id"], "variant_output": sample["variants"][variant],
        } for sample in samples]
        metrics_by_variant[variant] = {
            "overall": overall,
            "stratified": {
                "context": formal.stratified(variant_samples, macro_names, "context"),
                "primary_target": formal.stratified(variant_samples, macro_names, "primary_target"),
                "host_template": formal.stratified(variant_samples, macro_names, "host_template"),
            },
        }
    scalar_by_variant = {variant: formal.scalar_metrics(metrics_by_variant[variant]["overall"]) for variant in formal.VARIANTS}
    direct_deltas = {
        key: scalar_by_variant[formal.VARIANTS[1]][key] - scalar_by_variant[formal.VARIANTS[0]][key]
        for key in scalar_by_variant[formal.VARIANTS[0]]
    }
    bootstrap = optimized_bootstrap(samples, macro_names, replicates=1000)
    result = {
        "schema_version": "topocae_p1_c_holdout_v3_formal_report_1",
        "dataset_name": "Synthetic-Holdout-v3", "formal_inference_pass_count": 1,
        "execution_id": state["execution_id"], "sample_count": len(samples),
        "comparison_variants": list(formal.VARIANTS), "frozen_inputs": hashes,
        "metrics_by_variant": metrics_by_variant, "scalar_comparison": scalar_by_variant,
        "deltas_p1_c_minus_p1_b": direct_deltas, "paired_bootstrap": bootstrap,
        "review_package": {"path": str(formal.HOLDOUT / "human_review_36"), "step_count": 36, "human_gate": False, "used_for_return_work": False},
        "timing_seconds": time.perf_counter() - started,
        "postprocess_recovery": {
            "reason": "original bootstrap recomputed object matching per replicate and was terminated after all 1440 inference outputs were locked",
            "inference_recomputed": False,
            "locked_prediction_count": 1440,
            "recovery_script": str(Path(__file__).resolve()),
            "recovery_script_sha256": formal.digest(Path(__file__).resolve()),
        },
        "contracts": {
            "holdout_used_for_training_or_selection": False,
            "results_used_to_modify_model_data_rule_or_threshold": False,
            "truth_visible_to_inference": False, "one_formal_job_with_both_variants": True,
            "part_or_class_specific_thresholds": False, "llm_step_recognition": False,
            "test_part_s3_used": False, "second_formal_inference_forbidden": True,
        },
    }
    metrics_path = formal.REPORT / "metrics.json"
    formal.write_json(metrics_path, result)
    formal.write_csv(formal.REPORT / "overall_comparison.csv", [
        {"variant": variant, **scalar_by_variant[variant]} for variant in formal.VARIANTS
    ])
    class_rows = []
    for variant in formal.VARIANTS:
        for macro, row in metrics_by_variant[variant]["overall"]["object"]["per_class"].items():
            class_rows.append({"variant": variant, "macro_feature": macro, **row})
    formal.write_csv(formal.REPORT / "per_class_object_metrics.csv", class_rows)
    stratum_rows = []
    for variant in formal.VARIANTS:
        for dimension, groups in metrics_by_variant[variant]["stratified"].items():
            for group, values in groups.items():
                stratum_rows.append({"variant": variant, "dimension": dimension, "group": group, **values})
    formal.write_csv(formal.REPORT / "stratified_metrics.csv", stratum_rows)
    lines = [
        "# P1-C Synthetic-Holdout-v3 一次性正式报告", "",
        "正式推理次数：1；样本数：1440；P1-B 与 P1-C 在同一锁定作业中配对完成。", "",
        "| 变体 | face F1 | semantic F1 | edge F1 | proposal oracle | object F1 | exact | IoU | boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in formal.VARIANTS:
        row = scalar_by_variant[variant]
        lines.append(
            f"| {variant} | {row['face_macro_f1']:.6f} | {row['semantic_macro_f1']:.6f} | {row['same_instance_edge_f1']:.6f} | "
            f"{row['proposal_oracle_recall_iou_0_5']:.6f} | {row['object_macro_f1']:.6f} | {row['exact_instance_ratio']:.6f} | "
            f"{row['instance_iou']:.6f} | {row['boundary_f1']:.6f} |"
        )
    lines += ["", "P1-C−P1-B 差值及 1000 次配对 bootstrap 95% CI 见 `metrics.json`；结果不得触发任何回改。", ""]
    (formal.REPORT / "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")

    final_state = {
        **state, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_sample_count": 1440, "metrics_sha256": formal.digest(metrics_path),
        "postprocess_recovery_count": 1, "inference_recomputed_during_recovery": False,
        "postprocess_recovery_script_sha256": formal.digest(Path(__file__).resolve()),
    }
    generation_path = formal.HOLDOUT / "holdout_generation_manifest.json"
    generation = formal.read_json(generation_path)
    if int(generation.get("inference_pass_count", 0)) != 0:
        raise RuntimeError("Holdout-v3 inference counter changed before recovery commit")
    generation["inference_pass_count"] = 1
    generation["formal_inference"] = {
        "execution_id": state["execution_id"], "script_sha256": state["script_sha256"],
        "p1_b_selection_sha256": hashes["p1_b_selection_sha256"],
        "p1_c_selection_sha256": hashes["p1_c_selection_sha256"],
        "metrics_sha256": final_state["metrics_sha256"], "completed_at_utc": final_state["completed_at_utc"],
        "postprocess_recovery_without_inference": True,
    }
    formal.write_json(generation_path, generation)
    formal.write_json(formal.STATE, final_state)
    frozen_files = [
        metrics_path, formal.REPORT / "overall_comparison.csv", formal.REPORT / "per_class_object_metrics.csv",
        formal.REPORT / "stratified_metrics.csv", formal.REPORT / "report.zh-CN.md", formal.STATE,
        generation_path, formal.HOLDOUT / "reports" / "quality_report.json",
        formal.HOLDOUT / "reports" / "leakage_audit.json", formal.HOLDOUT / "human_review_36" / "review_package_manifest.json",
    ]
    formal.write_json(formal.REPORT / "freeze_manifest.json", {
        "schema_version": "topocae_p1_c_holdout_v3_freeze_manifest_1",
        "formal_inference_pass_count": 1,
        "files": [{"path": str(path), "sha256": formal.digest(path), "bytes": path.stat().st_size} for path in frozen_files],
        "sample_prediction_count": 1440, "review_step_count": 36, "second_run_forbidden": True,
    })
    print(json.dumps({
        "formal_inference_complete": True, "formal_inference_pass_count": 1,
        "sample_count": 1440, "metrics": scalar_by_variant,
        "deltas_p1_c_minus_p1_b": direct_deltas, "output": str(formal.REPORT),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
