from __future__ import annotations

"""Build the traceable Stage-6 comparison after the final permitted iteration."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parent
GNN_ROOT = ROOT / "reports" / "p1_c_strict_training_runs_v2"
ROUND0_ROOT = GNN_ROOT
ROUND1_ROOT = ROOT / "reports" / "p1_c_stage6_iteration1_runs_v1"
ROUND2_ROOT = ROOT / "reports" / "p1_c_stage6_iteration2_runs_v1"
OUTPUT = ROOT / "reports" / "p1_c_stage6_summary_v1"
SEEDS = (20260718, 20260719, 20260720)
FORMAL_VARIANTS = ("clean_5000", "clean_7200")
ABLATIONS = ("single_only", "no_instance_balanced_loss", "no_canonical_patch")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def run_name(variant: str, seed: int) -> str:
    return f"{variant}_seed_{seed}"


def pipeline_metrics(root: Path, variant: str, seed: int) -> dict[str, Any]:
    return read_json(root / run_name(variant, seed) / "synthetic_pipeline_v1" / "metrics.json")


def pipeline_row(root: Path, variant: str, seed: int, iteration: str) -> dict[str, Any]:
    path = root / run_name(variant, seed) / "synthetic_pipeline_v1" / "metrics.json"
    payload = read_json(path)
    split = payload["splits"]["val"]
    final = split["topology_verifier_global_decode"]
    class_name, class_record = min(final["per_class"].items(), key=lambda item: float(item[1]["f1"]))
    context_group = final["stratified"]["context_group"]
    single = float(context_group["single_context"]["object_macro_f1"])
    multi = float(context_group["multi_context"]["object_macro_f1"])
    return {
        "iteration": iteration,
        "variant": variant,
        "seed": seed,
        "object_macro_f1": float(final["object_macro_f1"]),
        "object_micro_f1": float(final["object_micro_f1"]),
        "exact_instance_ratio": float(final["exact_instance_ratio"]),
        "instance_iou": float(final["mean_best_instance_iou"]),
        "boundary_f1": float(final["boundary"]["f1"]),
        "proposal_oracle_recall_iou_0_5": float(split["candidate_proposals"]["oracle_recall_iou_0_5"]),
        "single_context_object_macro_f1": single,
        "multi_context_object_macro_f1": multi,
        "single_multi_gap_abs": abs(single - multi),
        "worst_class": class_name,
        "worst_class_object_f1": float(class_record["f1"]),
        "automatic_pass": bool(payload.get("contracts", {}).get("automatic_pass", False)),
        "metrics_path": str(path),
        "metrics_sha256": file_sha(path),
    }


def gnn_fields(variant: str, seed: int) -> dict[str, Any]:
    path = GNN_ROOT / run_name(variant, seed) / "metrics.json"
    payload = read_json(path)
    val = payload["evaluation"]["val"]
    robustness = payload["representation_robustness"]
    edge = val.get("same_instance_edge_selected_threshold", val["same_instance_edge_at_0_5"])
    return {
        "face_macro_f1": float(val["face"]["macro_f1_active"]),
        "semantic_role_macro_f1": float(val["semantic"]["macro_f1_active"]),
        "same_instance_edge_f1": float(edge["f1"]),
        "rotation_consistency": float(robustness["rotation_consistency"]),
        "scale_consistency": float(robustness["scale_consistency"]),
        "partition_consistency": float(robustness["partition_consistency"]),
        "cleanup_probe_signature_consistency": float(robustness["cleanup_probe_signature_consistency"]),
        "gnn_metrics_path": str(path),
        "gnn_metrics_sha256": file_sha(path),
    }


def candidate_fields(variant: str, seed: int) -> dict[str, Any]:
    path = ROUND1_ROOT / run_name(variant, seed) / "candidate_model_v1" / "metrics.json"
    payload = read_json(path)
    val = payload["evaluation"]["val"]
    proposal = payload["proposal_audit"]["val"]
    return {
        "candidate_macro_f1": float(val["macro"]["macro_f1"]),
        "candidate_validity_f1": float(val["validity"]["f1"]),
        "proposal_oracle_exact_recall": float(proposal["proposal_oracle_exact_recall"]),
        "all_seven_proposal_sources_present": bool(proposal["all_seven_sources_present"]),
        "candidate_metrics_path": str(path),
        "candidate_metrics_sha256": file_sha(path),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "face_macro_f1",
        "semantic_role_macro_f1",
        "same_instance_edge_f1",
        "candidate_macro_f1",
        "object_macro_f1",
        "object_micro_f1",
        "exact_instance_ratio",
        "instance_iou",
        "boundary_f1",
        "proposal_oracle_recall_iou_0_5",
        "single_multi_gap_abs",
        "worst_class_object_f1",
    )
    result: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {"mean": mean(values), "std": pstdev(values)}
    return result


def gates(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "face_macro_f1_ge_0_92": row["face_macro_f1"] >= 0.92,
        "semantic_role_macro_f1_ge_0_92": row["semantic_role_macro_f1"] >= 0.92,
        "object_macro_f1_ge_0_90": row["object_macro_f1"] >= 0.90,
        "exact_instance_ratio_ge_0_88": row["exact_instance_ratio"] >= 0.88,
        "same_instance_edge_f1_ge_0_96": row["same_instance_edge_f1"] >= 0.96,
        "proposal_oracle_recall_ge_0_95": row["proposal_oracle_recall_iou_0_5"] >= 0.95,
        "rotation_scale_partition_ge_0_98": min(
            row["rotation_consistency"], row["scale_consistency"], row["partition_consistency"]
        ) >= 0.98,
        "single_multi_gap_le_0_05": row["single_multi_gap_abs"] <= 0.05,
        "worst_class_object_f1_ge_0_80": row["worst_class_object_f1"] >= 0.80,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    round0 = [pipeline_row(ROUND0_ROOT, variant, seed, "initial") for variant in FORMAL_VARIANTS for seed in SEEDS]
    round1 = [pipeline_row(ROUND1_ROOT, variant, seed, "iteration_1_rejected") for variant in FORMAL_VARIANTS for seed in SEEDS]
    final_rows = []
    for variant in FORMAL_VARIANTS:
        for seed in SEEDS:
            row = pipeline_row(ROUND2_ROOT, variant, seed, "iteration_2_final")
            row.update(gnn_fields(variant, seed))
            row.update(candidate_fields(variant, seed))
            row["gates"] = gates(row)
            row["all_absolute_gates_pass"] = all(row["gates"].values())
            final_rows.append(row)

    p1_b_path = ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json"
    p1_b = read_json(p1_b_path)
    p1_b_mean = p1_b["three_seed_mean_std"]
    clean5000 = [row for row in final_rows if row["variant"] == "clean_5000"]
    clean7200 = [row for row in final_rows if row["variant"] == "clean_7200"]
    summaries = {"clean_5000": aggregate(clean5000), "clean_7200": aggregate(clean7200)}
    matched_noninferiority = {
        "face_margin": 0.01,
        "object_margin": 0.02,
        "p1_b_face_macro_f1_mean": float(p1_b_mean["face_macro_f1"]["mean"]),
        "p1_c_clean_5000_face_macro_f1_mean": summaries["clean_5000"]["face_macro_f1"]["mean"],
        "face_delta": summaries["clean_5000"]["face_macro_f1"]["mean"] - float(p1_b_mean["face_macro_f1"]["mean"]),
        "p1_b_object_macro_f1_mean": float(p1_b_mean["object_macro_f1"]["mean"]),
        "p1_c_clean_5000_object_macro_f1_mean": summaries["clean_5000"]["object_macro_f1"]["mean"],
        "object_delta": summaries["clean_5000"]["object_macro_f1"]["mean"] - float(p1_b_mean["object_macro_f1"]["mean"]),
    }
    matched_noninferiority["face_pass"] = matched_noninferiority["face_delta"] >= -matched_noninferiority["face_margin"]
    matched_noninferiority["object_pass"] = matched_noninferiority["object_delta"] >= -matched_noninferiority["object_margin"]

    ablations = []
    for variant in ABLATIONS:
        row = pipeline_row(ROUND0_ROOT, variant, 20260718, "frozen_ablation_initial_architecture")
        row.update(gnn_fields(variant, 20260718))
        candidate_path = ROUND0_ROOT / run_name(variant, 20260718) / "candidate_model_v1" / "metrics.json"
        candidate = read_json(candidate_path)
        row["candidate_macro_f1"] = float(candidate["evaluation"]["val"]["macro"]["macro_f1"])
        row["candidate_validity_f1"] = float(candidate["evaluation"]["val"]["validity"]["f1"])
        row["candidate_metrics_path"] = str(candidate_path)
        row["candidate_metrics_sha256"] = file_sha(candidate_path)
        ablations.append(row)

    iteration1_result = ROOT / "reports" / "p1_c_iterations_v1" / "stage6_iteration1_result_v2.json"
    iteration1_plan = ROOT / "configs" / "p1_c" / "stage6_iteration1_oriented_cylinder_polarity_v1.json"
    iteration2_plan = ROOT / "configs" / "p1_c" / "stage6_iteration2_candidate_verifier_calibration_v1.json"
    payload = {
        "schema_version": "topocae_p1_c_stage6_final_summary_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_round": round0,
        "iteration_1": {
            "decision": "rejected_fail_closed",
            "single_hypothesis": "add generic oriented-cylinder material polarity to candidate representation",
            "plan_path": str(iteration1_plan),
            "plan_sha256": file_sha(iteration1_plan),
            "formal_result_path": str(iteration1_result),
            "formal_result_sha256": file_sha(iteration1_result),
            "rows": round1,
            "reason": "5/6 formal seeds failed either worst-class or global-abstention gate",
        },
        "iteration_2": {
            "decision": "kept_final_permitted_iteration",
            "single_hypothesis": "candidate/verifier confidence interface was miscalibrated",
            "plan_path": str(iteration2_plan),
            "plan_sha256": file_sha(iteration2_plan),
            "rows": final_rows,
            "all_six_runs_pass": all(row["all_absolute_gates_pass"] for row in final_rows),
            "further_tuning_permitted": False,
        },
        "three_seed_mean_std": summaries,
        "p1_b_frozen_baseline": {
            "path": str(p1_b_path),
            "sha256": file_sha(p1_b_path),
            "face_macro_f1": p1_b_mean["face_macro_f1"],
            "object_macro_f1": p1_b_mean["object_macro_f1"],
            "exact_instance_ratio": p1_b_mean["exact_instance_ratio"],
            "same_instance_edge_f1": p1_b_mean["same_instance_edge_f1"],
        },
        "clean_5000_matched_size_noninferiority": matched_noninferiority,
        "ablations": ablations,
        "automatic_acceptance": {
            "all_six_formal_runs_pass_individually": all(row["all_absolute_gates_pass"] for row in final_rows),
            "clean_5000_noninferiority_pass": bool(matched_noninferiority["face_pass"] and matched_noninferiority["object_pass"]),
            "stage6_pass": bool(
                all(row["all_absolute_gates_pass"] for row in final_rows)
                and matched_noninferiority["face_pass"]
                and matched_noninferiority["object_pass"]
            ),
        },
        "contracts": {
            "selection_evidence": "Synthetic-Val only",
            "real_dev_used": False,
            "holdout_v2_predictions_or_errors_used": False,
            "holdout_v3_used": False,
            "per_class_or_part_thresholds": False,
            "test_part_s3_used": False,
            "llm_step_recognition": False,
            "superpower_skills_used": False,
        },
    }
    if not payload["automatic_acceptance"]["stage6_pass"]:
        raise RuntimeError("P1-C Stage-6 final acceptance failed")

    json_path = OUTPUT / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = OUTPUT / "formal_seed_metrics.csv"
    csv_fields = [
        "variant", "seed", "face_macro_f1", "semantic_role_macro_f1", "same_instance_edge_f1",
        "candidate_macro_f1", "proposal_oracle_recall_iou_0_5", "object_macro_f1", "object_micro_f1",
        "exact_instance_ratio", "instance_iou", "boundary_f1", "single_context_object_macro_f1",
        "multi_context_object_macro_f1", "single_multi_gap_abs", "worst_class", "worst_class_object_f1",
        "all_absolute_gates_pass",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    lines = [
        "# P1-C Stage6 Synthetic-Val 最终验收",
        "",
        "结论：最终允许的第 2 轮通用校准在 Clean-5000/Clean-7200 的 6 个正式种子上全部通过；不再允许继续调参。",
        "",
        "| 变体 | seed | face F1 | object F1 | exact | 最差类 | 最差类 F1 | single/multi gap |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['face_macro_f1']:.6f} | "
            f"{row['object_macro_f1']:.6f} | {row['exact_instance_ratio']:.6f} | {row['worst_class']} | "
            f"{row['worst_class_object_f1']:.6f} | {row['single_multi_gap_abs']:.6f} |"
        )
    lines += [
        "",
        "第 1 轮极性表示保留为候选表示基础，但其正式解码结果因 5/6 种子未通过而拒绝；第 2 轮只修复通用置信度接口并联合选择一个全局候选阈值与一个全局解码容差。",
        "",
        f"Clean-5000 相对 P1-B 的 face F1 差值为 {matched_noninferiority['face_delta']:+.6f}（容许 -0.01），object F1 差值为 {matched_noninferiority['object_delta']:+.6f}（容许 -0.02），均通过非劣门槛。",
        "",
        "全部数值、逐种子门禁、消融和哈希见 `summary.json` 与 `formal_seed_metrics.csv`。",
    ]
    (OUTPUT / "report.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage6_pass": True,
        "summary": str(json_path),
        "summary_sha256": file_sha(json_path),
        "formal_rows": len(final_rows),
        "clean_5000": summaries["clean_5000"],
        "clean_7200": summaries["clean_7200"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
