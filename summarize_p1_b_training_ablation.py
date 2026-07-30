from __future__ import annotations

"""Create traceable P1-B ablation and three-seed summary artifacts."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"
OUTPUT = ROOT / "reports" / "p1_b_training_ablation_summary_v1"
SEEDS = [20260718, 20260719, 20260720]


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    node = _read_json(RUN_ROOT / "node_only_seed_20260718" / "metrics.json")
    edge = _read_json(RUN_ROOT / "edge_attr_seed_20260718" / "metrics.json")
    final_runs = []
    seed_rows = []
    for seed in SEEDS:
        run = RUN_ROOT / f"same_instance_seed_{seed}"
        gnn = _read_json(run / "metrics.json")
        candidate = _read_json(run / "candidate_model_v1" / "metrics.json")
        pipeline = _read_json(run / "synthetic_pipeline_v1" / "metrics.json")
        selection = _read_json(run / "global_decode_v2_selection.json")
        val = gnn["evaluation"]["val"]
        full = pipeline["splits"]["val"]["topology_verifier_global_decode"]
        proposal = pipeline["splits"]["val"]["candidate_proposals"]
        diagnostic = pipeline["splits"]["val"]["topology_diagnostics"]
        row = {
            "seed": seed,
            "face_accuracy": val["face"]["accuracy"],
            "face_macro_accuracy": val["face"]["macro_accuracy_active"],
            "face_macro_f1": val["face"]["macro_f1_all"],
            "semantic_role_macro_f1": val["semantic"]["macro_f1_active"],
            "same_instance_edge_precision": val["same_instance_edge_selected_threshold"]["precision"],
            "same_instance_edge_recall": val["same_instance_edge_selected_threshold"]["recall"],
            "same_instance_edge_f1": val["same_instance_edge_selected_threshold"]["f1"],
            "proposal_oracle_recall_iou_0_5": proposal["oracle_recall_iou_0_5"],
            "proposal_oracle_exact_recall": proposal["oracle_exact_recall"],
            "object_micro_precision": full["object_micro_precision"],
            "object_micro_recall": full["object_micro_recall"],
            "object_micro_f1": full["object_micro_f1"],
            "object_macro_f1": full["object_macro_f1"],
            "exact_instance_ratio": full["exact_instance_ratio"],
            "instance_iou": full["mean_best_instance_iou"],
            "boundary_precision": full["boundary"]["precision"],
            "boundary_recall": full["boundary"]["recall"],
            "boundary_f1": full["boundary"]["f1"],
            "topology_invalid_rate": diagnostic["topology_invalid_rate"],
            "abstention_rate": diagnostic["abstention_rate"],
            "rotation_consistency": gnn["invariance"]["val"]["rotation_consistency"],
            "scale_consistency": gnn["invariance"]["val"]["scale_consistency"],
            "gnn_parameter_count": gnn["resource"]["parameter_count"],
            "candidate_parameter_count": candidate["resource"]["parameter_count"],
            "gnn_training_seconds": gnn["resource"]["training_seconds"],
            "candidate_training_seconds": candidate["resource"]["training_seconds"],
            "gnn_val_inference_ms_per_graph": val["inference_ms_per_graph"],
            "candidate_val_inference_ms_per_candidate": candidate["evaluation"]["val"]["inference_ms_per_candidate"],
            "peak_vram_bytes": max(gnn["resource"]["peak_vram_bytes"], candidate["resource"]["peak_vram_bytes"]),
            "global_decode_subset_tolerance": selection["global_subset_confidence_tolerance"],
        }
        seed_rows.append(row)
        final_runs.append({"seed": seed, "gnn": gnn, "candidate": candidate, "pipeline": pipeline, "row": row})
    aggregate = {
        key: {"mean": mean([row[key] for row in seed_rows]), "std": pstdev([row[key] for row in seed_rows])}
        for key in seed_rows[0]
        if key != "seed" and isinstance(seed_rows[0][key], (int, float))
    }
    first = final_runs[0]
    pipeline_val = first["pipeline"]["splits"]["val"]
    no_patch_candidate = _read_json(RUN_ROOT / "same_instance_seed_20260718" / "candidate_model_no_canonical_patch_v1" / "metrics.json")
    no_patch_pipeline = _read_json(RUN_ROOT / "same_instance_seed_20260718" / "synthetic_pipeline_no_canonical_patch_v1" / "metrics.json")
    ablations = [
        _ablation_row("1_node_only_gnn", node, None, None, "No edge attributes; no object assembly"),
        _ablation_row("2_plus_edge_attr", edge, None, None, "Graph-v2 edge attributes; no same-instance supervision"),
        _ablation_row(
            "3_plus_same_instance_assembly", first["gnn"],
            pipeline_val["same_instance_assembly"], None,
            "Global Synthetic-Val edge threshold and topology-constrained assembly",
        ),
        _ablation_row(
            "4_plus_candidate_classifier", first["gnn"],
            pipeline_val["candidate_classifier"], first["candidate"]["evaluation"]["val"],
            "All accepted candidate objects before overlap/global resolution; intentionally exposes duplicate-proposal precision",
        ),
        _ablation_row(
            "5_plus_verifier_global_decode", first["gnn"],
            pipeline_val["topology_verifier_global_decode"], first["candidate"]["evaluation"]["val"],
            "Verifier-v4 confidence correction plus global-decode-v2 selected on Synthetic-Val",
        ),
        _ablation_row(
            "6_no_canonical_patch", first["gnn"],
            no_patch_pipeline["splits"]["val"]["topology_verifier_global_decode"],
            no_patch_candidate["evaluation"]["val"],
            "Candidate canonical-patch count replaced by raw face count; full-model global tolerance kept frozen",
        ),
    ]
    acceptance = {
        "face_macro_f1_ge_0_90": aggregate["face_macro_f1"]["mean"] >= 0.90,
        "object_macro_f1_ge_0_88": aggregate["object_macro_f1"]["mean"] >= 0.88,
        "exact_instance_ratio_ge_0_85": aggregate["exact_instance_ratio"]["mean"] >= 0.85,
        "same_instance_edge_f1_ge_0_95": aggregate["same_instance_edge_f1"]["mean"] >= 0.95,
        "rotation_consistency_ge_0_98": aggregate["rotation_consistency"]["mean"] >= 0.98,
        "candidate_proposal_oracle_recall_ge_0_95": aggregate["proposal_oracle_exact_recall"]["mean"] >= 0.95,
        "all_three_seeds_pass_individually": all(
            row["face_macro_f1"] >= 0.90 and row["object_macro_f1"] >= 0.88
            and row["exact_instance_ratio"] >= 0.85 and row["same_instance_edge_f1"] >= 0.95
            and row["rotation_consistency"] >= 0.98 and row["proposal_oracle_exact_recall"] >= 0.95
            for row in seed_rows
        ),
    }
    payload = {
        "schema_version": "topocae_p1_b_training_ablation_summary_1",
        "seeds": SEEDS, "three_seed_rows": seed_rows, "three_seed_mean_std": aggregate,
        "ablations": ablations, "automatic_acceptance": acceptance,
        "canonical_patch_conclusion": {
            "full_object_macro_f1": pipeline_val["topology_verifier_global_decode"]["object_macro_f1"],
            "no_patch_object_macro_f1": no_patch_pipeline["splits"]["val"]["topology_verifier_global_decode"]["object_macro_f1"],
            "conclusion": "canonical patch count produced no measurable gain on this corrected synthetic validation set",
        },
        "contracts": {"selection_synthetic_val_only": True, "real_data_used": False, "holdout_used": False},
    }
    _write_json(OUTPUT / "summary.json", payload)
    _write_csv(OUTPUT / "three_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "ablation_metrics.csv", ablations)
    (OUTPUT / "report.zh-CN.md").write_text(_markdown(payload), encoding="utf-8")
    passed = all(acceptance.values())
    print(json.dumps({
        "automatic_pass": passed,
        "face_macro_f1_mean_std": aggregate["face_macro_f1"],
        "object_macro_f1_mean_std": aggregate["object_macro_f1"],
        "exact_instance_ratio_mean_std": aggregate["exact_instance_ratio"],
        "same_instance_edge_f1_mean_std": aggregate["same_instance_edge_f1"],
        "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0 if passed else 1


def _ablation_row(
    name: str, gnn: dict[str, Any], objects: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None, note: str,
) -> dict[str, Any]:
    val = gnn["evaluation"]["val"]
    return {
        "ablation": name,
        "face_macro_f1": val["face"]["macro_f1_all"],
        "semantic_macro_f1": val["semantic"]["macro_f1_active"],
        "same_instance_edge_f1": val.get("same_instance_edge_selected_threshold", {}).get("f1", ""),
        "candidate_head_macro_f1": candidate_metrics["macro"]["macro_f1"] if candidate_metrics else "",
        "candidate_validity_f1": candidate_metrics["validity"]["f1"] if candidate_metrics else "",
        "object_macro_f1": objects["object_macro_f1"] if objects else "",
        "object_micro_f1": objects["object_micro_f1"] if objects else "",
        "exact_instance_ratio": objects["exact_instance_ratio"] if objects else "",
        "instance_iou": objects["mean_best_instance_iou"] if objects else "",
        "boundary_f1": objects["boundary"]["f1"] if objects else "",
        "note": note,
    }


def _markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["three_seed_mean_std"]
    acceptance = payload["automatic_acceptance"]
    lines = [
        "# P1-B 训练、三种子与消融报告",
        "",
        "模型、early stopping、same-instance/candidate/global-decode 阈值全部仅由 Synthetic-Val 选择；Real-Dev 与 Holdout 均未参与。",
        "",
        "## 三固定种子核心结果（mean ± population std）",
        "",
    ]
    for key in ("face_macro_f1", "semantic_role_macro_f1", "same_instance_edge_f1", "proposal_oracle_exact_recall", "object_macro_f1", "object_micro_f1", "exact_instance_ratio", "instance_iou", "boundary_f1", "rotation_consistency", "scale_consistency"):
        lines.append(f"- {key}: {aggregate[key]['mean']:.6f} ± {aggregate[key]['std']:.6f}")
    lines.extend(["", "## 自动门槛", ""])
    for key, value in acceptance.items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend([
        "", "## Canonical patch 消融结论", "",
        f"完整模型 object Macro-F1={payload['canonical_patch_conclusion']['full_object_macro_f1']:.6f}；无 patch={payload['canonical_patch_conclusion']['no_patch_object_macro_f1']:.6f}。",
        "本数据上 canonical patch count 未产生可测收益；如实保留该阴性消融结论。",
        "", "原始逐种子与逐消融数字见同目录 CSV/JSON。",
    ])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
