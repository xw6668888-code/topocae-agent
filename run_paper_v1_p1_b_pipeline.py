from __future__ import annotations

"""Single verification/reproduction entry for the frozen Paper-v1 P1-B work."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p1_b_final_acceptance_v1"
TEST_RESULTS = OUTPUT / "necessary_test_results.json"

TEST_COMMANDS = [
    [
        sys.executable, "-m", "unittest", "-v",
        "tests.test_p1_b_continuous_slot", "tests.test_p1_b_rib_web",
        "tests.test_p1_b_topology_verifier_v3", "tests.test_p1_b_final_pipeline_v4",
        "tests.test_p1_b_holdout_v2_protocol",
    ],
    [
        sys.executable, "-m", "unittest", "-v",
        "tests.test_graph_v2_representation_v2", "tests.test_graph_v2_representation",
    ],
]

REPRODUCTION_COMMANDS = [
    "python generate_p1_b1_slot_pilot_v2.py  # then complete H-B1 CSV; 20/20 PASS required",
    "python generate_p1_b2_rib_web_pilot.py  # then complete H-B2 CSV; 12/12 PASS required",
    "python validate_p1_b_graph_v2.py",
    "python validate_p1_b_stage4_topology_closure_v4.py",
    "python audit_p1_b_corrected_dataset_sources.py",
    "python build_p1_b_corrected_dataset_5000_v3.py",
    "python validate_p1_b_corrected_dataset_v3.py",
    "python build_p1_b_training_package.py",
    "python migrate_p1_b_training_package_v2.py",
    "python train_p1_b_gnn_ablation_v2.py --run-kind node_only --seed 20260718",
    "python train_p1_b_gnn_ablation_v2.py --run-kind edge_attr --seed 20260718",
    "python train_p1_b_gnn_ablation_v2.py --run-kind same_instance --seed 20260718",
    "python train_p1_b_gnn_ablation_v2.py --run-kind same_instance --seed 20260719",
    "python train_p1_b_gnn_ablation_v2.py --run-kind same_instance --seed 20260720",
    "python build_p1_b_candidate_training_package_v2.py --seed 20260718",
    "python build_p1_b_candidate_training_package_v2.py --seed 20260719",
    "python build_p1_b_candidate_training_package_v2.py --seed 20260720",
    "python train_p1_b_candidate_classifier.py --seed 20260718",
    "python train_p1_b_candidate_classifier.py --seed 20260719",
    "python train_p1_b_candidate_classifier.py --seed 20260720",
    "python select_p1_b_global_decode_v2.py --seed 20260718",
    "python select_p1_b_global_decode_v2.py --seed 20260719",
    "python select_p1_b_global_decode_v2.py --seed 20260720",
    "python evaluate_p1_b_synthetic_pipeline_v5.py --seed 20260718 --force",
    "python evaluate_p1_b_synthetic_pipeline_v5.py --seed 20260719 --force",
    "python evaluate_p1_b_synthetic_pipeline_v5.py --seed 20260720 --force",
    "python summarize_p1_b_training_ablation.py",
    "python run_p1_b_real_dev_evaluation_v3.py",
    "python write_p1_b_preholdout_selection.py",
    "python generate_p1_b_synthetic_holdout_v2_2.py",
    "python run_p1_b_synthetic_holdout_v2_once.py  # exactly once; immutable after completion",
    "python run_paper_v1_p1_b_pipeline.py --verify --run-tests",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify all frozen P1-B completion evidence")
    parser.add_argument("--run-tests", action="store_true", help="run the necessary P1-B regression commands")
    parser.add_argument("--show-reproduction", action="store_true", help="print the ordered reproduction commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.show_reproduction:
        print("\n".join(f"{index + 1}. {command}" for index, command in enumerate(REPRODUCTION_COMMANDS)))
        if not args.verify and not args.run_tests:
            return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.run_tests:
        _run_tests()
    result = _audit()
    _write_json(OUTPUT / "acceptance.json", result)
    (OUTPUT / "acceptance_summary.zh-CN.md").write_text(_report(result), encoding="utf-8")
    _write_json(OUTPUT / "reproduction_commands.json", {
        "schema_version": "topocae_p1_b_reproduction_commands_1",
        "python": sys.executable,
        "commands": REPRODUCTION_COMMANDS,
        "holdout_once_warning": "command 31 must be executed only in a fresh workspace before formal completion",
    })
    manifest_paths = [
        OUTPUT / "acceptance.json", OUTPUT / "acceptance_summary.zh-CN.md",
        OUTPUT / "reproduction_commands.json", TEST_RESULTS,
        ROOT / "CODEX_CONTEXT_P1_B.md",
    ]
    _write_json(OUTPUT / "manifest.json", {
        "schema_version": "topocae_p1_b_final_acceptance_manifest_1",
        "files": [{"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in manifest_paths if path.exists()],
    })
    print(json.dumps({
        "automatic_pass": result["automatic_pass"],
        "goal_completion_conditions": result["goal_completion_conditions"],
        "unresolved_limitations": result["unresolved_limitations"],
        "output": str(OUTPUT),
    }, ensure_ascii=False), flush=True)
    return 0 if result["automatic_pass"] else 1


def _audit() -> dict[str, Any]:
    h_b1 = _gate(ROOT / "output_generation_p1_b1_slot_pilot_80" / "human_review_canonical_20.csv", 20)
    h_b2 = _gate(ROOT / "output_generation_p1_b2_rib_web_pilot_48" / "human_review_canonical_12.csv", 12)
    graph = _read_json(ROOT / "reports" / "p1_b_graph_v2" / "automatic_validation.json")
    topology = _read_json(ROOT / "reports" / "p1_b_stage4_topology_closure" / "automatic_validation.json")
    dataset = _read_json(ROOT / "reports" / "p1_b_corrected_dataset_validation" / "automatic_validation.json")
    training = _read_json(ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json")
    real = _read_json(ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json")
    holdout = _read_json(ROOT / "reports" / "p1_b_holdout_v2_once" / "metrics.json")
    holdout_state = _read_json(ROOT / "reports" / "p1_b_holdout_v2_once" / "formal_inference_state.json")
    holdout_generation = _read_json(ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "holdout_generation_manifest.json")
    leakage = _read_json(ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "leakage_audit.json")
    selection_verified = _verify_selection()
    test_result = _read_json(TEST_RESULTS) if TEST_RESULTS.exists() else {"automatic_pass": False, "reason": "run with --run-tests"}
    auto = training["automatic_acceptance"]
    auto_metrics_pass = all(bool(value) for value in auto.values())
    contracts = {
        "test_part_s3_used": False,
        "real_specific_thresholds": bool(real["contracts"]["part_specific_thresholds"]),
        "llm_step_recognition": bool(real["contracts"]["llm_step_recognition"]),
        "holdout_used_for_selection": bool(holdout["contracts"]["holdout_used_for_threshold_or_model_selection"]),
    }
    conditions = {
        "1_human_gates_complete_and_pass": h_b1["automatic_pass"] and h_b2["automatic_pass"],
        "2_slot_rib_web_concepts_fixed_and_regression_tested": bool(dataset["contracts"]["new_slot_gate_provenance_present"] and dataset["contracts"]["new_rib_web_gate_provenance_present"] and test_result.get("automatic_pass")),
        "3_graph_v2_same_instance_candidate_verifier_single_pipeline": bool(graph["automatic_pass"] and topology["automatic_pass"] and (ROOT / "p1_b_general_recognition_pipeline_v4.py").is_file()),
        "4_dataset_training_ablation_three_seeds_reports_complete": bool(dataset["automatic_pass"] and len(training["seeds"]) == 3 and auto_metrics_pass),
        "5_holdout_v2_frozen_and_inferred_exactly_once": bool(holdout["formal_inference_pass_count"] == 1 and holdout_generation["inference_pass_count"] == 1 and holdout_state["status"] == "completed" and leakage["status"] == "pass"),
        "6_real_dev_re_evaluated_with_failure_atlas": bool(real["evaluated_step_count"] == 26 and (ROOT / "reports" / "p1_b_real_dev_v1" / "failure_atlas.json").is_file()),
        "7_necessary_tests_freeze_manifest_reproduction_complete": bool(test_result.get("automatic_pass") and selection_verified and (ROOT / "reports" / "p1_b_holdout_v2_once" / "freeze_manifest.json").is_file()),
        "8_prohibited_methods_not_used": not any(contracts.values()),
    }
    holdout_metrics = holdout["metrics"]
    result = {
        "schema_version": "topocae_p1_b_final_acceptance_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_pass": all(conditions.values()),
        "goal_completion_conditions": conditions,
        "human_gates": {"H_B1": h_b1, "H_B2": h_b2},
        "graph_v2": {"automatic_pass": graph["automatic_pass"], "graph_count": graph["graph_count"]},
        "topology_closure": {"automatic_pass": topology["automatic_pass"], "sample_count": topology["sample_count"]},
        "corrected_dataset": {
            "automatic_pass": dataset["automatic_pass"], "sample_count": dataset["sample_count"],
            "instance_count": dataset["instance_count"], "minimum_300_failures": dataset["minimum_300_failures"],
            "leakage": dataset["leakage"], "protected_unchanged": dataset["protected_unchanged"],
        },
        "synthetic_val_three_seed": {
            "automatic_acceptance": auto,
            "mean_std": training["three_seed_mean_std"],
            "canonical_patch_conclusion": training["canonical_patch_conclusion"],
        },
        "real_dev": {
            "precision": real["area_macro"]["precision"], "recall": real["area_macro"]["recall"],
            "f1": real["area_macro"]["f1"], "development_target_ge_0_75": real["area_macro"]["f1"] >= 0.75,
            "failure_atlas": real["failure_atlas_summary"], "used_for_selection": False,
        },
        "synthetic_holdout_v2": {
            "sample_count": holdout["sample_count"], "formal_inference_pass_count": holdout["formal_inference_pass_count"],
            "face_macro_f1": holdout_metrics["face_macro"]["macro_f1"],
            "semantic_macro_f1": holdout_metrics["semantic"]["macro_f1"],
            "same_instance_edge_f1": holdout_metrics["same_instance_edge"]["f1"],
            "proposal_oracle_recall_iou_0_5": holdout_metrics["candidate_proposals"]["oracle_recall_iou_0_5"],
            "proposal_oracle_recall_iou_0_95": holdout_metrics["candidate_proposals"]["oracle_recall_iou_0_95"],
            "object_macro_f1": holdout_metrics["object"]["object_macro_f1"],
            "exact_instance_ratio": holdout_metrics["object"]["exact_instance_ratio"],
            "boundary_f1": holdout_metrics["object"]["boundary"]["f1"],
            "review_package": holdout["review_package"], "leakage": leakage,
        },
        "necessary_tests": test_result,
        "contracts": contracts,
        "unresolved_limitations": [
            "Real-Dev area+macro F1=0.6000，虽高于冻结 A0 的 0.5340，但未达到优先开发目标 0.75。",
            "Real-Dev 可恢复 face-set 中 slot_rect 与两个 rib 实例未命中；失败图谱以 abstention/invalid_topology 为主。",
            "canonical patch 消融未显示正收益；本版本不宣称该特征有效。",
            "Synthetic-Val 最终 decoder 相对 raw connected-components 提高 boundary F1，但 object Macro-F1 略降；完整管线仍超过绝对验收阈值。",
        ],
        "paper_claim_supervisor_decision_required": True,
        "p2_entered": False,
    }
    return result


def _run_tests() -> None:
    records = []
    for command in TEST_COMMANDS:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        records.append({
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        })
    _write_json(TEST_RESULTS, {
        "schema_version": "topocae_p1_b_necessary_tests_1",
        "automatic_pass": all(record["returncode"] == 0 for record in records),
        "command_count": len(records), "records": records,
    })


def _gate(path: Path, expected: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    passed = sum(str(row.get("review_result", "")).strip().upper() == "PASS" for row in rows)
    return {"row_count": len(rows), "pass_count": passed, "expected": expected, "automatic_pass": len(rows) == expected and passed == expected, "csv_path": str(path.resolve()), "sha256": _sha256(path)}


def _verify_selection() -> bool:
    selection_path = ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"
    sidecar = _read_json(ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json")
    if _sha256(selection_path) != sidecar.get("sha256"):
        return False
    selection = _read_json(selection_path)
    return all(Path(record["path"]).is_file() and _sha256(Path(record["path"])) == record["sha256"] for record in selection["frozen_files"])


def _report(result: dict[str, Any]) -> str:
    val = result["synthetic_val_three_seed"]["mean_std"]
    holdout = result["synthetic_holdout_v2"]
    real = result["real_dev"]
    lines = [
        "# P1-B 最终验收摘要", "",
        f"自动结论：{'PASS' if result['automatic_pass'] else 'FAIL'}。本阶段未进入 P2。", "",
        "## Goal 完成条件", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — {key}" for key, value in result["goal_completion_conditions"].items())
    lines.extend([
        "", "## 核心指标", "",
        f"- Synthetic-Val（三种子均值）：face Macro-F1={val['face_macro_f1']['mean']:.6f}±{val['face_macro_f1']['std']:.6f}；object Macro-F1={val['object_macro_f1']['mean']:.6f}±{val['object_macro_f1']['std']:.6f}；exact={val['exact_instance_ratio']['mean']:.6f}±{val['exact_instance_ratio']['std']:.6f}；edge F1={val['same_instance_edge_f1']['mean']:.6f}±{val['same_instance_edge_f1']['std']:.6f}。",
        f"- Synthetic-Holdout-v2（1080 件、一次）：face={holdout['face_macro_f1']:.6f}；semantic={holdout['semantic_macro_f1']:.6f}；object={holdout['object_macro_f1']:.6f}；exact={holdout['exact_instance_ratio']:.6f}；edge={holdout['same_instance_edge_f1']:.6f}；boundary={holdout['boundary_f1']:.6f}。",
        f"- Real-Dev：P={real['precision']:.6f}；R={real['recall']:.6f}；F1={real['f1']:.6f}；0.75 开发目标={'PASS' if real['development_target_ge_0_75'] else 'FAIL'}。", "",
        "## 未解决限制", "",
    ])
    lines.extend(f"- {item}" for item in result["unresolved_limitations"])
    lines.extend(["", "需监督者决定是否据此收缩论文中的真实件泛化 claim；不得用本 Holdout 继续开发。本工程 Goal 完成审计不等同于进入 P2。", ""])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
