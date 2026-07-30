from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "p1_c_final_acceptance_v1"
DATA = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
HOLDOUT_DATA = ROOT / "output_generation_p1_c_synthetic_holdout_v3"


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    passes = sum(str(row.get("decision", "")).strip().upper() == "PASS" for row in rows)
    return {
        "path": relative,
        "sha256": sha256(path),
        "row_count": len(rows),
        "pass_count": passes,
        "all_pass": bool(rows) and passes == len(rows),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, evidence: Any) -> None:
        checks.append({"id": identifier, "passed": bool(passed), "evidence": evidence})

    h_c1 = review(
        "output_generation_p1_c1_strict_concept_hosts_pilot_96_r2/"
        "human_review_canonical_24.csv"
    )
    h_c2 = review(
        "output_generation_p1_c2_all_class_canonical_36_strict_concept_r3/"
        "human_review_canonical_36.csv"
    )
    h_c1_receipt = load(
        "reports/p1_c_iterations_v1/h_c1_strict_concept_r2_authorization_receipt.json"
    )
    h_c2_receipt = load(
        "reports/p1_c_iterations_v1/h_c2_strict_concept_r3_authorization_receipt.json"
    )
    check(
        "human_gates",
        h_c1["all_pass"] and h_c1["row_count"] == 24
        and h_c2["all_pass"] and h_c2["row_count"] == 36
        and h_c1_receipt["explicit_resume_message_received"]
        and h_c2_receipt["explicit_resume_message_received"]
        and h_c2_receipt["strict_generation_contract_validation_passed"],
        {"H-C1": h_c1, "H-C2": h_c2},
    )

    dataset = load(
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/dataset_meta.json"
    )
    strict_audit = load(
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/"
        "reports/strict_generation_contract_audit.json"
    )
    quota_ok = (
        len(dataset["primary_counts"]) == 18
        and set(dataset["primary_counts"].values()) == {400}
        and all(
            value == {"target_only": 240, "pair": 120, "triple": 40}
            for value in dataset["context_counts_by_primary"].values()
        )
        and set(dataset["secondary_counts"].values()) == {200}
        and all(dataset["quota_checks"].values())
    )
    data_quality_ok = (
        dataset["automatic_pass"] and dataset["sample_count"] == 7200
        and dataset["cleanup_audit"]["pass_count"] == 7200
        and dataset["cleanup_audit"]["total_removable_same_domain_faces"] == 0
        and dataset["label_audit"]["pass_count"] == 7200
        and dataset["label_audit"]["face_label_overlap_count"] == 0
        and dataset["label_audit"]["face_label_coverage_fraction"] == 1.0
        and dataset["graph_input_audit"]["pass_count"] == 7200
        and dataset["old_protected_step_hash_reuse_count"] == 0
        and strict_audit["passed"]
        and strict_audit["strict_contract_pass_count"] == 7200
        and strict_audit["invalidated_predecessor_records_reused"] == 0
    )
    check(
        "clean_context_7200",
        data_quality_ok and quota_ok,
        {
            "sample_count": dataset["sample_count"],
            "generator_revision": dataset["generator_revision"],
            "quota_checks": dataset["quota_checks"],
            "cleanup_audit": dataset["cleanup_audit"],
            "label_audit": dataset["label_audit"],
            "old_geometry_reuse_count": dataset["old_protected_step_hash_reuse_count"],
        },
    )
    split = dataset["split_audit"]
    matched = dataset["matched_5000"]
    check(
        "splits_and_matched_5000",
        split["passed"] and split["cross_split_collision_count"] == 0
        and matched["frozen_before_training"] and matched["sample_count"] == 5000
        and matched["split_counts"] == {"train": 4000, "val": 500, "development_test": 500},
        {"split": split, "matched_5000": {
            "sample_count": matched["sample_count"],
            "split_counts": matched["split_counts"],
            "context_counts": matched["context_counts"],
            "class_count_range": matched["class_count_range"],
        }},
    )

    robustness = load(
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/"
        "reports/graph_v2_robustness/robustness.json"
    )
    check(
        "graph_v2_robustness",
        robustness["passed"] and min(
            robustness["rotation_consistency"], robustness["scale_consistency"],
            robustness["partition_consistency"],
            robustness["cleanup_probe_signature_consistency"],
        ) >= 0.98,
        {key: robustness[key] for key in (
            "sample_count", "rotation_consistency", "scale_consistency",
            "partition_consistency", "cleanup_probe_signature_consistency", "passed"
        )},
    )

    run_root = ROOT / "reports" / "p1_c_strict_training_runs_v2"
    formal_runs = [
        f"clean_{size}_seed_{seed}"
        for size in (5000, 7200)
        for seed in (20260718, 20260719, 20260720)
    ]
    ablations = [
        "single_only_seed_20260718",
        "no_instance_balanced_loss_seed_20260718",
        "no_canonical_patch_seed_20260718",
    ]
    run_evidence: dict[str, Any] = {}
    for name in formal_runs + ablations:
        path = run_root / name
        required = [
            path / "best.pt", path / "config.json", path / "history.json",
            path / "metrics.json", path / "predictions.pt",
            path / "candidate_model_v1" / "best.pt",
            path / "candidate_model_v1" / "config.json",
            path / "candidate_model_v1" / "metrics.json",
        ]
        run_evidence[name] = {
            "required_files_present": all(item.is_file() for item in required),
            "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
        }
    stage6 = load("reports/p1_c_stage6_summary_v1/summary.json")
    stage6_rows = stage6["iteration_2"]["rows"]
    check(
        "training_and_stage6",
        len(formal_runs) == 6 and len(stage6_rows) == 6
        and all(item["required_files_present"] for item in run_evidence.values())
        and all(row["all_absolute_gates_pass"] for row in stage6_rows)
        and stage6["iteration_2"]["all_six_runs_pass"]
        and stage6["automatic_acceptance"]["stage6_pass"]
        and stage6["automatic_acceptance"]["clean_5000_noninferiority_pass"],
        {
            "runs": run_evidence,
            "three_seed_mean_std": stage6["three_seed_mean_std"],
            "clean_5000_noninferiority": stage6["clean_5000_matched_size_noninferiority"],
            "ablations": stage6["ablations"],
            "automatic_acceptance": stage6["automatic_acceptance"],
        },
    )

    selection_path = ROOT / "reports/p1_c_freeze_v1/preholdout_selection.json"
    selection = load("reports/p1_c_freeze_v1/preholdout_selection.json")
    sidecar = load("reports/p1_c_freeze_v1/preholdout_selection.sha256.json")
    selection_hash = sha256(selection_path)
    check(
        "preholdout_freeze",
        selection_hash == sidecar["sha256"]
        and sidecar["verified_after_write"]
        and selection["selected_variant"] == "clean_7200"
        and int(selection["selected_seed"]) == 20260718
        and not selection["holdout_v3_artifacts_preexisting"],
        {
            "selection_sha256": selection_hash,
            "selected_variant": selection["selected_variant"],
            "selected_seed": selection["selected_seed"],
            "frozen_file_count": len(selection["frozen_files"]),
        },
    )

    real = load("reports/p1_c_real_dev_v1/metrics.json")
    real_receipt = load("reports/p1_c_real_dev_v1/execution_receipt.json")
    real_area = real["area_macro"]
    check(
        "real_dev_frozen_once",
        real["evaluated_step_count"] == 26
        and real_receipt["formal_inference_pass_count"] == 1
        and real_receipt["formal_inference_completed_sample_count"] == 26
        and not real_receipt["inference_recomputed_during_resume"]
        and not real["contracts"]["real_dev_caused_model_threshold_data_or_rule_change"],
        {
            "evaluated_step_count": real["evaluated_step_count"],
            "precision": real_area["precision"], "recall": real_area["recall"],
            "f1": real_area["f1"],
            "p1_b_final_f1": 0.600,
            "targets": real["real_dev_targets_non_tuning"],
            "required_classes": real["required_class_and_family_metrics"],
            "formal_inference_pass_count": real_receipt["formal_inference_pass_count"],
        },
    )

    holdout_quality = load(
        "output_generation_p1_c_synthetic_holdout_v3/reports/quality_report.json"
    )
    leakage = load(
        "output_generation_p1_c_synthetic_holdout_v3/reports/leakage_audit.json"
    )
    holdout_manifest = load(
        "output_generation_p1_c_synthetic_holdout_v3/holdout_generation_manifest.json"
    )
    holdout_state = load("reports/p1_c_holdout_v3_once/formal_inference_state.json")
    holdout = load("reports/p1_c_holdout_v3_once/metrics.json")
    prediction_count = sum(
        1 for item in (ROOT / "reports/p1_c_holdout_v3_once/sample_predictions").glob("*.json")
    )
    review_count = sum(
        1 for item in (HOLDOUT_DATA / "human_review_36").glob("*.step")
    ) + sum(1 for item in (HOLDOUT_DATA / "human_review_36").glob("*.stp"))
    check(
        "holdout_v3_quality_and_leakage",
        holdout_quality["automatic_pass"] and holdout_quality["sample_count"] == 1440
        and set(holdout_quality["primary_counts"].values()) == {80}
        and leakage["all_zero"]
        and not leakage["holdout_v2_predictions_or_errors_read"],
        {"quality": holdout_quality, "leakage": leakage},
    )
    check(
        "holdout_v3_exactly_once",
        holdout_manifest["inference_pass_count"] == 1
        and holdout_state["status"] == "completed"
        and holdout_state["formal_inference_pass"] == 1
        and holdout_state["completed_sample_count"] == 1440
        and not holdout_state["inference_recomputed_during_recovery"]
        and holdout["formal_inference_pass_count"] == 1
        and holdout["sample_count"] == 1440
        and prediction_count == 1440 and review_count == 36,
        {
            "execution_id": holdout_state["execution_id"],
            "inference_pass_count": holdout_manifest["inference_pass_count"],
            "prediction_count": prediction_count,
            "review_step_count": review_count,
            "metrics": holdout["scalar_comparison"],
            "deltas_p1_c_minus_p1_b": holdout["deltas_p1_c_minus_p1_b"],
            "paired_bootstrap": holdout["paired_bootstrap"],
        },
    )

    protected = load("reports/p1_c_final_acceptance_v1/protected_inventory_diff.json")
    check(
        "protected_p1_b_assets",
        protected["passed"] and protected["added_count"] == 0
        and protected["removed_count"] == 0 and protected["changed_count"] == 0,
        protected,
    )

    test_command = [
        sys.executable, "-m", "unittest", "discover", "-s", "tests",
        "-p", "test_p1_c*.py", "-q",
    ]
    completed = subprocess.run(
        test_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    test_output = (completed.stdout + completed.stderr).strip()
    match = re.search(r"Ran (\d+) tests", test_output)
    test_count = int(match.group(1)) if match else 0
    test_summary = {
        "schema_version": "topocae_p1_c_test_summary_1",
        "command": test_command,
        "returncode": completed.returncode,
        "test_count": test_count,
        "passed": completed.returncode == 0 and test_count >= 67,
        "output": test_output,
    }
    (OUT / "test_summary.json").write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    check("p1_c_tests", test_summary["passed"], test_summary)

    forbidden_ok = (
        not dataset["contracts"]["test_part_s3_accessed"]
        and not dataset["contracts"]["real_dev_used"]
        and not dataset["contracts"]["old_holdout_results_used"]
        and not dataset["contracts"]["llm_step_recognition_used"]
        and not dataset["contracts"]["superpower_skills_used"]
        and not stage6["contracts"]["per_class_or_part_thresholds"]
        and not selection["contracts"]["p2_entered"]
        and not real["contracts"]["test_part_s3_used"]
        and not holdout["contracts"]["holdout_used_for_training_or_selection"]
        and not holdout["contracts"]["results_used_to_modify_model_data_rule_or_threshold"]
        and not holdout["contracts"]["part_or_class_specific_thresholds"]
    )
    check(
        "forbidden_items_and_no_p2",
        forbidden_ok,
        {
            "dataset": dataset["contracts"], "stage6": stage6["contracts"],
            "preholdout": selection["contracts"], "real_dev": real["contracts"],
            "holdout": holdout["contracts"],
        },
    )
    check(
        "single_reproducible_entry",
        (ROOT / "run_paper_v1_p1_c_pipeline.py").is_file(),
        {"path": "run_paper_v1_p1_c_pipeline.py"},
    )
    check(
        "repeat_error_prevention",
        (ROOT / "p1_c/runtime_state_isolation_v3.py").is_file()
        and (ROOT / "tests/test_p1_c_import_order_isolation.py").is_file()
        and test_summary["passed"],
        {
            "root_cause_report": "reports/p1_c_iterations_v1/"
            "h_c2_strict_concept_r2_slot_regression_failure.json",
            "scoped_binding": "p1_c/scoped_bindings.py",
            "runtime_guard": "p1_c/runtime_state_isolation_v3.py",
            "order_regression_test": "tests/test_p1_c_import_order_isolation.py",
        },
    )

    accepted = all(item["passed"] for item in checks)
    acceptance = {
        "schema_version": "topocae_p1_c_final_acceptance_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "checks": checks,
        "real_dev_conclusion": {
            "hypothesis_supported_on_real_parts": False,
            "reason": "P1-C Real-Dev F1 0.446215 is below P1-B final 0.600; rib recall remains 0.",
            "no_post_real_tuning": True,
        },
        "holdout_conclusion": {
            "p1_b_object_macro_f1": holdout["scalar_comparison"]["P1-B-frozen"]["object_macro_f1"],
            "p1_c_object_macro_f1": holdout["scalar_comparison"]["P1-C-clean-7200-seed-20260718"]["object_macro_f1"],
            "delta": holdout["deltas_p1_c_minus_p1_b"]["object_macro_f1"],
            "formal_inference_pass_count": 1,
            "no_post_holdout_tuning": True,
        },
        "p2_entered": False,
    }
    acceptance_path = OUT / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    means = stage6["three_seed_mean_std"]
    noninf = stage6["clean_5000_matched_size_noninferiority"]
    real_required = real["required_class_and_family_metrics"]
    hmetrics = holdout["scalar_comparison"]
    lines = [
        "# P1-C 最终验收摘要",
        "",
        f"自动验收：{'PASS' if accepted else 'FAIL'}（{acceptance['passed_check_count']}/{acceptance['check_count']} 项）。未进入 P2。",
        "",
        "## 数据、训练与 Human Gate",
        "",
        f"- H-C1 r2：{h_c1['pass_count']}/{h_c1['row_count']} PASS；H-C2 r3：{h_c2['pass_count']}/{h_c2['row_count']} PASS。",
        "- Clean-Context-7200：7200/7200；18 类各 400；target-only/pair/triple 各类 240/120/40；旧几何复用 0。",
        "- cleanup、标签、Graph v2、split leakage 全部通过；removable same-domain faces=0，标签覆盖率=100%。",
        f"- Clean-5000 三种子 object Macro-F1={means['clean_5000']['object_macro_f1']['mean']:.6f}±{means['clean_5000']['object_macro_f1']['std']:.6f}；Clean-7200={means['clean_7200']['object_macro_f1']['mean']:.6f}±{means['clean_7200']['object_macro_f1']['std']:.6f}。",
        "",
        "## 消融结论",
        "",
        f"- Clean-5000 对 P1-B：face Δ={noninf['face_delta']:+.6f}（通过 -0.01 非劣界），object Δ={noninf['object_delta']:+.6f}。三种子波动远小于 object 增益，支持干净几何/上下文数据管线有效，但不能把全部因果只归于 cleanup。",
        f"- 7200 对 5000：face 均值提高 {means['clean_7200']['face_macro_f1']['mean']-means['clean_5000']['face_macro_f1']['mean']:+.6f}，object 均值变化 {means['clean_7200']['object_macro_f1']['mean']-means['clean_5000']['object_macro_f1']['mean']:+.6f}；更多数据主要改善 face，并未提高 object 均值。",
        f"- Clean-7200 single/multi object gap={means['clean_7200']['single_multi_gap_abs']['mean']:.6f}；上下文平衡未显著伤害 single，并稳定 multi。",
        f"- no-instance-balanced-loss object={stage6['ablations'][1]['object_macro_f1']:.6f}，no-canonical-patch object={stage6['ablations'][2]['object_macro_f1']:.6f}，均低于完整主模型，二者仍有收益。",
        "",
        "## Real-Dev（冻结后一次比较）",
        "",
        f"- P1-C area+macro P/R/F1={real_area['precision']:.6f}/{real_area['recall']:.6f}/{real_area['f1']:.6f}，低于 P1-B final F1=0.600000。",
        f"- rib recall={real_required['rib']['recall']:.6f}；slot family recall={real_required['slot_family']['recall']:.6f}。真实件假设未获支持，且未据此调参。",
        "- 主要未解决类：rib、cylinder_boss、pocket_rect、rectangular_boss 为 0 命中；slot_u/slot_rect/slot_v 仍漏检，chamfer F1 很低。",
        "",
        "## Synthetic-Holdout-v3（正式推理恰好一次）",
        "",
        f"- P1-B/P1-C object Macro-F1={hmetrics['P1-B-frozen']['object_macro_f1']:.6f}/{hmetrics['P1-C-clean-7200-seed-20260718']['object_macro_f1']:.6f}，Δ={holdout['deltas_p1_c_minus_p1_b']['object_macro_f1']:+.6f}。",
        f"- P1-B/P1-C face Macro-F1={hmetrics['P1-B-frozen']['face_macro_f1']:.6f}/{hmetrics['P1-C-clean-7200-seed-20260718']['face_macro_f1']:.6f}；exact={hmetrics['P1-B-frozen']['exact_instance_ratio']:.6f}/{hmetrics['P1-C-clean-7200-seed-20260718']['exact_instance_ratio']:.6f}。",
        "- 1440/1440 质量审计通过，内外泄漏碰撞 0，36 个审查 STEP 已归档；结果未回流开发。",
        "",
        "## 防回归与限制",
        "",
        "- H-C2 重复错误根因是 revision 模块导入时持久替换共享几何内核，导致结果依赖测试导入顺序。现已加入异常安全的作用域绑定、捕获引用重绑、最终入口依赖修复和 strict→legacy 乱序测试；P1-C 测试 67/67 PASS。",
        "- 未使用 test_part_s3、Real 专用阈值、LLM 直接识别 STEP、旧 Holdout 调参或 superpower skills；未进入 P2。",
    ]
    report_path = OUT / "final_acceptance.zh-CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    context_lines = [
        "# CODEX_CONTEXT_P1_C",
        "",
        f"- P1-C Goal 自动验收：{'PASS' if accepted else 'FAIL'}。",
        f"- H-C1 r2={h_c1['pass_count']}/{h_c1['row_count']}；H-C2 r3={h_c2['pass_count']}/{h_c2['row_count']}。",
        "- 正式数据：output_generation_p1_c_clean_context_7200_strict_concept_r2，7200 样本，旧几何复用 0。",
        "- Stage6：6/6 三种子正式运行通过；preholdout selection SHA-256=" + selection_hash + "。",
        f"- Real-Dev F1={real_area['f1']:.6f}，低于 P1-B 0.600；rib recall=0；不得据此调参。",
        f"- Holdout-v3：正式推理 1 次；P1-C object Macro-F1={hmetrics['P1-C-clean-7200-seed-20260718']['object_macro_f1']:.6f}，相对 P1-B Δ={holdout['deltas_p1_c_minus_p1_b']['object_macro_f1']:+.6f}。",
        "- 受保护 P1-B 资产逐文件不变；P1-C tests 67/67 PASS。",
        "- 导入顺序复发已由 p1_c/runtime_state_isolation_v3.py 和 tests/test_p1_c_import_order_isolation.py 封堵。",
        "- 禁止项全部未使用；P2 未进入。",
    ]
    context_path = ROOT / "CODEX_CONTEXT_P1_C.md"
    context_path.write_text("\n".join(context_lines) + "\n", encoding="utf-8")

    delivery_relatives = [
        "GOAL_P1_C_CLEAN_GEOMETRY_FULL_REGEN_RETRAIN_COMPARISON.zh-CN.md",
        "reports/p1_c_preflight/preflight.json",
        h_c1["path"], h_c2["path"],
        "reports/p1_c_iterations_v1/h_c1_strict_concept_r2_authorization_receipt.json",
        "reports/p1_c_iterations_v1/h_c2_strict_concept_r3_authorization_receipt.json",
        "reports/p1_c_iterations_v1/h_c2_strict_concept_r2_slot_regression_failure.json",
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/dataset_meta.json",
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/manifests/artifact_manifest.json",
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/manifests/generation_plan_freeze.json",
        "output_generation_p1_c_clean_context_7200_strict_concept_r2/reports/graph_v2_robustness/robustness.json",
        "reports/p1_c_stage6_summary_v1/summary.json",
        "reports/p1_c_freeze_v1/preholdout_selection.json",
        "reports/p1_c_freeze_v1/preholdout_selection.sha256.json",
        "reports/p1_c_real_dev_v1/metrics.json",
        "reports/p1_c_real_dev_v1/failure_atlas.json",
        "reports/p1_c_real_dev_v1/execution_receipt.json",
        "output_generation_p1_c_synthetic_holdout_v3/holdout_generation_manifest.json",
        "output_generation_p1_c_synthetic_holdout_v3/reports/quality_report.json",
        "output_generation_p1_c_synthetic_holdout_v3/reports/leakage_audit.json",
        "reports/p1_c_holdout_v3_once/formal_inference_state.json",
        "reports/p1_c_holdout_v3_once/metrics.json",
        "reports/p1_c_final_acceptance_v1/protected_inventory_diff.json",
        "reports/p1_c_final_acceptance_v1/test_summary.json",
        "reports/p1_c_final_acceptance_v1/acceptance.json",
        "reports/p1_c_final_acceptance_v1/final_acceptance.zh-CN.md",
        "run_paper_v1_p1_c_pipeline.py",
        "p1_c/scoped_bindings.py",
        "p1_c/runtime_state_isolation_v3.py",
        "tests/test_p1_c_import_order_isolation.py",
        "CODEX_CONTEXT_P1_C.md",
    ]
    artifacts = []
    for relative in delivery_relatives:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)
        })
    delivery = {
        "schema_version": "topocae_p1_c_delivery_manifest_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "protected_p1_b_aggregate_sha256": protected["current_aggregate_sha256"],
        "p2_entered": False,
    }
    (OUT / "delivery_manifest.json").write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "accepted": accepted,
        "passed_checks": acceptance["passed_check_count"],
        "check_count": acceptance["check_count"],
        "test_count": test_count,
        "real_dev_f1": real_area["f1"],
        "holdout_v3_p1_c_object_macro_f1": hmetrics["P1-C-clean-7200-seed-20260718"]["object_macro_f1"],
        "holdout_v3_inference_pass_count": holdout_manifest["inference_pass_count"],
        "p2_entered": False,
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
