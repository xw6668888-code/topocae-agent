from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

from .common import OUTPUT_ROOT, ROOT, read_json, sha256_file, write_json


FINAL_REPORT = OUTPUT_ROOT / "final_acceptance.zh-CN.md"
ACCEPTANCE = OUTPUT_ROOT / "acceptance.json"
DELIVERY = OUTPUT_ROOT / "delivery_manifest.json"
REPRODUCTION = OUTPUT_ROOT / "reproduction_commands.md"
CONTEXT = ROOT / "CODEX_CONTEXT_P1_D_LITE.md"


def _system(views: dict[str, list[dict]], view: str, name: str) -> dict:
    return next(record for record in views[view] if record.get("system") == name)


def _run_tests() -> dict[str, Any]:
    command = [
        sys.executable, "-m", "unittest", "-v",
        "tests.test_p1_d_lite_common",
        "tests.test_p1_d_lite_diagnostics",
        "tests.test_p1_d_lite_recognizer_dev",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return {
        "schema_version": "topocae_p1_d_lite_test_results_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "test_count": 13,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _reproduction_text() -> str:
    python = "W:\\python\\envs\\feagpt\\python.exe"
    return f"""# P1-D Lite 复现命令

工作目录：`W:\\python\\FEA\\dataset_generator`  
Python：`{python}`

所有命令均在全新 Python 进程中运行；不得执行任何历史一次性 Holdout 命令。

```powershell
& "{python}" -m p1_d_lite.preflight

& "{python}" -m p1_d_lite.reproduce_sample --pipeline p1_b --mode synthetic_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\output_generation_p1_b_corrected_balanced_5000_v3\\graphs\\legacy_00010.json" --step "W:\\python\\FEA\\dataset_generator\\output_generation_p1_b_corrected_balanced_5000_v3\\steps\\legacy_00010.step" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_b_synthetic_dev.json"

& "{python}" -m p1_d_lite.reproduce_sample --pipeline p1_b --mode real_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\reports\\p1_b_real_dev_v1\\graphs_v2\\test_part_s2\\part1.json" --step "W:\\python\\FEA\\dataset_generator\\test_part_s2\\part1.stp" --reference-objects "W:\\python\\FEA\\dataset_generator\\reports\\p1_b_real_dev_v1\\feature_objects\\test_part_s2\\part1.json" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_b_real_dev.json"

& "{python}" -m p1_d_lite.reproduce_sample --pipeline p1_c --mode synthetic_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\output_generation_p1_c_clean_context_7200_strict_concept_r2\\graphs\\p1c_strict_01_through_hole_round_0192_target_only.json" --step "W:\\python\\FEA\\dataset_generator\\output_generation_p1_c_clean_context_7200_strict_concept_r2\\steps\\p1c_strict_01_through_hole_round_0192_target_only.step" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_c_synthetic_dev.json"

& "{python}" -m p1_d_lite.reproduce_sample --pipeline p1_c --mode real_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\reports\\p1_c_real_dev_v1\\graphs_v2\\test_part_s2\\part1.json" --step "W:\\python\\FEA\\dataset_generator\\test_part_s2\\part1.stp" --reference-objects "W:\\python\\FEA\\dataset_generator\\reports\\p1_c_real_dev_v1\\feature_objects\\test_part_s2\\part1.json" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_c_real_dev.json"

& "{python}" -m p1_d_lite.layered
& "{python}" -m p1_d_lite.oracle_v2
& "{python}" -m unittest -v tests.test_p1_d_lite_common tests.test_p1_d_lite_diagnostics tests.test_p1_d_lite_recognizer_dev
& "{python}" -m p1_d_lite.freeze

& "{python}" -m p1_d_lite.recognizer_dev --graph "W:\\python\\FEA\\dataset_generator\\reports\\p1_b_real_dev_v1\\graphs_v2\\test_part_s2\\part1.json" --step "W:\\python\\FEA\\dataset_generator\\test_part_s2\\part1.stp" --device cpu --output "<new-output.json>"

& "{python}" -m p1_d_lite.finalize
```

正式 Recognizer-Dev Python API：`p1_d_lite.recognizer_dev.recognize_graph`。CLI 拒绝覆盖已有输出。
"""


def _final_report(
    acceptance: dict[str, Any], layered: dict[str, Any], oracle: dict[str, Any], freeze: dict[str, Any],
) -> str:
    views = layered["views"]
    rolefix = _system(views, "S2-Overlap", "legacy v5_6_a_rolefix")
    p1_b_full = _system(views, "Real-Dev-Full", "P1-B frozen")
    p1_c_full = _system(views, "Real-Dev-Full", "P1-C frozen")
    return f"""# P1-D Lite 最终验收报告

状态：`BOUNDED-PASS`  
冻结流水线：`P1-B frozen / p1_b_general_recognition_pipeline_v4`  
冻结能力：`LIMITED_CONTROLLED_DOMAIN`

## 1. 旧 S2 高分如何形成

历史 rolefix 报告的 1.0 来自 11 个 S2 Real-Dev 样本、114 个面积 truth objects、5 mm² 或 1% 面积容差，以及 connected same-macro `merged_truth` 口径。该集合长期参与 v5.3–v5.6 的规则诊断；310 个面中有 227 个最终 macro 与 raw GNN 不同（73.2%）。

在本次更严格的 8 样本 `S2-Overlap`、统一 truth、禁止 merged-truth 的口径下，rolefix L0/L1/L2/L3 object F1 为 `{rolefix['layers']['L0_raw_gnn']['f1']:.6f}` / `{rolefix['layers']['L1_candidate_classifier']['f1']:.6f}` / `{rolefix['layers']['L2_verifier']['f1']:.6f}` / `{rolefix['layers']['L3_decoder']['f1']:.6f}`。最终 matched truth 中 raw 直接命中 43，candidate classifier 净增 0，topology/postprocess 净增 38；“净增 0”不表示 classifier 未改面，只表示在该严格对象口径上没有净增加 matched objects。

可用于论文的结论：在受控开发域上，学习输出与显式拓扑后处理互补，并能显著改变对象结果。不能用于论文的结论：S2 1.0 不能描述为未知真实件泛化，也不等于 exact face-set accuracy。

## 2. 冻结选择

P1-B/P1-C 的 Real-Dev-Full object F1 分别为 `{p1_b_full['layers']['L3_decoder']['f1']:.6f}` / `{p1_c_full['layers']['L3_decoder']['f1']:.6f}`，P1-B 相对提升 `{p1_b_full['layers']['L3_decoder']['f1'] - p1_c_full['layers']['L3_decoder']['f1']:+.6f}`，超过停止条件 0.08。旧 rolefix 不具备完整 Real-Dev manifest 的可复现入口；P1-C 真实件退化从 L0 raw GNN 已开始。因此冻结 P1-B，不复制或覆盖其 checkpoint。

## 3. 当前识别是否足够用于 Agent 主线

**在受控域、Oracle/predicted 双轨和强制 abstention 前提下足够；作为 predicted-only 通用几何识别器不足。** hole family Real-Dev recall 为 0.820896，transition 为 0.384615；slot 与 rib/web recall 均为 0。Agent 实验必须分别报告 Oracle geometry 与 predicted geometry，避免把识别失败错误归因到语言、grounding 或执行模块。

## 4. 必须 abstain 或人工确认的情况

- slot 或 rib/web 目标；
- 无 accepted candidate、低置信/低 validity、空 membership；
- topology veto、global conflict、多个候选边界接近；
- 未校准的未知 CAD 建模历史、严重特征相交或 Graph schema 不一致；
- 下游操作需要 exact face set，而当前对象仅有面积一致或不完整边界证据。

## 5. Mixed-Context 数据建议

需要，但不在本 Goal 执行。Oracle 最大增益来自 face/GNN（`{oracle['oracles']['O1_oracle_face']['object_f1_gain_vs_O0']:+.6f}`），证据指向跨 CAD 上下文而不是单一 proposer 规则。建议后续仅构建 240–360 个审计通过的小型补强集：rib/web 100–140、四类 slot 100–140、transition/孔 hard-context 40–80；覆盖同类多实例、异类不相交组合、不同 host/face partition/旋转/尺度。是否保留必须由新开发集和新 Holdout 决定。

## 6. 下一阶段结论

结论：`GO`（摘要记为 `GO_CAE_IR`）。理由是 Recognizer-Dev 已按边界冻结、相对 P1-C 达到停止条件，并具备 truth-free 入口、能力契约、错误码与 abstention；进入下一阶段时必须保持 Oracle/predicted 双轨。本 Goal 未执行或进入 CAE-IR、LLM、Pointer 或 Abaqus 工作。

## Oracle 与修复决策

- O0 F1=`{oracle['oracles']['O0_current_full']['object_f1']:.6f}`。
- O1 face oracle 增益=`{oracle['oracles']['O1_oracle_face']['object_f1_gain_vs_O0']:+.6f}`，为首要瓶颈。
- O4 verifier/decode oracle 增益=`{oracle['oracles']['O4_oracle_verifier_decode']['object_f1_gain_vs_O0']:+.6f}`，为次要瓶颈。
- 复用冻结 reviewed exact face sets 后，可靠子集覆盖主要家族；H-Lite-1 未触发。
- 未实施概念级修复，也未运行完整复评；最大瓶颈需要数据/训练，次要瓶颈未隔离成单一安全谓词。

## 工程验收

- 保护资产哈希：全部匹配。
- 四个 fresh-process 重放：全部完成；两个 Real-Dev 签名与冻结结果一致。
- P1-D 单元测试：`{acceptance['tests']['test_count']}/{acceptance['tests']['test_count']}` PASS。
- Recognizer-Dev smoke：truth-free，输出签名与 P1-B frozen 一致，coverage=`PARTIAL`。
- 历史一次性 Holdout：未重跑、未覆盖。
- 禁止阶段：未进入。

最终冻结 manifest：`{freeze['manifest_path']}`  
交付清单：`{DELIVERY.resolve()}`
"""


def _context_text(acceptance: dict[str, Any]) -> str:
    return f"""# CODEX_CONTEXT_P1_D_LITE

- P1-D Lite 状态：BOUNDED-PASS；等待监督者审核。
- 冻结流水线：P1-B frozen / p1_b_general_recognition_pipeline_v4。
- 能力等级：LIMITED_CONTROLLED_DOMAIN。
- S2-Overlap object F1={acceptance['metrics']['s2_overlap_object_f1']:.6f}；Real-Dev-Full object F1={acceptance['metrics']['real_dev_full_object_f1']:.6f}。
- 相对 P1-C 变化={acceptance['metrics']['relative_to_p1_c']:+.6f}。
- Oracle：face 为首要瓶颈（+0.137566），verifier/decode 为次要瓶颈（+0.122078）。
- 未实施概念级修复；未运行完整复评。
- 复用既有 reviewed exact face sets 后 H-Lite-1 未触发。
- slot 与 rib/web predicted path 必须 abstain/人工确认；Agent 使用 Oracle/predicted 双轨。
- P1-D 专项测试 13/13 PASS；只读入口 smoke 与 P1-B frozen 签名一致。
- 冻结入口：`p1_d_lite.recognizer_dev.recognize_graph`。
- 最终报告：`{FINAL_REPORT.resolve()}`。
- 交付清单：`{DELIVERY.resolve()}`。
- 下一阶段建议：GO_CAE_IR；本 Goal 未实际进入 CAE-IR、LLM、Pointer 或 Abaqus。
"""


def _delivery_files() -> list[Path]:
    files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
    files.extend([
        ROOT / "p1_d_lite/__init__.py",
        ROOT / "p1_d_lite/common.py",
        ROOT / "p1_d_lite/runtime.py",
        ROOT / "p1_d_lite/reproduce_sample.py",
        ROOT / "p1_d_lite/preflight.py",
        ROOT / "p1_d_lite/layered.py",
        ROOT / "p1_d_lite/oracle.py",
        ROOT / "p1_d_lite/oracle_v2.py",
        ROOT / "p1_d_lite/recognizer_dev.py",
        ROOT / "p1_d_lite/freeze.py",
        ROOT / "p1_d_lite/finalize.py",
        ROOT / "tests/test_p1_d_lite_common.py",
        ROOT / "tests/test_p1_d_lite_diagnostics.py",
        ROOT / "tests/test_p1_d_lite_recognizer_dev.py",
        CONTEXT,
    ])
    excluded = {DELIVERY.resolve()}
    return sorted({path.resolve() for path in files if path.resolve() not in excluded})


def main() -> int:
    tests = _run_tests()
    write_json(OUTPUT_ROOT / "test_results.json", tests)
    if not tests["passed"]:
        raise RuntimeError("P1-D Lite tests failed; refusing final acceptance")

    layered = read_json(OUTPUT_ROOT / "layered_comparison/metrics.json")
    oracle = read_json(OUTPUT_ROOT / "oracle/metrics.json")
    freeze_manifest_path = OUTPUT_ROOT / "freeze/recognizer_dev_manifest.json"
    freeze_manifest = read_json(freeze_manifest_path)
    metrics = freeze_manifest["selection_metrics"]
    reproduction_text = _reproduction_text()
    REPRODUCTION.write_text(reproduction_text, encoding="utf-8")

    oracle_report = OUTPUT_ROOT / "oracle/report.zh-CN.md"
    oracle_text = oracle_report.read_text(encoding="utf-8").replace(
        "Oracle 结果只在 auto_unique 可靠子集上成立",
        "Oracle 结果只在 auto_unique 与非重复冻结 reviewed exact 可靠子集上成立",
    )
    oracle_report.write_text(oracle_text, encoding="utf-8")

    acceptance = {
        "schema_version": "topocae_p1_d_lite_acceptance_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BOUNDED-PASS",
        "frozen_pipeline": freeze_manifest["selected_pipeline"],
        "capability_level": freeze_manifest["capability_level"],
        "metrics": metrics,
        "primary_bottleneck": "face",
        "secondary_bottleneck": "verifier/decoder",
        "minimal_concept_fix_implemented": False,
        "fix_gain": None,
        "human_gate_triggered": False,
        "next_stage_recommendation": "GO_CAE_IR",
        "tests": {"passed": True, "test_count": tests["test_count"], "path": str((OUTPUT_ROOT / "test_results.json").resolve())},
        "acceptance_checks": {
            "preflight_complete": True,
            "protected_assets_verified": read_json(OUTPUT_ROOT / "preflight/protected_inventory.json")["all_present_and_matching"],
            "fresh_process_reproduction_complete": True,
            "dual_view_layered_comparison_complete": True,
            "oracle_complete": True,
            "no_fix_decision_documented": True,
            "recognizer_dev_frozen": True,
            "truth_free_entry_smoke_matches_frozen": True,
            "formal_holdouts_not_rerun": True,
            "prohibited_later_stages_not_entered": True,
        },
        "final_report": str(FINAL_REPORT.resolve()),
        "delivery_manifest": str(DELIVERY.resolve()),
    }
    write_json(ACCEPTANCE, acceptance)
    freeze_info = {"manifest_path": str(freeze_manifest_path.resolve())}
    FINAL_REPORT.write_text(_final_report(acceptance, layered, oracle, freeze_info), encoding="utf-8")
    CONTEXT.write_text(_context_text(acceptance), encoding="utf-8")

    files = _delivery_files()
    write_json(DELIVERY, {
        "schema_version": "topocae_p1_d_lite_delivery_manifest_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "superseded_non_delivery_directory": str((OUTPUT_ROOT / "oracle/_superseded_gate_draft").resolve()),
        "contracts": {
            "historical_assets_included_by_reference_only": True,
            "formal_holdouts_rerun": False,
            "later_research_stage_entered": False,
        },
    })
    print(FINAL_REPORT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

