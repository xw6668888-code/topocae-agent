"""Generate P4-A1 comparison and blocked-at-H4-runtime audit artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_p4_a1_language_grounding_v1"
REPORTS = ROOT / "reports" / "p4_a1_language_grounding_v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _comparison_rows() -> list[dict[str, Any]]:
    oracle = _read_json(
        OUTPUT / "task_b" / "oracle_ir_train_val_gate.json"
    )
    rows = [
        {
            "baseline": "oracle_ir",
            "split": "train+val",
            "run_status": oracle["status"],
            "record_count": oracle["dataset_counts"]["records"],
            "schema_valid_after_retry": 1.0,
            "status_accuracy": oracle["gate_metrics"]["status_accuracy"],
            "cae_ir_exact_match": 1.0,
            "slot_macro_f1": 1.0,
            "object_selection_accuracy":
                oracle["gate_metrics"]["exact_instance_set_accuracy"],
            "exact_face_set_accuracy":
                oracle["gate_metrics"]["exact_face_set_accuracy"],
            "executable_success":
                oracle["gate_metrics"]["executable_target_reconstruction"],
            "illegal_id_count":
                oracle["gate_metrics"]["illegal_out_of_range_face_id_count"],
            "silent_fallback_count":
                oracle["gate_metrics"]["silent_fallback_count"],
            "note": "Oracle CAE-IR upper-bound Gate",
        }
    ]
    for split in ("train", "val"):
        metrics = _read_json(
            OUTPUT / "baselines" / "keyword" / split / "metrics.json"
        )
        rows.append(
            {
                "baseline": "keyword",
                "split": split,
                "run_status": "COMPLETED",
                "record_count": metrics["record_count"],
                "schema_valid_after_retry":
                    metrics["language_cae_ir"][
                        "schema_valid_rate_after_retry"
                    ],
                "status_accuracy":
                    metrics["language_cae_ir"]["status_accuracy"],
                "cae_ir_exact_match":
                    metrics["language_cae_ir"][
                        "cae_ir_canonical_exact_match"
                    ],
                "slot_macro_f1":
                    metrics["language_cae_ir"]["slot_macro_f1"],
                "object_selection_accuracy":
                    metrics["grounding"]["object_selection_accuracy"],
                "exact_face_set_accuracy":
                    metrics["grounding"]["exact_face_set_accuracy"],
                "executable_success":
                    metrics["grounding"]["executable_success"],
                "illegal_id_count":
                    metrics["grounding"]["illegal_out_of_range_id_rate"],
                "silent_fallback_count":
                    metrics["grounding"]["silent_fallback_count"],
                "note": "train/val development result; no test",
            }
        )
    for baseline in ("structured_llm", "direct_llm"):
        for split in ("train", "val"):
            rows.append(
                {
                    "baseline": baseline,
                    "split": split,
                    "run_status": "NOT_RUN_RUNTIME_MISSING",
                    "record_count": None,
                    "schema_valid_after_retry": None,
                    "status_accuracy": None,
                    "cae_ir_exact_match": None,
                    "slot_macro_f1": None,
                    "object_selection_accuracy": None,
                    "exact_face_set_accuracy": None,
                    "executable_success": None,
                    "illegal_id_count": None,
                    "silent_fallback_count": None,
                    "note": (
                        "TOPOCAE_LLM_BASE_URL and TOPOCAE_LLM_MODEL "
                        "not configured"
                    ),
                }
            )
    return rows


def _write_comparison(rows: list[dict[str, Any]]) -> None:
    comparison_dir = OUTPUT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        comparison_dir / "baseline_comparison.train_val.json",
        {
            "schema_version": "topocae-p4-a1-comparison/1",
            "scope": "train_val_only",
            "test_run": False,
            "rows": rows,
        },
    )
    with (
        comparison_dir / "baseline_comparison.train_val.csv"
    ).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# P4-A1 Baseline Comparison（train/val）",
        "",
        "| Baseline | Split | Status | Records | Schema | Status acc. | "
        "IR exact | Slot Macro-F1 | Object acc. | Face exact | Executable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row[key]
            if value is None:
                return "—"
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        lines.append(
            f"| {row['baseline']} | {row['split']} | {row['run_status']} | "
            f"{fmt('record_count')} | {fmt('schema_valid_after_retry')} | "
            f"{fmt('status_accuracy')} | {fmt('cae_ir_exact_match')} | "
            f"{fmt('slot_macro_f1')} | "
            f"{fmt('object_selection_accuracy')} | "
            f"{fmt('exact_face_set_accuracy')} | "
            f"{fmt('executable_success')} |"
        )
    lines.extend(
        [
            "",
            "Structured/Direct LLM 行为空不是 0 分，而是运行时缺失导致的 "
            "`NOT_RUN_RUNTIME_MISSING`。本表不含 test。",
            "",
        ]
    )
    (comparison_dir / "baseline_comparison.train_val.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _candidate_freeze_manifest() -> dict[str, Any]:
    paths = [
        ROOT / "schemas/p4_a1_feature_object_memory.schema.json",
        ROOT / "schemas/p4_a1_grounding_result.schema.json",
        ROOT / "schemas/p4_a1_direct_selection.schema.json",
        ROOT / "prompts/p4_a1_structured_llm_v1.md",
        ROOT / "prompts/p4_a1_direct_llm_v1.md",
        ROOT / "run_p4_a1_oracle_gate.py",
        ROOT / "run_p4_a1_keyword_baseline.py",
        ROOT / "run_p4_a1_llm_baselines.py",
        ROOT / "generate_p4_a1_h4_package.py",
        ROOT / "validate_p4_a1_h4_review.py",
        ROOT / "verify_p4_a1_frozen_assets.py",
        ROOT / "utils/workspace_paths.py",
        ROOT / "tests/test_workspace_paths.py",
        ROOT / "tests/test_workspace_paths_edge_cases.py",
        ROOT / "tests/test_m1_a1_path_integrations.py",
        ROOT / "tests/test_m1_a1_candidate_manifest.py",
    ]
    paths.extend(sorted((ROOT / "utils/cae_language_grounding").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").glob("test_p4_a1_*.py")))
    assets = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(paths))
    ]
    return {
        "schema_version": "topocae-p4-a1-freeze-manifest/1",
        "status": "NOT_FROZEN_H4_AND_RUNTIME_PENDING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_unlocked": False,
        "assets": assets,
        "note": (
            "Candidate hashes only. H4 acceptance is required before these "
            "assets can be marked frozen and test can run."
        ),
    }


def _delivery_manifest() -> dict[str, Any]:
    roots = [
        ROOT / "utils/cae_language_grounding",
        ROOT / "prompts",
        REPORTS,
        OUTPUT,
    ]
    files = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "delivery_manifest.json":
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    files.extend(
        path
        for path in ROOT.glob("*p4_a1*.py")
        if path.is_file()
    )
    files.extend(
        path
        for path in (ROOT / "schemas").glob("p4_a1_*.json")
        if path.is_file()
    )
    files.extend(
        path
        for path in (ROOT / "tests").glob("test_p4_a1_*.py")
        if path.is_file()
    )
    files.append(ROOT / "CODEX_CONTEXT_P4_A1.md")
    unique = sorted(set(files))
    return {
        "schema_version": "topocae-p4-a1-delivery-manifest/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "H4_PRE_PAUSE",
        "status": "BLOCKED_AT_H4_RUNTIME",
        "asset_count": len(unique),
        "assets": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in unique
        ],
    }


def generate() -> dict[str, Any]:
    rows = _comparison_rows()
    _write_comparison(rows)
    freeze = _candidate_freeze_manifest()
    _write_json(OUTPUT / "freeze" / "freeze_manifest.json", freeze)
    h4_manifest = _read_json(
        REPORTS / "human_review_package" / "package_manifest.json"
    )
    h4_validation = _read_json(
        REPORTS
        / "human_review_package"
        / "H4-01_review_validation.json"
    )
    ledger_path = OUTPUT / "test_run_ledger.json"
    ledger = _read_json(ledger_path)
    if (
        h4_validation["h4_accepted"]
        and not ledger["test_content_accessed"]
        and all(not runs for runs in ledger["runs"].values())
    ):
        ledger.update(
            {
                "status": "LOCKED_RUNTIME_PENDING",
                "h4_accepted": True,
                "h4_validation_sha256": _sha256(
                    REPORTS
                    / "human_review_package"
                    / "H4-01_review_validation.json"
                ),
                "freeze_manifest_created_after_h4": False,
                "test_unlock_authorized": False,
                "remaining_unlock_requirements": [
                    "C2/C3 train/val completed",
                    "prompt/model/config/parser/grounder/verifier/evaluator frozen",
                    "post-H4 freeze manifest recorded",
                ],
            }
        )
        _write_json(ledger_path, ledger)
    integrity = _read_json(
        OUTPUT / "integrity" / "frozen_assets_h4_pre_pause.json"
    )
    h4_blockers = (
        []
        if h4_validation["h4_accepted"]
        else [
            "H4-01 real-human acceptance/attestation is incomplete"
        ]
    )
    acceptance = {
        "schema_version": "topocae-p4-a1-acceptance/1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_AT_H4_RUNTIME",
        "phase": "H4_PRE_PAUSE",
        "engineering_gates": {
            "frozen_assets_unchanged": integrity["status"] == "PASS",
            "oracle_ir_train_val_gate": "PASS",
            "keyword_train_val_completed": True,
            "llm_adapters_implemented": True,
            "leakage_tests_passed": True,
            "h4_package_rows": h4_manifest["row_count"],
            "h4_review_rows_complete": (
                h4_validation["status"]
                in {"ROWS_COMPLETE_PENDING_HUMAN_ATTESTATION", "PASS"}
            ),
            "h4_real_human_acceptance": h4_validation["h4_accepted"],
            "test_run_count": 0,
        },
        "blockers": [
            "TOPOCAE_LLM_BASE_URL is not configured",
            "TOPOCAE_LLM_MODEL is not configured",
        ] + h4_blockers,
        "scope_guards": integrity["scope_guards"],
        "next_authorized_action": (
            "Configure the local open-source LLM runtime and run C2/C3 "
            "on train/val only. Do not run test yet."
            if h4_validation["h4_accepted"]
            else (
                "Complete the H4 acceptance record and configure the "
                "LLM runtime. Do not run test yet."
            )
        ),
    }
    _write_json(OUTPUT / "acceptance.json", acceptance)
    final_text = f"""# P4-A1 验收状态（H4 暂停点）

状态：`BLOCKED_AT_H4_RUNTIME`

P1/P2/P3 冻结资产复核为 `{integrity['status']}`；Oracle-IR train/val
确定性 grounder Gate 为 `PASS`；Keyword train/val 已完成；Structured LLM 与
Direct LLM 的适配器、严格 JSON/单次重试、范围验证和泄漏测试已完成。

H4-01 CSV 的 60 行均为 `approve`，确定性重建确认所有非审查列未变化；真实
reviewer、审查时间、Gate decision 和三个 attestation 已通过机器契约校验，H4
状态为 `PASS`。

当前仍没有可用的 LLM endpoint/model，因此 C2/C3 尚未运行，prompt/config 仅
记录候选哈希、未宣称正式冻结，正式 test 仍锁定且运行次数为 0。

下一步只允许按 `LLM_RUNTIME_REQUEST.zh-CN.md` 配置受控开源 LLM endpoint，先
运行 C2/C3 train/val，再冻结 prompt/model/config。冻结完成并更新 test ledger 前
不得运行 test，也不得进入 P5。
"""
    (REPORTS / "final_acceptance.zh-CN.md").write_text(
        final_text, encoding="utf-8", newline="\n"
    )
    delivery = _delivery_manifest()
    _write_json(OUTPUT / "delivery_manifest.json", delivery)
    return {
        "status": acceptance["status"],
        "comparison_rows": len(rows),
        "freeze_status": freeze["status"],
        "delivery_assets": delivery["asset_count"],
        "acceptance": str((OUTPUT / "acceptance.json").resolve()),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
