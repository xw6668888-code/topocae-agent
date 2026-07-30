from __future__ import annotations

"""Build the final read-only P1-B delivery and review-package manifests."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "reports" / "p1_b_final_acceptance_v1"
HOLDOUT_REPORT = ROOT / "reports" / "p1_b_holdout_v2_once"


KEY_FILES = [
    ROOT / "GOAL_P1_B_GENERAL_GEOMETRY_RECOGNITION_REPAIR.zh-CN.md",
    ROOT / "CODEX_CONTEXT_P1_B.md",
    ROOT / "run_paper_v1_p1_b_pipeline.py",
    ROOT / "p1_b_general_recognition_pipeline_v4.py",
    ROOT / "output_generation_p1_b1_slot_pilot_80" / "human_review_canonical_20.csv",
    ROOT / "output_generation_p1_b2_rib_web_pilot_48" / "human_review_canonical_12.csv",
    ROOT / "reports" / "p1_b_graph_v2" / "automatic_validation.json",
    ROOT / "reports" / "p1_b_stage4_topology_closure" / "automatic_validation.json",
    ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "dataset_meta.json",
    ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "reports" / "dataset_card.zh-CN.md",
    ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "reports" / "artifact_manifest.json",
    ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json",
    ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json",
    ROOT / "reports" / "p1_b_real_dev_v1" / "failure_atlas.json",
    ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json",
    ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json",
    ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "holdout_generation_manifest.json",
    ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "leakage_audit.json",
    ROOT / "reports" / "p1_b_holdout_v2_once" / "metrics.json",
    ROOT / "reports" / "p1_b_holdout_v2_once" / "freeze_manifest.json",
    ROOT / "reports" / "p1_b_final_acceptance_v1" / "acceptance.json",
    ROOT / "reports" / "p1_b_final_acceptance_v1" / "acceptance_summary.zh-CN.md",
    ROOT / "reports" / "p1_b_final_acceptance_v1" / "necessary_test_results.json",
    ROOT / "reports" / "p1_b_final_acceptance_v1" / "reproduction_commands.json",
]


def main() -> int:
    missing = [str(path) for path in KEY_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing P1-B delivery files: {missing}")
    review_csv = HOLDOUT_REPORT / "human_review_stratified_20.csv"
    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    review_records = []
    for row in rows:
        path = HOLDOUT_REPORT / "human_review_stratified_20" / row["step_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        review_records.append({
            "review_index": int(row["review_index"]),
            "sample_id": row["sample_id"],
            "stratum_macro_feature": row["stratum_macro_feature"],
            "path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size,
        })
    macros = {record["stratum_macro_feature"] for record in review_records}
    review_manifest = {
        "schema_version": "topocae_p1_b_holdout_v2_review_package_manifest_1",
        "step_count": len(review_records),
        "stratum_class_count": len(macros),
        "all_18_classes_covered": len(macros) == 18,
        "slot_present": any(name.startswith("slot_") for name in macros),
        "rib_present": "rib" in macros,
        "web_present": "web" in macros,
        "csv_path": str(review_csv.resolve()),
        "csv_sha256": _sha256(review_csv),
        "records": review_records,
        "contract": "review package is post-inference evidence and must not modify the frozen result",
    }
    _write_json(HOLDOUT_REPORT / "review_package_manifest.json", review_manifest)
    holdout_freeze = _read_json(HOLDOUT_REPORT / "freeze_manifest.json")
    real = _read_json(ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json")
    acceptance = _read_json(FINAL / "acceptance.json")
    delivery = {
        "schema_version": "topocae_p1_b_final_delivery_manifest_1",
        "phase": "P1-B",
        "automatic_pass": bool(acceptance["automatic_pass"]),
        "p2_entered": False,
        "key_files": [_record(path) for path in KEY_FILES],
        "checkpoints": [
            _record(ROOT / "reports" / "p1_b_training_runs_v1" / f"same_instance_seed_{seed}" / "best.pt")
            for seed in (20260718, 20260719, 20260720)
        ] + [
            _record(ROOT / "reports" / "p1_b_training_runs_v1" / f"same_instance_seed_{seed}" / "candidate_model_v1" / "best.pt")
            for seed in (20260718, 20260719, 20260720)
        ],
        "counts": {
            "corrected_dataset_samples": 5000,
            "holdout_v2_samples": 1080,
            "holdout_v2_prediction_files": holdout_freeze["sample_prediction_count"],
            "holdout_v2_formal_inference_passes": 1,
            "real_dev_steps": real["evaluated_step_count"],
            "review_steps": review_manifest["step_count"],
        },
        "contracts": {
            "test_part_s3_used": False,
            "real_specific_thresholds": False,
            "llm_step_recognition": False,
            "holdout_used_for_model_or_threshold_selection": False,
            "superpower_skills_used": False,
        },
        "review_package_manifest": _record(HOLDOUT_REPORT / "review_package_manifest.json"),
        "unresolved_limitations": acceptance["unresolved_limitations"],
    }
    _write_json(FINAL / "delivery_manifest.json", delivery)
    print(json.dumps({"automatic_pass": delivery["automatic_pass"], "key_files": len(KEY_FILES), "checkpoints": 6, "review_steps": len(review_records), "output": str(FINAL / "delivery_manifest.json")}, ensure_ascii=False))
    return 0


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


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
