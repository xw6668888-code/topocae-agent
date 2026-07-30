from __future__ import annotations

"""Write the immutable P1-C preholdout selection before Real/Holdout-v3 access."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "p1_c_freeze_v1"
SELECTION = OUTPUT / "preholdout_selection.json"
SIDECAR = OUTPUT / "preholdout_selection.sha256.json"
SELECTED_VARIANT = "clean_7200"
SELECTED_SEED = 20260718
RUN_NAME = f"{SELECTED_VARIANT}_seed_{SELECTED_SEED}"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().lower()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if SELECTION.exists() or SIDECAR.exists():
        raise RuntimeError("P1-C preholdout selection is append-only and already exists")
    forbidden_existing = [
        ROOT / "output_generation_p1_c_synthetic_holdout_v3",
        ROOT / "output_generation_p1_c_holdout_v3",
        ROOT / "reports" / "p1_c_holdout_v3_once",
    ]
    existing = [str(path) for path in forbidden_existing if path.exists()]
    if existing:
        raise RuntimeError(f"Holdout-v3 artifact existed before freeze: {existing}")

    stage6_path = ROOT / "reports" / "p1_c_stage6_summary_v1" / "summary.json"
    stage6 = read_json(stage6_path)
    if not stage6["automatic_acceptance"]["stage6_pass"]:
        raise RuntimeError("Stage-6 did not pass")
    selected_row = next(
        row for row in stage6["iteration_2"]["rows"]
        if row["variant"] == SELECTED_VARIANT and int(row["seed"]) == SELECTED_SEED
    )
    if not selected_row["all_absolute_gates_pass"]:
        raise RuntimeError("predeclared primary P1-C seed did not pass")

    gnn_root = ROOT / "reports" / "p1_c_strict_training_runs_v2" / RUN_NAME
    candidate_root = ROOT / "reports" / "p1_c_stage6_iteration1_runs_v1" / RUN_NAME / "candidate_model_v1"
    iteration2_root = ROOT / "reports" / "p1_c_stage6_iteration2_runs_v1" / RUN_NAME
    dataset = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
    h_c1_csv = ROOT / "output_generation_p1_c1_strict_concept_hosts_pilot_96_r2" / "human_review_canonical_24.csv"
    h_c2_csv = ROOT / "output_generation_p1_c2_all_class_canonical_36_strict_concept_r3" / "human_review_canonical_36.csv"

    frozen_paths = [
        gnn_root / "best.pt",
        gnn_root / "config.json",
        gnn_root / "metrics.json",
        gnn_root / "history.json",
        gnn_root / "same_instance_threshold_selection.json",
        candidate_root / "best.pt",
        candidate_root / "config.json",
        candidate_root / "metrics.json",
        candidate_root / "history.json",
        iteration2_root / "candidate_thresholds_iteration2.json",
        iteration2_root / "global_candidate_decode_selection.json",
        iteration2_root / "synthetic_pipeline_v1" / "metrics.json",
        dataset / "dataset_meta.json",
        dataset / "manifests" / "artifact_manifest.json",
        dataset / "manifests" / "sample_manifest.jsonl",
        dataset / "manifests" / "compatibility_matrix.json",
        dataset / "reports" / "cleanup_audit.json",
        dataset / "reports" / "label_audit.json",
        dataset / "reports" / "split_audit.json",
        dataset / "reports" / "graph_input_audit.json",
        dataset / "reports" / "cooccurrence_matrix.json",
        dataset / "reports" / "strict_generation_contract_audit.json",
        dataset / "reports" / "graph_v2_robustness" / "robustness.json",
        dataset / "graph_polarity_v1_manifest.json",
        dataset / "splits" / "train.jsonl",
        dataset / "splits" / "val.jsonl",
        dataset / "splits" / "development_test.jsonl",
        dataset / "splits_matched_5000" / "matched_5000_meta.json",
        dataset / "splits_matched_5000" / "sample_manifest.jsonl",
        dataset / "training_packages" / "clean_7200" / "manifest.json",
        dataset / "training_packages" / "clean_7200" / "metadata.json",
        h_c1_csv,
        h_c2_csv,
        ROOT / "reports" / "p1_c_iterations_v1" / "h_c1_strict_concept_r2_authorization_receipt.json",
        ROOT / "reports" / "p1_c_iterations_v1" / "h_c2_strict_concept_r3_authorization_receipt.json",
        ROOT / "reports" / "p1_c_preflight" / "preflight.json",
        ROOT / "reports" / "p1_c_preflight" / "protected_inventory.json",
        stage6_path,
        ROOT / "configs" / "p1_c" / "training_plan_strict_v3.json",
        ROOT / "configs" / "p1_c" / "candidate_training_plan_strict_v2.json",
        ROOT / "configs" / "p1_c" / "stage6_iteration1_oriented_cylinder_polarity_v1.json",
        ROOT / "configs" / "p1_c" / "stage6_iteration2_candidate_verifier_calibration_v1.json",
        ROOT / "p1_c_general_recognition_pipeline_v1.py",
        ROOT / "generate_p1_c_graph_polarity_v1.py",
        ROOT / "p1_c" / "oriented_cylinder_polarity_v1.py",
        ROOT / "utils" / "p1_c_candidate_features_polarity_v1.py",
        ROOT / "utils" / "p1_c_topology_verifier_polarity_v1.py",
        ROOT / "utils" / "p1_b_instance_assembly.py",
        ROOT / "utils" / "p1_b_candidate_proposer.py",
        ROOT / "utils" / "p1_b_global_decode_v2.py",
        ROOT / "models" / "p1_b_candidate_topology_classifier.py",
        ROOT / "core" / "graph_v2_1.py",
        ROOT / "select_p1_c_candidate_decode_iteration2.py",
        ROOT / "p1_c" / "formal_generator_v7.py",
        ROOT / "p1_c" / "formal_plan_v6.py",
        ROOT / "p1_c" / "strict_generation_contract.py",
        ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json",
        ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json",
        ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json",
        ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json",
    ]
    frozen = [record(path) for path in frozen_paths]
    global_selection = read_json(iteration2_root / "global_candidate_decode_selection.json")
    candidate_thresholds = read_json(iteration2_root / "candidate_thresholds_iteration2.json")
    same_instance = read_json(gnn_root / "same_instance_threshold_selection.json")
    preflight = read_json(ROOT / "reports" / "p1_c_preflight" / "preflight.json")

    payload = {
        "schema_version": "topocae_p1_c_preholdout_selection_1",
        "phase": "P1-C",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_v3_artifacts_preexisting": False,
        "selected_variant": SELECTED_VARIANT,
        "selected_seed": SELECTED_SEED,
        "selection_basis": "predeclared primary seed; all model and threshold choices use Synthetic-Val only",
        "selection_function": {
            "default_variant": "clean_7200",
            "default_seed": 20260718,
            "aggregation_or_post_result_seed_switching": False,
            "source": str(ROOT / "configs" / "p1_c" / "training_plan_strict_v3.json"),
        },
        "selected_pipeline": "p1_c_general_recognition_pipeline_v1",
        "selected_gnn_checkpoint": str(gnn_root / "best.pt"),
        "selected_gnn_config": str(gnn_root / "config.json"),
        "selected_candidate_checkpoint": str(candidate_root / "best.pt"),
        "selected_candidate_config": str(candidate_root / "config.json"),
        "synthetic_val_thresholds": {
            "same_instance_path": str(gnn_root / "same_instance_threshold_selection.json"),
            "same_instance_global_threshold": same_instance["global_threshold"],
            "same_instance_selection_sha256": same_instance["selection_sha256"],
            "candidate_path": str(iteration2_root / "candidate_thresholds_iteration2.json"),
            "candidate_min_validity": candidate_thresholds["min_candidate_validity"],
            "candidate_membership_threshold": candidate_thresholds["membership_threshold"],
            "candidate_boundary_threshold": candidate_thresholds["boundary_threshold"],
            "candidate_selection_sha256": candidate_thresholds["selection_sha256"],
            "global_candidate_decode_path": str(iteration2_root / "global_candidate_decode_selection.json"),
            "min_candidate_confidence": global_selection["selected"]["min_candidate_confidence"],
            "global_subset_confidence_tolerance": global_selection["selected"]["subset_confidence_tolerance"],
            "global_candidate_decode_selection_sha256": global_selection["selection_sha256"],
        },
        "selected_synthetic_val_metrics": selected_row,
        "stage6_summary": {"path": str(stage6_path), "sha256": digest(stage6_path)},
        "human_gate_hashes": {"H_C1_r2": digest(h_c1_csv), "H_C2_r3": digest(h_c2_csv)},
        "dataset_and_representation": {
            "dataset_root": str(dataset),
            "dataset_meta_sha256": digest(dataset / "dataset_meta.json"),
            "sample_manifest_sha256": digest(dataset / "manifests" / "sample_manifest.jsonl"),
            "matched_5000_meta_sha256": digest(dataset / "splits_matched_5000" / "matched_5000_meta.json"),
            "clean_7200_training_manifest_sha256": digest(dataset / "training_packages" / "clean_7200" / "manifest.json"),
            "graph_polarity_manifest_sha256": digest(dataset / "graph_polarity_v1_manifest.json"),
            "quantiles_fitted_on_new_train_only": True,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "preflight_path": str(ROOT / "reports" / "p1_c_preflight" / "preflight.json"),
            "preflight_sha256": digest(ROOT / "reports" / "p1_c_preflight" / "preflight.json"),
            "preflight_environment": preflight.get("environment", preflight.get("runtime", {})),
        },
        "p1_b_frozen_baseline": {
            "freeze_path": str(ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"),
            "freeze_sha256": digest(ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json"),
            "training_summary_sha256": digest(ROOT / "reports" / "p1_b_training_ablation_summary_v1" / "summary.json"),
            "real_dev_metrics_sha256": digest(ROOT / "reports" / "p1_b_real_dev_v1" / "metrics.json"),
        },
        "holdout_v3_declared_comparison_variants": [
            {"id": "P1-B-frozen", "selection_manifest": str(ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.json")},
            {"id": "P1-C-clean-7200-seed-20260718", "selection_manifest": str(SELECTION)},
        ],
        "frozen_files": frozen,
        "contracts": {
            "real_data_used_for_training_or_selection": False,
            "real_dev_access_before_freeze": False,
            "holdout_v2_predictions_or_errors_used": False,
            "holdout_v3_access_before_freeze": False,
            "holdout_results_must_not_modify_selection": True,
            "part_or_class_specific_thresholds": False,
            "test_part_s3_used": False,
            "llm_step_recognition": False,
            "superpower_skills_used": False,
            "p2_entered": False,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SELECTION.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar = {
        "schema_version": "topocae_p1_c_preholdout_selection_sidecar_1",
        "selection_path": str(SELECTION),
        "sha256": digest(SELECTION),
        "bytes": SELECTION.stat().st_size,
        "verified_after_write": True,
    }
    SIDECAR.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    if digest(SELECTION) != sidecar["sha256"]:
        raise RuntimeError("preholdout selection verification failed")
    print(json.dumps({
        "selection": str(SELECTION),
        "sha256": sidecar["sha256"],
        "frozen_file_count": len(frozen),
        "selected_variant": SELECTED_VARIANT,
        "selected_seed": SELECTED_SEED,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
