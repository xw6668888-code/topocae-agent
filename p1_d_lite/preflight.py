from __future__ import annotations

from datetime import datetime, timezone
import platform
from pathlib import Path
import sys
from typing import Any

import numpy
import sklearn
import torch
import OCC

from .common import OUTPUT_ROOT, ROOT, read_json, sha256_file, write_json
from .runtime import PIPELINES


PREFLIGHT = OUTPUT_ROOT / "preflight"


def _asset(path: Path, category: str, declared: str | None = None) -> dict[str, Any]:
    exists = path.is_file()
    current = sha256_file(path) if exists else None
    return {
        "category": category,
        "path": str(path.resolve()),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "declared_sha256": declared.lower() if declared else None,
        "current_sha256": current,
        "status": "verified" if exists and (declared is None or current == declared.lower()) else "missing_or_mismatch",
    }


def protected_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for label, freeze_path in (
        ("p1_b", ROOT / "reports/p1_b_freeze_v1/preholdout_selection.json"),
        ("p1_c", ROOT / "reports/p1_c_freeze_v1/preholdout_selection.json"),
    ):
        freeze = read_json(freeze_path)
        records.append(_asset(freeze_path, f"{label}_freeze_manifest"))
        sidecar = freeze_path.with_name("preholdout_selection.sha256.json")
        records.append(_asset(sidecar, f"{label}_freeze_sidecar"))
        for frozen in freeze.get("frozen_files", []):
            records.append(_asset(Path(frozen["path"]), f"{label}_frozen_asset", str(frozen["sha256"])))

    for label, directory in (
        ("p1_b_final_report", ROOT / "reports/p1_b_final_acceptance_v1"),
        ("p1_c_final_report", ROOT / "reports/p1_c_final_acceptance_v1"),
    ):
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            records.append(_asset(path, label))

    for manifest_path, category in (
        (ROOT / "reports/p1_b_real_dev_v1/manifest.json", "real_dev_manifest"),
        (ROOT / "reports/p1_c_real_dev_v1/manifest.json", "real_dev_manifest"),
        (ROOT / "reports/p1_b_real_dev_v1/area_truth.json", "real_dev_truth"),
        (ROOT / "output_real_step_s2_v5_6_a_rolefix/predictions/edge_aware_v5_6_a_candidate_topology_post/prediction_manifest.json", "rolefix_prediction_manifest"),
        (ROOT / "output_real_step_s2_v5_6_a_rolefix/feature_objects/edge_aware_v5_6_a_candidate_topology_post/feature_object_manifest.json", "rolefix_feature_object_manifest"),
        (ROOT / "CAE_feature_ontology.yaml", "label_map"),
        (ROOT / "face_semantic_subtype_ontology.yaml", "label_map"),
    ):
        records.append(_asset(manifest_path, category))

    rolefix_manifest = read_json(ROOT / "output_real_step_s2_v5_6_a_rolefix/predictions/edge_aware_v5_6_a_candidate_topology_post/prediction_manifest.json")
    records.append(_asset(Path(rolefix_manifest["checkpoint"]), "rolefix_gnn_checkpoint"))
    records.append(_asset(Path(rolefix_manifest["candidate_classifier"]), "rolefix_candidate_classifier"))

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        existing = unique.get(record["path"])
        if existing is None or (existing["declared_sha256"] is None and record["declared_sha256"] is not None):
            unique[record["path"]] = record
    final = list(unique.values())
    return {
        "schema_version": "topocae_p1_d_lite_protected_inventory_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_count": len(final),
        "verified_count": sum(record["status"] == "verified" for record in final),
        "all_present_and_matching": all(record["status"] == "verified" for record in final),
        "assets": final,
        "contracts": {
            "historical_assets_are_read_only": True,
            "formal_holdouts_will_not_be_rerun": True,
            "excluded_research_data_accessed": False,
        },
    }


def pipeline_inventory() -> dict[str, Any]:
    rolefix_prediction_manifest = ROOT / "output_real_step_s2_v5_6_a_rolefix/predictions/edge_aware_v5_6_a_candidate_topology_post/prediction_manifest.json"
    rolefix_feature_manifest = ROOT / "output_real_step_s2_v5_6_a_rolefix/feature_objects/edge_aware_v5_6_a_candidate_topology_post/feature_object_manifest.json"
    return {
        "schema_version": "topocae_p1_d_lite_pipeline_inventory_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipelines": [
            {
                "name": "P1-B frozen",
                "actual_entry": str((ROOT / "p1_b_general_recognition_pipeline_v4.py").resolve()),
                "verification_entry": str((ROOT / "run_paper_v1_p1_b_pipeline.py").resolve()),
                "gnn_checkpoint": str(PIPELINES["p1_b"].gnn_run / "best.pt"),
                "candidate_classifier": str(PIPELINES["p1_b"].candidate_run / "best.pt"),
                "candidate_proposer": str((ROOT / "utils/p1_b_candidate_proposer.py").resolve()),
                "topology_verifier": str((ROOT / "utils/p1_b_topology_verifier_v4.py").resolve()),
                "decoder": str((ROOT / "utils/p1_b_global_decode_v2.py").resolve()),
                "truth_stripping_at_native_entry": False,
                "p1_d_wrapper_truth_stripping": True,
                "import_state_risk": "low in core pipeline; legacy Real-Dev wrapper replaces a module global at import time",
            },
            {
                "name": "P1-C frozen",
                "actual_entry": str((ROOT / "p1_c_general_recognition_pipeline_v1.py").resolve()),
                "verification_entry": str((ROOT / "run_paper_v1_p1_c_pipeline.py").resolve()),
                "gnn_checkpoint": str(PIPELINES["p1_c"].gnn_run / "best.pt"),
                "candidate_classifier": str(PIPELINES["p1_c"].candidate_run / "best.pt"),
                "candidate_proposer": str((ROOT / "utils/p1_b_candidate_proposer.py").resolve()),
                "topology_verifier": str((ROOT / "utils/p1_c_topology_verifier_polarity_v1.py").resolve()),
                "decoder": str((ROOT / "utils/p1_b_global_decode_v2.py").resolve()),
                "truth_stripping_at_native_entry": True,
                "p1_d_wrapper_truth_stripping": True,
                "import_state_risk": "formal core is explicit; historical Real-Dev runner remaps protocol globals and loaders at runtime",
            },
            {
                "name": "legacy v5_6_a_rolefix",
                "actual_entry": None,
                "gnn_checkpoint": read_json(rolefix_prediction_manifest)["checkpoint"],
                "candidate_classifier": read_json(rolefix_prediction_manifest)["candidate_classifier"],
                "candidate_proposer": "historical prediction-component/candidate postprocess recorded in prediction JSON",
                "topology_verifier": str((ROOT / "utils/topology_postprocess.py").resolve()),
                "decoder": "prediction_to_feature_objects connected same-macro components",
                "prediction_manifest": str(rolefix_prediction_manifest.resolve()),
                "feature_object_manifest": str(rolefix_feature_manifest.resolve()),
                "truth_stripping_at_native_entry": "not auditable from manifests",
                "import_state_risk": "high reproducibility risk: manifests omit the executable command, source revision and full threshold/config snapshot",
            },
        ],
        "current_best_runnable_pipeline_before_oracle": "P1-B frozen",
        "selection_status": "provisional; final choice requires layered and Oracle evidence",
    }


def environment() -> dict[str, Any]:
    package_versions = {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "OpenCASCADE": getattr(OCC, "VERSION", None),
    }
    return {
        "schema_version": "topocae_p1_d_lite_environment_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "random_seed": 20260720,
    }


def reproduction_commands() -> dict[str, Any]:
    python = str(Path(sys.executable).resolve())
    base = f'& "{python}" -m p1_d_lite.reproduce_sample'
    commands = [
        {
            "id": "p1_b_synthetic_dev",
            "command": base + ' --pipeline p1_b --mode synthetic_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\output_generation_p1_b_corrected_balanced_5000_v3\\graphs\\legacy_00010.json" --step "W:\\python\\FEA\\dataset_generator\\output_generation_p1_b_corrected_balanced_5000_v3\\steps\\legacy_00010.step" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_b_synthetic_dev.json"',
        },
        {
            "id": "p1_b_real_dev",
            "command": base + ' --pipeline p1_b --mode real_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\reports\\p1_b_real_dev_v1\\graphs_v2\\test_part_s2\\part1.json" --step "W:\\python\\FEA\\dataset_generator\\test_part_s2\\part1.stp" --reference-objects "W:\\python\\FEA\\dataset_generator\\reports\\p1_b_real_dev_v1\\feature_objects\\test_part_s2\\part1.json" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_b_real_dev.json"',
        },
        {
            "id": "p1_c_synthetic_dev",
            "command": base + ' --pipeline p1_c --mode synthetic_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\output_generation_p1_c_clean_context_7200_strict_concept_r2\\graphs\\p1c_strict_01_through_hole_round_0192_target_only.json" --step "W:\\python\\FEA\\dataset_generator\\output_generation_p1_c_clean_context_7200_strict_concept_r2\\steps\\p1c_strict_01_through_hole_round_0192_target_only.step" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_c_synthetic_dev.json"',
        },
        {
            "id": "p1_c_real_dev",
            "command": base + ' --pipeline p1_c --mode real_dev --device cpu --graph "W:\\python\\FEA\\dataset_generator\\reports\\p1_c_real_dev_v1\\graphs_v2\\test_part_s2\\part1.json" --step "W:\\python\\FEA\\dataset_generator\\test_part_s2\\part1.stp" --reference-objects "W:\\python\\FEA\\dataset_generator\\reports\\p1_c_real_dev_v1\\feature_objects\\test_part_s2\\part1.json" --output "W:\\python\\FEA\\dataset_generator\\reports\\p1_d_lite_v1\\preflight\\reproduction_p1_c_real_dev.json"',
        },
    ]
    return {
        "schema_version": "topocae_p1_d_lite_reproduction_commands_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": python,
        "commands": commands,
        "contracts": {
            "fresh_process_per_command": True,
            "writes_only_new_p1_d_output": True,
            "formal_holdout_commands_absent": True,
        },
    }


def main() -> int:
    inventory = protected_inventory()
    write_json(PREFLIGHT / "protected_inventory.json", inventory)
    write_json(PREFLIGHT / "pipeline_inventory.json", pipeline_inventory())
    write_json(PREFLIGHT / "environment.json", environment())
    write_json(PREFLIGHT / "reproduction_commands.json", reproduction_commands())
    print(PREFLIGHT.resolve())
    return 0 if inventory["all_present_and_matching"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

