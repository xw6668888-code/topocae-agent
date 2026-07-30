from __future__ import annotations

"""Single fail-closed entry for reproducing or verifying the P1-C Goal."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def _run(*arguments: str) -> None:
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def status() -> dict[str, object]:
    holdout_manifest = ROOT / "output_generation_p1_c_synthetic_holdout_v3" / "holdout_generation_manifest.json"
    holdout = json.loads(holdout_manifest.read_text(encoding="utf-8")) if holdout_manifest.is_file() else {}
    return {
        "h_c1_r2": (ROOT / "reports/p1_c_iterations_v1/h_c1_strict_concept_r2_authorization_receipt.json").is_file(),
        "h_c2_r3": (ROOT / "reports/p1_c_iterations_v1/h_c2_strict_concept_r3_authorization_receipt.json").is_file(),
        "dataset_7200": (ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2/dataset_meta.json").is_file(),
        "stage6_pass": (ROOT / "reports/p1_c_stage6_summary_v1/summary.json").is_file(),
        "preholdout_frozen": (ROOT / "reports/p1_c_freeze_v1/preholdout_selection.json").is_file(),
        "real_dev_complete": (ROOT / "reports/p1_c_real_dev_v1/metrics.json").is_file(),
        "holdout_v3_generated": bool(holdout),
        "holdout_v3_inference_pass_count": int(holdout.get("inference_pass_count", 0)),
        "holdout_v3_report_complete": (ROOT / "reports/p1_c_holdout_v3_once/metrics.json").is_file(),
        "p2_entered": False,
    }


def reproduce_data() -> None:
    output = ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2"
    if output.exists():
        raise RuntimeError("formal P1-C dataset already exists and is immutable; refusing overwrite")
    for command in ("prepare", "generate", "finalize", "validate"):
        _run("generate_p1_c_clean_context_7200_strict_concept.py", command)


def reproduce_training() -> None:
    if (ROOT / "reports/p1_c_freeze_v1/preholdout_selection.json").exists():
        raise RuntimeError("preholdout selection is already frozen; refusing to overwrite training artifacts")
    _run("build_p1_c_training_package_strict.py")
    variants = {
        "clean_5000": (20260718, 20260719, 20260720),
        "clean_7200": (20260718, 20260719, 20260720),
        "single_only": (20260718,),
        "no_instance_balanced_loss": (20260718,),
        "no_canonical_patch": (20260718,),
    }
    for variant, seeds in variants.items():
        for seed in seeds:
            _run("train_p1_c_gnn_strict_v2.py", "--variant", variant, "--seed", str(seed))
    _run("run_p1_c_candidate_matrix_strict_v3.py", "--stage", "all")
    _run("run_p1_c_stage6_iteration1_matrix.py", "--stage", "all")
    _run("run_p1_c_stage6_iteration2_matrix_v2.py", "--stage", "select")
    _run("run_p1_c_stage6_iteration2_matrix_v2.py", "--stage", "evaluate")
    _run("finalize_p1_c_stage6_summary_v3.py")


def holdout_once() -> None:
    manifest_path = ROOT / "output_generation_p1_c_synthetic_holdout_v3/holdout_generation_manifest.json"
    if manifest_path.is_file() and int(json.loads(manifest_path.read_text(encoding="utf-8")).get("inference_pass_count", 0)) != 0:
        raise RuntimeError("Holdout-v3 formal inference already completed; second run forbidden")
    _run("run_p1_c_synthetic_holdout_v3_once.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "verify", "reproduce-data", "reproduce-training", "holdout-once"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "verify":
        _run("verify_p1_c_final_acceptance_v1.py")
    elif args.command == "reproduce-data":
        reproduce_data()
    elif args.command == "reproduce-training":
        reproduce_training()
    else:
        holdout_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
