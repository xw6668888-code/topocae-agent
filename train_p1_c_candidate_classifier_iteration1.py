from __future__ import annotations

"""Train iteration-1 candidate models with unchanged balanced protocol/defaults."""

import argparse
import sys

import train_p1_c_candidate_classifier_v2 as entrypoint
from p1_c.iteration1_provenance import file_sha256, preflight_iteration_package, write_json
from utils.p1_c_iteration1_run_paths import IterationVariantSeedRunRoot, iteration_run_dir


def _extract_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def main() -> int:
    variant, seed = _extract_args()
    provenance = preflight_iteration_package(variant, seed)
    entrypoint.protocol.RUN_ROOT = IterationVariantSeedRunRoot(variant)
    code = int(entrypoint.main())
    if code != 0:
        return code
    run = iteration_run_dir(variant, seed)
    model = run / "candidate_model_v1"
    write_json(run / "iteration1_candidate_model_receipt.json", {
        "schema_version": "topocae_p1_c_stage6_iteration1_candidate_model_receipt_1",
        "variant": variant,
        "seed": seed,
        "iteration_plan_sha256": provenance["iteration_plan_sha256"],
        "candidate_package_manifest_sha256": provenance["candidate_package_manifest_sha256"],
        "candidate_model_config_sha256": file_sha256(model / "config.json"),
        "candidate_model_checkpoint_sha256": file_sha256(model / "best.pt"),
        "candidate_model_metrics_sha256": file_sha256(model / "metrics.json"),
        "candidate_model_predictions_sha256": file_sha256(model / "predictions.pt"),
        "contracts": {
            "candidate_instance_balanced_loss": True,
            "candidate_defaults_unchanged": True,
            "synthetic_val_selection_only": True,
            "real_data_used": False,
            "holdout_used": False
        }
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

