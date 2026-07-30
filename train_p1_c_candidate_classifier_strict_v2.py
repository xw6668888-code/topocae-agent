from __future__ import annotations

"""Strict candidate trainer with a true end-to-end no-balanced-loss branch."""

import argparse
import sys

from p1_c.strict_candidate_provenance import file_sha256, write_json
from p1_c.strict_candidate_provenance_v2 import preflight_candidate_package
from utils.p1_c_strict_run_paths import StrictVariantSeedRunRoot, VARIANT_SPECS, candidate_model_name, strict_run_dir


def _extract_wrapper_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def main() -> int:
    variant, seed = _extract_wrapper_args()
    provenance = preflight_candidate_package(variant, seed)
    spec = VARIANT_SPECS[variant]
    if not bool(spec["canonical_patch"]):
        sys.argv.append("--no-canonical-patch")
    if bool(spec["candidate_instance_balanced_loss"]):
        import train_p1_c_candidate_classifier_v2 as entrypoint_module

        protocol = entrypoint_module.protocol
        entrypoint = entrypoint_module.main
        implementation_revision = "p1_c_sample_instance_balanced_candidate_loss_v2"
    else:
        import train_p1_b_candidate_classifier as entrypoint_module

        protocol = entrypoint_module
        entrypoint = entrypoint_module.main
        implementation_revision = "p1_b_unbalanced_candidate_loss_reference"
    protocol.RUN_ROOT = StrictVariantSeedRunRoot(variant)
    code = int(entrypoint())
    if code != 0:
        return code
    run_dir = strict_run_dir(variant, seed)
    model_dir = run_dir / candidate_model_name(variant)
    write_json(run_dir / "strict_candidate_model_receipt.json", {
        "schema_version": "topocae_p1_c_strict_candidate_model_receipt_2",
        "variant": variant,
        "seed": seed,
        "candidate_plan_sha256": provenance["plan_sha256"],
        "candidate_package_manifest_sha256": provenance["candidate_package_manifest_sha256"],
        "candidate_model_config_sha256": file_sha256(model_dir / "config.json"),
        "candidate_model_checkpoint_sha256": file_sha256(model_dir / "best.pt"),
        "candidate_model_metrics_sha256": file_sha256(model_dir / "metrics.json"),
        "candidate_model_predictions_sha256": file_sha256(model_dir / "predictions.pt"),
        "candidate_instance_balanced_loss": bool(spec["candidate_instance_balanced_loss"]),
        "canonical_patch": bool(spec["canonical_patch"]),
        "implementation_revision": implementation_revision,
        "contracts": {
            "synthetic_val_selection_only": True,
            "no_instance_balanced_loss_branch_imported_balanced_protocol": False if not bool(spec["candidate_instance_balanced_loss"]) else None,
            "real_data_used": False,
            "holdout_used": False,
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

