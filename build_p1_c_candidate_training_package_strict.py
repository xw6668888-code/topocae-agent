from __future__ import annotations

"""Build candidate tensors only from an approved strict P1-C GNN run."""

import argparse
import sys

import build_p1_c_candidate_training_package_v3 as source
from p1_c.strict_candidate_provenance import file_sha256, preflight_gnn_source, write_json
from utils.p1_c_strict_run_paths import StrictVariantSeedRunRoot, strict_run_dir, training_metadata_path


def _extract_wrapper_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def main() -> int:
    variant, seed = _extract_wrapper_args()
    provenance = preflight_gnn_source(variant, seed)
    implementation = source.implementation
    implementation.RUN_ROOT = StrictVariantSeedRunRoot(variant)
    implementation.TRAINING_METADATA = training_metadata_path(variant)
    code = int(implementation.main())
    if code != 0:
        return code
    run_dir = strict_run_dir(variant, seed)
    manifest = run_dir / "candidate_training_v1" / "manifest.json"
    metadata = run_dir / "candidate_training_v1" / "metadata.json"
    write_json(run_dir / "strict_candidate_package_receipt.json", {
        "schema_version": "topocae_p1_c_strict_candidate_package_receipt_1",
        "variant": variant,
        "seed": seed,
        "candidate_plan_sha256": provenance["plan_sha256"],
        "source_predictions_sha256": provenance["predictions_sha256"],
        "source_config_sha256": provenance["config_sha256"],
        "source_training_package_manifest_sha256": provenance["training_package_manifest_sha256"],
        "candidate_package_manifest_sha256": file_sha256(manifest),
        "candidate_package_metadata_sha256": file_sha256(metadata),
        "automatic_pass": True,
        "contracts": {
            "strict_source_only": True,
            "invalidated_p1_c_data_used": False,
            "real_data_used": False,
            "holdout_used": False,
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

