from __future__ import annotations

"""Strict candidate-package builder with context-applicable hard-negative gates."""

import argparse
import json
import sys

import build_p1_c_candidate_training_package_v3 as source
from p1_c.strict_candidate_provenance import file_sha256, write_json
from p1_c.strict_candidate_provenance_v2 import preflight_gnn_source
from utils.p1_c_strict_run_paths import StrictVariantSeedRunRoot, strict_run_dir, training_metadata_path


GATE_PATH = training_metadata_path("clean_7200").parents[3] / "configs" / "p1_c" / "candidate_package_gate_strict_v1.json"
EXPECTED_GATE_SHA256 = "e16cf405dc22a4ff1e16ca47072bcf1a2e6332e4fa7b46761278c1f85b0dc3ac"
REQUIRED_BASE_NEGATIVES = {
    "missing_truth_face_subset",
    "adjacent_stock_or_transition_superset",
    "wrong_macro_hard_confusion",
}


def _extract_wrapper_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def _strict_gate(variant: str, metadata: dict) -> tuple[bool, dict]:
    train_negatives = set(metadata["splits"]["train"]["hard_negative_kind_counts"])
    val = metadata["splits"]["val"]
    common = (
        int(metadata["dimensions"]["face_embedding_dim"]) == 128
        and int(metadata["dimensions"]["edge_embedding_dim"]) <= 20
        and float(val["proposal_oracle_recall_iou_0_5"]) >= 0.95
        and bool(val["all_seven_sources_present"])
        and REQUIRED_BASE_NEGATIVES <= train_negatives
    )
    if variant == "single_only":
        negative_pass = len(train_negatives) >= 3
        applicability = "single_feature_only_three_applicable_kinds"
    else:
        negative_pass = len(train_negatives) >= 4 and bool(train_negatives - REQUIRED_BASE_NEGATIVES)
        applicability = "multi_context_instance_merge_required"
    checks = {
        "common_embedding_oracle_source_gate": common,
        "hard_negative_gate": negative_pass,
        "hard_negative_kinds": sorted(train_negatives),
        "hard_negative_applicability": applicability,
    }
    return bool(common and negative_pass), checks


def main() -> int:
    variant, seed = _extract_wrapper_args()
    if file_sha256(GATE_PATH) != EXPECTED_GATE_SHA256:
        raise RuntimeError("strict candidate-package gate drift")
    provenance = preflight_gnn_source(variant, seed)
    implementation = source.implementation
    implementation.RUN_ROOT = StrictVariantSeedRunRoot(variant)
    implementation.TRAINING_METADATA = training_metadata_path(variant)
    legacy_code = int(implementation.main())
    run_dir = strict_run_dir(variant, seed)
    package_dir = run_dir / "candidate_training_v1"
    metadata_path = package_dir / "metadata.json"
    manifest_path = package_dir / "manifest.json"
    if not metadata_path.exists() or not manifest_path.exists():
        return legacy_code if legacy_code else 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    passed, checks = _strict_gate(variant, metadata)
    write_json(run_dir / "strict_candidate_package_receipt.json", {
        "schema_version": "topocae_p1_c_strict_candidate_package_receipt_2",
        "variant": variant,
        "seed": seed,
        "candidate_plan_sha256": provenance["plan_sha256"],
        "candidate_package_gate_sha256": EXPECTED_GATE_SHA256,
        "source_predictions_sha256": provenance["predictions_sha256"],
        "source_config_sha256": provenance["config_sha256"],
        "source_training_package_manifest_sha256": provenance["training_package_manifest_sha256"],
        "candidate_package_manifest_sha256": file_sha256(manifest_path),
        "candidate_package_metadata_sha256": file_sha256(metadata_path),
        "legacy_builder_returncode": legacy_code,
        "checks": checks,
        "automatic_pass": passed,
        "contracts": {
            "strict_source_only": True,
            "hard_negative_applicability_is_context_structural": True,
            "invalidated_p1_c_data_used": False,
            "real_data_used": False,
            "holdout_used": False
        }
    })
    print(json.dumps({"strict_automatic_pass": passed, "variant": variant, "seed": seed, "checks": checks}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

