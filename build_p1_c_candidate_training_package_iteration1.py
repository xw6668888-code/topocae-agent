from __future__ import annotations

"""Build polarity-augmented candidate packages for the six formal clean runs."""

import argparse
import sys
from pathlib import Path

import build_p1_c_candidate_training_package_v3 as source
from p1_c.iteration1_provenance import file_sha256, preflight_iteration_source, write_json
from utils.p1_c_candidate_features_polarity_v1 import ENGINEERED_FEATURE_SCHEMA, build_candidate_tensor_record
from utils.p1_c_iteration1_run_paths import IterationVariantSeedRunRoot, iteration_run_dir
from utils.p1_c_strict_run_paths import DATASET_ROOT, training_metadata_path


POLARITY_GRAPHS = DATASET_ROOT / "graphs_polarity_v1"


def _extract_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--seed", str(known.seed), *remaining]
    return str(known.variant), int(known.seed)


def _mapped_graph_path(path: Path) -> Path:
    return POLARITY_GRAPHS / path.name if path.parent.name == "graphs" else path


def main() -> int:
    variant, seed = _extract_args()
    provenance = preflight_iteration_source(variant, seed)
    implementation = source.implementation
    implementation.RUN_ROOT = IterationVariantSeedRunRoot(variant)
    implementation.TRAINING_METADATA = training_metadata_path(variant)
    implementation.ENGINEERED_FEATURE_SCHEMA = ENGINEERED_FEATURE_SCHEMA
    implementation.build_candidate_tensor_record = build_candidate_tensor_record
    original_read = implementation._read_json
    original_build = implementation._build_record

    def mapped_read(path: Path):
        return original_read(_mapped_graph_path(Path(path)))

    def mapped_build(graph, prediction, proposal, truths, source_record, **kwargs):
        revised = dict(source_record)
        revised["source_graph_path"] = str(_mapped_graph_path(Path(source_record["source_graph_path"])).resolve())
        return original_build(graph, prediction, proposal, truths, revised, **kwargs)

    implementation._read_json = mapped_read
    implementation._build_record = mapped_build
    code = int(implementation.main())
    if code != 0:
        return code
    run = iteration_run_dir(variant, seed)
    metadata = run / "candidate_training_v1" / "metadata.json"
    manifest = run / "candidate_training_v1" / "manifest.json"
    write_json(run / "iteration1_candidate_package_receipt.json", {
        "schema_version": "topocae_p1_c_stage6_iteration1_candidate_package_receipt_1",
        "variant": variant,
        "seed": seed,
        "iteration_plan_sha256": provenance["iteration_plan_sha256"],
        "graph_polarity_manifest_sha256": provenance["graph_polarity_manifest_sha256"],
        "source_gnn_predictions_sha256": provenance["predictions_sha256"],
        "candidate_package_metadata_sha256": file_sha256(metadata),
        "candidate_package_manifest_sha256": file_sha256(manifest),
        "engineered_feature_dim": 71,
        "contracts": {"real_data_used": False, "holdout_used": False, "source_gnn_retrained": False}
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

