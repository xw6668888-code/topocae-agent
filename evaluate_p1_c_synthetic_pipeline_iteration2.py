from __future__ import annotations

"""Evaluate the final Stage 6 calibration with unchanged frozen model weights."""

import json
from pathlib import Path

import evaluate_p1_c_synthetic_pipeline_strict as protocol
from p1_c.iteration1_provenance import preflight_iteration_model
from utils.p1_c_iteration2_run_paths import Iteration2RunDir, Iteration2VariantSeedRunRoot
from utils.p1_c_strict_run_paths import DATASET_ROOT
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction


POLARITY_GRAPHS = DATASET_ROOT / "graphs_polarity_v1"
_original_read = protocol.implementation._read_json


def _mapped_read(path: Path):
    source = Path(path)
    if source.parent.name == "graphs":
        source = POLARITY_GRAPHS / source.name
    if source.name == "candidate_threshold_selection.json" and source.parent.name == "candidate_model_v1":
        run = source.parents[2]
        source = run / "candidate_thresholds_iteration2.json"
    return _original_read(source)


def _verify(graph_data, proposal, model_prediction, *, thresholds):
    augmented = dict(model_prediction)
    augmented["face_macro_predictions"] = protocol._face_macros_by_sample.get(str(graph_data.get("sample_id", "")), [])
    return verify_candidate_prediction(graph_data, proposal, augmented, thresholds=thresholds)


def _main() -> int:
    protocol.preflight_candidate_model = preflight_iteration_model
    protocol.strict_run_dir = lambda variant, seed: Iteration2RunDir(variant, seed)
    protocol.StrictVariantSeedRunRoot = Iteration2VariantSeedRunRoot
    protocol.implementation._read_json = _mapped_read
    protocol.verify_v4 = verify_candidate_prediction
    protocol._verify = _verify
    variant, seed = protocol._extract_wrapper_args()
    preflight_iteration_model(variant, seed)
    run = Iteration2RunDir(variant, seed)
    selection = json.loads((run / "global_candidate_decode_selection.json").read_text(encoding="utf-8"))
    protocol._selection = {
        "global_subset_confidence_tolerance": selection["selected"]["subset_confidence_tolerance"],
        "selection_sha256": selection["selection_sha256"],
        "selection_split": "synthetic_val"
    }
    protocol.implementation.RUN_ROOT = Iteration2VariantSeedRunRoot(variant)
    protocol.implementation.TRAINING_METADATA = protocol.training_metadata_path(variant)
    protocol.implementation._truth_objects = protocol._truth_objects
    protocol.implementation._boundary_faces_from_objects = protocol._boundary_faces_from_objects
    protocol.implementation._gnn_prediction_data = protocol._gnn_prediction_data
    protocol.implementation._object_metrics = protocol._object_metrics
    protocol.implementation.verify_candidate_prediction = _verify
    protocol.implementation.decode_feature_objects = protocol._decode
    return int(protocol.implementation.main())


if __name__ == "__main__":
    raise SystemExit(_main())

