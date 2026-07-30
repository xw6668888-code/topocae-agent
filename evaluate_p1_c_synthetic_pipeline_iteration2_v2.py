from __future__ import annotations

"""Corrected final Stage 6 evaluation with explicit selected-threshold routing."""

import json
from pathlib import Path

import evaluate_p1_c_synthetic_pipeline_strict as protocol
from p1_c.iteration1_provenance import preflight_iteration_model
from utils.p1_c_iteration2_run_paths import Iteration2RunDir, Iteration2VariantSeedRunRoot, iteration2_run_dir
from utils.p1_c_strict_run_paths import DATASET_ROOT
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction


POLARITY_GRAPHS = DATASET_ROOT / "graphs_polarity_v1"
_original_read = protocol.implementation._read_json
_selected_threshold_path: Path | None = None


def _mapped_read(path: Path):
    source = Path(path)
    if source.parent.name == "graphs":
        source = POLARITY_GRAPHS / source.name
    if source.name == "candidate_threshold_selection.json" and source.parent.name == "candidate_model_v1":
        if _selected_threshold_path is None:
            raise RuntimeError("iteration-2 selected threshold path is not initialized")
        source = _selected_threshold_path
    return _original_read(source)


def _verify(graph_data, proposal, model_prediction, *, thresholds):
    augmented = dict(model_prediction)
    augmented["face_macro_predictions"] = protocol._face_macros_by_sample.get(str(graph_data.get("sample_id", "")), [])
    return verify_candidate_prediction(graph_data, proposal, augmented, thresholds=thresholds)


def main() -> int:
    global _selected_threshold_path
    variant, seed = protocol._extract_wrapper_args()
    preflight_iteration_model(variant, seed)
    output = iteration2_run_dir(variant, seed)
    _selected_threshold_path = output / "candidate_thresholds_iteration2.json"
    selection = json.loads((output / "global_candidate_decode_selection.json").read_text(encoding="utf-8"))
    protocol._selection = {
        "global_subset_confidence_tolerance": selection["selected"]["subset_confidence_tolerance"],
        "selection_sha256": selection["selection_sha256"],
        "selection_split": "synthetic_val"
    }
    protocol.implementation.RUN_ROOT = Iteration2VariantSeedRunRoot(variant)
    protocol.implementation.TRAINING_METADATA = protocol.training_metadata_path(variant)
    protocol.implementation._read_json = _mapped_read
    protocol.implementation._truth_objects = protocol._truth_objects
    protocol.implementation._boundary_faces_from_objects = protocol._boundary_faces_from_objects
    protocol.implementation._gnn_prediction_data = protocol._gnn_prediction_data
    protocol.implementation._object_metrics = protocol._object_metrics
    protocol.implementation.verify_candidate_prediction = _verify
    protocol.implementation.decode_feature_objects = protocol._decode
    return int(protocol.implementation.main())


if __name__ == "__main__":
    raise SystemExit(main())

