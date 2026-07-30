from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from p1_b_general_recognition_pipeline_v4 import run_p1_b_general_recognition
from p1_c.oriented_cylinder_polarity_v1 import augment_graph_with_oriented_cylinder_polarity
from p1_c_general_recognition_pipeline_v1 import run_p1_c_general_recognition
import run_p1_b_real_dev_evaluation as base_runtime
from utils.step_solid_validation import read_step_shape

from .common import ROOT, read_json, strip_graph_truth, strip_prediction_truth, validate_graph


@dataclass(frozen=True)
class FrozenPipelineSpec:
    name: str
    pipeline: str
    gnn_run: Path
    candidate_run: Path
    training_package: Path
    same_instance_selection: Path
    candidate_selection: Path
    global_selection: Path


PIPELINES = {
    "p1_b": FrozenPipelineSpec(
        name="P1-B frozen",
        pipeline="p1_b_general_recognition_pipeline_v4",
        gnn_run=ROOT / "reports/p1_b_training_runs_v1/same_instance_seed_20260718",
        candidate_run=ROOT / "reports/p1_b_training_runs_v1/same_instance_seed_20260718/candidate_model_v1",
        training_package=ROOT / "output_generation_p1_b_corrected_balanced_5000_v3/training_p1_b_v2",
        same_instance_selection=ROOT / "reports/p1_b_training_runs_v1/same_instance_seed_20260718/same_instance_threshold_selection.json",
        candidate_selection=ROOT / "reports/p1_b_training_runs_v1/same_instance_seed_20260718/candidate_model_v1/candidate_threshold_selection.json",
        global_selection=ROOT / "reports/p1_b_training_runs_v1/same_instance_seed_20260718/global_decode_v2_selection.json",
    ),
    "p1_c": FrozenPipelineSpec(
        name="P1-C frozen",
        pipeline="p1_c_general_recognition_pipeline_v1",
        gnn_run=ROOT / "reports/p1_c_strict_training_runs_v2/clean_7200_seed_20260718",
        candidate_run=ROOT / "reports/p1_c_stage6_iteration1_runs_v1/clean_7200_seed_20260718/candidate_model_v1",
        training_package=ROOT / "output_generation_p1_c_clean_context_7200_strict_concept_r2/training_packages/clean_7200",
        same_instance_selection=ROOT / "reports/p1_c_strict_training_runs_v2/clean_7200_seed_20260718/same_instance_threshold_selection.json",
        candidate_selection=ROOT / "reports/p1_c_stage6_iteration2_runs_v1/clean_7200_seed_20260718/candidate_thresholds_iteration2.json",
        global_selection=ROOT / "reports/p1_c_stage6_iteration2_runs_v1/clean_7200_seed_20260718/global_candidate_decode_selection.json",
    ),
}


def get_spec(name: str) -> FrozenPipelineSpec:
    try:
        return PIPELINES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen pipeline: {name}") from exc


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_gnn(spec: FrozenPipelineSpec, device: torch.device):
    previous_run = base_runtime.RUN
    previous_package = base_runtime.TRAINING_PACKAGE
    try:
        base_runtime.RUN = spec.gnn_run
        base_runtime.TRAINING_PACKAGE = spec.training_package
        return base_runtime._load_gnn(device)
    finally:
        base_runtime.RUN = previous_run
        base_runtime.TRAINING_PACKAGE = previous_package


def load_candidate_model(spec: FrozenPipelineSpec, device: torch.device) -> CandidateTopologyClassifier:
    checkpoint = torch.load(spec.candidate_run / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CandidateTopologyClassifier(
        CandidateModelDimensions(**config["dimensions"]),
        macro_classes=config["macro_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_faces(
    spec: FrozenPipelineSpec,
    graph: Mapping[str, Any],
    gnn,
    gnn_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    previous_run = base_runtime.RUN
    previous_package = base_runtime.TRAINING_PACKAGE
    try:
        base_runtime.RUN = spec.gnn_run
        base_runtime.TRAINING_PACKAGE = spec.training_package
        metadata = read_json(spec.training_package / "metadata.json")
        return base_runtime._predict_gnn(dict(graph), gnn, dict(gnn_config), metadata, device)
    finally:
        base_runtime.RUN = previous_run
        base_runtime.TRAINING_PACKAGE = previous_package


def prepare_graph(spec: FrozenPipelineSpec, graph: Mapping[str, Any], step_path: Path | None = None) -> dict[str, Any]:
    prepared = dict(graph)
    protocol = prepared.get("graph_v2_protocol", {})
    if spec.pipeline.startswith("p1_c_") and protocol.get("oriented_cylinder_polarity_revision") != "p1_c_oriented_cylinder_polarity_v1":
        if step_path is None:
            raise ValueError("P1-C graph requires a STEP path to add generic cylinder polarity")
        shape = read_step_shape(step_path)
        prepared = augment_graph_with_oriented_cylinder_polarity(shape, prepared)
    prepared = strip_graph_truth(prepared)
    validate_graph(prepared)
    return prepared


def run_frozen_pipeline(
    spec: FrozenPipelineSpec,
    graph: Mapping[str, Any],
    face_prediction: Mapping[str, Any],
    candidate_model: CandidateTopologyClassifier,
    device: torch.device,
) -> dict[str, Any]:
    graph_clean = strip_graph_truth(graph)
    prediction_clean = strip_prediction_truth(face_prediction)
    same = read_json(spec.same_instance_selection)
    candidate = read_json(spec.candidate_selection)
    global_selection = read_json(spec.global_selection)
    if spec.pipeline == "p1_b_general_recognition_pipeline_v4":
        return run_p1_b_general_recognition(
            graph_clean,
            prediction_clean,
            candidate_model,
            same_instance_thresholds=same,
            candidate_thresholds=candidate,
            global_decode_selection=global_selection,
            device=str(device),
        )
    return run_p1_c_general_recognition(
        graph_clean,
        prediction_clean,
        candidate_model,
        same_instance_thresholds=same,
        candidate_thresholds=candidate,
        global_candidate_decode_selection=global_selection,
        device=str(device),
    )

