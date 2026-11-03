"""G09 STEP recognition entry: one fixed sparse-R system and role/parameter output."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from candidate_pool import capture_legacy_stages
from g09_common import ARTIFACT, FROZEN, ROOT, checked, descriptor, now, read, write_json


MODEL = FROZEN / "model_packages/p1_e12_reviewed_legacy_v1"
CANDIDATE = (
    FROZEN / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718"
    / "candidate_model/candidate_threshold_selection.json"
)
GLOBAL = CANDIDATE.parent.parent / "global_decode_selection.json"
EXPECTED_CHECKPOINT_SHA256 = "491655bfec5591c5a6a1b7f6300294f4818c29bf60b4c1b8dd5ff6078a057054"


def _legacy_inference_module():
    source = FROZEN / "scripts/run_inference.py"
    spec = importlib.util.spec_from_file_location("g09_frozen_sparse_r_inference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen sparse R entry could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _role_module():
    source = ROOT / "tools/aei_paper_goals_v1/g07"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import engineering_reference
    return engineering_reference


def _sources() -> dict:
    paths = {
        "checkpoint": MODEL / "model.pt",
        "normalization": MODEL / "metadata.json",
        "same_instance": MODEL / "same_instance_threshold.json",
        "candidate_config": CANDIDATE,
        "global_config": GLOBAL,
        "frozen_inference": FROZEN / "scripts/run_inference.py",
        "frozen_graph_extractor": FROZEN / "core/graph_v2_2_1.py",
        "frozen_candidate_proposer": FROZEN / "utils/p1_b_candidate_proposer.py",
        "role_adapter": ROOT / "tools/aei_paper_goals_v1/g07/engineering_reference.py",
        "role_helper": ROOT / "tools/aei_case_suite/geometry.py",
    }
    result = {key: descriptor(path) for key, path in paths.items()}
    if result["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen old R checkpoint identity changed")
    return result


def _role_and_parameter_objects(step: Path, graph: dict, prediction: dict,
                                feature_objects: list[dict]) -> tuple[list[dict], dict]:
    adapter = _role_module()
    faces = adapter.measured_faces(step)
    if len(faces) != len(graph["nodes"]) or len(faces) != len(prediction["faces"]):
        raise ValueError("STEP geometry, graph and model face indices do not align")
    for index, face in enumerate(faces):
        face["semantic_label"] = int(prediction["faces"][index]["semantic_label"])
        face["predicted_class"] = prediction["faces"][index]["macro_feature"]
    objects = adapter.adapt_objects(feature_objects, graph)
    data = {
        "part": step.name,
        "source_sha256": descriptor(step)["sha256"],
        "faces": faces,
        "objects": objects,
        "_graph": graph,
    }
    output = []
    role_classes = {
        "top_face": {"cylinder_boss", "rectangular_boss"},
        "inner_walls": {"through_hole_round", "obround_hole"},
        "cavity_interior": {"pocket_circular"},
    }
    for obj in objects:
        roles = {"whole_feature": adapter.H["roles"](data, [obj], "whole_feature")[0]}
        unavailable = {}
        for role, classes in role_classes.items():
            if obj["class"] not in classes:
                continue
            try:
                roles[role] = adapter.H["roles"](data, [obj], role)[0]
            except adapter.PlanError as error:
                unavailable[role] = error.code
        output.append({
            "object_index": obj["object_index"],
            "class": obj["class"],
            "graph_node_indices": [face - 1 for face in obj["original_face_numbers"]],
            "original_face_numbers": obj["original_face_numbers"],
            "roles": roles,
            "unavailable_roles": unavailable,
            "parameters": adapter.parameter_values(data, obj),
        })
    spacing = {}
    counterbores = [obj for obj in objects if obj["class"] == "counterbore_hole"]
    if len(counterbores) == 3:
        try:
            spacing = adapter.spacing_values(data, counterbores)
        except adapter.PlanError as error:
            spacing = {"unavailable_reason": error.code}
    return output, spacing


class SparseRService:
    """A fixed model instance reused across arbitrary STEP files."""

    def __init__(self, device: str = "cpu"):
        import torch
        self.sources = _sources()
        self.legacy = _legacy_inference_module()
        self.device = torch.device(device)
        self.model = self.legacy.build_model().to(self.device)
        checkpoint = torch.load(checked(self.sources["checkpoint"]), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.metadata = read(checked(self.sources["normalization"]))
        self.same = read(checked(self.sources["same_instance"]))
        self.candidate = read(checked(self.sources["candidate_config"]))
        self.global_decode = read(checked(self.sources["global_config"]))

    def infer(self, step: str | Path) -> dict:
        step = Path(step).resolve()
        if not step.is_file() or step.suffix.lower() not in (".stp", ".step"):
            raise ValueError("Input must be an existing STEP file")
        started = time.perf_counter()
        shape = self.legacy.read_step_shape(step)
        graph = self.legacy.guard_degenerate_projection(
            self.legacy.extract_graph_v2_2_1(shape, build_patches=True)
        )
        prediction = self.legacy.predict(graph, self.model, self.metadata, self.device)
        stages = capture_legacy_stages(
            graph, prediction, self.same, self.candidate, self.global_decode
        )
        feature_objects = stages["final"]
        output, spacing = _role_and_parameter_objects(step, graph, prediction, feature_objects)
        elapsed = time.perf_counter() - started
        return {
            "schema": "g09_sparse_r_step_service_v1",
            "at_utc": now(),
            "input_step": descriptor(step),
            "system": "frozen_sparse_R_seed_20260719_eight_stage_recovery_and_role_completion",
            "device": str(self.device),
            "source_identity": self.sources,
            "feature_objects": output,
            "spacing_if_three_counterbores": spacing,
            "evidence_trace": {
                "candidate_stage_counts": {stage: len(objects) for stage, objects in stages.items()},
                "stage_objects": stages,
                "prediction": prediction,
                "face_index_base": 0,
                "step_face_number_base": 1,
            },
            "inference_seconds": elapsed,
            "truth_used": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="+", type=Path, help="One or more arbitrary STEP files")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT / "inference")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(ARTIFACT.resolve()):
        parser.error("G09 inference evidence must be written under its G09 artifact directory")
    service = SparseRService(args.device)
    for step in args.steps:
        result = service.infer(step)
        output_path = output_dir / f"{step.stem}_{result['input_step']['sha256'][:12]}.json"
        write_json(output_path, result)
        print(json.dumps({
            "input": str(step.resolve()), "output": str(output_path),
            "feature_objects": len(result["feature_objects"]),
            "seconds": result["inference_seconds"], "truth_used": False,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
