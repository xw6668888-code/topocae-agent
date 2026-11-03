from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_2_1 import extract_graph_v2_2_1
from p1_e6r import evaluate as legacy_evaluate
from p1_e6r.common import MACRO_CLASSES
from p1_e6r2_r3.instance_boundary import constrain_feature_objects_to_instance_components
from p1_e6r2_r3.training_common import build_model
from utils.step_solid_validation import read_step_shape
from utils.training_export_v2_2_1 import export_graph_v2_2_1_training_record


MODEL_DIR = ROOT / "model_packages" / "p1_e12_reviewed_legacy_v1"
CANDIDATE_PATH = ROOT / "reports" / "p1_e3c_seed_matched_candidate_v1" / "pipelines" / "old_same_instance_seed_20260718" / "candidate_model" / "candidate_threshold_selection.json"
GLOBAL_PATH = CANDIDATE_PATH.parents[1] / "global_decode_selection.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def guard_degenerate_projection(graph: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    result = copy.deepcopy(graph)
    changed = []
    for index, node in enumerate(result.get("nodes", [])):
        attributes = node.get("attributes", {})
        descriptors = attributes.get("shape_descriptors", {})
        long_extent = float(descriptors.get("projected_long_extent", 0.0))
        short_extent = float(descriptors.get("projected_short_extent", 0.0))
        if long_extent <= 0.0 or short_extent > 1.0e-10 * long_extent:
            continue
        recognition = attributes.get("recognition_features_v2", {})
        changed.append({
            "graph_node_index": index,
            "previous": float(recognition.get("log1p_projected_area_fill_ratio", 0.0)),
        })
        descriptors["projected_area_fill_ratio"] = 0.0
        recognition["projected_area_fill_ratio"] = 0.0
        recognition["log1p_projected_area_fill_ratio"] = 0.0
    result["projection_numeric_guard"] = changed
    return result


def graph_tensors(graph: dict[str, Any], metadata: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    exported = export_graph_v2_2_1_training_record(graph)
    transform = metadata["quantile_transform"]
    normalization = metadata["normalization"]
    node = np.asarray(exported["node_features"], dtype=np.float32)
    edge = np.asarray(exported["edge_features"], dtype=np.float32)
    node = np.clip(node, np.asarray(transform["node_lower"], dtype=np.float32), np.asarray(transform["node_upper"], dtype=np.float32))
    edge = np.clip(edge, np.asarray(transform["edge_lower"], dtype=np.float32), np.asarray(transform["edge_upper"], dtype=np.float32))
    node = (node - np.asarray(normalization["node_mean"], dtype=np.float32)) / np.asarray(normalization["node_std"], dtype=np.float32)
    edge = (edge - np.asarray(normalization["edge_mean"], dtype=np.float32)) / np.asarray(normalization["edge_std"], dtype=np.float32)
    edge_index = torch.tensor(exported["edges"], dtype=torch.long).t().contiguous() if exported["edges"] else torch.empty((2, 0), dtype=torch.long)
    primitive = torch.tensor([int(item.get("primitive_surface_id", 0)) for item in graph.get("nodes", [])], dtype=torch.long)
    return torch.from_numpy(node), edge_index, primitive, torch.from_numpy(edge)


def predict(graph: dict[str, Any], model: torch.nn.Module, metadata: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    tensors = [value.to(device) for value in graph_tensors(graph, metadata)]
    with torch.inference_mode():
        output = model(tensors[0], tensors[1], tensors[2], edge_features=tensors[3])
    macro = torch.softmax(output["macro_logits"], dim=1).cpu()
    semantic = torch.softmax(output["semantic_logits"], dim=1).cpu()
    faces = []
    for index in range(macro.shape[0]):
        macro_id = int(macro[index].argmax())
        semantic_id = int(semantic[index].argmax())
        faces.append({
            "idx": index,
            "macro_feature": MACRO_CLASSES[macro_id],
            "macro_class_id": macro_id,
            "macro_confidence": float(macro[index, macro_id]),
            "macro_probabilities": {name: float(macro[index, class_id]) for class_id, name in enumerate(MACRO_CLASSES)},
            "semantic_label": semantic_id,
            "semantic_confidence": float(semantic[index, semantic_id]),
            "embedding": [],
        })
    return {"faces": faces, "edge_probabilities": torch.sigmoid(output["edge_same_instance_logits"]).cpu().tolist()}


def decode(graph: dict[str, Any], prediction: Mapping[str, Any], same: Mapping[str, Any]) -> list[dict[str, Any]]:
    face_prediction = {
        "faces": list(prediction["faces"]),
        "node_embeddings": [],
        "edge_same_instance_probabilities": list(prediction["edge_probabilities"]),
        "contracts": {"truth_used": False, "latent_candidate_classifier_enabled": False},
    }
    objects, _, _, _, _ = legacy_evaluate.explicit_topology_objects(
        graph, face_prediction, same, read_json(CANDIDATE_PATH), read_json(GLOBAL_PATH), "manual_step"
    )
    constrained = constrain_feature_objects_to_instance_components(
        graph, objects, prediction["edge_probabilities"], float(same["global_threshold"])
    )
    return constrained["feature_objects"]


def rotate_shape(shape: Any) -> tuple[Any, dict[int, int]]:
    original_faces = list(TopologyExplorer(shape).faces())
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 2, 3)), 0.473)
    builder = BRepBuilderAPI_Transform(shape, transform, True)
    builder.Build()
    rotated = builder.Shape()
    rotated_faces = list(TopologyExplorer(rotated).faces())
    mapping: dict[int, int] = {}
    used = set()
    for source_index, source_face in enumerate(original_faces):
        modified = builder.ModifiedShape(source_face)
        matches = [index for index, face in enumerate(rotated_faces) if face.IsSame(modified)]
        if len(matches) != 1 or matches[0] in used:
            raise RuntimeError("rotation face correspondence is not bijective")
        mapping[source_index] = matches[0]
        used.add(matches[0])
    return rotated, mapping


def canonical_objects(objects: list[Mapping[str, Any]], inverse: Mapping[int, int] | None = None) -> list[tuple[str, tuple[int, ...]]]:
    inverse = inverse or {}
    return sorted((str(item.get("macro_feature", "")), tuple(sorted(inverse.get(int(face), int(face)) for face in item.get("face_indices", [])))) for item in objects)


def edge_decisions(graph: Mapping[str, Any], prediction: Mapping[str, Any], threshold: float, mapping: Mapping[int, int]) -> dict[tuple[int, int], list[bool]]:
    groups: defaultdict[tuple[int, int], list[bool]] = defaultdict(list)
    for edge, probability in zip(graph.get("edges", []), prediction["edge_probabilities"]):
        groups[(mapping[int(edge[0])], mapping[int(edge[1])])].append(float(probability) >= threshold)
    return {key: sorted(values) for key, values in groups.items()}


def geometry_summary(graph: Mapping[str, Any], faces: list[int]) -> dict[str, Any]:
    nodes = [graph["nodes"][index] for index in faces]
    areas = [float(node["attributes"]["physical"]["area"]) for node in nodes]
    centers = [node["attributes"]["physical"]["center_of_mass"] for node in nodes]
    total = sum(areas)
    return {
        "member_face_count": len(faces),
        "total_area": total,
        "surface_family_counts": {family: sum(node["primitive_surface"] == family for node in nodes) for family in sorted({node["primitive_surface"] for node in nodes})},
        "area_weighted_center": [sum(area * center[axis] for area, center in zip(areas, centers)) / total for axis in range(3)] if total else [0.0, 0.0, 0.0],
        "positive_radii": sorted({float(node["attributes"]["specific"].get("radius", 0.0)) for node in nodes if float(node["attributes"]["specific"].get("radius", 0.0)) > 0.0}),
    }


def run(input_path: Path, output_dir: Path, compare_rotation: bool, device_name: str | None) -> Path:
    input_path = input_path.resolve()
    if input_path.suffix.lower() not in {".step", ".stp"} or not input_path.is_file():
        raise ValueError(f"not a STEP file: {input_path}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = MODEL_DIR / "model.pt"
    metadata_path = MODEL_DIR / "metadata.json"
    threshold_path = MODEL_DIR / "same_instance_threshold.json"
    if sha256(checkpoint_path) != "491655BFEC5591C5A6A1B7F6300294F4818C29BF60B4C1B8DD5FF6078A057054":
        raise RuntimeError("best model hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = read_json(metadata_path)
    same = read_json(threshold_path)
    shape = read_step_shape(input_path)
    graph = guard_degenerate_projection(extract_graph_v2_2_1(shape, build_patches=True))
    prediction = predict(graph, model, metadata, device)
    objects = decode(graph, prediction, same)
    output_objects = []
    for object_index, item in enumerate(objects):
        indices = sorted({int(value) for value in item.get("face_indices", [])})
        output_objects.append({
            "object_index": object_index,
            "class": str(item.get("macro_feature", "")),
            "graph_node_indices": indices,
            "original_face_numbers": [index + 1 for index in indices],
            "topology_valid": bool(item.get("topology_valid", False)),
            "boundary_decision": str(item.get("boundary_decision", "retained")),
            "class_confidence": sum(float(prediction["faces"][index]["macro_probabilities"].get(str(item.get("macro_feature", "")), 0.0)) for index in indices) / len(indices) if indices else 0.0,
            "geometry_summary": geometry_summary(graph, indices),
        })
    faces = []
    for index, (node, item) in enumerate(zip(graph["nodes"], prediction["faces"])):
        faces.append({
            "original_face_number": index + 1,
            "graph_node_index": index,
            "class": item["macro_feature"],
            "confidence": item["macro_confidence"],
            "semantic_label": item["semantic_label"],
            "geometry_properties": {
                "surface_family": node["primitive_surface"],
                **node["attributes"]["physical"],
                **node["attributes"]["specific"],
            },
        })
        node["inference_prediction"] = item
    rotation = None
    if compare_rotation:
        rotated_shape, mapping = rotate_shape(shape)
        rotated_graph = guard_degenerate_projection(extract_graph_v2_2_1(rotated_shape, build_patches=True))
        rotated_prediction = predict(rotated_graph, model, metadata, device)
        rotated_objects = decode(rotated_graph, rotated_prediction, same)
        inverse = {target: source for source, target in mapping.items()}
        identity = {index: index for index in range(len(graph["nodes"]))}
        threshold = float(same["global_threshold"])
        rotation = {
            "face_prediction_equal": all(prediction["faces"][source]["macro_class_id"] == rotated_prediction["faces"][target]["macro_class_id"] for source, target in mapping.items()),
            "same_instance_edge_decision_equal": edge_decisions(graph, prediction, threshold, identity) == edge_decisions(rotated_graph, rotated_prediction, threshold, inverse),
            "feature_object_prediction_equal": canonical_objects(objects) == canonical_objects(rotated_objects, inverse),
        }
        rotation["passed"] = all(rotation.values())
    result = {
        "schema_version": "topocae_legacy_baseline_manual_inference_1",
        "input": {"absolute_path": str(input_path), "bytes": input_path.stat().st_size, "sha256": sha256(input_path)},
        "model": {"name": "p1_e6r2_r3_legacy_seed_20260719", "checkpoint_sha256": sha256(checkpoint_path), "device": str(device)},
        "feature_objects": output_objects,
        "faces": faces,
        "rotation_comparison": rotation,
        "truth_used": False,
        "face_number_contract": "one-based OCC STEP traversal, not NX internal IDs",
    }
    graph["inference_overlay"] = {"feature_objects": output_objects, "truth_used": False}
    destination = output_dir / f"{input_path.stem}_recognition.json"
    write_json(destination, result)
    write_json(output_dir / f"{input_path.stem}_graph.json", graph)
    return destination


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="TopoCAE old baseline STEP recognition")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "manual_inference")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--compare-rotation", action="store_true")
    args = parser.parse_args()
    destination = run(args.input, args.output_dir, args.compare_rotation, args.device)
    result = read_json(destination)
    print(f"识别结果：{destination}")
    for item in result["feature_objects"]:
        print(f"[{item['object_index']}] {item['class']} | 面编号={item['original_face_numbers']} | 总面积={item['geometry_summary']['total_area']:.4f} | 置信度={item['class_confidence']:.3f}")
    if result["rotation_comparison"] is not None:
        print("旋转一致性：", result["rotation_comparison"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
