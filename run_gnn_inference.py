from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.gnn_baseline import EdgeAwareMultiHeadGNN, MultiHeadGraphSAGE
from utils.candidate_classifier_postprocess import (
    apply_candidate_classifier_postprocess,
    load_candidate_classifier_bundle,
)
from utils.instance_candidate_postprocess import (
    apply_instance_candidate_postprocess,
    load_instance_candidate_classifier_bundle,
)
from utils.rib_candidate_proposer_postprocess import (
    apply_rib_candidate_proposer_postprocess,
    load_rib_candidate_proposer_bundle,
)
from utils.rib_boundary_classifier_postprocess import (
    apply_rib_boundary_classifier_postprocess,
    load_rib_boundary_classifier_bundle,
)
from utils.topology_postprocess import apply_topology_postprocess
from utils.training_export import (
    DEFAULT_NODE_FEATURE_MODE,
    build_node_feature_vector,
    build_numeric_edge_feature_vector,
)


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run GNN baseline inference on unlabeled STEP-derived graph json files."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-round-ndigits", type=int, default=6)
    parser.add_argument("--disable-topology-postprocess", action="store_true")
    parser.add_argument("--same-instance-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-classifier", type=Path, default=None)
    parser.add_argument("--candidate-classifier-min-confidence", type=float, default=0.5)
    parser.add_argument("--instance-candidate-classifier", type=Path, default=None)
    parser.add_argument("--instance-candidate-min-confidence", type=float, default=0.5)
    parser.add_argument("--same-instance-assembly-threshold", type=float, default=0.55)
    parser.add_argument("--same-instance-weak-threshold", type=float, default=0.35)
    parser.add_argument("--rib-candidate-proposer", type=Path, default=None)
    parser.add_argument("--rib-candidate-proposer-min-confidence", type=float, default=0.75)
    parser.add_argument("--rib-boundary-classifier", type=Path, default=None)
    parser.add_argument("--rib-boundary-classifier-min-confidence", type=float, default=0.8)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collect_graph_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(
        path
        for path in globber("*.json")
        if path.is_file() and _is_graph_input_file(path)
    )


def _is_graph_input_file(path: Path) -> bool:
    ignored_names = {
        "inference_graph_manifest.json",
        "prediction_manifest.json",
        "feature_object_manifest.json",
        "primitive_surface_label_map.json",
    }
    if path.name in ignored_names:
        return False
    return not path.name.endswith("_label_map.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def resolve_output_path(input_root: Path, graph_path: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / graph_path.name
    return output_dir / graph_path.relative_to(input_root)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> MultiHeadGraphSAGE:
    config = checkpoint["config"]
    label_spaces = config["label_spaces"]
    common_kwargs = {
        "input_dim": len(config["normalization"]["feature_mean"]),
        "num_primitive_surfaces": len(label_spaces["primitive_surface_id_to_key"]),
        "num_macro_classes": len(label_spaces["macro_feature_id_to_key"]),
        "num_semantic_classes": len(label_spaces["semantic_subtype_id_to_key"]),
        "derived_target_dim": len(label_spaces["derived_target_ids"]),
        "idealization_target_dim": len(label_spaces["idealization_target_ids"]),
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "primitive_embedding_dim": config["primitive_embedding_dim"],
        "dropout": config["dropout"],
    }
    if config.get("model_kind") == "edge_aware":
        model = EdgeAwareMultiHeadGNN(
            edge_feature_dim=int(config["edge_feature_dim"]),
            **common_kwargs,
        )
    else:
        model = MultiHeadGraphSAGE(**common_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def graph_to_tensors(graph_data: dict, checkpoint_config: dict, ndigits: int, device: torch.device) -> dict[str, torch.Tensor]:
    mean = torch.tensor(checkpoint_config["normalization"]["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(checkpoint_config["normalization"]["feature_std"], dtype=torch.float32, device=device)
    node_feature_mode = checkpoint_config.get("node_feature_mode", DEFAULT_NODE_FEATURE_MODE)

    node_features = torch.tensor(
        [
            build_node_feature_vector(node, ndigits, node_feature_mode=node_feature_mode)
            for node in graph_data["nodes"]
        ],
        dtype=torch.float32,
        device=device,
    )
    node_features = (node_features - mean) / std

    primitive_surface_ids = torch.tensor(
        [node["primitive_surface_id"] for node in graph_data["nodes"]],
        dtype=torch.long,
        device=device,
    )

    if graph_data.get("edges"):
        edge_index = torch.tensor(graph_data["edges"], dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_mean_values = checkpoint_config.get("normalization", {}).get("edge_feature_mean", [])
    edge_std_values = checkpoint_config.get("normalization", {}).get("edge_feature_std", [])
    edge_feature_dim = int(checkpoint_config.get("edge_feature_dim") or len(edge_mean_values))
    if edge_feature_dim > 0:
        edge_attrs = list(graph_data.get("edge_attributes", []))
        edge_features = [
            build_numeric_edge_feature_vector(edge_attrs[index] if index < len(edge_attrs) else {}, ndigits)
            for index, _edge in enumerate(graph_data.get("edges", []))
        ]
        aligned_features = []
        for row in edge_features:
            if len(row) < edge_feature_dim:
                row = row + [0.0] * (edge_feature_dim - len(row))
            aligned_features.append(row[:edge_feature_dim])
        edge_tensor = torch.tensor(aligned_features, dtype=torch.float32, device=device) if aligned_features else torch.zeros((0, edge_feature_dim), dtype=torch.float32, device=device)
        if edge_mean_values and edge_std_values:
            edge_mean = torch.tensor(edge_mean_values[:edge_feature_dim], dtype=torch.float32, device=device)
            edge_std = torch.tensor(edge_std_values[:edge_feature_dim], dtype=torch.float32, device=device)
            edge_tensor = (edge_tensor - edge_mean) / edge_std
    else:
        edge_tensor = torch.zeros((edge_index.size(1), 0), dtype=torch.float32, device=device)

    return {
        "node_features": node_features,
        "primitive_surface_ids": primitive_surface_ids,
        "edge_index": edge_index,
        "edge_features": edge_tensor,
    }


def decode_multilabel(probabilities: torch.Tensor, target_ids: list[int], id_to_key: dict[str, str], threshold: float = 0.5) -> tuple[list[int], list[str], list[float]]:
    label_ids = []
    label_keys = []
    label_scores = []
    for idx, probability in enumerate(probabilities.tolist()):
        if probability >= threshold:
            label_id = int(target_ids[idx])
            label_ids.append(label_id)
            label_keys.append(id_to_key[str(label_id)])
            label_scores.append(float(probability))
    return label_ids, label_keys, label_scores


def decode_edge_same_instance_predictions(outputs: dict, graph_data: dict) -> list[dict]:
    logits = outputs.get("edge_same_instance_logits")
    if logits is None:
        return []

    probabilities = torch.sigmoid(logits).detach().cpu().tolist()
    predictions = []
    for index, edge in enumerate(graph_data.get("edges", [])):
        if index >= len(probabilities) or not isinstance(edge, list) or len(edge) != 2:
            continue
        predictions.append({
            "source_face": int(edge[0]),
            "target_face": int(edge[1]),
            "same_instance_probability": float(probabilities[index]),
        })
    return predictions


def predict_one_graph(
    model: MultiHeadGraphSAGE,
    graph_data: dict,
    graph_path: Path,
    checkpoint_config: dict,
    ndigits: int,
    device: torch.device,
    topology_postprocess: bool = True,
    same_instance_threshold: float = 0.5,
    candidate_classifier_bundle: dict | None = None,
    candidate_classifier_min_confidence: float = 0.5,
    instance_candidate_classifier_bundle: dict | None = None,
    instance_candidate_min_confidence: float = 0.5,
    same_instance_assembly_threshold: float = 0.55,
    same_instance_weak_threshold: float = 0.35,
    rib_candidate_proposer_bundle: dict | None = None,
    rib_candidate_proposer_min_confidence: float = 0.75,
    rib_boundary_classifier_bundle: dict | None = None,
    rib_boundary_classifier_min_confidence: float = 0.8,
) -> dict:
    tensors = graph_to_tensors(graph_data, checkpoint_config, ndigits, device)
    outputs = model(
        node_features=tensors["node_features"],
        edge_index=tensors["edge_index"],
        primitive_surface_ids=tensors["primitive_surface_ids"],
        edge_features=tensors["edge_features"],
    )

    label_spaces = checkpoint_config["label_spaces"]
    macro_id_to_key = label_spaces["macro_feature_id_to_key"]
    semantic_id_to_key = label_spaces["semantic_subtype_id_to_key"]
    idealization_id_to_key = label_spaces["idealization_tag_id_to_key"]
    derived_target_ids = label_spaces["derived_target_ids"]
    idealization_target_ids = label_spaces["idealization_target_ids"]

    macro_prob = torch.softmax(outputs["macro_logits"], dim=-1)
    semantic_prob = torch.softmax(outputs["semantic_logits"], dim=-1)
    derived_prob = torch.sigmoid(outputs["derived_logits"]) if "derived_logits" in outputs else None
    idealization_prob = torch.sigmoid(outputs["idealization_logits"]) if "idealization_logits" in outputs else None
    edge_same_instance = decode_edge_same_instance_predictions(outputs, graph_data)

    faces = []
    for node_idx, node in enumerate(graph_data["nodes"]):
        macro_id = int(torch.argmax(macro_prob[node_idx]).item())
        semantic_id = int(torch.argmax(semantic_prob[node_idx]).item())

        face_record = {
            "idx": node["idx"],
            "primitive_surface": node["primitive_surface"],
            "primitive_surface_id": node["primitive_surface_id"],
            "macro_feature_id": macro_id,
            "macro_feature": macro_id_to_key[str(macro_id)],
            "macro_confidence": float(macro_prob[node_idx, macro_id].item()),
            "semantic_subtype_id": semantic_id,
            "semantic_subtype": semantic_id_to_key[str(semantic_id)],
            "semantic_confidence": float(semantic_prob[node_idx, semantic_id].item()),
        }

        if derived_prob is not None:
            label_ids, label_keys, label_scores = decode_multilabel(
                derived_prob[node_idx],
                derived_target_ids,
                macro_id_to_key,
            )
            face_record["derived_macro_feature_ids"] = label_ids
            face_record["derived_macro_features"] = label_keys
            face_record["derived_macro_feature_scores"] = label_scores

        if idealization_prob is not None:
            label_ids, label_keys, label_scores = decode_multilabel(
                idealization_prob[node_idx],
                idealization_target_ids,
                idealization_id_to_key,
            )
            face_record["idealization_tag_ids"] = label_ids
            face_record["idealization_tags"] = label_keys
            face_record["idealization_tag_scores"] = label_scores

        faces.append(face_record)

    prediction = {
        "sample_id": graph_path.stem,
        "graph_mode": "prediction",
        "source_step_path": graph_data.get("source_step_path"),
        "source_graph_path": str(graph_path.resolve()),
        "source_graph_mode": graph_data.get("graph_mode", "unknown"),
        "face_count": len(faces),
        "faces": faces,
        "edge_same_instance": edge_same_instance,
    }
    if instance_candidate_classifier_bundle is not None:
        prediction = apply_instance_candidate_postprocess(
            prediction,
            graph_data,
            instance_candidate_classifier_bundle,
            min_confidence=instance_candidate_min_confidence,
            strong_threshold=same_instance_assembly_threshold,
            weak_threshold=same_instance_weak_threshold,
        )
    if candidate_classifier_bundle is not None:
        prediction = apply_candidate_classifier_postprocess(
            prediction,
            graph_data,
            candidate_classifier_bundle,
            same_instance_threshold=same_instance_threshold,
            min_confidence=candidate_classifier_min_confidence,
        )
    if topology_postprocess:
        graph_for_postprocess = {
            **graph_data,
            "prediction_edge_same_instance": edge_same_instance,
        }
        prediction = apply_topology_postprocess(
            prediction,
            graph_for_postprocess,
            same_instance_threshold=same_instance_threshold,
        )
    if rib_candidate_proposer_bundle is not None:
        prediction = apply_rib_candidate_proposer_postprocess(
            prediction,
            graph_data,
            rib_candidate_proposer_bundle,
            min_confidence=rib_candidate_proposer_min_confidence,
        )
    if rib_boundary_classifier_bundle is not None:
        prediction = apply_rib_boundary_classifier_postprocess(
            prediction,
            graph_data,
            rib_boundary_classifier_bundle,
            min_confidence=rib_boundary_classifier_min_confidence,
        )
    return prediction


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)
    candidate_classifier_bundle = (
        load_candidate_classifier_bundle(args.candidate_classifier)
        if args.candidate_classifier is not None
        else None
    )
    instance_candidate_classifier_bundle = (
        load_instance_candidate_classifier_bundle(args.instance_candidate_classifier)
        if args.instance_candidate_classifier is not None
        else None
    )
    rib_candidate_proposer_bundle = (
        load_rib_candidate_proposer_bundle(args.rib_candidate_proposer)
        if args.rib_candidate_proposer is not None
        else None
    )
    rib_boundary_classifier_bundle = (
        load_rib_boundary_classifier_bundle(args.rib_boundary_classifier)
        if args.rib_boundary_classifier is not None
        else None
    )

    graph_files = collect_graph_files(args.input_path, args.recursive)
    if not graph_files:
        raise RuntimeError(f"No graph files found under: {args.input_path}")

    manifest_entries = []
    for graph_path in graph_files:
        graph_data = load_json(graph_path)
        if "nodes" not in graph_data or "edges" not in graph_data:
            continue
        prediction = predict_one_graph(
            model=model,
            graph_data=graph_data,
            graph_path=graph_path,
            checkpoint_config=checkpoint["config"],
            ndigits=args.feature_round_ndigits,
            device=device,
            topology_postprocess=not args.disable_topology_postprocess,
            same_instance_threshold=args.same_instance_threshold,
            candidate_classifier_bundle=candidate_classifier_bundle,
            candidate_classifier_min_confidence=args.candidate_classifier_min_confidence,
            instance_candidate_classifier_bundle=instance_candidate_classifier_bundle,
            instance_candidate_min_confidence=args.instance_candidate_min_confidence,
            same_instance_assembly_threshold=args.same_instance_assembly_threshold,
            same_instance_weak_threshold=args.same_instance_weak_threshold,
            rib_candidate_proposer_bundle=rib_candidate_proposer_bundle,
            rib_candidate_proposer_min_confidence=args.rib_candidate_proposer_min_confidence,
            rib_boundary_classifier_bundle=rib_boundary_classifier_bundle,
            rib_boundary_classifier_min_confidence=args.rib_boundary_classifier_min_confidence,
        )
        output_path = resolve_output_path(args.input_path, graph_path, args.output_dir).resolve()
        write_json(output_path, prediction)
        manifest_entries.append({
            "input_graph_path": str(graph_path.resolve()),
            "output_prediction_path": str(output_path),
            "face_count": prediction["face_count"],
        })
        print(f"prediction written: {output_path}")

    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "candidate_classifier": str(args.candidate_classifier.resolve()) if args.candidate_classifier else None,
            "instance_candidate_classifier": str(args.instance_candidate_classifier.resolve()) if args.instance_candidate_classifier else None,
            "rib_candidate_proposer": str(args.rib_candidate_proposer.resolve()) if args.rib_candidate_proposer else None,
            "rib_boundary_classifier": str(args.rib_boundary_classifier.resolve()) if args.rib_boundary_classifier else None,
            "prediction_count": len(manifest_entries),
            "samples": manifest_entries,
        },
    )


if __name__ == "__main__":
    main()




