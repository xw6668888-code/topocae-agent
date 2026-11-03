from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import train_p1_b_gnn_ablation as legacy
from models.gnn_baseline import EdgeAwareMultiHeadGNN
from p1_e6.common import aggregate_object_counts, binary_metrics, classification_metrics, object_counts, truth_objects
from p1_m2.boundary import enforce_base_candidate_boundaries
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction

from .candidate_decision import resolve_candidate_macro
from .common import MACRO_CLASSES, REPORT, ROOT, TARGET_CLASSES, read_json
from .training_common import BATCH_SIZE, adapt_record, build_model


SHARED_CANDIDATE_THRESHOLDS = (
    ROOT / "reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718/candidate_model/candidate_threshold_selection.json"
)
SHARED_GLOBAL_DECODE = SHARED_CANDIDATE_THRESHOLDS.parents[1] / "global_decode_selection.json"


def load_model(run_dir: Path, device: torch.device) -> EdgeAwareMultiHeadGNN:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_records(records: Sequence[dict[str, Any]], model: EdgeAwareMultiHeadGNN, device: torch.device) -> list[dict[str, Any]]:
    loader = DataLoader(
        legacy.Records(list(records)), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=legacy._collate, pin_memory=True,
    )
    results = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["node_features"].to(device, non_blocking=True),
                batch["edge_index"].to(device, non_blocking=True),
                batch["primitive"].to(device, non_blocking=True),
                edge_features=batch["edge_features"].to(device, non_blocking=True),
            )
            macro = torch.softmax(output["macro_logits"], dim=1).cpu()
            semantic = torch.softmax(output["semantic_logits"], dim=1).cpu()
            edge = torch.sigmoid(output["edge_same_instance_logits"]).cpu()
            for record, node_slice, edge_slice in zip(batch["records"], batch["node_slices"], batch["edge_slices"]):
                ns, ne = node_slice
                es, ee = edge_slice
                macro_rows = macro[ns:ne]
                semantic_rows = semantic[ns:ne]
                faces = []
                for index in range(ne - ns):
                    macro_id = int(macro_rows[index].argmax())
                    semantic_id = int(semantic_rows[index].argmax())
                    faces.append({
                        "idx": index,
                        "macro_feature": MACRO_CLASSES[macro_id],
                        "macro_class_id": macro_id,
                        "macro_confidence": float(macro_rows[index, macro_id]),
                        "macro_probabilities": {name: float(macro_rows[index, class_id]) for class_id, name in enumerate(MACRO_CLASSES)},
                        "semantic_label": semantic_id,
                        "semantic_confidence": float(semantic_rows[index, semantic_id]),
                        "embedding": [],
                    })
                results.append({"record": record, "faces": faces, "edge_probabilities": edge[es:ee].tolist()})
    return results


def explicit_topology_objects(
    graph: Mapping[str, Any],
    face_prediction: Mapping[str, Any],
    same_threshold: Mapping[str, Any],
    candidate_thresholds: Mapping[str, Any],
    global_decode: Mapping[str, Any],
    sample_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    assembly = assemble_same_instance_candidates(
        graph, face_prediction,
        global_threshold=float(same_threshold["global_threshold"]),
        threshold_selection_sha256=str(same_threshold["selection_sha256"]),
    )
    raw = generate_candidate_proposals(graph, face_prediction, assembly)["proposals"]
    boundary = enforce_base_candidate_boundaries(raw, assembly)
    face_macros = {int(face["idx"]): str(face["macro_feature"]) for face in face_prediction["faces"]}
    verified = []
    decisions = []
    for proposal in boundary["proposals"]:
        decision = resolve_candidate_macro(proposal, face_prediction)
        decisions.append({"candidate_id": proposal.get("candidate_id"), **decision.to_dict()})
        if decision.status != "selected" or decision.selected_macro not in TARGET_CLASSES:
            continue
        faces = sorted(set(int(value) for value in proposal.get("face_indices", [])))
        face_set = set(faces)
        macro = str(decision.selected_macro)
        synthetic = {
            "candidate_id": proposal.get("candidate_id"),
            "face_indices": faces,
            "macro_classes": list(TARGET_CLASSES),
            "macro_logits": [20.0 if name == macro else -20.0 for name in TARGET_CLASSES],
            "macro_probabilities": [1.0 if name == macro else 0.0 for name in TARGET_CLASSES],
            "validity_score": 1.0,
            "membership_probabilities": [1.0] * len(faces),
            "boundary_probabilities": [1.0 if face in _boundary_faces(graph, face_set) else 0.0 for face in faces],
            "face_macro_predictions": face_macros,
        }
        result = verify_candidate_prediction(graph, proposal, synthetic, thresholds=candidate_thresholds)
        result["candidate_macro_decision"] = decision.to_dict()
        verified.append(result)
    decoded = decode_feature_objects(
        graph, verified,
        subset_confidence_tolerance=float(global_decode["global_subset_confidence_tolerance"]),
        selection_sha256=str(global_decode["selection_sha256"]),
    )
    objects = [
        {"sample_id": sample_id, **dict(item), "face_indices": sorted(set(int(value) for value in item.get("face_indices", [])))}
        for item in decoded["feature_objects"]
    ]
    return objects, assembly, verified, boundary, decisions


def candidate_objects(assembly: Mapping[str, Any], face_prediction: Mapping[str, Any], sample_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects, decisions = [], []
    for item in assembly.get("candidates", []):
        proposal = {
            "face_indices": list(item.get("face_indices", [])),
            "macro_hints": list(TARGET_CLASSES),
            "topology_evidence": {"source": "same_instance_candidate_assembly"},
        }
        decision = resolve_candidate_macro(proposal, face_prediction)
        decisions.append(decision.to_dict())
        if decision.status == "selected" and decision.selected_macro in TARGET_CLASSES:
            objects.append({
                "sample_id": sample_id,
                "macro_feature": decision.selected_macro,
                "face_indices": sorted(set(int(value) for value in item.get("face_indices", []))),
                "source": "same_instance_candidate_assembly",
            })
    return objects, decisions


def evaluate_sample(
    prediction: Mapping[str, Any], graph: Mapping[str, Any], same_threshold: Mapping[str, Any],
    candidate_thresholds: Mapping[str, Any], global_decode: Mapping[str, Any],
) -> dict[str, Any]:
    record = prediction["record"]
    sample_id = str(record["sample_id"])
    face_prediction = {
        "faces": list(prediction["faces"]), "node_embeddings": [],
        "edge_same_instance_probabilities": list(prediction["edge_probabilities"]),
        "contracts": {"truth_used": False, "latent_candidate_classifier_enabled": False},
    }
    full, assembly, verified, boundary, decisions = explicit_topology_objects(
        graph, face_prediction, same_threshold, candidate_thresholds, global_decode, sample_id,
    )
    candidates, assembly_decisions = candidate_objects(assembly, face_prediction, sample_id)
    truths = truth_objects(record["feature_sets"])
    threshold = float(same_threshold["global_threshold"])
    edge_prediction = [int(value >= threshold) for value in prediction["edge_probabilities"]]
    validity = _validity(full, len(graph.get("nodes", [])))
    return {
        "sample_id": sample_id,
        "stock_type": str(record["stock_type"]),
        "feature_count": len(record["feature_sets"]),
        "feature_macros": [str(item["macro_feature"]) for item in record["feature_sets"]],
        "face_truth": record["macro_targets"].tolist(),
        "face_prediction": [int(face["macro_class_id"]) for face in prediction["faces"]],
        "semantic_truth": record["semantic_targets"].tolist(),
        "semantic_prediction": [int(face["semantic_label"]) for face in prediction["faces"]],
        "edge_truth": record["edge_same_instance_targets"].long().tolist(),
        "edge_prediction": edge_prediction,
        "edge_mask": record["edge_supervision_mask"].bool().tolist(),
        "truth_objects": truths,
        "candidate_objects": candidates,
        "full_objects": full,
        "candidate_counts": object_counts(truths, candidates),
        "full_counts": object_counts(truths, full),
        "validity": validity,
        "diagnostics": {
            "assembly_candidate_count": len(assembly.get("candidates", [])),
            "proposal_count": len(boundary.get("proposals", [])),
            "verified_candidate_count": len(verified),
            "uncertain_candidate_decision_count": sum(item["status"] == "uncertain" for item in decisions),
            "uncertain_assembly_decision_count": sum(item["status"] == "uncertain" for item in assembly_decisions),
            "candidate_decisions": decisions,
            "latent_candidate_classifier_inference_count": 0,
        },
    }


def aggregate_samples(samples: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    face_truth = [value for sample in samples for value in sample["face_truth"]]
    face_prediction = [value for sample in samples for value in sample["face_prediction"]]
    semantic_truth = [value for sample in samples for value in sample["semantic_truth"]]
    semantic_prediction = [value for sample in samples for value in sample["semantic_prediction"]]
    edge_truth = [value for sample in samples for value, mask in zip(sample["edge_truth"], sample["edge_mask"]) if mask]
    edge_prediction = [value for sample in samples for value, mask in zip(sample["edge_prediction"], sample["edge_mask"]) if mask]
    validity = Counter()
    for sample in samples:
        validity.update(sample["validity"])
    candidate = aggregate_object_counts(samples, "candidate_counts")
    full = aggregate_object_counts(samples, "full_counts")
    return {
        "sample_count": len(samples),
        "face": classification_metrics(face_truth, face_prediction, MACRO_CLASSES),
        "semantic": classification_metrics(semantic_truth, semantic_prediction),
        "same_instance_edge": {"threshold": threshold, **binary_metrics(edge_truth, edge_prediction)},
        "candidate_objects": candidate,
        "full_pipeline_objects": full,
        "validity": dict(validity),
        "cross_truth_instance_overmerge_count": full["overmerge_count"],
        "latent_candidate_classifier_inference_count": 0,
    }


def _boundary_faces(graph: Mapping[str, Any], faces: set[int]) -> set[int]:
    result = set()
    for left, right in graph.get("edges", []):
        left, right = int(left), int(right)
        if left in faces and right not in faces:
            result.add(left)
        if right in faces and left not in faces:
            result.add(right)
    return result


def _validity(objects: Sequence[Mapping[str, Any]], node_count: int) -> dict[str, int]:
    illegal = duplicate = empty = 0
    for item in objects:
        faces = [int(value) for value in item.get("face_indices", [])]
        empty += int(not faces)
        illegal += sum(value < 0 or value >= node_count for value in faces)
        duplicate += len(faces) - len(set(faces))
    return {
        "illegal_face_id_count": illegal,
        "duplicate_face_id_inside_object_count": duplicate,
        "empty_object_count": empty,
    }


__all__ = [
    "SHARED_CANDIDATE_THRESHOLDS", "SHARED_GLOBAL_DECODE", "aggregate_samples",
    "candidate_objects", "evaluate_sample", "explicit_topology_objects", "infer_records", "load_model",
]
