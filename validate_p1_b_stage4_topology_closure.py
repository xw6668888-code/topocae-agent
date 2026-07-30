from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from models.p1_b_candidate_topology_classifier import CandidateModelDimensions, CandidateTopologyClassifier
from utils.ontology import load_layer_index, load_ontology_maps
from utils.p1_b_candidate_features import ENGINEERED_FEATURE_SCHEMA, MACRO_CLASSES, build_candidate_tensor_record
from utils.p1_b_candidate_hard_negatives import generate_candidate_hard_negatives
from utils.p1_b_candidate_proposer import evaluate_proposal_oracle_recall, generate_candidate_proposals
from utils.p1_b_global_decode import decode_feature_objects
from utils.p1_b_instance_assembly import (
    assemble_same_instance_candidates,
    evaluate_same_instance_edges,
    select_global_same_instance_threshold,
)
from utils.p1_b_topology_verifier import select_global_candidate_thresholds, verify_candidate_prediction
from utils.training_export_v2_1 import GRAPH_V2_EDGE_FEATURE_SCHEMA, GRAPH_V2_NODE_FEATURE_SCHEMA
from validate_p1_b_graph_v2 import load_p1_b_semantic_index, rib_web_graphs, slot_graphs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports" / "p1_b_stage4_topology_closure"
PROVISIONAL_PERTURBATION = {
    "slot_u": "slot_rect",
    "slot_rect": "slot_v",
    "slot_v": "slot_o",
    "slot_o": "slot_u",
    "rib": "web",
    "web": "rib",
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    semantic_index = load_p1_b_semantic_index()
    ontology_maps = load_ontology_maps(ROOT / "CAE_feature_ontology.yaml")
    primitive_index = load_layer_index(ROOT / "CAE_feature_ontology.yaml", "primitive_surface")
    source_records = slot_graphs(semantic_index, ontology_maps, primitive_index) + rib_web_graphs(
        semantic_index, ontology_maps, primitive_index
    )

    threshold_records = []
    prepared = []
    for source in source_records:
        graph = source["graph"]
        prediction = _perfect_edge_prediction_with_cross_macro_perturbation(graph, source["sample_id"])
        for edge, probability in zip(graph.get("edges", []), prediction["edge_same_instance_probabilities"]):
            left, right = int(edge[0]), int(edge[1])
            left_instance = int(graph["nodes"][left].get("instance_id", 0) or 0)
            right_instance = int(graph["nodes"][right].get("instance_id", 0) or 0)
            threshold_records.append({
                "split": "synthetic_val",
                "probability": probability,
                "label": int(left_instance > 0 and left_instance == right_instance),
            })
        prepared.append((source, _neutralize_truth_labels(graph), prediction, _truth_instances(graph)))
    same_instance_selection = select_global_same_instance_threshold(threshold_records)
    candidate_selection = select_global_candidate_thresholds(
        [
            {
                "split": "synthetic_val",
                "macro_confidence": 0.99,
                "validity_score": 0.99,
                "accept_label": 1,
                "membership_probabilities": [0.99, 0.01],
                "membership_labels": [1, 0],
                "boundary_probabilities": [0.99, 0.01],
                "boundary_labels": [1, 0],
            },
            {
                "split": "synthetic_val",
                "macro_confidence": 0.05,
                "validity_score": 0.05,
                "accept_label": 0,
            },
        ],
        threshold_grid=[index / 100.0 for index in range(5, 96, 5)],
    )

    sample_audits = []
    all_tensor_records = []
    for source, inference_graph, prediction, truths in prepared:
        edge_metrics = evaluate_same_instance_edges(
            source["graph"], prediction, global_threshold=same_instance_selection["global_threshold"]
        )
        assembly = assemble_same_instance_candidates(
            inference_graph,
            prediction,
            global_threshold=same_instance_selection["global_threshold"],
            threshold_selection_sha256=same_instance_selection["selection_sha256"],
        )
        proposal_result = generate_candidate_proposals(inference_graph, prediction, assembly)
        proposal_metrics = evaluate_proposal_oracle_recall(proposal_result["proposals"], truths)
        hard_negatives = generate_candidate_hard_negatives(source["graph"], truths)
        tensor_records = [
            build_candidate_tensor_record(inference_graph, prediction, proposal)
            for proposal in proposal_result["proposals"]
        ]
        all_tensor_records.extend(tensor_records)
        verified_truth_candidates = []
        for truth_position, truth in enumerate(truths):
            proposal = {
                "candidate_id": f"oracle_truth_{truth_position}",
                "face_indices": truth["face_indices"],
                "proposal_sources": ["same_instance_strong_component"],
            }
            verified_truth_candidates.append(verify_candidate_prediction(
                inference_graph,
                proposal,
                _structural_oracle_prediction(truth),
                thresholds=candidate_selection,
            ))
        decoded = decode_feature_objects(inference_graph, verified_truth_candidates)
        decoded_sets = {
            (record["macro_feature"], tuple(record["face_indices"]))
            for record in decoded["feature_objects"]
        }
        truth_sets = {(record["macro_feature"], tuple(record["face_indices"])) for record in truths}
        exact_object_ratio = len(decoded_sets & truth_sets) / len(truth_sets) if truth_sets else 1.0
        cross_macro_candidates = sum(
            bool(candidate.get("cross_provisional_macro")) for candidate in assembly["candidates"]
        )
        tensor_finite = all(
            record["trace"]["all_features_finite"] and record["trace"]["max_abs"] <= 32.0
            for record in tensor_records
        )
        passed = (
            edge_metrics["f1"] >= 0.95
            and proposal_metrics["oracle_recall"] >= 0.95
            and exact_object_ratio >= 0.85
            and tensor_finite
            and not decoded["unexplained_face_overlap"]
        )
        sample_audits.append({
            "sample_id": source["sample_id"],
            "family": source["family"],
            "passed": passed,
            "same_instance_edge_metrics": edge_metrics,
            "assembly_candidate_count": assembly["candidate_count"],
            "cross_provisional_macro_candidate_count": cross_macro_candidates,
            "proposal_count": proposal_result["candidate_count"],
            "proposal_source_counts": proposal_result["source_counts"],
            "proposal_oracle": proposal_metrics,
            "hard_negative_count": hard_negatives["record_count"],
            "candidate_tensor_count": len(tensor_records),
            "candidate_tensor_finite": tensor_finite,
            "oracle_structural_exact_object_ratio": exact_object_ratio,
            "oracle_structural_abstention_count": decoded["abstention_count"],
            "unexplained_face_overlap": decoded["unexplained_face_overlap"],
        })

    model_smoke = _candidate_model_smoke(all_tensor_records)
    same_f1_values = [record["same_instance_edge_metrics"]["f1"] for record in sample_audits]
    proposal_values = [record["proposal_oracle"]["oracle_recall"] for record in sample_audits]
    exact_values = [record["oracle_structural_exact_object_ratio"] for record in sample_audits]
    source_union = sorted({source for record in sample_audits for source in record["proposal_source_counts"]})
    required_sources = {
        "same_instance_strong_component",
        "coaxial_cylinder_cone_chain",
        "loop_material_side_guided",
        "concave_boundary_region",
        "elongated_swept_channel",
        "thin_wall_rib_web",
        "parent_surface_transition",
    }
    source_coverage = sorted(required_sources & set(source_union))
    automatic_pass = (
        all(record["passed"] for record in sample_audits)
        and model_smoke["passed"]
        and min(same_f1_values, default=0.0) >= 0.95
        and min(proposal_values, default=0.0) >= 0.95
        and min(exact_values, default=0.0) >= 0.85
        and len(source_coverage) >= 5
    )
    payload = {
        "schema_version": "topocae_p1_b_stage4_topology_closure_validation_1",
        "automatic_pass": automatic_pass,
        "validation_scope": "synthetic_canonical_oracle_upper_bound_and_pipeline_contract",
        "not_a_trained_model_claim": True,
        "sample_count": len(sample_audits),
        "same_instance_threshold_selection": same_instance_selection,
        "candidate_threshold_selection": candidate_selection,
        "metrics_summary": {
            "same_instance_edge_f1_min": min(same_f1_values, default=0.0),
            "same_instance_edge_f1_mean": _mean(same_f1_values),
            "proposal_oracle_recall_min": min(proposal_values, default=0.0),
            "proposal_oracle_recall_mean": _mean(proposal_values),
            "oracle_structural_exact_object_ratio_min": min(exact_values, default=0.0),
            "oracle_structural_exact_object_ratio_mean": _mean(exact_values),
            "cross_provisional_macro_candidate_count": sum(record["cross_provisional_macro_candidate_count"] for record in sample_audits),
            "hard_negative_count": sum(record["hard_negative_count"] for record in sample_audits),
            "candidate_tensor_count": len(all_tensor_records),
            "proposal_source_union": source_union,
            "required_source_coverage": source_coverage,
        },
        "candidate_model_smoke": model_smoke,
        "sample_audits": sample_audits,
        "contracts": {
            "legacy_same_macro_component_fallback": False,
            "real_data_used_for_threshold_selection": False,
            "truth_labels_neutralized_before_pipeline_features": True,
            "rules_hard_relabel_classes": False,
            "absolute_area_constants": False,
            "original_face_ids_preserved": True,
        },
    }
    result_path = OUTPUT_DIR / "automatic_validation.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = OUTPUT_DIR / "automatic_validation_summary.zh-CN.md"
    summary_path.write_text(_summary(payload), encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_b_stage4_validation_manifest_1",
        "automatic_pass": automatic_pass,
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (result_path, summary_path)
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "automatic_pass": automatic_pass,
        "sample_count": len(sample_audits),
        "same_instance_f1_min": payload["metrics_summary"]["same_instance_edge_f1_min"],
        "proposal_recall_min": payload["metrics_summary"]["proposal_oracle_recall_min"],
        "exact_object_min": payload["metrics_summary"]["oracle_structural_exact_object_ratio_min"],
        "source_coverage": len(source_coverage),
    }, ensure_ascii=False))
    return 0 if automatic_pass else 1


def _perfect_edge_prediction_with_cross_macro_perturbation(graph: dict[str, Any], sample_id: str) -> dict[str, Any]:
    instance_faces: dict[int, list[int]] = defaultdict(list)
    for node in graph["nodes"]:
        instance_id = int(node.get("instance_id", 0) or 0)
        if instance_id > 0:
            instance_faces[instance_id].append(int(node["idx"]))
    perturbed = {}
    for faces in instance_faces.values():
        if len(faces) > 1:
            index = sorted(faces)[0]
            macro = str(graph["nodes"][index].get("macro_feature", "unknown_feature"))
            perturbed[index] = PROVISIONAL_PERTURBATION.get(macro, macro)
    faces = []
    for node in graph["nodes"]:
        index = int(node["idx"])
        faces.append({
            "idx": index,
            "macro_feature": perturbed.get(index, str(node.get("macro_feature", "stock"))),
            "raw_macro_feature": perturbed.get(index, str(node.get("macro_feature", "stock"))),
        })
    probabilities = []
    for edge in graph.get("edges", []):
        left, right = int(edge[0]), int(edge[1])
        left_instance = int(graph["nodes"][left].get("instance_id", 0) or 0)
        right_instance = int(graph["nodes"][right].get("instance_id", 0) or 0)
        probabilities.append(0.99 if left_instance > 0 and left_instance == right_instance else 0.01)
    return {
        "sample_id": sample_id,
        "faces": faces,
        "edge_same_instance_probabilities": probabilities,
        "simulation_scope": "synthetic_canonical_representation_contract_only",
    }


def _neutralize_truth_labels(graph: dict[str, Any]) -> dict[str, Any]:
    neutral = deepcopy(graph)
    for node in neutral["nodes"]:
        node["instance_id"] = 0
        node["macro_feature"] = "stock"
        node["macro_feature_id"] = 0
        node["semantic_label"] = 0
    for patch in neutral.get("canonical_patches", []):
        patch["instance_ids"] = [0]
        patch["macro_features"] = ["stock"]
    return neutral


def _truth_instances(graph: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[int, list[int]] = defaultdict(list)
    macros = {}
    for node in graph["nodes"]:
        instance_id = int(node.get("instance_id", 0) or 0)
        if instance_id > 0:
            groups[instance_id].append(int(node["idx"]))
            macros[instance_id] = str(node.get("macro_feature", "unknown_feature"))
    return [
        {"instance_id": instance_id, "macro_feature": macros[instance_id], "face_indices": sorted(faces)}
        for instance_id, faces in sorted(groups.items())
    ]


def _structural_oracle_prediction(truth: dict[str, Any]) -> dict[str, Any]:
    probabilities = [0.001] * len(MACRO_CLASSES)
    probabilities[MACRO_CLASSES.index(truth["macro_feature"])] = 0.999
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    count = len(truth["face_indices"])
    return {
        "macro_classes": MACRO_CLASSES,
        "macro_probabilities": probabilities,
        "macro_logits": probabilities,
        "validity_score": 0.99,
        "membership_probabilities": [0.99] * count,
        "boundary_probabilities": [0.99] * count,
    }


def _candidate_model_smoke(tensor_records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = next((record for record in tensor_records if record["face_embeddings"]), None)
    if usable is None:
        return {"passed": False, "reason": "no_candidate_tensor"}
    dimensions = CandidateModelDimensions(
        face_embedding_dim=len(GRAPH_V2_NODE_FEATURE_SCHEMA),
        edge_embedding_dim=len(GRAPH_V2_EDGE_FEATURE_SCHEMA),
        engineered_feature_dim=len(ENGINEERED_FEATURE_SCHEMA),
        hidden_dim=64,
        semantic_role_count=8,
    )
    model = CandidateTopologyClassifier(dimensions)
    face_tensor = torch.tensor(usable["face_embeddings"], dtype=torch.float32)
    internal = _edge_tensor(usable["internal_edge_embeddings"], dimensions.edge_embedding_dim)
    boundary = _edge_tensor(usable["boundary_edge_embeddings"], dimensions.edge_embedding_dim)
    engineered = torch.tensor(usable["engineered_features"], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(face_tensor, internal, boundary, engineered)
    finite = all(torch.isfinite(value).all().item() for value in outputs.values())
    return {
        "passed": bool(finite and outputs["macro_logits"].numel() == 18),
        "parameter_count": model.parameter_count(),
        "metadata": model.metadata(),
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "all_outputs_finite": finite,
    }


def _edge_tensor(values: list[list[float]], dimension: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32) if values else torch.empty((0, dimension), dtype=torch.float32)


def _summary(payload: dict[str, Any]) -> str:
    metrics = payload["metrics_summary"]
    return "\n".join([
        "# P1-B 阶段 4 自动闭环验证",
        "",
        f"- 自动结论：{'PASS' if payload['automatic_pass'] else 'FAIL'}",
        f"- 合成 canonical 样本：{payload['sample_count']}",
        f"- same-instance edge F1：min={metrics['same_instance_edge_f1_min']:.6f}, mean={metrics['same_instance_edge_f1_mean']:.6f}",
        f"- candidate proposal oracle recall：min={metrics['proposal_oracle_recall_min']:.6f}, mean={metrics['proposal_oracle_recall_mean']:.6f}",
        f"- topology/verifier oracle structural exact object ratio：min={metrics['oracle_structural_exact_object_ratio_min']:.6f}, mean={metrics['oracle_structural_exact_object_ratio_mean']:.6f}",
        f"- 跨 provisional macro same-instance 候选：{metrics['cross_provisional_macro_candidate_count']}",
        f"- hard negatives：{metrics['hard_negative_count']}",
        f"- 候选张量：{metrics['candidate_tensor_count']}",
        f"- 候选来源并集：{', '.join(metrics['proposal_source_union'])}",
        "- 阈值选择接口严格限制为 Synthetic-Val；验证前已中和 graph 中的 truth macro/instance 字段。",
        "- 本报告验证实现闭环与 oracle 上界，不代表候选模型已经训练；训练实测指标将在修正数据集与消融阶段给出。",
        "",
    ])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
