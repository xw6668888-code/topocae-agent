from __future__ import annotations

import unittest

import torch

from models.p1_b_candidate_topology_classifier import (
    CandidateModelDimensions,
    CandidateTopologyClassifier,
    candidate_multitask_loss,
)
from p1_b_general_recognition_pipeline import run_p1_b_general_recognition
from utils.p1_b_candidate_features import ENGINEERED_FEATURE_SCHEMA, MACRO_CLASSES, build_candidate_tensor_record
from utils.p1_b_candidate_hard_negatives import generate_candidate_hard_negatives
from utils.p1_b_candidate_proposer import evaluate_proposal_oracle_recall, generate_candidate_proposals
from utils.p1_b_global_decode import decode_feature_objects
from utils.p1_b_instance_assembly import (
    assemble_same_instance_candidates,
    evaluate_same_instance_edges,
    select_global_same_instance_threshold,
    symmetrize_directed_probabilities,
)
from utils.p1_b_topology_verifier import select_global_candidate_thresholds, verify_candidate_prediction
from utils.training_export_v2_1 import GRAPH_V2_EDGE_FEATURE_SCHEMA, GRAPH_V2_NODE_FEATURE_SCHEMA


class P1BGeneralRecognitionPipelineTests(unittest.TestCase):
    def test_bidirectional_mean_and_cross_macro_same_instance_assembly(self) -> None:
        graph, prediction = _graph_and_prediction()
        selection = select_global_same_instance_threshold(
            [
                {"split": "synthetic_val", "probability": 0.98, "label": 1},
                {"split": "synthetic_val", "probability": 0.20, "label": 0},
            ],
            allowed_thresholds=[0.5, 0.9],
        )
        symmetrized = symmetrize_directed_probabilities(graph, prediction)
        self.assertAlmostEqual(symmetrized[(0, 1)]["probability"], 0.98)
        self.assertEqual(symmetrized[(0, 1)]["direction_count"], 2)
        assembled = assemble_same_instance_candidates(
            graph,
            prediction,
            global_threshold=selection["global_threshold"],
            threshold_selection_sha256=selection["selection_sha256"],
        )
        candidate = next(record for record in assembled["candidates"] if record["face_indices"] == [0, 1])
        self.assertTrue(candidate["cross_provisional_macro"])
        self.assertNotIn(2, candidate["face_indices"])
        metrics = evaluate_same_instance_edges(graph, prediction, global_threshold=selection["global_threshold"])
        self.assertEqual(metrics["f1"], 1.0)

    def test_threshold_selectors_reject_non_synthetic_val(self) -> None:
        with self.assertRaisesRegex(ValueError, "synthetic_val"):
            select_global_same_instance_threshold([{"split": "development_test", "probability": 0.9, "label": 1}])
        with self.assertRaisesRegex(ValueError, "synthetic_val"):
            select_global_candidate_thresholds([
                {"split": "real_dev", "macro_confidence": 0.9, "validity_score": 0.9, "accept_label": 1}
            ])

    def test_candidate_sources_features_and_oracle_recall(self) -> None:
        graph, prediction = _graph_and_prediction()
        selection = _same_instance_selection()
        assembly = assemble_same_instance_candidates(
            graph,
            prediction,
            global_threshold=selection["global_threshold"],
            threshold_selection_sha256=selection["selection_sha256"],
        )
        proposals = generate_candidate_proposals(graph, prediction, assembly)
        sources = set(proposals["source_counts"])
        self.assertIn("same_instance_strong_component", sources)
        self.assertIn("coaxial_cylinder_cone_chain", sources)
        self.assertIn("loop_material_side_guided", sources)
        tensor = build_candidate_tensor_record(graph, prediction, proposals["proposals"][0])
        self.assertEqual(len(tensor["engineered_features"]), len(ENGINEERED_FEATURE_SCHEMA))
        self.assertTrue(tensor["trace"]["all_features_finite"])
        self.assertLessEqual(tensor["trace"]["max_abs"], 32.0)
        oracle = evaluate_proposal_oracle_recall(
            proposals["proposals"],
            [{"instance_id": 1, "macro_feature": "countersink_hole", "face_indices": [0, 1]}],
        )
        self.assertEqual(oracle["oracle_recall"], 1.0)

    def test_candidate_model_has_all_required_heads_and_multitask_loss(self) -> None:
        dimensions = CandidateModelDimensions(
            face_embedding_dim=len(GRAPH_V2_NODE_FEATURE_SCHEMA),
            edge_embedding_dim=len(GRAPH_V2_EDGE_FEATURE_SCHEMA),
            engineered_feature_dim=len(ENGINEERED_FEATURE_SCHEMA),
            hidden_dim=32,
            semantic_role_count=5,
        )
        model = CandidateTopologyClassifier(dimensions)
        outputs = model(
            torch.zeros((3, dimensions.face_embedding_dim)),
            torch.zeros((2, dimensions.edge_embedding_dim)),
            torch.zeros((4, dimensions.edge_embedding_dim)),
            torch.zeros((dimensions.engineered_feature_dim,)),
        )
        self.assertEqual(tuple(outputs["macro_logits"].shape), (18,))
        self.assertEqual(tuple(outputs["membership_logits"].shape), (3,))
        self.assertEqual(tuple(outputs["boundary_logits"].shape), (3,))
        self.assertEqual(tuple(outputs["semantic_role_logits"].shape), (3, 5))
        losses = candidate_multitask_loss(
            outputs,
            {
                "validity": torch.tensor(1.0),
                "macro_class": torch.tensor(4),
                "membership": torch.ones(3),
                "boundary": torch.tensor([0.0, 1.0, 0.0]),
                "semantic_role": torch.tensor([0, 1, 2]),
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(model.parameter_count(), 0)

    def test_verifier_only_masks_vetoes_corrects_or_abstains(self) -> None:
        graph, _ = _graph_and_prediction()
        thresholds = _candidate_threshold_selection()
        proposal = {"candidate_id": "p", "face_indices": [0, 1]}
        probabilities = [0.001] * len(MACRO_CLASSES)
        probabilities[MACRO_CLASSES.index("rib")] = 0.90
        probabilities[MACRO_CLASSES.index("countersink_hole")] = 0.083
        prediction = {
            "macro_classes": MACRO_CLASSES,
            "macro_probabilities": probabilities,
            "macro_logits": probabilities,
            "validity_score": 0.95,
            "membership_probabilities": [0.95, 0.95],
            "boundary_probabilities": [0.1, 0.9],
        }
        verified = verify_candidate_prediction(graph, proposal, prediction, thresholds=thresholds)
        self.assertEqual(verified["model_unmasked_top1"], "rib")
        self.assertEqual(verified["masked_top1"], "countersink_hole")
        self.assertFalse(verified["verifier_contract"]["hard_class_rewrite"])
        low_validity = dict(prediction, validity_score=0.01)
        abstained = verify_candidate_prediction(graph, proposal, low_validity, thresholds=thresholds)
        self.assertTrue(abstained["abstain"])

    def test_global_decode_and_single_pipeline_have_traceable_objects(self) -> None:
        graph, prediction = _graph_and_prediction()
        decoded = decode_feature_objects(
            graph,
            [
                _verified("main", "countersink_hole", [0, 1], 0.95),
                _verified("lower_overlap", "counterbore_hole", [1], 0.70),
                _verified("transition", "chamfer", [1], 0.80),
            ],
        )
        macros = {record["macro_feature"] for record in decoded["feature_objects"]}
        self.assertIn("countersink_hole", macros)
        self.assertIn("chamfer", macros)
        self.assertEqual(decoded["global_conflict_reject_count"], 1)

        pipeline = run_p1_b_general_recognition(
            graph,
            prediction,
            _oracle_candidate_predictor,
            same_instance_thresholds=_same_instance_selection(),
            candidate_thresholds=_candidate_threshold_selection(),
        )
        self.assertFalse(pipeline["pipeline_manifest"]["legacy_same_macro_component_fallback"])
        self.assertFalse(pipeline["pipeline_manifest"]["truth_used_during_inference"])
        self.assertEqual(
            pipeline["decoded_feature_objects"]["grouping_method"],
            "same_instance_plus_candidate_topology_global_decode",
        )
        self.assertTrue(pipeline["decoded_feature_objects"]["feature_objects"])

    def test_hard_negatives_use_generic_topology_relations(self) -> None:
        graph, _ = _graph_and_prediction()
        negatives = generate_candidate_hard_negatives(
            graph,
            [
                {"instance_id": 1, "macro_feature": "countersink_hole", "face_indices": [0, 1]},
                {"instance_id": 2, "macro_feature": "rib", "face_indices": [2, 3]},
            ],
        )
        kinds = {record["negative_kind"] for record in negatives["records"]}
        self.assertIn("missing_truth_face_subset", kinds)
        self.assertIn("adjacent_instance_merge", kinds)
        self.assertFalse(negatives["contract"]["uses_sample_names"])
        self.assertFalse(negatives["contract"]["uses_real_part_face_ids"])


def _same_instance_selection() -> dict:
    return select_global_same_instance_threshold(
        [
            {"split": "synthetic_val", "probability": 0.98, "label": 1},
            {"split": "synthetic_val", "probability": 0.20, "label": 0},
        ],
        allowed_thresholds=[0.5],
    )


def _candidate_threshold_selection() -> dict:
    return select_global_candidate_thresholds(
        [
            {
                "split": "synthetic_val",
                "macro_confidence": 0.90,
                "validity_score": 0.90,
                "accept_label": 1,
                "membership_probabilities": [0.9, 0.1],
                "membership_labels": [1, 0],
                "boundary_probabilities": [0.1, 0.9],
                "boundary_labels": [0, 1],
            },
            {
                "split": "synthetic_val",
                "macro_confidence": 0.20,
                "validity_score": 0.20,
                "accept_label": 0,
            },
        ],
        threshold_grid=[0.5],
    )


def _oracle_candidate_predictor(tensor_record: dict, proposal: dict) -> dict:
    probabilities = [0.001] * len(MACRO_CLASSES)
    probabilities[MACRO_CLASSES.index("countersink_hole")] = 0.983
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    return {
        "macro_classes": MACRO_CLASSES,
        "macro_probabilities": probabilities,
        "macro_logits": probabilities,
        "validity_score": 0.99,
        "membership_probabilities": [0.99] * len(proposal["face_indices"]),
        "boundary_probabilities": [0.6] * len(proposal["face_indices"]),
        "semantic_role_probabilities": [[1.0] for _ in proposal["face_indices"]],
    }


def _verified(candidate_id: str, macro: str, faces: list[int], confidence: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "status": "accepted",
        "predicted_macro": macro,
        "refined_face_indices": faces,
        "boundary_face_indices": [],
        "original_face_indices": faces,
        "final_confidence": confidence,
        "macro_confidence_before_correction": confidence,
        "candidate_validity_score": 1.0,
        "topology_ambiguity_penalty": 1.0,
        "topology_evidence": {},
        "class_mask": {},
        "threshold_selection_sha256": "SYNTHETIC_VAL_ONLY",
    }


def _graph_and_prediction() -> tuple[dict, dict]:
    nodes = [
        _node(0, "cylinder", "countersink_hole", 1, inner_loops=0),
        _node(1, "cone", "counterbore_hole", 1, inner_loops=0),
        _node(2, "plane", "stock", 0, inner_loops=1),
        _node(3, "plane", "rib", 2, inner_loops=0),
    ]
    directed_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    edge_attributes = [
        _edge_attr("concave", "circle", "inner", "inner_hole_wall"),
        _edge_attr("concave", "circle", "inner", "inner_hole_wall"),
        _edge_attr("concave", "circle", "inner", "inner_hole_wall"),
        _edge_attr("concave", "circle", "inner", "inner_hole_wall"),
        _edge_attr("convex", "line", "outer", "outer_cylinder_wall"),
        _edge_attr("convex", "line", "outer", "outer_cylinder_wall"),
    ]
    graph = {
        "schema_version": "topocae_graph_v2_relative_1_1",
        "sample_id": "synthetic_unit",
        "nodes": nodes,
        "edges": [list(pair) for pair in directed_pairs],
        "edge_attributes": edge_attributes,
        "part_metadata_v2": {
            "normalization": {"part_total_area": 100.0, "part_characteristic_length": 10.0, "part_bbox_diagonal": 12.0}
        },
        "face_to_canonical_patch": {str(index): index for index in range(4)},
        "canonical_patches": [{"patch_id": index, "face_indices": [index]} for index in range(4)],
    }
    prediction = {
        "sample_id": "synthetic_unit",
        "faces": [
            {"idx": 0, "macro_feature": "countersink_hole"},
            {"idx": 1, "macro_feature": "counterbore_hole"},
            {"idx": 2, "macro_feature": "stock"},
            {"idx": 3, "macro_feature": "rib"},
        ],
        "edge_same_instance_probabilities": [0.99, 0.97, 0.99, 0.99, 0.20, 0.20],
    }
    return graph, prediction


def _node(index: int, primitive: str, macro: str, instance_id: int, inner_loops: int) -> dict:
    axis = primitive in {"cylinder", "cone"}
    geometry = {
        "surface_family": primitive,
        "axis_direction": [0.0, 0.0, 1.0],
        "axis_location": [0.0, 0.0, 0.0],
    } if axis else {"surface_family": primitive}
    return {
        "idx": index,
        "primitive_surface": primitive,
        "primitive_surface_id": {"plane": 0, "cylinder": 1, "cone": 2}.get(primitive, 10),
        "macro_feature": macro,
        "instance_id": instance_id,
        "attributes": {
            "physical": {"area": 10.0},
            "specific": {"radius": 2.0 if axis else 0.0},
            "raw_physical_v2": {"area": 10.0, "radius": 2.0 if axis else 0.0},
            "topology_v2": {
                "wire_count": 1 + inner_loops,
                "outer_loop_count": 1,
                "inner_loop_count": inner_loops,
                "outer_loop_edge_count": 4,
                "inner_loop_edge_count": inner_loops,
            },
            "geometry_v2": geometry,
            "recognition_features_v2": {
                "face_area_over_part_total_area": 0.1,
                "face_area_over_part_scale_sq": 0.1,
                "radius_over_part_scale": 0.2 if axis else 0.0,
                "pca_long_over_part_scale": 0.5,
                "pca_middle_over_part_scale": 0.2,
                "pca_short_over_part_scale": 0.02,
                "local_extent_long_middle_ratio": 2.5,
                "projected_aspect_ratio": 2.5,
                "uv_aspect_ratio": 2.5,
                "vertex_count": 4.0,
                "edge_count": 4.0,
                "neighbor_count": 2.0,
                "wire_count": 1.0 + inner_loops,
                "outer_loop_count": 1.0,
                "inner_loop_count": float(inner_loops),
                "outer_loop_edge_count": 4.0,
                "inner_loop_edge_count": float(inner_loops),
            },
        },
    }


def _edge_attr(convexity: str, curve: str, loop_role: str, wall_role: str) -> dict:
    return {
        "length": 2.0,
        "circle_radius": 2.0 if curve == "circle" else 0.0,
        "curve_type": curve,
        "convexity": convexity,
        "dihedral_angle": 1.0,
        "adjacent_plane_loop_role": loop_role,
        "cylindrical_wall_role_hint": wall_role,
        "recognition_features_v2": {
            "edge_length_over_part_scale": 0.2,
            "circle_radius_over_part_scale": 0.2 if curve == "circle" else 0.0,
            "dihedral_angle_over_pi": 0.318,
            "material_side_inner_evidence": 1.0 if loop_role == "inner" else 0.0,
            "material_side_outer_evidence": 1.0 if loop_role == "outer" else 0.0,
            "material_side_ambiguous_evidence": 0.0,
            "log1p_edge_length_over_part_scale": 0.182,
            "log1p_circle_radius_over_part_scale": 0.182 if curve == "circle" else 0.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
