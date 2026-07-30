import json
from decimal import Decimal
import unittest
from unittest.mock import patch

from utils.instance_candidate_postprocess import apply_instance_candidate_postprocess


def graph_node(idx, primitive="plane", area=10.0):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "attributes": {
            "physical": {
                "area": area,
                "center_of_mass": [float(idx), 0.0, 0.0],
            },
            "specific": {"normal": [0.0, 0.0, 1.0]},
            "shape_descriptors": {},
        },
    }


def prediction_face(idx, macro="web", macro_id=21):
    return {
        "idx": idx,
        "macro_feature_id": macro_id,
        "macro_feature": macro,
        "semantic_subtype_id": 0,
        "semantic_subtype": "stock_face",
    }


def graph(nodes, edges=None):
    return {
        "nodes": nodes,
        "edges": edges or [],
        "edge_attributes": [{} for _ in edges or []],
        "feature_sets": [
            {
                "macro_feature": "truth_label_must_not_be_used",
                "instance_id": 123,
                "node_indices": [node["idx"] for node in nodes],
            }
        ],
    }


class ConstantEstimator:
    classes_ = [0, 7]

    def __init__(self, label_id=7, confidence=0.9):
        self.label_id = label_id
        self.confidence = confidence

    def predict(self, features):
        return [self.label_id for _ in features]

    def predict_proba(self, features):
        if self.label_id == 0:
            return [[self.confidence, 1.0 - self.confidence] for _ in features]
        return [[1.0 - self.confidence, self.confidence] for _ in features]



class MultiElementArrayLike:
    def item(self):
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    def tolist(self):
        return [1, 2]

class FinalEstimatorOnlyClasses:
    classes_ = [0, 7]


class PipelineLikeEstimatorWithoutTopLevelClasses:
    steps = [("classifier", FinalEstimatorOnlyClasses())]
    named_steps = {"classifier": steps[-1][1]}

    def predict(self, features):
        return [7 for _ in features]

    def predict_proba(self, features):
        return [[0.80, 0.20] for _ in features]


class AreaConfidenceEstimator:
    classes_ = [0, 7]

    def predict(self, features):
        return [7 for _ in features]

    def predict_proba(self, features):
        probabilities = []
        for feature_vector in features:
            candidate_area = feature_vector[0]
            confidence = 0.95 if candidate_area < 10.0 else 0.70
            probabilities.append([1.0 - confidence, confidence])
        return probabilities


class InstanceCandidatePostprocessTest(unittest.TestCase):
    def test_high_confidence_candidate_remaps_faces(self):
        graph_data = graph(
            [graph_node(0), graph_node(1), graph_node(2)],
            edges=[[0, 1]],
        )
        prediction = {
            "faces": [
                prediction_face(0, "web", 21),
                prediction_face(1, "web", 21),
                prediction_face(2, "stock", 0),
            ],
            "edge_same_instance": [0.90],
        }
        bundle = {
            "estimator": ConstantEstimator(label_id=7, confidence=0.92),
            "metadata": {
                "feature_schema": ["face_count"],
                "id_to_label": {"0": "reject", "7": "rib"},
            },
        }

        corrected = apply_instance_candidate_postprocess(prediction, graph_data, bundle)

        self.assertEqual([corrected["faces"][idx]["macro_feature"] for idx in [0, 1]], ["rib", "rib"])
        self.assertEqual(corrected["faces"][0]["raw_macro_feature"], "web")
        self.assertEqual(corrected["instance_candidate_postprocess"]["candidate_count"], 1)
        self.assertEqual(corrected["instance_candidate_postprocess"]["accepted_count"], 1)
        self.assertEqual(corrected["instance_candidate_postprocess"]["rejected_count"], 0)
        record = corrected["instance_candidate_postprocess"]["records"][0]
        self.assertEqual(record["action"], "accepted_remap")
        self.assertEqual(record["predicted_label"], "rib")
        self.assertEqual(record["classification"]["label"], "rib")
        self.assertIn("features", record)

    def test_low_confidence_or_reject_candidate_does_not_remap(self):
        graph_data = graph([graph_node(0), graph_node(1)], edges=[[0, 1]])
        prediction = {
            "faces": [prediction_face(0, "web", 21), prediction_face(1, "web", 21)],
            "edge_same_instance": [0.90],
        }

        cases = [
            (ConstantEstimator(label_id=7, confidence=0.40), "skipped_low_confidence"),
            (ConstantEstimator(label_id=0, confidence=0.95), "rejected_label"),
        ]
        for estimator, expected_action in cases:
            with self.subTest(expected_action=expected_action):
                bundle = {
                    "estimator": estimator,
                    "metadata": {
                        "feature_schema": ["face_count"],
                        "id_to_label": {"0": "reject", "7": "rib"},
                    },
                }

                corrected = apply_instance_candidate_postprocess(prediction, graph_data, bundle)

                self.assertEqual([face["macro_feature"] for face in corrected["faces"]], ["web", "web"])
                self.assertEqual(corrected["instance_candidate_postprocess"]["accepted_count"], 0)
                self.assertEqual(corrected["instance_candidate_postprocess"]["rejected_count"], 1)
                self.assertEqual(corrected["instance_candidate_postprocess"]["records"][0]["action"], expected_action)

    def test_overlapping_accepted_candidates_keep_highest_confidence(self):
        graph_data = graph(
            [graph_node(0, area=10.0), graph_node(1, area=1.0), graph_node(2, area=2.0)],
            edges=[[0, 1], [1, 2]],
        )
        prediction = {
            "faces": [
                prediction_face(0, "web", 21),
                prediction_face(1, "web", 21),
                prediction_face(2, "web", 21),
            ],
            "edge_same_instance": [0.90, 0.90],
        }
        proposals = [
            {"source": "test_low", "proposal_macro": "web", "face_indices": [0, 1]},
            {"source": "test_high", "proposal_macro": "web", "face_indices": [1, 2]},
        ]
        bundle = {
            "estimator": AreaConfidenceEstimator(),
            "metadata": {
                "feature_schema": ["candidate_area"],
                "id_to_label": {"0": "reject", "7": "rib"},
            },
        }

        with patch("utils.instance_candidate_postprocess.assemble_instance_candidates", return_value=proposals):
            corrected = apply_instance_candidate_postprocess(prediction, graph_data, bundle)

        self.assertEqual([face["macro_feature"] for face in corrected["faces"]], ["web", "rib", "rib"])
        self.assertEqual(corrected["instance_candidate_postprocess"]["accepted_count"], 1)
        self.assertEqual(corrected["instance_candidate_postprocess"]["rejected_count"], 1)
        records_by_source = {
            record["source"]: record
            for record in corrected["instance_candidate_postprocess"]["records"]
        }
        self.assertEqual(records_by_source["test_high"]["action"], "accepted_remap")
        self.assertEqual(records_by_source["test_low"]["action"], "skipped_overlap")
        self.assertEqual(records_by_source["test_low"]["overlap_face_indices"], [1])



    def test_pipeline_final_estimator_classes_are_used_for_confidence(self):
        graph_data = graph([graph_node(0), graph_node(1)], edges=[[0, 1]])
        prediction = {
            "faces": [prediction_face(0, "web", 21), prediction_face(1, "web", 21)],
            "edge_same_instance": [0.90],
        }
        bundle = {
            "estimator": PipelineLikeEstimatorWithoutTopLevelClasses(),
            "metadata": {
                "feature_schema": ["face_count"],
                "id_to_label": {"0": "reject", "7": "rib"},
            },
        }

        corrected = apply_instance_candidate_postprocess(
            prediction,
            graph_data,
            bundle,
            min_confidence=0.5,
        )

        self.assertEqual([face["macro_feature"] for face in corrected["faces"]], ["web", "web"])
        record = corrected["instance_candidate_postprocess"]["records"][0]
        self.assertEqual(record["action"], "skipped_low_confidence")
        self.assertAlmostEqual(record["confidence"], 0.20)

    def test_diagnostic_records_are_json_serializable(self):
        graph_data = graph([graph_node(0), graph_node(1)], edges=[[0, 1]])
        prediction = {
            "faces": [prediction_face(0, "web", 21), prediction_face(1, "web", 21)],
            "edge_same_instance": [0.90],
        }
        bundle = {
            "estimator": ConstantEstimator(label_id=7, confidence=0.92),
            "metadata": {
                "feature_schema": ["face_count"],
                "id_to_label": {"0": "reject", "7": "rib"},
            },
        }

        with patch(
            "utils.instance_candidate_postprocess.extract_instance_candidate_features",
            return_value={"face_count": Decimal("2.0"), "decimal_metric": Decimal("1.25"), "array_metric": MultiElementArrayLike()},
        ):
            corrected = apply_instance_candidate_postprocess(prediction, graph_data, bundle)

        json.dumps(corrected["instance_candidate_postprocess"])
        features = corrected["instance_candidate_postprocess"]["records"][0]["features"]
        self.assertEqual(features["face_count"], 2.0)
        self.assertEqual(features["decimal_metric"], 1.25)
        self.assertEqual(features["array_metric"], [1, 2])

    def test_pocket_circular_candidate_uses_default_macro_id_when_absent_from_prediction(self):
        graph_data = graph([graph_node(0, "cylinder"), graph_node(1, "plane")], edges=[[0, 1]])
        prediction = {
            "faces": [prediction_face(0, "web", 21), prediction_face(1, "web", 21)],
            "edge_same_instance": [0.90],
        }
        bundle = {
            "estimator": ConstantEstimator(label_id=12, confidence=0.92),
            "metadata": {
                "feature_schema": ["face_count"],
                "id_to_label": {"0": "reject", "12": "pocket_circular"},
            },
        }

        corrected = apply_instance_candidate_postprocess(prediction, graph_data, bundle)

        self.assertEqual([face["macro_feature"] for face in corrected["faces"]], ["pocket_circular", "pocket_circular"])
        self.assertEqual([face["macro_feature_id"] for face in corrected["faces"]], [12, 12])


if __name__ == "__main__":
    unittest.main()



