import unittest

from tests.test_topology_postprocess import edge, face, graph, graph_node
from utils.candidate_classifier_postprocess import apply_candidate_classifier_postprocess


class FakeEstimator:
    classes_ = [0, 7]

    def predict(self, features):
        return [7 for _ in range(len(features))]

    def predict_proba(self, features):
        return [[0.1, 0.9] for _ in range(len(features))]


class RejectEstimator:
    classes_ = [0, 6]

    def predict(self, features):
        return [0 for _ in range(len(features))]

    def predict_proba(self, features):
        return [[0.95, 0.05] for _ in range(len(features))]


class CandidateClassifierPostprocessTest(unittest.TestCase):
    def test_candidate_classifier_can_remap_large_cylinder_candidate_to_slot_o(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, -1, 0], radius=2.5, descriptors={"area": 3416.28}),
                graph_node(1, "plane", normal=[0, 0, 1]),
                graph_node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (0, 1, "line", "convex", 435.0),
                (1, 0, "line", "convex", 435.0),
                (0, 2, "line", "convex", 435.0),
                (2, 0, "line", "convex", 435.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(0, macro_id=25, macro="fillet")],
            "edge_same_instance": [],
        }
        bundle = {
            "estimator": FakeEstimator(),
            "metadata": {
                "feature_schema": ["area_ratio", "proposal_macro_slot_o"],
                "id_to_label": {"0": "reject", "7": "slot_o"},
            },
        }

        corrected = apply_candidate_classifier_postprocess(prediction, graph_data, bundle)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "slot_o")
        self.assertEqual(corrected["faces"][0]["raw_macro_feature"], "fillet")
        self.assertEqual(corrected["candidate_classifier_postprocess"]["accepted_count"], 1)

    def test_rejected_prediction_component_is_demoted_to_stock(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 10000.0}),
                graph_node(1, "plane", normal=[0, -1, 0], descriptors={"area": 800.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 12000.0}),
            ],
            [
                (0, 1, "line", "convex", 167.0),
                (1, 0, "line", "convex", 167.0),
                (1, 2, "line", "convex", 167.0),
                (2, 1, "line", "convex", 167.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=20, macro="rib") for idx in [0, 1, 2]],
            "edge_same_instance": [],
        }
        bundle = {
            "estimator": RejectEstimator(),
            "metadata": {
                "feature_schema": ["area_ratio", "proposal_macro_rib"],
                "id_to_label": {"0": "reject", "6": "rib"},
            },
        }

        corrected = apply_candidate_classifier_postprocess(prediction, graph_data, bundle)

        self.assertEqual([item["macro_feature"] for item in corrected["faces"]], ["stock", "stock", "stock"])
        self.assertEqual(corrected["candidate_classifier_postprocess"]["rejected_prediction_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
