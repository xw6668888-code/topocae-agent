import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.rib_candidate_proposer_postprocess import apply_rib_candidate_proposer_postprocess


class RibCandidateEstimator:
    classes_ = [0, 1]

    def __init__(self, feature_schema):
        self.area_ratio_index = feature_schema.index("candidate_area_ratio")

    def predict(self, features):
        return [1 if row[self.area_ratio_index] < 0.2 else 0 for row in features]

    def predict_proba(self, features):
        rows = []
        for row in features:
            if row[self.area_ratio_index] < 0.2:
                rows.append([0.05, 0.95])
            else:
                rows.append([0.95, 0.05])
        return rows


class RibCandidateProposerPostprocessTest(unittest.TestCase):
    def bundle(self):
        feature_schema = ["candidate_area_ratio"]
        return {
            "estimator": RibCandidateEstimator(feature_schema),
            "metadata": {
                "feature_schema": feature_schema,
                "id_to_label": {"0": "not_rib_candidate", "1": "rib_candidate"},
            },
        }

    def test_promotes_stock_like_face_to_rib_candidate(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 500.0, "projected_aspect_ratio": 1.2}),
            ],
            [
                (0, 1, "line", "convex", 12.0),
                (1, 0, "line", "convex", 12.0),
            ],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_rib_candidate_proposer_postprocess(
            prediction,
            graph_data,
            self.bundle(),
            min_confidence=0.8,
        )

        self.assertEqual(corrected["faces"][0]["macro_feature"], "rib")
        self.assertEqual(corrected["faces"][0]["raw_macro_feature"], "stock")
        self.assertEqual(corrected["faces"][1]["macro_feature"], "stock")
        self.assertEqual(corrected["rib_candidate_proposer_postprocess"]["promoted_count"], 1)

    def test_does_not_overwrite_protected_hole_or_slot_predictions(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 500.0, "projected_aspect_ratio": 1.2}),
            ],
            [(0, 1, "line", "convex", 12.0)],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=2, macro="through_hole_round"),
                face(1, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_rib_candidate_proposer_postprocess(
            prediction,
            graph_data,
            self.bundle(),
            min_confidence=0.8,
        )

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["rib_candidate_proposer_postprocess"]["skipped_protected_count"], 1)


    def test_promotes_transition_like_face_only_when_adjacent_to_rib(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "cylinder", axis=[0, 1, 0], descriptors={"area": 5.0, "bbox_aspect_ratio": 5.0}),
                graph_node(2, "cylinder", axis=[0, 1, 0], descriptors={"area": 5.0, "bbox_aspect_ratio": 5.0}),
            ],
            [
                (0, 1, "line", "convex", 12.0),
                (1, 0, "line", "convex", 12.0),
            ],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=25, macro="fillet"),
                face(2, macro_id=25, macro="fillet"),
            ],
        }

        corrected = apply_rib_candidate_proposer_postprocess(
            prediction,
            graph_data,
            self.bundle(),
            min_confidence=0.8,
        )

        self.assertEqual(corrected["faces"][1]["macro_feature"], "rib")
        self.assertEqual(corrected["faces"][2]["macro_feature"], "fillet")
if __name__ == "__main__":
    unittest.main()

