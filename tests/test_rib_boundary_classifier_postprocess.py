import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.rib_boundary_classifier_postprocess import apply_rib_boundary_classifier_postprocess


class AreaRatioEstimator:
    classes_ = [0, 1]

    def __init__(self, feature_schema):
        self.feature_schema = feature_schema
        self.ratio_index = feature_schema.index("candidate_to_seed_area_ratio")

    def predict(self, features):
        return [0 if row[self.ratio_index] > 5.0 else 1 for row in features]

    def predict_proba(self, features):
        rows = []
        for row in features:
            if row[self.ratio_index] > 5.0:
                rows.append([0.95, 0.05])
            else:
                rows.append([0.05, 0.95])
        return rows


class RibBoundaryClassifierPostprocessTest(unittest.TestCase):
    def bundle(self):
        feature_schema = ["candidate_to_seed_area_ratio"]
        return {
            "estimator": AreaRatioEstimator(feature_schema),
            "metadata": {
                "feature_schema": feature_schema,
                "id_to_label": {"0": "not_rib_member", "1": "rib_member"},
            },
        }

    def test_prunes_large_stock_face_from_rib_component(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 500.0}),
            ],
            [
                (0, 1, "line", "convex", 10.0),
                (1, 0, "line", "convex", 10.0),
                (0, 2, "line", "convex", 20.0),
                (2, 0, "line", "convex", 20.0),
                (1, 2, "line", "convex", 20.0),
                (2, 1, "line", "convex", 20.0),
            ],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=20, macro="rib"),
            ],
        }

        corrected = apply_rib_boundary_classifier_postprocess(
            prediction,
            graph_data,
            self.bundle(),
            min_confidence=0.8,
        )

        self.assertEqual(corrected["faces"][2]["macro_feature"], "stock")
        self.assertEqual(corrected["rib_boundary_classifier_postprocess"]["pruned_count"], 1)

    def test_expands_adjacent_missing_rib_face(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0}),
            ],
            [
                (0, 1, "line", "convex", 10.0),
                (1, 0, "line", "convex", 10.0),
                (0, 2, "line", "convex", 10.0),
                (2, 0, "line", "convex", 10.0),
                (1, 2, "line", "convex", 10.0),
                (2, 1, "line", "convex", 10.0),
            ],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_rib_boundary_classifier_postprocess(
            prediction,
            graph_data,
            self.bundle(),
            min_confidence=0.8,
        )

        self.assertEqual(corrected["faces"][2]["macro_feature"], "rib")
        self.assertEqual(corrected["rib_boundary_classifier_postprocess"]["expanded_count"], 1)



    def test_logs_low_confidence_boundary_decision_without_action(self):
        class LowConfidenceRibEstimator(AreaRatioEstimator):
            def predict(self, features):
                return [1 for _ in features]

            def predict_proba(self, features):
                return [[0.45, 0.55] for _ in features]

        feature_schema = ["candidate_to_seed_area_ratio"]
        bundle = {
            "estimator": LowConfidenceRibEstimator(feature_schema),
            "metadata": {
                "feature_schema": feature_schema,
                "id_to_label": {"0": "not_rib_member", "1": "rib_member"},
            },
        }
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0}),
            ],
            [
                (0, 1, "line", "convex", 10.0),
                (1, 0, "line", "convex", 10.0),
                (0, 2, "line", "convex", 10.0),
                (2, 0, "line", "convex", 10.0),
                (1, 2, "line", "convex", 10.0),
                (2, 1, "line", "convex", 10.0),
            ],
        )
        prediction = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_rib_boundary_classifier_postprocess(
            prediction,
            graph_data,
            bundle,
            min_confidence=0.8,
        )

        post = corrected["rib_boundary_classifier_postprocess"]
        self.assertEqual(corrected["faces"][2]["macro_feature"], "stock")
        self.assertEqual(post["expanded_count"], 0)
        self.assertEqual(post["decision_count"], 1)
        self.assertEqual(post["decision_records"][0]["action"], "skipped_low_confidence")
        self.assertEqual(post["decision_records"][0]["candidate_face_idx"], 2)
if __name__ == "__main__":
    unittest.main()


