import unittest

from evaluate_gnn_predictions import aggregate_reports, evaluate_prediction


class PredictionEvaluationTest(unittest.TestCase):
    def test_evaluate_prediction_reports_macro_and_semantic_face_metrics(self):
        prediction = {
            "sample_id": "mock",
            "faces": [
                {
                    "idx": 0,
                    "macro_feature_id": 1,
                    "macro_feature": "through_hole_round",
                    "semantic_subtype_id": 5,
                    "semantic_subtype": "through_hole_cylinder",
                },
                {
                    "idx": 1,
                    "macro_feature_id": 2,
                    "macro_feature": "blind_hole_round",
                    "semantic_subtype_id": 6,
                    "semantic_subtype": "blind_hole_cylinder",
                },
            ],
        }
        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "macro_feature_id": 1,
                    "macro_feature": "through_hole_round",
                    "semantic_label": 5,
                },
                {
                    "idx": 1,
                    "macro_feature_id": 3,
                    "macro_feature": "stepped_hole_round",
                    "semantic_label": 1,
                },
            ],
            "label_maps": {
                "macro_feature": {
                    "id_to_key": {
                        "1": "through_hole_round",
                        "2": "blind_hole_round",
                        "3": "stepped_hole_round",
                    }
                },
                "face_semantic_subtype": {
                    "id_to_key": {
                        "1": "stepped_hole_large_hole_cylinder",
                        "5": "through_hole_cylinder",
                        "6": "blind_hole_cylinder",
                    }
                },
            },
        }

        report = evaluate_prediction(prediction, graph)

        self.assertEqual(report["face_count"], 2)
        self.assertEqual(report["macro"]["accuracy"], 0.5)
        self.assertEqual(report["semantic"]["accuracy"], 0.5)
        self.assertEqual(report["macro"]["per_class"]["through_hole_round"]["tp"], 1)
        self.assertEqual(report["macro"]["per_class"]["stepped_hole_round"]["fn"], 1)

    def test_aggregate_reports_averages_sample_metrics(self):
        aggregate = aggregate_reports([
            {
                "face_count": 2,
                "macro": {"accuracy": 0.5, "macro_f1": 0.25},
                "semantic": {"accuracy": 0.0, "macro_f1": 0.0},
            },
            {
                "face_count": 2,
                "macro": {"accuracy": 1.0, "macro_f1": 1.0},
                "semantic": {"accuracy": 1.0, "macro_f1": 1.0},
            },
        ])

        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["face_count"], 4)
        self.assertEqual(aggregate["macro"]["mean_sample_accuracy"], 0.75)
        self.assertEqual(aggregate["semantic"]["mean_sample_macro_f1"], 0.5)
        self.assertEqual(aggregate["macro"]["global_accuracy"], 0.0)

    def test_aggregate_reports_computes_global_per_class_metrics(self):
        aggregate = aggregate_reports([
            {
                "face_count": 2,
                "macro": {
                    "accuracy": 0.5,
                    "macro_f1": 0.25,
                    "per_class": {
                        "stock": {"id": 0, "tp": 1, "fp": 0, "fn": 0},
                        "blind_hole_round": {"id": 2, "tp": 0, "fp": 1, "fn": 0},
                        "through_hole_round": {"id": 1, "tp": 0, "fp": 0, "fn": 1},
                    },
                },
                "semantic": {
                    "accuracy": 0.5,
                    "macro_f1": 0.25,
                    "per_class": {
                        "stock_face": {"id": 0, "tp": 1, "fp": 0, "fn": 0},
                        "through_hole_cylinder": {"id": 5, "tp": 0, "fp": 1, "fn": 1},
                    },
                },
            }
        ])

        self.assertEqual(aggregate["macro"]["global_accuracy"], 0.5)
        self.assertAlmostEqual(aggregate["macro"]["per_class"]["stock"]["f1"], 1.0)
        self.assertEqual(aggregate["macro"]["per_class"]["through_hole_round"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
