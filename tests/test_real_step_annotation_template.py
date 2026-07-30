import unittest

from generate_real_step_annotation_template import build_annotation_template


class RealStepAnnotationTemplateTest(unittest.TestCase):
    def test_build_annotation_template_copies_predictions_into_editable_ground_truth_fields(self):
        feature_object_doc = {
            "sample_id": "part1",
            "source_step_path": "W:/python/FEA/dataset_generator/test_part/part1.stp",
            "source_graph_path": "W:/python/FEA/dataset_generator/output_real_step_test_part/graphs/part1.json",
            "source_prediction_path": "W:/python/FEA/dataset_generator/output_real_step_test_part/predictions/part1.json",
            "feature_object_count": 1,
            "stock_face_indices": [0, 1, 2],
            "feature_objects": [
                {
                    "feature_object_id": "part1__feature_000",
                    "macro_feature": "blind_hole_round",
                    "face_indices": [3, 4],
                    "selectable_regions": [
                        {
                            "role": "hole_cylindrical_wall",
                            "face_indices": [3],
                            "source_semantic_subtypes": ["blind_hole_cylinder"],
                            "intended_uses": ["fixed_support", "pressure_load"],
                            "is_default_for": ["hole_wall_selection"],
                            "selection_priority": 10,
                        }
                    ],
                    "confidence": {
                        "macro_confidence_mean": 0.91,
                        "semantic_confidence_mean": 0.88,
                    },
                    "fingerprint": {
                        "area_total": 12.3,
                        "area_weighted_centroid": [1.0, 2.0, 3.0],
                        "primitive_surface_histogram": {"cylinder": 1, "plane": 1},
                        "semantic_subtype_histogram": {"blind_hole_cylinder": 1},
                    },
                }
            ],
        }

        template = build_annotation_template([feature_object_doc])

        self.assertEqual("cae_real_step_annotation_template_v1", template["annotation_schema"])
        self.assertEqual(1, template["sample_count"])
        sample = template["samples"][0]
        self.assertEqual("part1", sample["sample_id"])
        self.assertEqual(feature_object_doc["source_step_path"], sample["source_step_path"])
        self.assertEqual([], sample["missing_ground_truth_features"])

        predicted_feature = sample["predicted_features"][0]
        self.assertEqual("unreviewed", predicted_feature["review_status"])
        self.assertTrue(predicted_feature["is_in_scope"])
        self.assertEqual("blind_hole_round", predicted_feature["predicted_macro_feature"])
        self.assertEqual("blind_hole_round", predicted_feature["ground_truth_macro_feature"])
        self.assertEqual([3, 4], predicted_feature["ground_truth_face_indices"])

        ground_truth_region = predicted_feature["ground_truth_selectable_regions"][0]
        self.assertEqual("hole_cylindrical_wall", ground_truth_region["role"])
        self.assertEqual([3], ground_truth_region["face_indices"])
        self.assertEqual("unreviewed", ground_truth_region["review_status"])


if __name__ == "__main__":
    unittest.main()
