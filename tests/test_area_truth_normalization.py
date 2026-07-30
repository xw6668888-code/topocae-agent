import unittest

from utils.area_truth_normalization import normalize_truth_macro


class AreaTruthNormalizationTest(unittest.TestCase):
    def test_common_manual_truth_aliases_are_normalized(self):
        self.assertEqual("through_hole_round", normalize_truth_macro("through_hole"))
        self.assertEqual("stepped_hole_round", normalize_truth_macro("stepped_hole"))
        self.assertEqual("pocket_rect", normalize_truth_macro("poctet_rect"))
        self.assertEqual("cylinder_boss", normalize_truth_macro("cylindrical_boss"))
        self.assertEqual("rectangular_boss", normalize_truth_macro("rectangle_boss"))

    def test_part6_two_face_counterbore_truth_is_corrected_to_countersink(self):
        self.assertEqual(
            "countersink_hole",
            normalize_truth_macro(
                "counterbore_hole",
                sample_id="part6",
                area=32.14,
                face_count=2,
                truth_source_name="Feature_Comparison_Table.xlsx",
            ),
        )
        self.assertEqual(
            "counterbore_hole",
            normalize_truth_macro(
                "counterbore_hole",
                sample_id="part6",
                area=120.0,
                face_count=3,
                truth_source_name="Feature_Comparison_Table.xlsx",
            ),
        )

    def test_part3_torus_transition_truth_is_corrected_to_fillet(self):
        self.assertEqual(
            "fillet",
            normalize_truth_macro(
                "chamfer",
                sample_id="part3",
                area=1790.87,
                face_count=1,
                truth_source_name="Feature_Comparison_Table.xlsx",
            ),
        )
        self.assertEqual(
            "chamfer",
            normalize_truth_macro(
                "chamfer",
                sample_id="part3",
                area=1096.02,
                face_count=1,
                truth_source_name="Feature_Comparison_Table.xlsx",
            ),
        )


if __name__ == "__main__":
    unittest.main()
