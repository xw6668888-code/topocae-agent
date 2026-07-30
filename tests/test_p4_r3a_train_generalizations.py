from __future__ import annotations

import unittest

from utils.cae_text_evidence import CAETextEvidenceExtractor


class TrainGeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.extractor = CAETextEvidenceExtractor()

    @staticmethod
    def values(result, category):
        return [row["normalized"] for row in result["evidence"] if row["category"] == category]

    def test_local_refinement_and_material_usage_are_action_cues(self):
        result = self.extractor.extract({
            "text": "Use material Alloy-X with E=80 GPa and Poisson ratio 0.31, then locally refine every fillet blend face to 0.6 mm.",
            "language": "en-US",
        })
        self.assertEqual(self.values(result, "operation_cue"), ["material_assignment", "local_mesh_size"])
        self.assertEqual(self.values(result, "material_name"), ["alloy-x"])

    def test_only_quantifier_and_shaped_slots_use_frozen_enums(self):
        cases = (("the only U-shaped slot", "slot_u"), ("the only V-shaped slot", "slot_v"), ("the only O-shaped slot", "slot_o"))
        for phrase, feature in cases:
            with self.subTest(feature=feature):
                result = self.extractor.extract({"text": f"Locally refine {phrase} complete feature region to 1 mm.", "language": "en-US"})
                self.assertIn({"selector": "exactly_one"}, self.values(result, "selector"))
                self.assertIn(feature, self.values(result, "feature_mention"))

    def test_chinese_only_and_local_refinement_are_recognized(self):
        text = "\u5c06\u552f\u4e00\u4e00\u4e2aV \u5f62\u69fd\u7684\u5b8c\u6574\u7279\u5f81\u533a\u57df\u5c40\u90e8\u52a0\u5bc6\u81f3 0.5 mm \u7f51\u683c\u3002"
        result = self.extractor.extract({"text": text, "language": "zh-CN"})
        self.assertIn("local_mesh_size", self.values(result, "operation_cue"))
        self.assertIn({"selector": "exactly_one"}, self.values(result, "selector"))
        self.assertIn("slot_v", self.values(result, "feature_mention"))

    def test_specific_policy_roles_win_over_generic_hole_wall(self):
        cases = (
            ("large-hole wall", "large_hole_wall"),
            ("small-hole wall", "small_hole_wall"),
            ("through-hole wall", "through_hole_wall"),
            ("through small-hole wall", "through_small_hole_wall"),
            ("hole bottom", "hole_bottom"),
            ("step face", "step_face"),
            ("conical countersink face", "countersink_conical_face"),
            ("complete slot inner wall", "complete_slot_inner_wall"),
        )
        for phrase, role in cases:
            with self.subTest(role=role):
                result = self.extractor.extract({"text": f"Locally refine every counterbore hole {phrase} to 0.5 mm.", "language": "en-US"})
                self.assertEqual(self.values(result, "face_role_mention")[0], role)

    def test_opposite_normal_is_distinct_from_surface_normal(self):
        result = self.extractor.extract({
            "text": "Apply 2 MPa pressure opposite to the surface normal to every slot bottom face.",
            "language": "en-US",
        })
        self.assertEqual(self.values(result, "direction"), [{"kind": "opposite_normal"}])


if __name__ == "__main__":
    unittest.main()
