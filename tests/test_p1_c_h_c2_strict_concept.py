from __future__ import annotations

import unittest

from p1_c.all_features_h_c2_v2 import (
    GENERATOR_REVISION,
    STRICT_H_C2_HOST_BY_MACRO,
    build_feature_sample,
)


class StrictConceptHC2Tests(unittest.TestCase):
    def test_rib_and_web_target_only_use_human_approved_strict_hosts(self) -> None:
        for index, macro in enumerate(("rib", "web")):
            with self.subTest(macro=macro):
                result = build_feature_sample(macro, 2026090000 + index)
                self.assertEqual(result.parameters["generator_revision"], GENERATOR_REVISION)
                self.assertEqual(
                    result.parameters["strict_h_c2_host_family"],
                    STRICT_H_C2_HOST_BY_MACRO[macro],
                )
                self.assertEqual(result.parameters["source_engineering_macro"], macro)
                self.assertNotIn("semantic_relabel", result.parameters)
                self.assertTrue(result.cleanup["passed"])
                self.assertEqual({tag.instance_id for tag in result.face_tags}, {1})
                self.assertEqual({tag.macro_feature for tag in result.face_tags}, {macro})

    def test_rib_and_web_pair_holes_are_independent_instances(self) -> None:
        for index, macro in enumerate(("rib", "web")):
            with self.subTest(macro=macro):
                result = build_feature_sample(
                    macro,
                    2026090100 + index,
                    secondary_features=("through_hole_round",),
                )
                self.assertTrue(result.cleanup["passed"])
                self.assertGreater(result.parameters["composition_clearance"], 2.0)
                self.assertEqual({tag.instance_id for tag in result.face_tags}, {1, 2})
                self.assertEqual(
                    {tag.macro_feature for tag in result.face_tags if tag.instance_id == 1},
                    {macro},
                )
                self.assertEqual(
                    {tag.macro_feature for tag in result.face_tags if tag.instance_id == 2},
                    {"through_hole_round"},
                )


if __name__ == "__main__":
    unittest.main()
