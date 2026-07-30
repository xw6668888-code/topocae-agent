from __future__ import annotations

import unittest

from p1_c.all_features_h_c2 import build_feature_sample


class Hc2CompositionTests(unittest.TestCase):
    def test_rib_and_web_secondary_holes_are_in_host_not_thin_wall(self) -> None:
        for index, macro in enumerate(("rib", "web")):
            with self.subTest(macro=macro):
                pair = build_feature_sample(
                    macro,
                    2026075000 + index,
                    secondary_features=("through_hole_round",),
                )
                self.assertTrue(pair.cleanup["passed"])
                self.assertGreater(pair.parameters["composition_clearance"], 2.0)
                self.assertEqual(pair.parameters["features"][1]["operation"], "non_intersecting_host_through_hole_cut")
                self.assertEqual({tag.instance_id for tag in pair.face_tags}, {1, 2})


if __name__ == "__main__":
    unittest.main()
