from __future__ import annotations

import unittest

from p1_c.formal_generator_v5 import GENERATOR_REVISION, build_planned_result
from p1_c.formal_plan_v4 import build_plan


class FormalGeneratorV5RegressionTests(unittest.TestCase):
    def test_two_support_composite_chamfer_uses_clean_outer_edge(self) -> None:
        sample = next(
            item
            for item in build_plan()
            if item.sample_id == "p1c_02_blind_hole_round_0366_triple"
        )
        result = build_planned_result(sample)
        chamfer = next(record for record in result.parameters["features"] if record["macro_feature"] == "chamfer")
        self.assertEqual(result.parameters["generator_revision"], GENERATOR_REVISION)
        self.assertTrue(result.cleanup["passed"])
        self.assertTrue(chamfer["cleanup_after_operation"]["passed"])
        self.assertEqual(chamfer["feature_face_count"], 1)
        self.assertGreaterEqual(chamfer["parameters"]["candidate_rank"], 1)


if __name__ == "__main__":
    unittest.main()
