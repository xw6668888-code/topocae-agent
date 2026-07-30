from __future__ import annotations

import unittest

from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.rib_web import SCENARIO_SPECS, build_rib_web_sample


class CleanRibWebGeneratorTests(unittest.TestCase):
    def test_scenario_matrix_is_exact(self) -> None:
        self.assertEqual(len(SCENARIO_SPECS), 12)
        self.assertEqual(sum(item[1] == "rib" for item in SCENARIO_SPECS), 9)
        self.assertEqual(sum(item[1] == "web" for item in SCENARIO_SPECS), 3)

    def test_all_scenarios_are_clean_and_supported(self) -> None:
        for index, (scenario_key, macro, host_family, variant) in enumerate(SCENARIO_SPECS):
            with self.subTest(scenario=scenario_key):
                result = build_rib_web_sample(scenario_key, 2026071800 + index)
                self.assertTrue(result.cleanup["passed"])
                self.assertEqual(result.cleanup["removable_same_domain_face_count"], 0)
                self.assertEqual(result.cleanup["removable_same_domain_edge_count"], 0)
                self.assertEqual(len(list(TopologyExplorer(result.shape).solids())), 1)
                self.assertEqual(result.parameters["macro_feature"], macro)
                self.assertEqual(result.parameters["host_family"], host_family)
                self.assertEqual(result.parameters["variant"], variant)
                self.assertGreaterEqual(result.parameters["positive_support_contact_count"], 2)
                self.assertTrue(result.parameters["host_cleanup_before_feature"]["passed"])
                self.assertNotIn("overlapping", result.parameters["construction_method"])


if __name__ == "__main__":
    unittest.main()
