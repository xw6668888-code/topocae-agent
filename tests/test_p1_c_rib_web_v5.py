from __future__ import annotations

import unittest

from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.rib_web_v5 import SCENARIO_SPECS, build_rib_web_sample


class FullHeightSupportedWebTests(unittest.TestCase):
    def test_all_scenarios_have_clean_full_support_connections(self) -> None:
        self.assertEqual(len(SCENARIO_SPECS), 24)
        for index, (scenario, macro, host_family, variant) in enumerate(SCENARIO_SPECS):
            with self.subTest(scenario=scenario):
                result = build_rib_web_sample(scenario, 2026088000 + index)
                parameters = result.parameters
                self.assertEqual(parameters["macro_feature"], macro)
                self.assertEqual(parameters["host_family"], host_family)
                self.assertEqual(parameters["variant"], variant)
                self.assertTrue(result.cleanup["passed"])
                self.assertTrue(parameters["host_cleanup_before_feature"]["passed"])
                self.assertEqual(parameters["exposed_feature_end_face_count"], 0)
                self.assertTrue(parameters["full_height_two_support_connection"])
                self.assertGreaterEqual(parameters["positive_support_contact_count"], 2)
                self.assertEqual(len(list(TopologyExplorer(result.shape).solids())), 1)

    def test_rejected_web_families_no_longer_leave_end_faces(self) -> None:
        rejected_families = {"two_supports_tapered", "two_supports_buttressed"}
        for index, (scenario, macro, host_family, _variant) in enumerate(SCENARIO_SPECS):
            if host_family not in rejected_families:
                continue
            with self.subTest(scenario=scenario):
                result = build_rib_web_sample(scenario, 2026089000 + index)
                self.assertEqual(macro, "web")
                self.assertEqual(result.parameters["feature_face_count"], 5 if scenario.endswith("transition") else 3)
                self.assertEqual(result.parameters["exposed_feature_end_face_count"], 0)
                self.assertTrue(
                    result.parameters["host_metadata"]["inner_support_interfaces_vertical"]
                )
                self.assertTrue(
                    result.parameters["host_metadata"]["inner_support_interfaces_full_height"]
                )


if __name__ == "__main__":
    unittest.main()
