from __future__ import annotations

import unittest

from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.formal_generator_v6 import GENERATOR_REVISION, build_planned_result
from p1_c.formal_plan_v5 import audit_plan, build_plan
from p1_c.rib_web_v3 import RIB_HOST_FAMILIES, SCENARIO_SPECS, WEB_HOST_FAMILIES, build_rib_web_sample
from p1_c.supported_wall_families_v2 import HOST_FAMILIES_BY_MACRO, build_supported_wall_sample


class StrictConceptHostTests(unittest.TestCase):
    def test_all_24_rib_web_scenarios_are_clean_and_source_true(self) -> None:
        self.assertEqual(len(SCENARIO_SPECS), 24)
        for index, (scenario, macro, host_family, variant) in enumerate(SCENARIO_SPECS):
            with self.subTest(scenario=scenario):
                result = build_rib_web_sample(scenario, 2026082000 + index)
                allowed = RIB_HOST_FAMILIES if macro == "rib" else WEB_HOST_FAMILIES
                self.assertIn(host_family, allowed)
                self.assertEqual(result.parameters["macro_feature"], macro)
                self.assertEqual(result.parameters["host_family"], host_family)
                self.assertEqual(result.parameters["variant"], variant)
                self.assertTrue(result.parameters["concept_contract"]["target_macro_equals_source_engineering_macro"])
                self.assertFalse(result.parameters["concept_contract"]["semantic_relabel_used"])
                self.assertTrue(result.cleanup["passed"])
                self.assertTrue(result.parameters["host_cleanup_before_feature"]["passed"])
                self.assertGreaterEqual(result.parameters["positive_support_contact_count"], 2)
                self.assertEqual(len(list(TopologyExplorer(result.shape).solids())), 1)

    def test_supported_wall_adapter_rejects_cross_concept_hosts(self) -> None:
        with self.assertRaises(ValueError):
            build_supported_wall_sample("rib", WEB_HOST_FAMILIES[0], "plain", 1)
        with self.assertRaises(ValueError):
            build_supported_wall_sample("web", RIB_HOST_FAMILIES[0], "plain", 1)

    def test_supported_wall_adapter_never_semantically_relabels(self) -> None:
        for macro, hosts in HOST_FAMILIES_BY_MACRO.items():
            for index, host in enumerate(hosts):
                with self.subTest(macro=macro, host=host):
                    result = build_supported_wall_sample(macro, host, "plain", 2026084000 + index)
                    self.assertEqual(result.parameters["source_engineering_macro"], macro)
                    self.assertNotIn("semantic_relabel", result.parameters)
                    self.assertEqual(
                        {tag.macro_feature for tag in result.face_tags if tag.instance_id == 1},
                        {macro},
                    )

    def test_formal_plan_has_zero_concept_incompatible_hosts(self) -> None:
        plan = build_plan()
        audit = audit_plan(plan)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["sample_count"], 7200)
        self.assertEqual(audit["concept_incompatible_supported_wall_count"], 0)
        self.assertEqual(audit["directed_pair_max"], 13)
        self.assertEqual(audit["supported_wall_source_macro_counts"], {"rib": 600, "web": 600})
        self.assertEqual(set(audit["host_family_counts_by_primary"]["rib"]), set(RIB_HOST_FAMILIES))
        self.assertEqual(set(audit["host_family_counts_by_primary"]["web"]), set(WEB_HOST_FAMILIES))

    def test_formal_generator_preserves_macro_for_all_eight_hosts(self) -> None:
        plan = build_plan()
        for macro, hosts in HOST_FAMILIES_BY_MACRO.items():
            for host in hosts:
                with self.subTest(macro=macro, host=host):
                    sample = next(
                        record
                        for record in plan
                        if record.primary_target == macro
                        and record.context == "target_only"
                        and record.host_family == host
                    )
                    result = build_planned_result(sample)
                    source = result.parameters["supported_wall"]["source_parameters"]
                    self.assertEqual(result.parameters["generator_revision"], GENERATOR_REVISION)
                    self.assertEqual(source["source_engineering_macro"], macro)
                    self.assertNotIn("semantic_relabel", source)
                    self.assertTrue(result.cleanup["passed"])
                    self.assertEqual({tag.macro_feature for tag in result.face_tags}, {macro})


if __name__ == "__main__":
    unittest.main()
