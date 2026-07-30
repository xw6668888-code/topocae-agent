from __future__ import annotations

import unittest

from p1_c import all_features as feature_core
from p1_c import formal_plan as plan_core
from p1_c import formal_plan_v4, formal_plan_v6
from p1_c import rib_web as rib_core
from p1_c import rib_web_v3
from p1_c.runtime_state_isolation_v1 import install


class ImportOrderIsolationTests(unittest.TestCase):
    def test_strict_then_legacy_entrypoints_restore_shared_kernels(self) -> None:
        install()
        rib_snapshot = (
            rib_core._l_corner_host,
            rib_core._two_support_host,
            rib_core._feature_profile,
        )
        feature_snapshot = feature_core._feature_tool
        plan_snapshot = (
            plan_core.PLAN_REVISION,
            plan_core.SEED_NAMESPACE,
            plan_core.build_plan,
            plan_core.audit_plan,
            plan_core.compatibility_matrix,
        )

        strict = rib_web_v3.build_rib_web_sample(
            "web_two_supports_equal_plain", 2026099901
        )
        legacy = rib_core.build_rib_web_sample(
            "web_two_supports_plain", 2026099902
        )
        self.assertTrue(strict.cleanup["passed"])
        self.assertTrue(legacy.cleanup["passed"])

        strict_plan = formal_plan_v6.build_plan()
        legacy_plan = formal_plan_v4.build_plan()
        self.assertEqual(len(strict_plan), 7200)
        self.assertEqual(len(legacy_plan), 7200)
        self.assertTrue(strict_plan[0].sample_id.startswith("p1c_strict_"))
        self.assertTrue(legacy_plan[0].sample_id.startswith("p1c_01_"))

        self.assertEqual(
            rib_snapshot,
            (
                rib_core._l_corner_host,
                rib_core._two_support_host,
                rib_core._feature_profile,
            ),
        )
        self.assertIs(feature_snapshot, feature_core._feature_tool)
        self.assertEqual(
            plan_snapshot,
            (
                plan_core.PLAN_REVISION,
                plan_core.SEED_NAMESPACE,
                plan_core.build_plan,
                plan_core.audit_plan,
                plan_core.compatibility_matrix,
            ),
        )


if __name__ == "__main__":
    unittest.main()
