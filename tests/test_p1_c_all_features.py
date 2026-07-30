from __future__ import annotations

import unittest

from p1_c.all_features import H_C2_SECONDARY_BY_PRIMARY, HOST_FAMILIES, MACRO_CLASSES, build_feature_sample
from p1_c.supported_wall_families import SUPPORTED_WALL_HOST_FAMILIES, build_supported_wall_sample


class CleanAllFeatureGeneratorTests(unittest.TestCase):
    def test_exact_18_class_registry(self) -> None:
        self.assertEqual(len(MACRO_CLASSES), 18)
        self.assertEqual(len(set(MACRO_CLASSES)), 18)

    def test_all_h_c2_single_and_pair_contexts_are_clean(self) -> None:
        for index, macro in enumerate(MACRO_CLASSES):
            with self.subTest(macro=macro, context="single"):
                single = build_feature_sample(macro, 2026072000 + index * 2)
                self.assertTrue(single.cleanup["passed"])
                self.assertEqual(single.parameters["secondary_features"], [])
            with self.subTest(macro=macro, context="pair"):
                secondary = H_C2_SECONDARY_BY_PRIMARY[macro]
                pair = build_feature_sample(macro, 2026072001 + index * 2, secondary_features=(secondary,))
                self.assertTrue(pair.cleanup["passed"])
                self.assertEqual(pair.parameters["secondary_features"], [secondary])

    def test_generic_classes_cover_four_clean_host_families(self) -> None:
        for macro_index, macro in enumerate(item for item in MACRO_CLASSES if item not in {"rib", "web"}):
            for family_index, host_family in enumerate(HOST_FAMILIES):
                with self.subTest(macro=macro, host_family=host_family):
                    result = build_feature_sample(
                        macro,
                        2026073000 + macro_index * 10 + family_index,
                        host_family=host_family,
                    )
                    self.assertEqual(result.parameters["host_family"], host_family)
                    self.assertTrue(result.cleanup["passed"])

    def test_rib_and_web_cover_four_clean_engineering_hosts(self) -> None:
        for macro_index, macro in enumerate(("rib", "web")):
            for family_index, host_family in enumerate(SUPPORTED_WALL_HOST_FAMILIES):
                with self.subTest(macro=macro, host_family=host_family):
                    result = build_supported_wall_sample(
                        macro,
                        host_family,
                        "plain",
                        2026074000 + macro_index * 10 + family_index,
                    )
                    self.assertEqual(result.parameters["host_family"], host_family)
                    self.assertTrue(result.cleanup["passed"])


if __name__ == "__main__":
    unittest.main()
