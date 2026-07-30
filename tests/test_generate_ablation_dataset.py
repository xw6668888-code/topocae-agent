import unittest

from generate_ablation_dataset import build_generation_cfg, generation_context_for_candidate, resolve_target_macros
from utils.generation_context import build_generation_context


class GenerateAblationDatasetTest(unittest.TestCase):
    def test_resolve_target_macros_defaults_to_all_targets(self):
        targets = resolve_target_macros("")

        self.assertIn("through_hole_round", targets)
        self.assertIn("chamfer", targets)
        self.assertIn("rib", targets)

    def test_resolve_target_macros_filters_explicit_comma_list(self):
        targets = resolve_target_macros("fillet,chamfer,slot_v,slot_o,rib")

        self.assertEqual(["fillet", "chamfer", "slot_v", "slot_o", "rib"], targets)

    def test_resolve_target_macros_rejects_unknown_target(self):
        with self.assertRaises(KeyError):
            resolve_target_macros("fillet,unknown_feature")

    def test_rib_single_feature_candidate_forces_l_plate_corner_context(self):
        base_context = build_generation_context(build_generation_cfg("single_feature", max_features_per_sample=1))

        rib_context = generation_context_for_candidate(
            base_context,
            target_macro="rib",
            generator_names=["RibGenerator"],
        )

        self.assertEqual(1.0, rib_context["stock_profiles"]["weights"]["l_plate"])
        self.assertEqual(0.0, rib_context["stock_profiles"]["weights"]["block"])
        self.assertEqual(["corner"], rib_context["rib_features"]["placement_modes"])
        self.assertEqual(1.0, rib_context["rib_features"]["corner_probability"])

    def test_rib_with_pocket_candidate_uses_cavity_rib_context(self):
        base_context = build_generation_context(build_generation_cfg("nonintersecting_multi", max_features_per_sample=2))

        rib_context = generation_context_for_candidate(
            base_context,
            target_macro=None,
            generator_names=["RectPocketGenerator", "RibGenerator"],
        )

        self.assertEqual(0.0, rib_context["stock_profiles"]["weights"]["l_plate"])
        self.assertEqual(["cavity"], rib_context["rib_features"]["placement_modes"])
        self.assertEqual(1.0, rib_context["rib_features"]["cavity_probability"])


if __name__ == "__main__":
    unittest.main()
