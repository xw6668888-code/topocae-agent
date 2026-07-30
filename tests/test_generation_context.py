import unittest

from main import instantiate_generators
from utils.generation_context import build_generation_context, cap_requested_instances, order_generator_names
from utils.ontology import load_layer_index


class GenerationContextTest(unittest.TestCase):
    def test_feature_generators_are_ordered_by_cae_construction_stage(self):
        generator_names = [
            "ChamferGenerator",
            "ThroughHoleGenerator",
            "RectPocketGenerator",
            "RibGenerator",
            "BossFeatureGenerator",
            "SlotRectGenerator",
        ]

        ordered = order_generator_names(generator_names)

        self.assertEqual(
            ordered,
            [
                "RectPocketGenerator",
                "ThroughHoleGenerator",
                "BossFeatureGenerator",
                "SlotRectGenerator",
                "RibGenerator",
                "ChamferGenerator",
            ],
        )

    def test_generation_context_defaults_include_stock_and_orientation_policies(self):
        context = build_generation_context({})

        self.assertTrue(context["stock_profiles"]["enabled"])
        self.assertIn("cylinder", context["stock_profiles"]["weights"])
        self.assertIn("composite", context["stock_profiles"]["weights"])
        self.assertEqual(["z", "x", "y"], context["feature_orientation"]["hole_axes"])
        self.assertTrue(context["feature_ordering"]["enabled"])
        self.assertTrue(context["feature_density"]["enabled"])
        self.assertGreater(context["feature_density"]["max_total_instances_per_sample"], 0)

    def test_cap_requested_instances_respects_remaining_budget_and_generator_limit(self):
        self.assertEqual(1, cap_requested_instances(2, remaining_budget=5, max_instances_per_generator=1))
        self.assertEqual(2, cap_requested_instances(3, remaining_budget=2, max_instances_per_generator=3))
        self.assertEqual(0, cap_requested_instances(1, remaining_budget=0, max_instances_per_generator=1))

    def test_instantiate_generators_applies_ordering_and_context(self):
        subtype_index = load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype")
        context = build_generation_context({
            "feature_orientation": {
                "hole_axes": ["x"],
            }
        })

        generators = instantiate_generators(
            ["ChamferGenerator", "ThroughHoleGenerator", "RectPocketGenerator"],
            subtype_index,
            generation_context=context,
        )

        self.assertEqual(
            ["RectPocketGenerator", "ThroughHoleGenerator", "ChamferGenerator"],
            [generator.__class__.__name__ for generator in generators],
        )
        self.assertEqual(["x"], generators[1].generation_context["feature_orientation"]["hole_axes"])


if __name__ == "__main__":
    unittest.main()
