import random
import unittest

from utils.stock_factory import sample_stock_spec


class StockFactoryTest(unittest.TestCase):
    def test_sample_stock_spec_can_force_cylindrical_stock_with_safe_planar_workspace(self):
        spec = sample_stock_spec(
            L=120.0,
            W=100.0,
            H=50.0,
            config={"enabled": True, "weights": {"cylinder": 1.0}},
            rng=random.Random(11),
        )

        self.assertEqual("cylinder", spec["stock_type"])
        self.assertEqual(50.0, spec["dimensions"]["H"])
        self.assertLess(spec["feature_workspace"]["L"], spec["dimensions"]["diameter"])
        self.assertLess(spec["feature_workspace"]["W"], spec["dimensions"]["diameter"])
        self.assertEqual("top_safe_square", spec["feature_workspace"]["placement_mode"])

    def test_sample_stock_spec_can_force_composite_stock_components(self):
        spec = sample_stock_spec(
            L=120.0,
            W=100.0,
            H=50.0,
            config={
                "enabled": True,
                "weights": {"composite": 1.0},
                "composite_component_count_range": [3, 3],
            },
            rng=random.Random(17),
        )

        self.assertEqual("composite", spec["stock_type"])
        self.assertEqual(3, spec["component_count"])
        self.assertEqual(3, len(spec["components"]))
        self.assertEqual("base_top", spec["feature_workspace"]["placement_mode"])

    def test_plate_stock_is_visibly_thin(self):
        spec = sample_stock_spec(
            L=120.0,
            W=100.0,
            H=50.0,
            config={"enabled": True, "weights": {"block": 0.0, "plate": 1.0, "beam": 0.0, "cylinder": 0.0, "composite": 0.0}},
            rng=random.Random(23),
        )

        dims = spec["dimensions"]
        self.assertEqual("plate", spec["stock_type"])
        self.assertLessEqual(dims["H"], min(dims["L"], dims["W"]) * 0.14)

    def test_beam_stock_is_visibly_long_relative_to_cross_section(self):
        spec = sample_stock_spec(
            L=120.0,
            W=100.0,
            H=50.0,
            config={"enabled": True, "weights": {"block": 0.0, "plate": 0.0, "beam": 1.0, "cylinder": 0.0, "composite": 0.0}},
            rng=random.Random(29),
        )

        dims = spec["dimensions"]
        self.assertEqual("beam", spec["stock_type"])
        self.assertGreaterEqual(dims["L"], max(dims["W"], dims["H"]) * 6.0)

    def test_l_plate_stock_is_two_large_base_components_with_rib_context(self):
        spec = sample_stock_spec(
            L=120.0,
            W=100.0,
            H=50.0,
            config={"enabled": True, "weights": {"l_plate": 1.0}},
            rng=random.Random(31),
        )

        self.assertEqual("l_plate", spec["stock_type"])
        self.assertEqual(2, spec["component_count"])
        self.assertEqual(["base_plate", "vertical_plate"], [component["role"] for component in spec["components"]])
        self.assertEqual("l_plate_corner", spec["feature_workspace"]["placement_mode"])

        base_plate, vertical_plate = spec["components"]
        self.assertGreater(base_plate["size"][0], vertical_plate["size"][2])
        self.assertGreater(vertical_plate["size"][2], base_plate["size"][2] * 3.0)
        self.assertGreater(vertical_plate["size"][0] * vertical_plate["size"][1], 0.0)

        rib_context = spec["rib_context"]
        self.assertEqual("l_plate_corner", rib_context["type"])
        self.assertEqual(spec["feature_workspace"]["H"], rib_context["base_plane_z"])
        self.assertEqual("stock", rib_context["support_label"])


if __name__ == "__main__":
    unittest.main()
