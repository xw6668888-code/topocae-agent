import random
import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

from features.counterbore_hole import CounterboreHoleGenerator
from features.countersink_hole import CountersinkHoleGenerator
from features.obround_hole import ObroundHoleGenerator
from features.stepped_hole import SteppedHoleGenerator
from utils.generation_context import build_generation_context
from utils.ontology import load_layer_index


def semantic_index():
    return load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype")


class RemainingHoleGeneratorsMultiAxisTest(unittest.TestCase):
    def setUp(self):
        self.context = build_generation_context({
            "feature_orientation": {
                "hole_axes": ["x"],
            }
        })
        self.stock = BRepPrimAPI_MakeBox(
            gp_Pnt(-60.0, -55.0, 0.0),
            gp_Pnt(60.0, 55.0, 50.0),
        ).Shape()

    def assert_generator_adds_axis_feature(self, generator_cls, axis_key):
        random.seed(20260603)
        generator = generator_cls(semantic_index())
        generator.set_generation_context(build_generation_context({
            "feature_orientation": {
                "hole_axes": [axis_key],
            },
            "stock_profiles": {
                "enabled": False,
            },
        }))

        _stock, added, _zones, params = generator.apply_feature(
            self.stock,
            120.0,
            110.0,
            50.0,
            1,
            1,
            [],
        )

        self.assertEqual(1, added)
        self.assertEqual(axis_key, params[0]["axis"])

    def test_stepped_hole_supports_x_axis_generation(self):
        self.assert_generator_adds_axis_feature(SteppedHoleGenerator, "x")

    def test_stepped_hole_supports_y_axis_generation(self):
        self.assert_generator_adds_axis_feature(SteppedHoleGenerator, "y")

    def test_counterbore_hole_supports_x_axis_generation(self):
        self.assert_generator_adds_axis_feature(CounterboreHoleGenerator, "x")

    def test_counterbore_hole_supports_y_axis_generation(self):
        self.assert_generator_adds_axis_feature(CounterboreHoleGenerator, "y")

    def test_countersink_hole_supports_x_axis_generation(self):
        self.assert_generator_adds_axis_feature(CountersinkHoleGenerator, "x")

    def test_countersink_hole_supports_y_axis_generation(self):
        self.assert_generator_adds_axis_feature(CountersinkHoleGenerator, "y")

    def test_obround_hole_supports_x_axis_generation(self):
        self.assert_generator_adds_axis_feature(ObroundHoleGenerator, "x")

    def test_obround_hole_supports_y_axis_generation(self):
        self.assert_generator_adds_axis_feature(ObroundHoleGenerator, "y")


if __name__ == "__main__":
    unittest.main()
