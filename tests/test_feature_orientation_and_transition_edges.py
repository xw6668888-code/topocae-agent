import unittest
from unittest.mock import patch

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from features.fillet_feature import FilletGenerator
from features.through_hole import ThroughHoleGenerator
from utils.ontology import load_layer_index


def semantic_index():
    return load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype")


class FeatureOrientationAndTransitionEdgesTest(unittest.TestCase):
    def test_hole_generators_read_multi_axis_orientation_policy(self):
        generator = ThroughHoleGenerator(semantic_index())
        generator.set_generation_context({
            "feature_orientation": {
                "hole_axes": ["x"],
            }
        })

        self.assertEqual("x", generator.select_feature_axis("hole"))

    def test_hole_axis_policy_restricts_cylinder_stock_to_top_entry_axis(self):
        generator = ThroughHoleGenerator(semantic_index())
        generator.set_generation_context({
            "feature_orientation": {
                "hole_axes": ["x", "y"],
            },
            "current_stock_profile": {
                "stock_type": "cylinder",
                "dimensions": {"radius": 50.0, "H": 40.0},
            },
        })

        self.assertEqual("z", generator.select_feature_axis("hole"))

    def test_hole_axis_policy_keeps_side_axes_on_rectangular_stock(self):
        generator = ThroughHoleGenerator(semantic_index())
        generator.set_generation_context({
            "feature_orientation": {
                "hole_axes": ["x"],
            },
            "current_stock_profile": {
                "stock_type": "block",
                "dimensions": {"L": 120.0, "W": 100.0, "H": 50.0},
            },
        })

        self.assertEqual("x", generator.select_feature_axis("hole"))

    def test_fillet_candidates_include_circular_feature_edges(self):
        box = BRepPrimAPI_MakeBox(gp_Pnt(-50.0, -50.0, 0.0), gp_Pnt(50.0, 50.0, 40.0)).Shape()
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0.0, 0.0, -1.0), gp_Dir(0, 0, 1)),
            10.0,
            42.0,
        ).Shape()
        cut = BRepAlgoAPI_Cut(box, tool)
        cut.Build()

        generator = FilletGenerator(semantic_index())
        candidates = generator._edge_candidates(cut.Shape(), 100.0, 100.0, 40.0, [])

        self.assertTrue(
            any(candidate["curve_type"] == "circle" for candidate in candidates),
            "Expected circular hole rim edges to be valid fillet candidates.",
        )

    def test_fillet_generation_skips_occ_build_failures(self):
        class FailingFilletBuilder:
            def __init__(self, stock):
                self.stock = stock

            def Add(self, radius, edge):
                return None

            def Build(self):
                raise RuntimeError("There are no suitable edges for chamfer or fillet")

            def IsDone(self):
                return False

        box = BRepPrimAPI_MakeBox(gp_Pnt(-50.0, -50.0, 0.0), gp_Pnt(50.0, 50.0, 40.0)).Shape()
        generator = FilletGenerator(semantic_index())

        with patch("features.fillet_feature.BRepFilletAPI_MakeFillet", FailingFilletBuilder):
            stock, added, _zones, params = generator.apply_feature(box, 100.0, 100.0, 40.0, 1, 1, [])

        self.assertEqual(0, added)
        self.assertEqual([], params)
        self.assertIsNotNone(stock)


if __name__ == "__main__":
    unittest.main()
