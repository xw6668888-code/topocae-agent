import unittest
from unittest.mock import patch

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

from features.circular_pocket import CircularPocketGenerator
from features.rect_pocket import RectPocketGenerator
from features.rib import RibGenerator
from features.web import WebGenerator
from utils.ontology import load_layer_index
from utils.stock_factory import build_stock_shape, sample_stock_spec


SEMANTIC_INDEX = {
    "alias_to_key": {
        "stock_face": "stock_face",
        "rib_side_plane": "rib_side_plane",
        "rib_top_plane": "rib_top_plane",
        "web_side_plane": "web_side_plane",
        "web_top_plane": "web_top_plane",
    },
    "key_to_id": {
        "stock_face": 0,
        "rib_side_plane": 1,
        "rib_top_plane": 2,
        "web_side_plane": 3,
        "web_top_plane": 4,
    },
}


def centered_box():
    return BRepPrimAPI_MakeBox(
        gp_Pnt(-60.0, -50.0, 0.0),
        gp_Pnt(60.0, 50.0, 40.0),
    ).Shape()


def l_plate_stock():
    spec = sample_stock_spec(
        L=120.0,
        W=100.0,
        H=50.0,
        config={"enabled": True, "weights": {"l_plate": 1.0}},
    )
    return build_stock_shape(spec), spec


class RibWebPlacementRetryTest(unittest.TestCase):
    def test_rib_retries_after_first_conflicting_position(self):
        generator = RibGenerator(SEMANTIC_INDEX)
        generator.set_generation_context({
            "rib_features": {
                "placement_modes": ["boundary"],
                "corner_probability": 0.0,
            }
        })
        occupied_zones = [{"cx": -45.0, "cy": 0.0, "radius": 5.0}]

        with patch("features.rib.random.choice", return_value="left"), patch(
            "features.rib.random.uniform",
            side_effect=[2.0, 20.0, 30.0, 0.0, 2.0, 20.0, 30.0, 30.0],
        ):
            _shape, added, _zones, params = generator.apply_feature(
                centered_box(),
                120.0,
                100.0,
                40.0,
                num_instances=1,
                start_inst_id=1,
                occupied_zones=occupied_zones,
            )

        self.assertEqual(1, added)
        self.assertEqual(30.0, params[0]["cy"])

    def test_rib_corner_mode_builds_triangular_gusset_on_l_bracket_context(self):
        stock, spec = l_plate_stock()
        generator = RibGenerator(SEMANTIC_INDEX)
        generator.set_generation_context({
            "current_stock_profile": spec,
            "rib_features": {
                "placement_modes": ["corner"],
                "corner_probability": 1.0,
            }
        })

        _shape, added, _zones, params = generator.apply_feature(
            stock,
            spec["feature_workspace"]["L"],
            spec["feature_workspace"]["W"],
            spec["feature_workspace"]["H"],
            num_instances=1,
            start_inst_id=1,
            occupied_zones=[],
        )

        self.assertEqual(1, added)
        self.assertEqual("corner", params[0]["placement_mode"])
        self.assertEqual("l_bracket_base_wall", params[0]["support_context"])
        self.assertEqual("right_triangle_gusset", params[0]["rib_body_kind"])
        self.assertEqual(2, params[0]["support_plate_count"])
        self.assertEqual(["base_plate", "vertical_wall"], params[0]["corner_sides"])
        self.assertTrue(params[0]["connects_main_faces"])
        self.assertEqual("existing_l_plate_stock", params[0]["support_source"])
        self.assertNotEqual("l_corner", params[0].get("footprint_kind"))
        self.assertGreaterEqual(params[0]["h"], params[0]["wall_h"] * 0.40)
        self.assertGreaterEqual(params[0]["l"], params[0]["wall_h"] * 0.48)

    def test_rib_corner_mode_does_not_generate_on_single_block_stock(self):
        generator = RibGenerator(SEMANTIC_INDEX)
        generator.set_generation_context({
            "current_stock_profile": {"stock_type": "block"},
            "rib_features": {
                "placement_modes": ["corner"],
                "corner_probability": 1.0,
            }
        })

        _shape, added, _zones, params = generator.apply_feature(
            centered_box(),
            120.0,
            100.0,
            40.0,
            num_instances=1,
            start_inst_id=1,
            occupied_zones=[],
        )

        self.assertEqual(0, added)
        self.assertEqual([], params)

    def test_rectangular_pocket_registers_cavity_rib_context(self):
        context = {"cavity_rib_contexts": []}
        generator = RectPocketGenerator(_full_semantic_index())
        generator.set_generation_context(context)

        _shape, added, _zones, params = generator.apply_feature(
            centered_box(),
            120.0,
            100.0,
            40.0,
            num_instances=1,
            start_inst_id=1,
            occupied_zones=[],
        )

        self.assertEqual(1, added)
        self.assertEqual(1, len(context["cavity_rib_contexts"]))
        cavity = context["cavity_rib_contexts"][0]
        self.assertEqual("rect_pocket", cavity["type"])
        self.assertEqual("pocket_rect", cavity["parent_macro_feature"])
        self.assertEqual(params[0]["inst"], cavity["parent_instance_id"])
        self.assertEqual(40.0, cavity["top_z"])
        self.assertAlmostEqual(40.0 - params[0]["depth"], cavity["bottom_z"])
        self.assertTrue(cavity["supports_cavity_rib"])

    def test_circular_pocket_registers_cavity_rib_context(self):
        context = {"cavity_rib_contexts": []}
        generator = CircularPocketGenerator(_full_semantic_index())
        generator.set_generation_context(context)

        _shape, added, _zones, params = generator.apply_feature(
            centered_box(),
            120.0,
            100.0,
            40.0,
            num_instances=1,
            start_inst_id=1,
            occupied_zones=[],
        )

        self.assertEqual(1, added)
        self.assertEqual(1, len(context["cavity_rib_contexts"]))
        cavity = context["cavity_rib_contexts"][0]
        self.assertEqual("circular_pocket", cavity["type"])
        self.assertEqual("pocket_circular", cavity["parent_macro_feature"])
        self.assertEqual(params[0]["inst"], cavity["parent_instance_id"])
        self.assertEqual(params[0]["R"], cavity["radius"])
        self.assertTrue(cavity["supports_cavity_rib"])

    def test_rib_cavity_mode_builds_wall_rib_from_existing_pocket_context(self):
        context = {
            "cavity_rib_contexts": [],
            "rib_features": {
                "placement_modes": ["cavity"],
                "cavity_probability": 1.0,
            },
        }
        pocket = RectPocketGenerator(_full_semantic_index())
        pocket.set_generation_context(context)
        stock, added, occupied_zones, pocket_params = pocket.apply_feature(
            centered_box(),
            120.0,
            100.0,
            40.0,
            num_instances=1,
            start_inst_id=1,
            occupied_zones=[],
        )
        self.assertEqual(1, added)

        rib = RibGenerator(SEMANTIC_INDEX)
        rib.set_generation_context(context)
        _shape, rib_added, _zones, rib_params = rib.apply_feature(
            stock,
            120.0,
            100.0,
            40.0,
            num_instances=1,
            start_inst_id=2,
            occupied_zones=occupied_zones,
        )

        self.assertEqual(1, rib_added)
        self.assertEqual("cavity", rib_params[0]["placement_mode"])
        self.assertEqual("cavity_wall_rib", rib_params[0]["rib_body_kind"])
        self.assertEqual("existing_pocket_context", rib_params[0]["support_source"])
        self.assertEqual("pocket_rect", rib_params[0]["parent_macro_feature"])
        self.assertEqual(pocket_params[0]["inst"], rib_params[0]["parent_instance_id"])

    def test_web_retries_after_first_conflicting_strip(self):
        generator = WebGenerator(SEMANTIC_INDEX)
        occupied_zones = [{"cx": 0.0, "cy": 0.0, "radius": 10.0}]

        with patch("features.web.random.choice", return_value="x_span"), patch(
            "features.web.random.uniform",
            side_effect=[2.0, 20.0, 0.0, 2.0, 20.0, 30.0],
        ):
            _shape, added, _zones, params = generator.apply_feature(
                centered_box(),
                120.0,
                100.0,
                40.0,
                num_instances=1,
                start_inst_id=1,
                occupied_zones=occupied_zones,
            )

        self.assertEqual(1, added)
        self.assertEqual(30.0, params[0]["cy"])


def _full_semantic_index():
    return load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype")


if __name__ == "__main__":
    unittest.main()
