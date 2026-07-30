import unittest

import main as dataset_main
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line, GeomAbs_SurfaceOfRevolution
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from features.swept_channel_slot import SweptSlotGenerator
from features.swept_channel_slot import (
    _path_metadata_from_points,
    _point_is_inside_path_corridor,
    build_overrun_slot_tool_path,
)
from features.swept_slot import (
    build_path_wire_from_points,
    build_vertical_swept_slot_profile_wire,
    build_vertical_swept_slot_tool,
    build_vertical_swept_slot_tool_from_path,
)
from utils.ontology import load_layer_index


class SweptSlotV2Test(unittest.TestCase):
    def test_vertical_u_profile_uses_one_true_arc_and_three_straight_edges(self):
        wire = build_vertical_swept_slot_profile_wire(
            profile_shape="u",
            start_x=0.0,
            start_y=0.0,
            path_angle=0.35,
            width=8.0,
            depth=6.0,
            surface_z=20.0,
            top_margin=1.0,
        )

        curve_types = _edge_curve_types(wire)

        self.assertEqual(len(curve_types), 4)
        self.assertEqual(curve_types.count(GeomAbs_Line), 3)
        self.assertEqual(curve_types.count(GeomAbs_Circle), 1)

    def test_vertical_o_profile_is_one_true_closed_circle(self):
        wire = build_vertical_swept_slot_profile_wire(
            profile_shape="o",
            start_x=0.0,
            start_y=0.0,
            path_angle=0.35,
            width=8.0,
            depth=6.0,
            surface_z=20.0,
            top_margin=1.0,
        )

        curve_types = _edge_curve_types(wire)

        self.assertEqual(len(curve_types), 1)
        self.assertEqual(curve_types[0], GeomAbs_Circle)

    def test_vertical_swept_slot_tool_supports_required_profile_family(self):
        for profile_shape in ("u", "v", "o"):
            with self.subTest(profile_shape=profile_shape):
                tool = build_vertical_swept_slot_tool(
                    profile_shape=profile_shape,
                    cx=0.0,
                    cy=0.0,
                    surface_z=20.0,
                    path_length=60.0,
                    path_angle=0.35,
                    width=8.0,
                    depth=6.0,
                    top_margin=1.0,
                )

                self.assertIsNotNone(tool)

    def test_path_wire_from_points_contains_one_edge_per_segment(self):
        wire = build_path_wire_from_points([(-25.0, 0.0), (-10.0, 8.0), (10.0, 8.0), (25.0, 0.0)], surface_z=20.0)

        self.assertEqual(len(list(TopologyExplorer(wire).edges())), 3)

    def test_vertical_swept_slot_tool_supports_circular_arc_path_points(self):
        path_points = [(-25.0, 0.0), (-10.0, 8.0), (10.0, 8.0), (25.0, 0.0)]

        tool = build_vertical_swept_slot_tool_from_path(
            profile_shape="u",
            path_points_xy=path_points,
            surface_z=20.0,
            width=8.0,
            depth=6.0,
            top_margin=1.0,
        )

        self.assertIsNotNone(tool)

    def test_swept_slot_generator_exposes_all_slot_macro_types(self):
        self.assertEqual(set(SweptSlotGenerator.supported_profile_shapes), {"u", "v", "o"})
        self.assertEqual(SweptSlotGenerator.profile_to_macro_key["u"], "slot_u")
        self.assertEqual(SweptSlotGenerator.profile_to_macro_key["v"], "slot_v")
        self.assertEqual(SweptSlotGenerator.profile_to_macro_key["o"], "slot_o")

    def test_catalog_uses_swept_generator_for_u_v_o_slots(self):
        self.assertEqual(dataset_main.CATALOG_GENERATOR_BY_MACRO["slot_u"], "SweptSlotGenerator")
        self.assertEqual(dataset_main.CATALOG_GENERATOR_BY_MACRO["slot_v"], "SweptSlotGenerator")
        self.assertEqual(dataset_main.CATALOG_GENERATOR_BY_MACRO["slot_o"], "SweptSlotGenerator")
        self.assertIn("SweptSlotGenerator", dataset_main.STATIC_STRUCTURAL_V1_GENERATORS)
        self.assertNotIn("USlotGenerator", dataset_main.STATIC_STRUCTURAL_V1_GENERATORS)

    def test_curved_slot_path_metadata_records_arc_kind(self):
        path = _path_metadata_from_points([(-20.0, 0.0), (0.0, 6.0), (20.0, 0.0)], path_curve_kind="circular_arc")

        self.assertEqual(path["path_curve_kind"], "circular_arc")
        self.assertEqual(path["path_type"], "curved")
        self.assertGreater(path["length"], 40.0)
        self.assertEqual(path["path_points_xy"][1], (0.0, 6.0))

    def test_slot_tool_path_overruns_visible_boundary_path(self):
        visible_path = _path_metadata_from_points([(-40.0, 0.0), (40.0, 0.0)], path_curve_kind="line")

        tool_path = build_overrun_slot_tool_path(visible_path, slot_width=8.0)

        self.assertEqual(visible_path["path_points_xy"], [(-40.0, 0.0), (40.0, 0.0)])
        self.assertLess(tool_path["tool_path_points_xy"][0][0], -40.0)
        self.assertGreater(tool_path["tool_path_points_xy"][-1][0], 40.0)
        self.assertGreater(tool_path["tool_path_length"], visible_path["length"])
        self.assertTrue(tool_path["tool_path_extends_visible_path"])

    def test_curved_slot_tool_path_keeps_visible_path_for_labels(self):
        visible_path = _path_metadata_from_points(
            [(-35.0, 0.0), (0.0, 12.0), (35.0, 0.0)],
            path_curve_kind="circular_arc",
        )

        tool_path = build_overrun_slot_tool_path(visible_path, slot_width=8.0)

        self.assertEqual(tool_path["path_points_xy"], visible_path["path_points_xy"])
        self.assertEqual(tool_path["path_curve_kind"], "circular_arc")
        self.assertEqual(tool_path["tool_path_curve_kind"], "polyline_overrun")
        self.assertLess(tool_path["tool_path_points_xy"][0][0], visible_path["path_points_xy"][0][0])
        self.assertGreater(tool_path["tool_path_points_xy"][-1][0], visible_path["path_points_xy"][-1][0])

    def test_point_to_slot_path_corridor_accepts_near_curved_path(self):
        params = {"path_points_xy": [(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)], "w": 6.0, "depth": 5.0}

        self.assertTrue(_point_is_inside_path_corridor(params, 0.0, 8.2, tolerance=0.5))
        self.assertFalse(_point_is_inside_path_corridor(params, 0.0, 20.0, tolerance=0.5))

    def test_curved_u_slot_surface_of_revolution_matches_bottom_region(self):
        generator = SweptSlotGenerator(load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype"))
        params = [{
            "profile_shape": "u",
            "path_points_xy": [(-20.0, 0.0), (0.0, 8.0), (20.0, 0.0)],
            "w": 6.0,
            "depth": 8.0,
            "top_margin": 1.0,
            "inst": 4,
            "macro_key": "slot_u",
        }]

        label, inst, macro = generator.match_face(
            _FakeSurface(GeomAbs_SurfaceOfRevolution),
            gp_Pnt(0.0, 8.0, 12.0),
            0.0,
            0.0,
            20.0,
            params,
        )

        self.assertEqual(generator.semantic_map["slot_u_bottom_plane"], label)
        self.assertEqual(4, inst)
        self.assertEqual("slot_u", macro)


def _edge_curve_types(wire):
    return [BRepAdaptor_Curve(edge).GetType() for edge in TopologyExplorer(wire).edges()]


class _FakeSurface:
    def __init__(self, surface_type):
        self._surface_type = surface_type

    def GetType(self):
        return self._surface_type


if __name__ == "__main__":
    unittest.main()
