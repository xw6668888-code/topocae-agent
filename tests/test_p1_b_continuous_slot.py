from __future__ import annotations

import math
import random
import unittest

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from features.p1_b_continuous_slot import (
    GUIDE_END_COMBINATIONS,
    PROFILE_SHAPES,
    TERMINAL_ROLE,
    all_slot_combinations,
    build_continuous_slot_cut,
    build_slot_profile_wire,
)


class P1BContinuousSlotTests(unittest.TestCase):
    L = 100.0
    W = 80.0
    H = 25.0

    def _stock(self):
        return BRepPrimAPI_MakeBox(gp_Pnt(-self.L / 2.0, -self.W / 2.0, 0.0), self.L, self.W, self.H).Shape()

    def test_cross_product_contains_exactly_twenty_required_combinations(self):
        combinations = all_slot_combinations()
        self.assertEqual(len(combinations), 20)
        self.assertEqual(len(set(combinations)), 20)
        self.assertEqual({item[0] for item in combinations}, set(PROFILE_SHAPES))
        self.assertEqual({item[1:] for item in combinations}, set(GUIDE_END_COMBINATIONS))

    def test_profiles_are_exact_closed_analytic_profiles(self):
        expected = {
            "rect": (4, [GeomAbs_Line] * 4),
            "u": (4, [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Line, GeomAbs_Line]),
            "v": (3, [GeomAbs_Line] * 3),
            "o": (1, [GeomAbs_Circle]),
        }
        for profile_shape, (edge_count, curve_types) in expected.items():
            with self.subTest(profile_shape=profile_shape):
                wire = build_slot_profile_wire(
                    profile_shape=profile_shape,
                    start_xy=(0.0, 0.0),
                    tangent_angle=0.0,
                    surface_z=self.H,
                    width=8.0,
                    depth=6.0 if profile_shape != "o" else 4.0,
                    top_margin=1.2,
                )
                edges = list(TopologyExplorer(wire).edges())
                self.assertEqual(len(edges), edge_count)
                self.assertEqual(sorted(BRepAdaptor_Curve(edge).GetType() for edge in edges), sorted(curve_types))
                self.assertTrue(wire.Closed())

    def test_all_twenty_combinations_are_valid_single_solid_and_label_complete(self):
        for index, (profile_shape, guide_family, end_condition) in enumerate(all_slot_combinations()):
            with self.subTest(profile_shape=profile_shape, guide_family=guide_family, end_condition=end_condition):
                result = build_continuous_slot_cut(
                    self._stock(),
                    L=self.L,
                    W=self.W,
                    H=self.H,
                    profile_shape=profile_shape,
                    guide_family=guide_family,
                    end_condition=end_condition,
                    seed=2026071800 + index,
                )
                self.assertTrue(BRepCheck_Analyzer(result.shape).IsValid())
                self.assertEqual(len(list(TopologyExplorer(result.shape).solids())), 1)
                props = GProp_GProps()
                brepgprop.VolumeProperties(result.shape, props)
                self.assertGreater(props.Mass(), 0.0)

                params = result.parameters
                final_hashes = {hash(face) for face in TopologyExplorer(result.shape).faces()}
                labelled_hashes = set(params["_face_role_by_hash"])
                self.assertTrue(labelled_hashes)
                self.assertTrue(labelled_hashes.issubset(final_hashes))
                self.assertEqual(params["feature_face_count"], len(labelled_hashes))
                expected_terminal_count = 1 if end_condition == "open_closed" else 0
                self.assertEqual(params["terminal_face_count"], expected_terminal_count)
                self.assertEqual(
                    sum(role == TERMINAL_ROLE for role in params["_face_role_by_hash"].values()),
                    expected_terminal_count,
                )

                guide_audit = params["guide_audit"]
                self.assertTrue(guide_audit["single_edge"])
                self.assertTrue(guide_audit["analytic_curve_type_ok"])
                self.assertTrue(guide_audit["at_least_c1"])
                self.assertFalse(guide_audit["polyline"])
                self.assertGreaterEqual(params["path_length_profile_width_ratio"], 4.0)
                if guide_family != "straight":
                    self.assertNotIn(params["feature_face_count"], {16, 32, 48})

    def test_end_conditions_follow_host_boundary_contract(self):
        half_u = self.L / 2.0
        half_v = self.W / 2.0

        def on_boundary(point):
            return math.isclose(abs(point[0]), half_u, abs_tol=1e-7) or math.isclose(abs(point[1]), half_v, abs_tol=1e-7)

        def inside(point, margin=0.0):
            return abs(point[0]) <= half_u - margin and abs(point[1]) <= half_v - margin

        for index, (profile_shape, guide_family, end_condition) in enumerate(all_slot_combinations()):
            result = build_continuous_slot_cut(
                self._stock(),
                L=self.L,
                W=self.W,
                H=self.H,
                profile_shape=profile_shape,
                guide_family=guide_family,
                end_condition=end_condition,
                seed=2026071900 + index,
            )
            params = result.parameters
            visible_start = params["visible_start_xy"]
            visible_end = params["visible_end_xy"]
            if end_condition == "open_open":
                self.assertTrue(on_boundary(visible_start))
                self.assertTrue(on_boundary(visible_end))
            elif end_condition == "open_closed":
                self.assertTrue(on_boundary(visible_start))
                self.assertTrue(inside(visible_end))
                self.assertFalse(on_boundary(visible_end))
            else:
                self.assertTrue(params["path_closed"])
                definition = params["guide_definition"]
                radius = definition["circle_radius"] + params["profile_width"] / 2.0
                center = definition["circle_center_xy"]
                self.assertTrue(inside((center[0] + radius, center[1]), margin=1e-7))
                self.assertTrue(inside((center[0] - radius, center[1]), margin=1e-7))
                self.assertTrue(inside((center[0], center[1] + radius), margin=1e-7))
                self.assertTrue(inside((center[0], center[1] - radius), margin=1e-7))

    def test_seed_is_local_and_reproducible(self):
        random.seed(91)
        expected_next = random.random()
        random.seed(91)
        first = build_continuous_slot_cut(
            self._stock(), L=self.L, W=self.W, H=self.H,
            profile_shape="u", guide_family="curved_open", end_condition="open_closed", seed=44,
        )
        self.assertEqual(random.random(), expected_next)
        second = build_continuous_slot_cut(
            self._stock(), L=self.L, W=self.W, H=self.H,
            profile_shape="u", guide_family="curved_open", end_condition="open_closed", seed=44,
        )
        for key in ("profile_width", "profile_depth", "path_length", "guide_definition"):
            self.assertEqual(first.parameters[key], second.parameters[key])


if __name__ == "__main__":
    unittest.main()
