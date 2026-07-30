from __future__ import annotations

import random
import unittest

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line
from OCC.Extend.TopologyUtils import TopologyExplorer

from p1_c.all_features_sweep_v2 import build_feature_sample
from p1_c.swept_slots import build_profile_wire, build_swept_slot_tool


class P1CCleanSweptSlotTests(unittest.TestCase):
    def test_four_closed_profiles_are_analytic(self) -> None:
        expected = {
            "rect": (4, [GeomAbs_Line] * 4),
            "u": (4, [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Line, GeomAbs_Line]),
            "v": (3, [GeomAbs_Line] * 3),
            "o": (1, [GeomAbs_Circle]),
        }
        for profile, (edge_count, curve_types) in expected.items():
            with self.subTest(profile=profile):
                wire = build_profile_wire(
                    profile,
                    start=(-20.0, 0.0),
                    tangent=(1.0, 0.0),
                    surface_z=25.0,
                    width=6.0,
                    depth=3.0 if profile == "o" else 5.0,
                    top_margin=1.0,
                )
                edges = list(TopologyExplorer(wire).edges())
                self.assertTrue(wire.Closed())
                self.assertEqual(len(edges), edge_count)
                self.assertEqual(sorted(BRepAdaptor_Curve(edge).GetType() for edge in edges), sorted(curve_types))

    def test_tools_use_surface_guides_and_pipe_sweeps(self) -> None:
        host = {"top_z": 25.0, "length": 100.0, "width": 80.0, "height": 25.0}
        for index, macro in enumerate(("slot_u", "slot_rect", "slot_v", "slot_o")):
            with self.subTest(macro=macro):
                swept = build_swept_slot_tool(macro, (0.0, 0.0), host, random.Random(2026077000 + index))
                params = swept.parameters
                self.assertEqual(params["construction_method"], "p1_c_continuous_closed_profile_sweep_cut")
                self.assertTrue(params["guide_on_existing_planar_face"])
                self.assertTrue(params["profile_closed"])
                self.assertEqual(params["guide_edge_count"], 1)
                self.assertFalse(params["guide_polyline"])
                self.assertGreaterEqual(params["path_length_profile_width_ratio"], 4.0)

    def test_single_and_pair_samples_are_clean_for_all_slot_classes(self) -> None:
        for index, macro in enumerate(("slot_u", "slot_rect", "slot_v", "slot_o")):
            for context, secondaries in (("single", ()), ("pair", ("through_hole_round",))):
                with self.subTest(macro=macro, context=context):
                    result = build_feature_sample(
                        macro,
                        2026077100 + index * 2 + (context == "pair"),
                        secondary_features=secondaries,
                    )
                    self.assertTrue(result.cleanup["passed"])
                    feature = next(item for item in result.parameters["features"] if item["macro_feature"] == macro)
                    self.assertEqual(feature["parameters"]["construction_method"], "p1_c_continuous_closed_profile_sweep_cut")


if __name__ == "__main__":
    unittest.main()
