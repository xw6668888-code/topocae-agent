import math
import unittest

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Extend.TopologyUtils import TopologyExplorer

from features.swept_slot import (
    build_capsule_swept_profile_wire,
    build_dovetail_swept_profile_wire,
    build_o_swept_profile_wire,
    build_rectangular_swept_profile_wire,
    build_u_swept_profile_wire,
    build_v_swept_profile_wire,
    straight_boundary_path_on_circle,
    straight_boundary_path_on_box,
    swept_profile_metadata,
)


class SweptSlotProfilesTest(unittest.TestCase):
    def test_rectangular_swept_profile_metadata_records_path_and_profile_scale(self):
        metadata = swept_profile_metadata(
            profile_shape="rectangle",
            sweep_path_length=32.0,
            profile_width=5.0,
            profile_depth=4.0,
            path_type="straight",
            path_on_face=True,
            path_start_on_boundary=True,
            path_end_on_boundary=True,
            anchor_face_kind="existing_planar_face",
        )

        self.assertEqual("swept_closed_profile", metadata["construction_method"])
        self.assertEqual("rectangle", metadata["profile_shape"])
        self.assertGreater(metadata["sweep_to_profile_ratio"], 6.0)
        self.assertEqual("straight", metadata["path_type"])
        self.assertTrue(metadata["path_on_face"])
        self.assertTrue(metadata["path_start_on_boundary"])
        self.assertTrue(metadata["path_end_on_boundary"])
        self.assertEqual("existing_planar_face", metadata["anchor_face_kind"])

    def test_swept_profile_wire_builders_return_closed_wires(self):
        rect_wire = build_rectangular_swept_profile_wire(0.0, 0.0, 10.0, 6.0, 36.0, 0.2)
        capsule_wire = build_capsule_swept_profile_wire(0.0, 0.0, 10.0, 4.0, 28.0, 0.5)
        u_wire = build_u_swept_profile_wire(0.0, 0.0, 10.0, 4.0, 28.0, 0.5)
        v_wire = build_v_swept_profile_wire(0.0, 0.0, 10.0, 6.0, 32.0, 0.1)
        dovetail_wire = build_dovetail_swept_profile_wire(0.0, 0.0, 10.0, 8.0, 5.0, 32.0, 0.4)
        o_wire = build_o_swept_profile_wire(0.0, 0.0, 10.0, 5.0, 32.0, 0.3)

        self.assertIsNotNone(rect_wire)
        self.assertIsNotNone(capsule_wire)
        self.assertIsNotNone(u_wire)
        self.assertIsNotNone(v_wire)
        self.assertIsNotNone(dovetail_wire)
        self.assertIsNotNone(o_wire)

    def test_o_swept_profile_wire_uses_a_true_circle_edge(self):
        wire = build_o_swept_profile_wire(0.0, 0.0, 10.0, 5.0, 32.0, 0.3)
        curve_types = [BRepAdaptor_Curve(edge).GetType() for edge in TopologyExplorer(wire).edges()]

        self.assertEqual(curve_types, [GeomAbs_Circle])

    def test_straight_boundary_path_on_box_starts_and_ends_on_box_boundary(self):
        path = straight_boundary_path_on_box(seed_x=3.0, seed_y=-8.0, angle=math.radians(25.0), L=120.0, W=80.0)

        for point_key in ("start", "end"):
            x, y = path[point_key]
            on_x_boundary = abs(abs(x) - 60.0) < 1e-6
            on_y_boundary = abs(abs(y) - 40.0) < 1e-6
            self.assertTrue(on_x_boundary or on_y_boundary)

        self.assertGreater(path["length"], 80.0)

    def test_straight_boundary_path_on_circle_starts_and_ends_on_circle_boundary(self):
        path = straight_boundary_path_on_circle(seed_x=5.0, seed_y=-7.0, angle=math.radians(70.0), radius=50.0)

        for point_key in ("start", "end"):
            x, y = path[point_key]
            self.assertAlmostEqual(math.hypot(x, y), 50.0, places=6)

        self.assertGreater(path["length"], 80.0)


if __name__ == "__main__":
    unittest.main()
