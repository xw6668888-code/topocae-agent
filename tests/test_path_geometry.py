import math
import unittest

from utils.path_geometry import (
    path_tangent_angle,
    point_to_polyline_distance,
    polyline_length,
    sample_circular_arc_points,
)


class PathGeometryTest(unittest.TestCase):
    def test_sample_circular_arc_points_keeps_endpoints_and_length_above_chord(self):
        points = sample_circular_arc_points((-10.0, 0.0), (10.0, 0.0), sagitta=5.0, sample_count=9)

        self.assertEqual(points[0], (-10.0, 0.0))
        self.assertEqual(points[-1], (10.0, 0.0))
        self.assertGreater(polyline_length(points), 20.0)

    def test_sample_circular_arc_points_supports_negative_sagitta(self):
        positive = sample_circular_arc_points((-10.0, 0.0), (10.0, 0.0), sagitta=4.0, sample_count=5)
        negative = sample_circular_arc_points((-10.0, 0.0), (10.0, 0.0), sagitta=-4.0, sample_count=5)

        self.assertGreater(positive[2][1], 0.0)
        self.assertLess(negative[2][1], 0.0)

    def test_point_to_polyline_distance_is_zero_on_path(self):
        points = [(-10.0, 0.0), (0.0, 4.0), (10.0, 0.0)]

        self.assertAlmostEqual(point_to_polyline_distance(0.0, 4.0, points), 0.0)

    def test_path_tangent_angle_uses_first_segment(self):
        points = [(-10.0, 0.0), (-6.0, 3.0), (10.0, 0.0)]

        self.assertAlmostEqual(path_tangent_angle(points), math.atan2(3.0, 4.0))

    def test_path_tangent_angle_can_use_last_segment(self):
        points = [(-10.0, 0.0), (6.0, 3.0), (10.0, 0.0)]

        self.assertAlmostEqual(path_tangent_angle(points, at_end=True), math.atan2(-3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
