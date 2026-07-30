import unittest

from OCC.Core.GeomAbs import GeomAbs_Plane

from features.obround_hole import ObroundHoleGenerator


SEMANTIC_INDEX = {
    "alias_to_key": {
        "stock_face": "stock_face",
        "obround_side_plane": "obround_hole_side_plane",
        "obround_end_cylinder": "obround_hole_end_cylinder",
    },
    "key_to_id": {
        "stock_face": 0,
        "obround_hole_side_plane": 8,
        "obround_hole_end_cylinder": 9,
    },
}


class FakePoint:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def X(self):
        return self._x

    def Y(self):
        return self._y

    def Z(self):
        return self._z


class FakeDirection:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def X(self):
        return self._x

    def Y(self):
        return self._y

    def Z(self):
        return self._z


class FakeAxis:
    def __init__(self, direction):
        self._direction = direction

    def Direction(self):
        return self._direction


class FakePlane:
    def __init__(self, normal):
        self._axis = FakeAxis(FakeDirection(*normal))

    def Axis(self):
        return self._axis


class FakePlaneSurface:
    def __init__(self, normal):
        self._plane = FakePlane(normal)

    def GetType(self):
        return GeomAbs_Plane

    def Plane(self):
        return self._plane


class ObroundHoleAxisMatchingTest(unittest.TestCase):
    def test_x_axis_obround_side_plane_can_have_z_normal(self):
        generator = ObroundHoleGenerator(SEMANTIC_INDEX)
        params = [{
            "inst": 3,
            "cx": 0.0,
            "cy": 0.0,
            "cz": 0.0,
            "R": 5.0,
            "dist": 20.0,
            "angle": 0.0,
            "axis": "x",
        }]

        result = generator.match_face(
            FakePlaneSurface((0.0, 0.0, 1.0)),
            FakePoint(0.0, 0.0, 5.0),
            0.0,
            0.0,
            40.0,
            params,
        )

        self.assertEqual((8, 3, "obround_hole"), result)

    def test_y_axis_obround_side_plane_can_have_z_normal(self):
        generator = ObroundHoleGenerator(SEMANTIC_INDEX)
        params = [{
            "inst": 4,
            "cx": 0.0,
            "cy": 0.0,
            "cz": 0.0,
            "R": 5.0,
            "dist": 20.0,
            "angle": 0.0,
            "axis": "y",
        }]

        result = generator.match_face(
            FakePlaneSurface((0.0, 0.0, 1.0)),
            FakePoint(0.0, 0.0, 5.0),
            0.0,
            0.0,
            40.0,
            params,
        )

        self.assertEqual((8, 4, "obround_hole"), result)


if __name__ == "__main__":
    unittest.main()
