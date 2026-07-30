import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Extend.TopologyUtils import TopologyExplorer

from utils.shape_quality import inspect_generation_candidate, inspect_shape_quality


class ShapeQualityTest(unittest.TestCase):
    def test_box_solid_is_acceptable(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()

        report = inspect_shape_quality(shape)

        self.assertTrue(report["acceptable"])
        self.assertEqual([], report["issues"])
        self.assertEqual(1, report["solid_count"])
        self.assertGreater(report["signed_volume"], 0.0)

    def test_reversed_solid_is_acceptable_because_writer_reorients_volume(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape().Reversed()

        report = inspect_shape_quality(shape)

        self.assertTrue(report["acceptable"])
        self.assertLess(report["signed_volume"], 0.0)

    def test_closed_shell_without_solid_is_rejected(self):
        solid = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()
        shell = list(TopologyExplorer(solid).shells())[0]

        report = inspect_shape_quality(shell)

        self.assertFalse(report["acceptable"])
        self.assertIn("no_solid", report["issues"])

    def test_generation_candidate_rejects_empty_feature_sample(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()

        report = inspect_generation_candidate(
            shape,
            feature_count=0,
            require_nonempty_sample=True,
        )

        self.assertFalse(report["acceptable"])
        self.assertIn("empty_sample", report["issues"])


if __name__ == "__main__":
    unittest.main()
