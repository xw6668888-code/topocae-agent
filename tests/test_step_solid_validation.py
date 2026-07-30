import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Extend.TopologyUtils import TopologyExplorer

from utils import step_solid_validation
from utils.step_solid_validation import (
    inspect_step_file,
    orient_shape_for_positive_volume,
    signed_solid_volume,
    solid_body_for_step_export,
    write_step_with_positive_volume,
)


class StepSolidValidationTest(unittest.TestCase):
    def test_orient_shape_for_positive_volume_reverses_negative_volume_solid(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape().Reversed()

        oriented = orient_shape_for_positive_volume(shape)

        props = GProp_GProps()
        brepgprop.VolumeProperties(oriented, props)
        self.assertGreater(props.Mass(), 0.0)

    def test_signed_solid_volume_handles_closed_shell_without_solid_topology(self):
        solid = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()
        shell = list(TopologyExplorer(solid).shells())[0].Reversed()

        volume = signed_solid_volume(shell)

        self.assertLess(volume, 0.0)

    def test_write_step_with_positive_volume_writes_readable_positive_solid(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape().Reversed()
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "box.step"

            write_step_with_positive_volume(shape, output_path)

            report = inspect_step_file(output_path)
            self.assertTrue(report["readable"])
            self.assertGreater(report["solid_count"], 0)
            self.assertGreater(report["signed_volume"], 0.0)

    def test_solid_body_for_step_export_extracts_unique_solid_from_compound(self):
        solid = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()
        orphan_face = next(iter(TopologyExplorer(BRepPrimAPI_MakeBox(2.0, 2.0, 1.0).Shape()).faces()))
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, solid)
        builder.Add(compound, orphan_face)

        export_shape = solid_body_for_step_export(compound)

        self.assertEqual(TopAbs_SOLID, export_shape.ShapeType())

    def test_write_step_with_positive_volume_exports_single_solid_body_from_compound(self):
        solid = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()
        orphan_face = next(iter(TopologyExplorer(BRepPrimAPI_MakeBox(2.0, 2.0, 1.0).Shape()).faces()))
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, solid)
        builder.Add(compound, orphan_face)

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "compound.step"

            write_step_with_positive_volume(compound, output_path)

            report = inspect_step_file(output_path)
            self.assertEqual(1, report["solid_count"])
            self.assertEqual(TopAbs_SOLID, report["top_shape_type"])
            self.assertEqual([], report["issues"])

    def test_write_step_with_positive_volume_accepts_string_path(self):
        shape = BRepPrimAPI_MakeBox(10.0, 8.0, 6.0).Shape()
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "box.step"

            write_step_with_positive_volume(shape, str(output_path))

            self.assertTrue(output_path.exists())

    def test_validate_step_directory_includes_step_and_stp_files(self):
        with TemporaryDirectory() as temp_dir:
            steps_dir = Path(temp_dir)
            (steps_dir / "part_a.step").write_text("dummy", encoding="utf-8")
            (steps_dir / "part_b.stp").write_text("dummy", encoding="utf-8")
            (steps_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            inspected_names = []

            def fake_inspect(path):
                inspected_names.append(Path(path).name)
                return {
                    "step_path": str(path),
                    "readable": True,
                    "solid_count": 1,
                    "shell_count": 1,
                    "face_count": 6,
                    "edge_count": 12,
                    "signed_volume": 1.0,
                    "valid": True,
                    "issues": [],
                }

            with patch.object(step_solid_validation, "inspect_step_file", fake_inspect):
                report = step_solid_validation.validate_step_directory(steps_dir)

            self.assertEqual(["part_a.step", "part_b.stp"], inspected_names)
            self.assertEqual(2, report["step_count"])


if __name__ == "__main__":
    unittest.main()
