from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.workspace_paths import resolve_workspace_path


class ResolveWorkspacePathTests(unittest.TestCase):
    def test_relative_path_is_rooted_at_workspace_not_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as cwd:
            root = Path(root_dir)
            with patch.dict(os.environ, {"TOPOCAE_WORKSPACE_ROOT": str(root)}), patch(
                "pathlib.Path.cwd", return_value=Path(cwd)
            ):
                actual = resolve_workspace_path("data/sample.json")
        self.assertEqual((root / "data" / "sample.json").resolve(), actual)

    def test_windows_frozen_path_maps_to_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            with patch.dict(
                os.environ,
                {
                    "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": r"W:\python\FEA\dataset_generator",
                    "TOPOCAE_WORKSPACE_ROOT": str(root),
                },
            ):
                actual = resolve_workspace_path(
                    r"w:\python\FEA\dataset_generator\graphs\sample.json"
                )
        self.assertEqual((root / "graphs" / "sample.json").resolve(), actual)

    def test_native_absolute_path_inside_workspace_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            child = root / "graphs" / "sample.json"
            with patch.dict(os.environ, {"TOPOCAE_WORKSPACE_ROOT": str(root)}):
                actual = resolve_workspace_path(child)
        self.assertEqual(child.resolve(), actual)

    def test_similar_windows_prefix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, patch.dict(
            os.environ,
            {
                "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": r"W:\python\FEA\dataset_generator",
                "TOPOCAE_WORKSPACE_ROOT": root_dir,
            },
        ):
            with self.assertRaises(ValueError):
                resolve_workspace_path(
                    r"W:\python\FEA\dataset_generator_evil\sample.json"
                )

    def test_different_drive_and_unc_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, patch.dict(
            os.environ,
            {
                "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": r"W:\python\FEA\dataset_generator",
                "TOPOCAE_WORKSPACE_ROOT": root_dir,
            },
        ):
            for value in (
                r"D:\python\FEA\dataset_generator\sample.json",
                r"\\server\share\sample.json",
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    resolve_workspace_path(value)

    def test_relative_and_windows_parent_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, patch.dict(
            os.environ,
            {
                "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": r"W:\python\FEA\dataset_generator",
                "TOPOCAE_WORKSPACE_ROOT": root_dir,
            },
        ):
            for value in (
                "../outside.json",
                r"W:\python\FEA\dataset_generator\..\outside.json",
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    resolve_workspace_path(value)

    def test_native_absolute_path_outside_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside:
            with patch.dict(os.environ, {"TOPOCAE_WORKSPACE_ROOT": root_dir}):
                with self.assertRaises(ValueError):
                    resolve_workspace_path(Path(outside) / "sample.json")

    def test_existing_symlink_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside:
            root = Path(root_dir)
            link = root / "outside_link"
            try:
                link.symlink_to(Path(outside), target_is_directory=True)
            except OSError:
                result = subprocess.run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), outside],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    self.skipTest(
                        "symlink and junction creation unavailable: "
                        + result.stderr.strip()
                    )
            with patch.dict(os.environ, {"TOPOCAE_WORKSPACE_ROOT": str(root)}):
                with self.assertRaises(ValueError):
                    resolve_workspace_path("outside_link/sample.json")


if __name__ == "__main__":
    unittest.main()
