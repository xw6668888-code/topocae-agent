from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from utils.workspace_paths import resolve_workspace_path


class ResolveWorkspacePathEdgeCaseTests(unittest.TestCase):
    def test_drive_relative_and_rooted_windows_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, patch.dict(
            os.environ,
            {
                "TOPOCAE_ORIGINAL_WORKSPACE_ROOT": r"W:\python\FEA\dataset_generator",
                "TOPOCAE_WORKSPACE_ROOT": root_dir,
            },
        ):
            for value in (r"C:relative\sample.json", r"\rooted\sample.json"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    resolve_workspace_path(value)

    def test_workspace_root_must_be_absolute(self) -> None:
        with patch.dict(os.environ, {"TOPOCAE_WORKSPACE_ROOT": "relative/root"}):
            with self.assertRaises(ValueError):
                resolve_workspace_path("sample.json")

    def test_empty_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_workspace_path("")


if __name__ == "__main__":
    unittest.main()
