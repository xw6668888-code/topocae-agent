import unittest

from generate_p4_a1_interim_reports import _candidate_freeze_manifest


class M1CandidateManifestTests(unittest.TestCase):
    def test_path_portability_assets_are_candidate_hashed(self) -> None:
        manifest = _candidate_freeze_manifest()
        paths = {asset["path"] for asset in manifest["assets"]}

        self.assertEqual("NOT_FROZEN_H4_AND_RUNTIME_PENDING", manifest["status"])
        self.assertFalse(manifest["test_unlocked"])
        self.assertTrue(
            {
                "utils/workspace_paths.py",
                "verify_p4_a1_frozen_assets.py",
                "tests/test_workspace_paths.py",
                "tests/test_workspace_paths_edge_cases.py",
                "tests/test_m1_a1_path_integrations.py",
            }.issubset(paths)
        )


if __name__ == "__main__":
    unittest.main()
