from __future__ import annotations

import json
from pathlib import Path
import unittest

from run_paper_recognition_pipeline import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ExcludedResearchArtifactError,
    assert_path_allowed,
    load_runtime_config,
    verify_label_schema,
)


class PaperV1FreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.policy = load_runtime_config(DEFAULT_CONFIG)

    def test_frozen_model_paths_exist_and_are_outside_excluded_artifacts(self):
        for name, record in self.config["models"].items():
            if not record.get("enabled"):
                continue
            path = assert_path_allowed(
                REPO_ROOT / record["path"],
                self.policy,
                purpose=name,
            )
            self.assertTrue(path.is_file(), name)

    def test_label_schema_source_and_canonical_hashes_match(self):
        result = verify_label_schema(self.config, self.policy)
        self.assertEqual(result["counts"]["macro_feature"], 36)
        self.assertEqual(result["counts"]["semantic_subtype"], 63)

    def test_direct_excluded_input_is_rejected(self):
        with self.assertRaises(ExcludedResearchArtifactError):
            assert_path_allowed(
                REPO_ROOT / "test_part_s3" / "part1.stp",
                self.policy,
                purpose="test input",
                recursive=True,
            )

    def test_derived_excluded_output_is_rejected(self):
        with self.assertRaises(ExcludedResearchArtifactError):
            assert_path_allowed(
                REPO_ROOT / "output_real_step_s3_example" / "graphs",
                self.policy,
                purpose="test input",
                recursive=True,
            )

    def test_recursive_repository_root_is_rejected_before_enumeration(self):
        with self.assertRaises(ExcludedResearchArtifactError):
            assert_path_allowed(
                REPO_ROOT,
                self.policy,
                purpose="test input",
                recursive=True,
            )

    def test_real_dev_s2_graph_is_allowed(self):
        path = assert_path_allowed(
            REPO_ROOT / "output_real_step_s2_v5_4_g_looprole" / "graphs" / "part1.json",
            self.policy,
            purpose="test input",
            recursive=True,
        )
        self.assertTrue(path.is_file())

    def test_paper_config_contains_no_secret_values(self):
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(payload["security"]["credentials_required"])
        self.assertEqual(
            payload["security"]["credential_source_if_future_service_is_enabled"],
            "environment_variables_only",
        )


if __name__ == "__main__":
    unittest.main()
