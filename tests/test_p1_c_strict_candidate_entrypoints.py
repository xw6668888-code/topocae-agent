from __future__ import annotations

import json
import unittest

from p1_c.strict_candidate_provenance_v2 import PLAN_PATH, file_sha256, preflight_gnn_source
from utils.p1_c_strict_run_paths import StrictVariantSeedRunRoot, VARIANT_SPECS, candidate_model_name, strict_run_dir, training_metadata_path


class StrictCandidateEntrypointTests(unittest.TestCase):
    def test_exactly_nine_approved_runs_are_frozen(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        expected = {f"{variant}_seed_{seed}" for variant, spec in VARIANT_SPECS.items() for seed in spec["seeds"]}
        self.assertEqual(set(plan["source_gnn_runs"]), expected)
        self.assertEqual(len(expected), 9)

    def test_every_frozen_gnn_source_passes_hash_preflight(self) -> None:
        for variant, spec in VARIANT_SPECS.items():
            for seed in spec["seeds"]:
                result = preflight_gnn_source(variant, seed)
                self.assertEqual(result["run_name"], f"{variant}_seed_{seed}")

    def test_legacy_path_shape_is_mapped_only_to_selected_variant(self) -> None:
        adapter = StrictVariantSeedRunRoot("clean_7200")
        self.assertEqual(adapter / "same_instance_seed_20260718", strict_run_dir("clean_7200", 20260718))
        with self.assertRaises(ValueError):
            _ = adapter / "iteration2_seed_20260718"

    def test_variant_metadata_and_model_name_are_fail_closed(self) -> None:
        self.assertIn("strict_concept_r2", str(training_metadata_path("clean_5000")))
        self.assertEqual(candidate_model_name("no_canonical_patch"), "candidate_model_no_canonical_patch_v1")
        self.assertEqual(candidate_model_name("clean_7200"), "candidate_model_v1")
        with self.assertRaises(ValueError):
            training_metadata_path("old_invalid_output")

    def test_plan_hash_is_stable_and_contracts_forbid_leakage(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(PLAN_PATH), "053d335682f9dbe46b0097e46a6cc39074f5a245ed53a5a12bf337215c773a7c")
        self.assertTrue(plan["frozen_before_candidate_models"])
        self.assertFalse(plan["contracts"]["real_dev_used"])
        self.assertFalse(plan["contracts"]["holdout_v3_used"])
        self.assertFalse(plan["contracts"]["test_part_s3_accessed"])


if __name__ == "__main__":
    unittest.main()
