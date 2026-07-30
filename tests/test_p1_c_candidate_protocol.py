from __future__ import annotations

import unittest

import torch

from train_p1_c_candidate_classifier_v2 import SampleInstanceBalancedCandidateRecords
from utils.p1_c_run_paths import LegacySeedRunRoot, p1_c_run_dir


def _record(sample_id: str, valid: bool, instance_id: int | None) -> dict:
    return {
        "sample_id": sample_id,
        "validity_target": torch.tensor(float(valid), dtype=torch.float32),
        "matched_truth_instance_id": instance_id,
        "engineered_features": torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32),
    }


class CandidateWeightProtocolTests(unittest.TestCase):
    def test_each_sample_has_unit_total_mass(self) -> None:
        records = [
            _record("single", True, 1),
            _record("single", False, None),
            _record("multi", True, 1),
            _record("multi", True, 1),
            _record("multi", True, 2),
            _record("multi", False, None),
            _record("multi", False, None),
        ]
        dataset = SampleInstanceBalancedCandidateRecords(records, False)
        totals = {"single": 0.0, "multi": 0.0}
        for record in dataset.records:
            totals[record["sample_id"]] += float(record["candidate_loss_weight"])
        self.assertAlmostEqual(totals["single"], 1.0, places=6)
        self.assertAlmostEqual(totals["multi"], 1.0, places=6)

    def test_positive_and_negative_mass_are_balanced(self) -> None:
        records = [
            _record("sample", True, 1),
            _record("sample", True, 1),
            _record("sample", True, 2),
            _record("sample", False, None),
            _record("sample", False, None),
        ]
        dataset = SampleInstanceBalancedCandidateRecords(records, False)
        weights = [float(record["candidate_loss_weight"]) for record in dataset.records]
        self.assertAlmostEqual(sum(weights[:3]), 0.5, places=6)
        self.assertAlmostEqual(sum(weights[3:]), 0.5, places=6)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2], places=6)

    def test_no_canonical_patch_replaces_only_patch_count(self) -> None:
        dataset = SampleInstanceBalancedCandidateRecords(
            [_record("sample", True, 1)], no_canonical_patch=True
        )
        record = dataset[0]
        self.assertEqual(float(record["engineered_features"][0]), 3.0)
        self.assertEqual(float(record["engineered_features"][1]), 3.0)
        self.assertEqual(float(record["engineered_features"][2]), 1.0)


class RunPathProtocolTests(unittest.TestCase):
    def test_legacy_seed_path_routes_to_frozen_p1_c_variant(self) -> None:
        seed = 20260718
        self.assertEqual(
            LegacySeedRunRoot() / f"same_instance_seed_{seed}",
            p1_c_run_dir(seed),
        )

    def test_unexpected_legacy_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = LegacySeedRunRoot() / "unexpected"


if __name__ == "__main__":
    unittest.main()
