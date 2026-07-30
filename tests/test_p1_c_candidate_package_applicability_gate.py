from __future__ import annotations

import json
from pathlib import Path
import unittest

from build_p1_c_candidate_training_package_strict_v2 import EXPECTED_GATE_SHA256, GATE_PATH, _strict_gate
from p1_c.strict_candidate_provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]


class CandidatePackageApplicabilityGateTests(unittest.TestCase):
    def test_gate_file_hash_is_frozen(self) -> None:
        self.assertEqual(file_sha256(GATE_PATH), EXPECTED_GATE_SHA256)

    def test_real_single_only_package_passes_only_applicable_kinds(self) -> None:
        path = ROOT / "reports" / "p1_c_strict_training_runs_v2" / "single_only_seed_20260718" / "candidate_training_v1" / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        passed, checks = _strict_gate("single_only", metadata)
        self.assertTrue(passed)
        self.assertEqual(checks["hard_negative_applicability"], "single_feature_only_three_applicable_kinds")

    def test_single_only_does_not_waive_common_oracle_gate(self) -> None:
        metadata = {
            "dimensions": {"face_embedding_dim": 128, "edge_embedding_dim": 20},
            "splits": {
                "train": {"hard_negative_kind_counts": {name: 1 for name in (
                    "missing_truth_face_subset", "adjacent_stock_or_transition_superset", "wrong_macro_hard_confusion"
                )}},
                "val": {"proposal_oracle_recall_iou_0_5": 0.94, "all_seven_sources_present": True}
            }
        }
        passed, _ = _strict_gate("single_only", metadata)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
