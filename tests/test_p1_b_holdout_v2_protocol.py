from __future__ import annotations

from pathlib import Path
import json
import unittest

import run_p1_b_synthetic_holdout_v2_once as protocol


ROOT = Path(__file__).resolve().parents[1]


class TestP1BHoldoutV2Protocol(unittest.TestCase):
    def test_truth_strip_removes_all_labels(self) -> None:
        graph = {
            "feature_sets": [{"macro_feature": "rib", "node_indices": [0]}],
            "nodes": [{"idx": 0, "semantic_label": 7, "instance_id": 1, "macro_feature": "rib", "macro_feature_id": 12, "semantic_role": "rib_side_plane", "attributes": {}}],
            "edges": [],
            "p1_b_holdout_provenance": {"target": "rib"},
        }
        stripped = protocol._strip_truth(graph)
        protocol._assert_truth_free(stripped)
        self.assertEqual(stripped["feature_sets"], [])
        self.assertIn("macro_feature", graph["nodes"][0])

    def test_frozen_selection_and_zero_leakage(self) -> None:
        selection = json.loads((ROOT / "reports" / "p1_b_freeze_v1" / "preholdout_selection.sha256.json").read_text(encoding="utf-8"))
        self.assertTrue(selection["verified_after_write"])
        leakage = json.loads((ROOT / "output_generation_p1_b_synthetic_holdout_v2" / "leakage_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(leakage["status"], "pass")
        self.assertEqual(leakage["base_collision_with_training"], 0)
        self.assertEqual(leakage["exact_collision_with_training"], 0)
        self.assertEqual(leakage["near_collision_with_training"], 0)

    def test_binary_and_multiclass_metric_helpers(self) -> None:
        self.assertEqual(protocol._binary_metrics([1, 1, 0], [1, 0, 0])["f1"], 2 / 3)
        metrics = protocol._classification_metrics([0, 1, 1], [0, 1, 0])
        self.assertEqual(metrics["active_class_count"], 2)
        self.assertGreater(metrics["macro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
