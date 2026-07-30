from __future__ import annotations

import json
from pathlib import Path
import unittest

from utils.p1_c_topology_verifier_polarity_v1 import polarity_strength
from utils.p1_c_strict_run_paths import DATASET_ROOT, ROOT


PLAN = ROOT / "configs" / "p1_c" / "stage6_iteration2_candidate_verifier_calibration_v1.json"


class Iteration2CalibrationTests(unittest.TestCase):
    def test_plan_is_final_allowed_iteration_and_forbids_leakage(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["iteration_index"], 2)
        self.assertEqual(plan["maximum_allowed_iterations"], 2)
        self.assertFalse(plan["contracts"]["real_dev_used"])
        self.assertFalse(plan["contracts"]["holdout_v3_used"])
        self.assertFalse(plan["change_scope"]["class_names_used"])
        self.assertFalse(plan["change_scope"]["thresholds_per_class_or_part"])

    def test_truth_feature_cylinder_has_decisive_strength_without_class_input(self) -> None:
        graph_path = next((DATASET_ROOT / "graphs_polarity_v1").glob("*pocket_circular*target_only*.json"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        feature = next(record for record in graph["feature_sets"] if record["macro_feature"] == "pocket_circular")
        self.assertGreater(polarity_strength(graph, feature["node_indices"]), 0.999)

    def test_non_cylindrical_candidate_has_zero_strength(self) -> None:
        graph_path = next((DATASET_ROOT / "graphs_polarity_v1").glob("*chamfer*target_only*.json"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        feature = next(record for record in graph["feature_sets"] if record["macro_feature"] == "chamfer")
        self.assertEqual(polarity_strength(graph, feature["node_indices"]), 0.0)


if __name__ == "__main__":
    unittest.main()
