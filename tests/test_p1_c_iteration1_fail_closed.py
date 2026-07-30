from __future__ import annotations

import json
import unittest

from p1_c.iteration1_provenance import GRAPH_MANIFEST, GRAPH_MANIFEST_SHA256, PLAN, PLAN_SHA256, file_sha256, preflight_iteration_source
from utils.p1_c_iteration1_run_paths import IterationRunDir


class Iteration1FailClosedTests(unittest.TestCase):
    def test_frozen_plan_and_graph_inventory_hashes(self) -> None:
        self.assertEqual(file_sha256(PLAN), PLAN_SHA256)
        self.assertEqual(file_sha256(GRAPH_MANIFEST), GRAPH_MANIFEST_SHA256)
        manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["graph_count"], 7200)
        self.assertFalse(manifest["contracts"]["feature_uses_labels"])

    def test_all_six_sources_pass_preflight(self) -> None:
        for variant in ("clean_5000", "clean_7200"):
            for seed in (20260718, 20260719, 20260720):
                self.assertEqual(preflight_iteration_source(variant, seed)["run_name"], f"{variant}_seed_{seed}")

    def test_path_proxy_keeps_gnn_read_only_and_iteration_outputs_separate(self) -> None:
        run = IterationRunDir("clean_7200", 20260718)
        self.assertIn("p1_c_strict_training_runs_v2", str(run / "predictions.pt"))
        self.assertIn("p1_c_stage6_iteration1_runs_v1", str(run / "candidate_training_v1"))


if __name__ == "__main__":
    unittest.main()
