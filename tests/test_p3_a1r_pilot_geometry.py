from __future__ import annotations

from pathlib import Path
import unittest

from utils.cae_query_ground_p3_a1r.pilot_geometry import (
    build_graph_with_metric_evidence,
    build_multi_instance_result,
    metric_table,
)


ROOT = Path(__file__).resolve().parents[1]


class PilotGeometryTests(unittest.TestCase):
    def test_three_real_same_type_holes_have_graph_instances_and_metrics(self) -> None:
        result = build_multi_instance_result("through_hole_round", 3, 2026072201)
        graph, _labels = build_graph_with_metric_evidence(result, ROOT)
        self.assertEqual(3, len(graph["feature_sets"]))
        self.assertEqual({"through_hole_round"}, {item["macro_feature"] for item in graph["feature_sets"]})
        self.assertEqual({1, 2, 3}, {item["instance_id"] for item in graph["feature_sets"]})
        self.assertTrue(all(item["geometry_parameters"]["diameter"] > 0 for item in graph["feature_sets"]))
        self.assertEqual(3, len(metric_table(graph, "through_hole_round")))

    def test_tie_design_uses_equal_true_geometry_not_metadata_only(self) -> None:
        result = build_multi_instance_result("slot_o", 2, 2026072202, tie=True)
        graph, _labels = build_graph_with_metric_evidence(result, ROOT)
        rows = metric_table(graph, "slot_o")
        self.assertAlmostEqual(rows[0]["comparison_value"], rows[1]["comparison_value"], places=9)
        self.assertAlmostEqual(rows[0]["nearest_distance"], rows[1]["nearest_distance"], places=6)

    def test_two_supported_ribs_are_distinct_graph_instances(self) -> None:
        result = build_multi_instance_result("rib", 2, 2026072203)
        graph, _labels = build_graph_with_metric_evidence(result, ROOT)
        self.assertEqual(2, len(graph["feature_sets"]))
        self.assertEqual({"rib"}, {item["macro_feature"] for item in graph["feature_sets"]})
        self.assertEqual({1, 2}, {item["instance_id"] for item in graph["feature_sets"]})


if __name__ == "__main__":
    unittest.main()
