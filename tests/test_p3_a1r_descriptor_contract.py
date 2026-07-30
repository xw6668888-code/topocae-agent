from __future__ import annotations

from pathlib import Path
import unittest

from utils.cae_query_ground_p3_a1r.pilot_geometry import (
    COMPARISON_METRIC,
    build_graph_with_metric_evidence,
    build_multi_instance_result,
    metric_table,
)


ROOT = Path(__file__).resolve().parents[1]


class P3A1RDescriptorContractTests(unittest.TestCase):
    def build(self, feature_type: str, seed: int) -> tuple[dict, list[dict]]:
        result = build_multi_instance_result(feature_type, 2, seed)
        graph, _labels = build_graph_with_metric_evidence(result, ROOT)
        return graph, metric_table(graph, feature_type)

    def test_comparison_descriptor_is_explicitly_not_a_complete_feature_description(self) -> None:
        graph, rows = self.build("blind_hole_round", 2026072301)
        for feature_set, row in zip(graph["feature_sets"], rows, strict=True):
            contract = feature_set["descriptor_contract"]
            self.assertEqual("same_type_instance_ordering_only", contract["scope"])
            self.assertFalse(contract["is_complete_feature_description"])
            self.assertEqual(["diameter", "depth"], contract["complete_dimension_keys"])
            self.assertIn("diameter", feature_set["geometry_parameters"])
            self.assertIn("depth", feature_set["geometry_parameters"])
            self.assertFalse(row["is_complete_feature_description"])

    def test_human_recommended_engineering_metrics_drive_formal_generation(self) -> None:
        expected = {
            "rib": ("thickness", "rib_thickness"),
            "fillet": ("radius", "fillet_radius"),
            "chamfer": ("length", "chamfer_leg_length"),
        }
        for offset, (feature_type, (metric, semantics)) in enumerate(expected.items()):
            with self.subTest(feature_type=feature_type):
                graph, rows = self.build(feature_type, 2026072310 + offset)
                self.assertEqual(metric, COMPARISON_METRIC[feature_type])
                self.assertTrue(all(row["comparison_semantics"] == semantics for row in rows))
                for feature_set, row in zip(graph["feature_sets"], rows, strict=True):
                    self.assertAlmostEqual(
                        feature_set["geometry_parameters"][metric],
                        row["comparison_value"],
                    )

    def test_counterbore_and_countersink_diameter_means_small_hole_diameter(self) -> None:
        for offset, feature_type in enumerate(("counterbore_hole", "countersink_hole")):
            with self.subTest(feature_type=feature_type):
                graph, rows = self.build(feature_type, 2026072320 + offset)
                for feature_set, row in zip(graph["feature_sets"], rows, strict=True):
                    params = feature_set["geometry_parameters"]
                    self.assertAlmostEqual(params["small_diameter"], params["diameter"])
                    self.assertGreater(params["large_diameter"], params["small_diameter"])
                    self.assertAlmostEqual(params["small_diameter"], row["comparison_value"])

    def test_v_slot_width_is_finished_opening_not_construction_profile_width(self) -> None:
        graph, rows = self.build("slot_v", 2026072330)
        for feature_set, row in zip(graph["feature_sets"], rows, strict=True):
            params = feature_set["geometry_parameters"]
            expected = (
                params["profile_width"]
                * params["profile_depth"]
                / (params["profile_depth"] + params["top_margin"])
            )
            self.assertAlmostEqual(expected, params["opening_width"])
            self.assertLess(params["opening_width"], params["profile_width"])
            self.assertAlmostEqual(params["opening_width"], row["comparison_value"])


if __name__ == "__main__":
    unittest.main()
