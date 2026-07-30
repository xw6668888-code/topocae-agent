import unittest

from utils.ablation_dataset_policy import (
    audit_ablation_sample,
    balanced_target_sequence,
    feature_set_footprint,
)


def feature_set(instance_id, macro_feature, params):
    return {
        "instance_id": instance_id,
        "macro_feature": macro_feature,
        "node_indices": [instance_id],
        "raw_parameters": params,
        "geometry_parameters": params,
    }


class AblationDatasetPolicyTest(unittest.TestCase):
    def test_single_feature_policy_requires_one_feature_and_no_composite_stock(self):
        graph = {
            "stock_profile": {"stock_type": "block"},
            "feature_sets": [
                feature_set(1, "through_hole_round", {"cx": 0.0, "cy": 0.0, "R": 4.0, "axis": "z"}),
            ],
        }

        report = audit_ablation_sample(graph, policy="single_feature")

        self.assertTrue(report["valid"])

        graph["feature_sets"].append(
            feature_set(2, "blind_hole_round", {"cx": 20.0, "cy": 0.0, "R": 4.0, "axis": "z"})
        )
        report = audit_ablation_sample(graph, policy="single_feature")

        self.assertFalse(report["valid"])
        self.assertIn("feature_set_count_not_1", report["reasons"])

        graph["feature_sets"] = graph["feature_sets"][:1]
        graph["stock_profile"] = {"stock_type": "composite"}
        report = audit_ablation_sample(graph, policy="single_feature")

        self.assertFalse(report["valid"])
        self.assertIn("composite_stock_disallowed", report["reasons"])

    def test_nonintersecting_policy_rejects_overlapping_feature_footprints(self):
        graph = {
            "stock_profile": {"stock_type": "plate"},
            "feature_sets": [
                feature_set(1, "through_hole_round", {"cx": 0.0, "cy": 0.0, "R": 6.0, "axis": "z"}),
                feature_set(2, "blind_hole_round", {"cx": 8.0, "cy": 0.0, "R": 5.0, "axis": "z"}),
            ],
        }

        report = audit_ablation_sample(graph, policy="nonintersecting_multi", max_feature_sets=4)

        self.assertFalse(report["valid"])
        self.assertIn("feature_footprint_conflict", report["reasons"])

    def test_nonintersecting_policy_allows_cavity_rib_parent_pocket_overlap(self):
        graph = {
            "stock_profile": {"stock_type": "plate"},
            "feature_sets": [
                feature_set(1, "pocket_rect", {"cx": 0.0, "cy": 0.0, "w": 30.0, "l": 40.0}),
                feature_set(
                    2,
                    "rib",
                    {
                        "cx": 0.0,
                        "cy": 0.0,
                        "w": 3.0,
                        "l": 30.0,
                        "placement_mode": "cavity",
                        "rib_body_kind": "cavity_wall_rib",
                        "parent_macro_feature": "pocket_rect",
                        "parent_instance_id": 1,
                    },
                ),
            ],
        }

        report = audit_ablation_sample(graph, policy="nonintersecting_multi", max_feature_sets=4)

        self.assertTrue(report["valid"], report["reasons"])

    def test_nonintersecting_policy_accepts_separated_features(self):
        graph = {
            "stock_profile": {"stock_type": "beam"},
            "feature_sets": [
                feature_set(1, "through_hole_round", {"cx": -30.0, "cy": 0.0, "R": 4.0, "axis": "z"}),
                feature_set(2, "pocket_rect", {"cx": 35.0, "cy": 0.0, "width": 12.0, "length": 20.0}),
            ],
        }

        report = audit_ablation_sample(graph, policy="nonintersecting_multi", max_feature_sets=4)

        self.assertTrue(report["valid"])

    def test_feature_set_footprint_uses_axis_specific_planes_for_side_holes(self):
        fp = feature_set_footprint(
            feature_set(1, "through_hole_round", {"cy": 10.0, "cz": 20.0, "R": 3.0, "axis": "x"})
        )

        self.assertEqual(fp["zone_plane"], "yz")
        self.assertEqual(fp["cx"], 10.0)
        self.assertEqual(fp["cy"], 20.0)

    def test_balanced_target_sequence_repeats_targets_evenly(self):
        sequence = balanced_target_sequence(["a", "b", "c"], 8, seed=7)
        counts = {target: sequence.count(target) for target in ["a", "b", "c"]}

        self.assertEqual(len(sequence), 8)
        self.assertEqual(sum(counts.values()), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
