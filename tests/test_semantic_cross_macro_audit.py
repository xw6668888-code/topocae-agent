from __future__ import annotations

import unittest

from semantic_cross_macro_audit import (
    classify_review_category,
    macro_pair_key,
    simulate_policy,
    summarize_cross_macro_misses,
)


class SemanticCrossMacroAuditTest(unittest.TestCase):
    def test_macro_pair_key_is_ordered_and_stable(self):
        self.assertEqual(
            macro_pair_key("counterbore_hole", "through_hole_round"),
            "counterbore_hole|through_hole_round",
        )
        self.assertEqual(
            macro_pair_key("through_hole_round", "counterbore_hole"),
            "counterbore_hole|through_hole_round",
        )

    def test_classifies_hole_subfeature_vs_composite_hole(self):
        self.assertEqual(
            classify_review_category("through_hole_round", "counterbore_hole"),
            "hole_subfeature_vs_composite_hole",
        )
        self.assertEqual(
            classify_review_category("countersink_hole", "through_hole_round"),
            "hole_subfeature_vs_composite_hole",
        )

    def test_classifies_rectangular_recess_slot_vs_pocket(self):
        self.assertEqual(
            classify_review_category("slot_rect", "pocket_rect"),
            "rectangular_recess_slot_vs_pocket",
        )

    def test_classifies_rib_vs_boss_or_web(self):
        self.assertEqual(
            classify_review_category("rib", "rectangular_boss"),
            "thinwall_rib_vs_boss_or_web",
        )
        self.assertEqual(
            classify_review_category("web", "slot_v"),
            "thinwall_rib_vs_boss_or_web",
        )

    def test_classifies_transition_conflict(self):
        self.assertEqual(
            classify_review_category("fillet", "blind_hole_round"),
            "transition_vs_nontransition_conflict",
        )

    def test_summarize_cross_macro_misses_counts_pairs_and_categories(self):
        misses = [
            {
                "sample": "part1",
                "macro": "through_hole_round",
                "area": 100.0,
                "miss_cause": "cross_macro_area_compatible",
                "nearest_cross_macro": [{"proposal_macro": "counterbore_hole", "area": 100.1, "relative_error": 0.001}],
            },
            {
                "sample": "part2",
                "macro": "slot_rect",
                "area": 72.0,
                "miss_cause": "cross_macro_area_compatible",
                "nearest_cross_macro": [{"proposal_macro": "pocket_rect", "area": 72.0, "relative_error": 0.0}],
            },
        ]

        rows, summary = summarize_cross_macro_misses(misses)

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["cross_macro_compatible_count"], 2)
        self.assertEqual(summary["by_pair"]["counterbore_hole|through_hole_round"], 1)
        self.assertEqual(summary["by_category"]["hole_subfeature_vs_composite_hole"], 1)
        self.assertEqual(summary["by_category"]["rectangular_recess_slot_vs_pocket"], 1)

    def test_simulate_policy_recovers_only_allowed_categories(self):
        rows = [
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "counterbore_hole|through_hole_round"},
            {"review_category": "rectangular_recess_slot_vs_pocket", "pair": "pocket_rect|slot_rect"},
            {"review_category": "transition_vs_nontransition_conflict", "pair": "fillet|through_hole_round"},
        ]

        result = simulate_policy("safe_hole_and_rectangular_recess", rows, baseline_miss_count=65)

        self.assertEqual(result["policy"], "safe_hole_and_rectangular_recess")
        self.assertEqual(result["recovered_count"], 2)
        self.assertEqual(result["remaining_miss_count"], 63)
        self.assertEqual(result["recovered_by_category"]["hole_subfeature_vs_composite_hole"], 1)
        self.assertEqual(result["recovered_by_category"]["rectangular_recess_slot_vs_pocket"], 1)


if __name__ == "__main__":
    unittest.main()
