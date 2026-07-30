from __future__ import annotations

import unittest

from proposal_miss_atlas import (
    area_compatible,
    classify_no_proposal_failure,
    nearest_candidates,
    summarize_misses,
)


class ProposalMissAtlasTest(unittest.TestCase):
    def test_area_compatible_uses_absolute_or_relative_tolerance(self):
        self.assertTrue(area_compatible(100.0, 109.0, absolute_tolerance=10.0, relative_tolerance=0.02))
        self.assertTrue(area_compatible(1000.0, 1015.0, absolute_tolerance=10.0, relative_tolerance=0.02))
        self.assertFalse(area_compatible(100.0, 115.0, absolute_tolerance=10.0, relative_tolerance=0.02))

    def test_nearest_candidates_sorts_by_relative_error(self):
        candidates = [
            {"proposal_macro": "rib", "area": 80.0, "source": "same_instance", "face_indices": [1]},
            {"proposal_macro": "rib", "area": 105.0, "source": "topology_rule", "face_indices": [2]},
            {"proposal_macro": "fillet", "area": 101.0, "source": "topology_rule", "face_indices": [3]},
        ]

        nearest = nearest_candidates(100.0, candidates, macro="rib", limit=2)

        self.assertEqual([item["area"] for item in nearest], [105.0, 80.0])
        self.assertAlmostEqual(nearest[0]["relative_error"], 0.05)

    def test_classifies_no_same_macro_candidates(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "fillet", "area": 10.0, "source": "topology_rule", "face_indices": [7]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "no_same_macro_candidates")
        self.assertEqual(result["same_macro_candidate_count"], 0)

    def test_classifies_same_macro_area_near_miss(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "rib", "area": 120.0, "source": "same_instance", "face_indices": [1, 2]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "same_macro_area_near_miss")
        self.assertAlmostEqual(result["nearest_same_macro"][0]["relative_error"], 0.20)

    def test_classifies_same_macro_split_or_merge_suspected(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [
            {"proposal_macro": "rib", "area": 45.0, "source": "same_instance", "face_indices": [1]},
            {"proposal_macro": "rib", "area": 55.0, "source": "same_instance", "face_indices": [2]},
        ]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "same_macro_split_or_merge_suspected")
        self.assertAlmostEqual(result["same_macro_area_sum"], 100.0)

    def test_classifies_cross_macro_area_compatible(self):
        failure = {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3}
        candidates = [{"proposal_macro": "web", "area": 101.0, "source": "same_instance", "face_indices": [5]}]

        result = classify_no_proposal_failure(
            failure,
            candidates,
            absolute_tolerance=10.0,
            relative_tolerance=0.02,
        )

        self.assertEqual(result["miss_cause"], "cross_macro_area_compatible")
        self.assertEqual(result["nearest_cross_macro"][0]["proposal_macro"], "web")

    def test_summarize_misses_counts_macro_part_and_cause(self):
        misses = [
            {"sample": "part1", "macro": "rib", "miss_cause": "no_same_macro_candidates"},
            {"sample": "part1", "macro": "rib", "miss_cause": "same_macro_area_near_miss"},
            {"sample": "part2", "macro": "fillet", "miss_cause": "no_same_macro_candidates"},
        ]

        summary = summarize_misses(misses)

        self.assertEqual(summary["miss_count"], 3)
        self.assertEqual(summary["by_macro"]["rib"], 2)
        self.assertEqual(summary["by_part"]["part1"], 2)
        self.assertEqual(summary["by_cause"]["no_same_macro_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
