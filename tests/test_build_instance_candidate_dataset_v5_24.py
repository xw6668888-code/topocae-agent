from __future__ import annotations

import unittest

from build_instance_candidate_dataset import (
    assign_candidate_sample_weight,
    build_assembly_reason_positive_clone_proposals,
    parse_assembly_reason_macro_pairs,
)


class BuildInstanceCandidateDatasetV524Test(unittest.TestCase):
    def test_parse_assembly_reason_macro_pairs(self):
        pairs = parse_assembly_reason_macro_pairs(
            "topology_review_targeted_rib_residual:rib,topology_review_targeted_fillet_residual:fillet"
        )

        self.assertEqual(
            pairs,
            {
                ("topology_review_targeted_rib_residual", "rib"),
                ("topology_review_targeted_fillet_residual", "fillet"),
            },
        )

    def test_build_assembly_reason_positive_clone_proposals(self):
        truth_instances = [
            {"macro_feature": "rib", "face_indices": [1, 2, 3]},
            {"macro_feature": "fillet", "face_indices": [4]},
        ]

        clones = build_assembly_reason_positive_clone_proposals(
            truth_instances,
            {("topology_review_targeted_rib_residual", "rib")},
        )

        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0]["proposal_macro"], "rib")
        self.assertEqual(clones[0]["source"], "topology_rule")
        self.assertEqual(clones[0]["face_indices"], [1, 2, 3])
        self.assertEqual(clones[0]["assembly_reasons"], ["topology_review_targeted_rib_residual"])
        self.assertEqual(clones[0]["augmentation"], "assembly_reason_positive_clone")

    def test_assembly_reason_positive_clone_gets_clone_weight(self):
        weight, action = assign_candidate_sample_weight(
            source="topology_rule",
            proposal_macro="rib",
            label="rib",
            augmentation="assembly_reason_positive_clone",
            source_macro_positive_clone_pairs=set(),
            reject_downweight_source_macro_pairs=set(),
            default_sample_weight=1.0,
            positive_nontruth_weight=3.0,
            positive_clone_weight=8.0,
            downweighted_reject_weight=0.35,
        )

        self.assertEqual(weight, 8.0)
        self.assertEqual(action, "positive_assembly_reason_clone")


if __name__ == "__main__":
    unittest.main()
