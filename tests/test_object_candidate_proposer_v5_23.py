from __future__ import annotations

import unittest
from unittest.mock import patch


class ObjectCandidateProposerV523Test(unittest.TestCase):
    def test_topology_rule_proposals_include_review_targeted_residual_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {"nodes": [], "edges": [], "edge_attributes": []}
        faces = []
        finders = [
            "find_through_hole_opening_candidates", "find_blind_hole_candidates",
            "find_counterbore_candidates", "find_countersink_candidates",
            "find_cylinder_boss_candidates", "find_obround_hole_candidates",
            "find_rectangular_prism_candidates", "find_supported_rectangular_boss_candidates",
            "find_u_slot_candidates", "find_rectangular_slot_candidates",
            "find_o_slot_candidates", "find_v_slot_stock_candidates",
            "find_mixed_rib_candidates", "find_rib_stock_candidates",
            "find_existing_rib_web_cap_completion_candidates",
            "find_transition_cap_rib_candidates", "find_pocket_embedded_rib_candidates",
            "find_existing_rib_transition_absorption_candidates",
            "find_fillet_stock_candidates", "find_chamfer_stock_candidates",
            "find_review_targeted_rib_candidates",
            "find_review_targeted_fillet_candidates",
            "find_review_targeted_chamfer_candidates",
        ]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for f in finders:
                val = [[1, 2, 3]] if "rib" in f else [[4]] if "fillet" in f else [[5]] if "chamfer" in f else []
                stack.enter_context(patch("utils.object_candidate_proposer." + f, return_value=val))
            proposals = topology_rule_proposals(graph, faces)

        simplified = sorted(
            (proposal["proposal_macro"], proposal["face_indices"], proposal["assembly_reasons"])
            for proposal in proposals
        )
        self.assertIn(("chamfer", [5], ["topology_review_targeted_chamfer_residual"]), simplified)
        self.assertIn(("fillet", [4], ["topology_review_targeted_fillet_residual"]), simplified)
        self.assertIn(("rib", [1, 2, 3], ["topology_review_targeted_rib_residual"]), simplified)


if __name__ == "__main__":
    unittest.main()
