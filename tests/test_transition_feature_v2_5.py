import math
import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

from features.chamfer_feature import ChamferGenerator
from features.fillet_feature import FilletGenerator
from features.transition_edge_selection import (
    compute_transition_size_limits,
    extract_transition_edge_candidates,
    public_transition_parameters,
)
from utils.ontology import load_layer_index


def semantic_index():
    return load_layer_index("face_semantic_subtype_ontology.yaml", "face_semantic_subtype")


class TransitionFeatureV25Test(unittest.TestCase):
    def test_transition_size_limit_keeps_hole_mouth_treatments_small(self):
        min_size, max_size = compute_transition_size_limits(
            {
                "curve_type": "circle",
                "curve_radius": 10.0,
                "length": 2.0 * math.pi * 10.0,
            },
            L=120.0,
            W=100.0,
            H=50.0,
            config={"min_size": 0.8, "max_size": 4.0},
        )

        self.assertGreaterEqual(min_size, 0.8)
        self.assertLessEqual(max_size, 2.2)
        self.assertGreater(max_size, min_size)

    def test_edge_candidates_obey_minimum_length(self):
        box = BRepPrimAPI_MakeBox(gp_Pnt(-50.0, -50.0, 0.0), gp_Pnt(50.0, 50.0, 40.0)).Shape()

        candidates = extract_transition_edge_candidates(
            box,
            existing_params=[],
            min_edge_length=60.0,
            midpoint_clearance=0.0,
        )

        self.assertTrue(candidates)
        self.assertTrue(all(candidate["length"] >= 60.0 for candidate in candidates))

    def test_existing_transition_midpoints_are_not_reused(self):
        box = BRepPrimAPI_MakeBox(gp_Pnt(-50.0, -50.0, 0.0), gp_Pnt(50.0, 50.0, 40.0)).Shape()
        candidates = extract_transition_edge_candidates(
            box,
            existing_params=[],
            min_edge_length=60.0,
            midpoint_clearance=0.0,
        )
        blocked_midpoint = candidates[0]["midpoint"]

        filtered = extract_transition_edge_candidates(
            box,
            existing_params=[{"midpoint": blocked_midpoint}],
            min_edge_length=60.0,
            midpoint_clearance=1.0,
        )

        self.assertFalse(any(candidate["midpoint"] == blocked_midpoint for candidate in filtered))

    def test_public_parameters_do_not_expose_transition_context(self):
        params = public_transition_parameters(
            inst=7,
            size_key="radius",
            size_value=1.5,
            candidate={
                "midpoint": [1.0, 2.0, 3.0],
                "axis": [0.0, 0.0, 1.0],
                "curve_type": "circle",
                "internal_edge_context": "hole_mouth",
            },
        )

        self.assertEqual(
            {"inst", "radius", "midpoint", "axis", "curve_type"},
            set(params.keys()),
        )

    def test_generators_read_transition_feature_policy(self):
        context = {
            "transition_features": {
                "min_edge_length": 18.0,
                "min_size": 0.9,
                "max_size": 2.4,
                "midpoint_clearance_factor": 5.0,
            }
        }

        fillet = FilletGenerator(semantic_index())
        chamfer = ChamferGenerator(semantic_index())
        fillet.set_generation_context(context)
        chamfer.set_generation_context(context)

        self.assertEqual(18.0, fillet._transition_policy()["min_edge_length"])
        self.assertEqual(2.4, chamfer._transition_policy()["max_size"])


if __name__ == "__main__":
    unittest.main()
