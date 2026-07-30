from __future__ import annotations

from copy import deepcopy
import unittest

from utils.cae_hybrid_resolver import HybridCAEQueryFrameResolver
from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import compile_query_frame, validate_query_frame


def pressure_frame(*, selector="all", magnitude=1.0, role="hole_wall"):
    return {
        "frame_version": "cae-query-frame/1.0.0",
        "status": "valid",
        "language": "en-US",
        "operations": [{
            "operation_key": "op1",
            "action": "pressure",
            "feature": "through_hole_round",
            "face_role": role,
            "selector": selector,
            "magnitude": {"value": magnitude, "unit": "MPa"},
            "direction": {"kind": "surface_normal"},
        }],
    }


class HybridResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = HybridCAEQueryFrameResolver()

    def _assert_contracts(self, result):
        self.assertTrue(validate_query_frame(result["frame"])["valid"])
        compiled = compile_query_frame(result["frame"])
        self.assertTrue(validate_intent_result(compiled)["valid"])
        self.assertEqual(result["compiled_result"], compiled)
        self.assertTrue(result["frame_valid"])
        self.assertTrue(result["p2_valid"])

    def test_accepts_policy_compatible_proposal(self):
        result = self.resolver.resolve(
            {"text": "Apply 1 MPa pressure normal to all hole inner walls.", "language": "en-US"},
            pressure_frame(),
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["status"], "valid")
        decisions = {row["field_path"]: row["decision"] for row in result["decision_trace"]}
        self.assertEqual(decisions["/operations/0/action"], "accept")
        self.assertEqual(decisions["/operations/0/feature"], "accept")
        self.assertEqual(decisions["/operations/0/face_role"], "accept")

    def test_explicit_selector_and_magnitude_override_llm_values(self):
        result = self.resolver.resolve(
            {"text": "Apply 5 MPa pressure normal to exactly one hole inner wall.", "language": "en-US"},
            pressure_frame(selector="all", magnitude=1.0),
        )
        self._assert_contracts(result)
        operation = result["frame"]["operations"][0]
        self.assertEqual(operation["selector"], "exactly_one")
        self.assertEqual(operation["magnitude"], {"value": 5.0, "unit": "MPa"})
        overrides = [row for row in result["decision_trace"] if row["decision"] == "override_by_explicit_evidence"]
        self.assertEqual({row["field_path"] for row in overrides}, {"/operations/0/selector", "/operations/0/magnitude"})
        self.assertTrue(all(row["evidence_span"] is not None and row["rule_id"] for row in overrides))

    def test_policy_incompatible_role_is_repaired_by_unique_explicit_role(self):
        proposal = {
            "frame_version": "cae-query-frame/1.0.0",
            "status": "valid",
            "language": "en-US",
            "operations": [{
                "operation_key": "op1", "action": "displacement",
                "feature": "through_hole_round", "face_role": "feature_region", "selector": "exactly_one",
                "displacements": [{"dof": "U1", "value": 0.5, "unit": "mm"}],
                "load_frame": "global_cartesian",
            }],
        }
        result = self.resolver.resolve(
            {"text": "Apply displacement U1=0.5 mm in the global frame to exactly one hole inner wall.", "language": "en-US"},
            proposal,
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["operations"][0]["face_role"], "hole_wall")
        trace = next(row for row in result["decision_trace"] if row["field_path"] == "/operations/0/face_role")
        self.assertEqual(trace["decision"], "override_by_explicit_evidence")
        self.assertTrue(trace["p2_policy_reference"].startswith("RP-"))

    def test_unresolved_incompatible_role_needs_clarification(self):
        proposal = {
            "frame_version": "cae-query-frame/1.0.0", "status": "valid", "language": "en-US",
            "operations": [{
                "operation_key": "op1", "action": "displacement", "feature": "slot_rect",
                "face_role": "bottom_face", "selector": "all",
                "displacements": [{"dof": "U1", "value": 1.0, "unit": "mm"}],
                "load_frame": "global_cartesian",
            }],
        }
        result = self.resolver.resolve(
            {"text": "Apply displacement U1=1 mm in the global frame to all slot faces.", "language": "en-US"},
            proposal,
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["status"], "needs_clarification")
        self.assertTrue(any(row["decision"] == "clarify" for row in result["decision_trace"]))

    def test_set_difference_is_invalid(self):
        result = self.resolver.resolve(
            {"text": "Apply 1 MPa pressure normal to all holes except the nearest hole.", "language": "en-US"},
            pressure_frame(),
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["status"], "invalid")
        self.assertTrue(any(row["decision"] == "reject" for row in result["decision_trace"]))

    def test_nearest_without_spatial_reference_needs_clarification(self):
        proposal = pressure_frame(selector="nearest")
        proposal["operations"][0]["geometry_filters"] = {"nearest_to": {"kind": "datum_axis", "axis": "X"}}
        proposal["operations"][0]["selection_frame"] = "global_cartesian"
        result = self.resolver.resolve(
            {"text": "Apply 1 MPa pressure normal to the nearest hole inner wall.", "language": "en-US"},
            proposal,
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["status"], "needs_clarification")
        self.assertIn("M_REFERENCE", [row["code"] for row in result["frame"]["missing_information"]])

    def test_multi_operation_parameters_remain_in_their_clauses(self):
        proposal = pressure_frame(magnitude=1.0)
        proposal["operations"].append({
            "operation_key": "op2", "action": "local_mesh_size", "feature": "fillet",
            "face_role": "blend_face", "selector": "all", "mesh_size": {"value": 5.0, "unit": "mm"},
        })
        result = self.resolver.resolve(
            {"text": "Apply 8 MPa pressure normal to all hole walls and then set local mesh size on every fillet blend face to 0.8 mm.", "language": "en-US"},
            proposal,
        )
        self._assert_contracts(result)
        first, second = result["frame"]["operations"]
        self.assertEqual(first["magnitude"], {"value": 8.0, "unit": "MPa"})
        self.assertEqual(second["mesh_size"], {"value": 0.8, "unit": "mm"})
        self.assertNotIn("mesh_size", first)
        self.assertNotIn("magnitude", second)

    def test_identity_bearing_proposal_is_rejected_without_identity_output(self):
        proposal = deepcopy(pressure_frame())
        proposal["operations"][0]["face_id"] = 42
        result = self.resolver.resolve(
            {"text": "Apply 1 MPa pressure normal to all hole inner walls.", "language": "en-US"},
            proposal,
        )
        self._assert_contracts(result)
        self.assertEqual(result["frame"]["status"], "invalid")
        self.assertNotIn("42", str(result["frame"]))
        self.assertTrue(any(row["decision"] == "reject" for row in result["decision_trace"]))


if __name__ == "__main__":
    unittest.main()
