from __future__ import annotations

import json
import unittest

import execute_p4_api_r2b_20_val_smoke_v2 as delivery


PASS_PROTOCOL = {
    "post_repair_frame_valid_count": 20,
    "compiled_result_count": 20,
    "p2_valid_count": 20,
}
PASS_SEMANTIC = {
    "status_correct_count": 18,
    "invalid_prediction_correct": True,
    "needs_clarification_prediction_correct": True,
    "action_macro_f1": 0.9,
    "feature_class_macro_f1": 0.8,
    "selector_accuracy": 0.8,
    "canonical_semantic_exact_count": 14,
    "multi_operation_exact_count": 3,
}
PASS_GROUNDING = {
    "object_selection_accuracy": 0.8,
    "exact_face_set_accuracy": 0.8,
    "illegal_or_out_of_range_id_rate": 0.0,
}


class R2BDeliveryTests(unittest.TestCase):
    def test_ready_requires_every_gate(self) -> None:
        status, checks = delivery.determine_status(
            PASS_PROTOCOL, PASS_SEMANTIC, PASS_GROUNDING, actual_cost_cny=1.0
        )
        self.assertEqual(status, "READY_FOR_P4_API_R2C_100_VAL_REVIEW")
        self.assertTrue(all(checks.values()))

    def test_protocol_gate_failure_has_priority(self) -> None:
        status, _ = delivery.determine_status(
            {**PASS_PROTOCOL, "post_repair_frame_valid_count": 18},
            PASS_SEMANTIC,
            PASS_GROUNDING,
            actual_cost_cny=1.0,
        )
        self.assertEqual(status, "R2B_PROTOCOL_GATE_FAILED_WITH_EVIDENCE")

    def test_semantic_gate_failure_is_reported(self) -> None:
        status, _ = delivery.determine_status(
            PASS_PROTOCOL,
            {**PASS_SEMANTIC, "canonical_semantic_exact_count": 11},
            PASS_GROUNDING,
            actual_cost_cny=1.0,
        )
        self.assertEqual(status, "R2B_SEMANTIC_GATE_FAILED_WITH_EVIDENCE")

    def test_external_blocker_overrides_metric_decision(self) -> None:
        status, _ = delivery.determine_status(
            PASS_PROTOCOL,
            PASS_SEMANTIC,
            PASS_GROUNDING,
            actual_cost_cny=0.0,
            external_blocker="E_HTTP_401",
        )
        self.assertEqual(status, "BLOCKED_WITH_EVIDENCE")

    def test_sanitized_result_excludes_payloads_and_raw_frame(self) -> None:
        raw = {
            "sample_id": "s1",
            "frame": {"status": "valid"},
            "compiled_result": {
                "status": "VALID",
                "intent": {
                    "operations": [
                        {
                            "operation_id": "op1",
                            "operation_type": "pressure",
                            "target": {
                                "feature_type": "through_hole_round",
                                "face_role": "side_wall",
                                "quantifier": "all",
                            },
                        }
                    ]
                },
            },
            "first_round_json_parse_valid": True,
            "first_round_frame_schema_valid": True,
            "post_repair_frame_valid": True,
            "p2_valid": True,
            "repair_count": 0,
            "failure_phase": None,
            "failure_code": None,
            "error_path": None,
            "requests": [{"stage": "first_round", "request_hash": "a" * 64}],
        }
        sanitized = delivery.sanitize_outcome(raw)
        serialized = json.dumps(sanitized)
        for forbidden in (
            '"text"', '"query"', '"ground_truth"', '"frame"',
            '"compiled_result"', '"raw_model_response"'
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(sanitized["predicted"]["actions"], ["pressure"])


if __name__ == "__main__":
    unittest.main()
