from __future__ import annotations

from pathlib import Path
import unittest

from utils.cae_query_ground_p3_a1r.semantic_coverage import audit_dataset, audit_records


ROOT = Path(__file__).resolve().parents[1]


class SemanticCoverageAuditTests(unittest.TestCase):
    def test_frozen_p3_reproduces_supervisor_defect_counts(self) -> None:
        result = audit_dataset(ROOT / "output_cae_query_ground_v1")
        self.assertEqual("SEMANTIC-COVERAGE-FAIL", result["status"])
        self.assertEqual(20700, result["counts"]["valid_feature_targets"])
        self.assertEqual(0, result["counts"]["candidate_count_ge_2"])
        self.assertEqual(0, result["counts"]["count_quantifier"])
        self.assertEqual(0, result["counts"]["valid_L4"])

    def test_multi_instance_contract_checks_semantics_not_level_presence(self) -> None:
        record = {
            "query": {"difficulty_level": "L4"},
            "intent_result": {
                "status": "VALID",
                "intent": {"operations": [{
                    "operation_id": "op1",
                    "target": {
                        "quantifier": "largest",
                        "comparison_metric": "area",
                        "filters": {},
                    },
                }]},
            },
            "ground_truth": {
                "intent_status": "VALID",
                "targets": [{
                    "grounding_required": True,
                    "target_contract": "feature_face_set",
                    "operation_id": "op1",
                    "quantifier": "largest",
                    "uniqueness_contract": {"candidate_count": 1, "selected_count": 1},
                }],
            },
            "source_ref": {"canonical_record_id": "bad-l4"},
        }
        result = audit_records([record])
        self.assertEqual(1, result["counts"]["valid_L4"])
        self.assertEqual(1, result["contract_failure_counts"]["L4_candidate_count_lt_2"])
        self.assertEqual("SEMANTIC-COVERAGE-FAIL", result["status"])


if __name__ == "__main__":
    unittest.main()
