from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"


class QueryFrameAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_p4_api_r2a_compact_query_frame as runner

        cls.runner = runner
        cls.audits = runner.build_audits()

    def test_authorized_record_scope_is_exact(self):
        roundtrip = self.audits["compiler_roundtrip"]
        self.assertEqual(21885, roundtrip["total_records"])
        self.assertEqual(265, roundtrip["source_counts"]["p2_fixtures"])
        self.assertEqual(19180, roundtrip["source_counts"]["p3_train"])
        self.assertEqual(2440, roundtrip["source_counts"]["p3_val"])
        self.assertEqual(0, roundtrip["manifest_count_difference"])

    def test_all_roundtrip_gates_are_exact(self):
        audit = self.audits["compiler_roundtrip"]
        for field in (
            "frame_schema_valid_rate", "canonical_schema_valid_rate",
            "semantic_exact_rate", "deterministic_reproduction_rate",
        ):
            self.assertEqual(1.0, audit[field], field)
        self.assertEqual(21885, audit["semantic_exact_count"])

    def test_compactness_gate_is_record_median_ratio(self):
        audit = self.audits["compactness"]
        self.assertEqual("per-record nearest-rank", audit["median_method"])
        self.assertLessEqual(audit["schema_utf8_bytes"], 10 * 1024)
        self.assertLessEqual(audit["frame_to_canonical_ratio_p50"], 0.65)
        self.assertEqual(21885, audit["record_count"])

    def test_coverage_has_all_required_dimensions(self):
        coverage = self.audits["coverage"]
        self.assertEqual({"en-US", "zh-CN", "fixture_unlabeled"}, set(coverage["language_counts"]))
        self.assertTrue({"L1", "L2", "L3", "L4", "L5", "fixture"}.issubset(coverage["difficulty_counts"]))
        self.assertEqual({"VALID", "NEEDS_CLARIFICATION", "INVALID"}, set(coverage["status_counts"]))
        self.assertEqual({
            "fixed_support", "displacement", "pressure", "surface_traction",
            "material_assignment", "global_mesh_size", "local_mesh_size",
        }, set(coverage["operation_type_counts"]))
        self.assertTrue({
            "all", "any_one", "exactly_one", "count", "largest", "smallest", "nearest",
        }.issubset(coverage["quantifier_counts"]))

    def test_leakage_and_formal_access_are_zero(self):
        audit = self.audits["leakage"]
        self.assertEqual(0, audit["train_val_canonical_id_overlap_count"])
        self.assertEqual(0, audit["duplicate_identity_count"])
        self.assertEqual(0, audit["decoded_test_lines"])
        self.assertEqual(0, audit["formal_test_access_count"])
        self.assertFalse(audit["formal_test_payload_decoded"])
        self.assertEqual(audit["formal_test_ledger_before"], audit["formal_test_ledger_after"])

    def test_freeze_hashes_have_no_changes(self):
        audit = self.audits["freeze"]
        self.assertEqual(0, audit["change_count"])
        self.assertTrue(audit["p1_unchanged"])
        self.assertTrue(audit["p2_unchanged"])
        self.assertTrue(audit["p3_unchanged"])
        self.assertTrue(audit["historical_reports_unchanged"])

    def test_delivery_contract_files_are_declared(self):
        required = {
            "final_acceptance.zh-CN.md", "acceptance.json", "query_frame.schema.json",
            "query_frame_contract.zh-CN.md", "compiler_roundtrip_audit.json",
            "compactness_audit.json", "coverage_audit.json", "leakage_audit.json",
            "regression_results.json", "freeze_audit.json", "delivery_manifest.json",
            "CODEX_CONTEXT_P4_API_R2A.md",
        }
        self.assertTrue(required.issubset(self.runner.REQUIRED_DELIVERY_FILES))

    def test_prompt_and_repair_protocol_are_offline_and_bounded(self):
        prompt = (REPORT / "query_frame_prompt.zh-CN.md").read_text(encoding="utf-8")
        repair = (REPORT / "repair_protocol.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("{text, language}", prompt)
        self.assertIn("仅输出", prompt)
        self.assertNotIn("http", prompt.lower())
        self.assertIn("最多一次", repair)
        self.assertIn("error_code", repair)
        self.assertIn("schema_summary", repair)
        self.assertIn("禁止 silent repair", repair)
        self.assertNotIn("while", repair.lower())

    def test_saved_json_audits_match_recomputed_values_after_delivery(self):
        if not (REPORT / "compiler_roundtrip_audit.json").exists():
            self.skipTest("delivery is generated after the audit runner green phase")
        pairs = {
            "compiler_roundtrip_audit.json": "compiler_roundtrip",
            "compactness_audit.json": "compactness",
            "coverage_audit.json": "coverage",
            "leakage_audit.json": "leakage",
            "freeze_audit.json": "freeze",
        }
        for filename, key in pairs.items():
            with self.subTest(filename=filename):
                saved = json.loads((REPORT / filename).read_text(encoding="utf-8"))
                self.assertEqual(self.audits[key], saved)


if __name__ == "__main__":
    unittest.main()
