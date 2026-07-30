from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "reports"
    / "p4_a1_language_grounding_v1"
    / "human_review_package"
)


class P4A1H4PackageTests(unittest.TestCase):
    def test_review_csv_is_complete_and_review_fields_are_valid(self):
        path = PACKAGE / "H4-01_val_language_review_60.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            {row["language"] for row in rows}, {"zh-CN", "en-US"}
        )
        self.assertEqual(
            {row["oracle_status"] for row in rows},
            {"VALID", "NEEDS_CLARIFICATION", "INVALID"},
        )
        self.assertTrue(all(row["baseline_name"] == "keyword" for row in rows))
        self.assertTrue(
            all(
                row["reviewer_decision"].strip().lower()
                in {"approve", "edit", "reject"}
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["reviewer_note"].strip()
                or row["reviewer_decision"].strip().lower() == "approve"
                for row in rows
            )
        )
        self.assertNotIn("face_id", rows[0])
        self.assertNotIn("face_ids", rows[0])

    def test_package_manifest_declares_val_only_full_coverage(self):
        value = json.loads(
            (PACKAGE / "package_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(value["source_split"], "val")
        self.assertEqual(value["row_count"], 60)
        self.assertEqual(
            value["distributions"]["language"],
            {"en-US": 30, "zh-CN": 30},
        )
        self.assertEqual(len(value["distributions"]["feature_type"]), 17)
        self.assertTrue(
            value["coverage_checks"]["raw_face_id_review_required"] is False
        )

    def test_test_ledger_remains_locked_and_empty(self):
        value = json.loads(
            (
                ROOT
                / "output_p4_a1_language_grounding_v1"
                / "test_run_ledger.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(value["status"], "LOCKED_RUNTIME_PENDING")
        self.assertFalse(value["test_content_accessed"])
        self.assertTrue(value["h4_accepted"])
        self.assertFalse(value["test_unlock_authorized"])
        self.assertFalse(value["freeze_manifest_created_after_h4"])
        self.assertTrue(all(not runs for runs in value["runs"].values()))

    def test_review_validation_accepts_completed_human_attestation(self):
        value = json.loads(
            (PACKAGE / "H4-01_review_validation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            value["status"], "PASS"
        )
        self.assertEqual(value["decision_counts"]["approve"], 60)
        self.assertTrue(value["protected_columns_unchanged"] )
        self.assertTrue(value["h4_accepted"] )
        self.assertTrue(value["h4_prerequisite_satisfied"] )
        self.assertFalse(value["test_unlock_authorized"] )


if __name__ == "__main__":
    unittest.main()
