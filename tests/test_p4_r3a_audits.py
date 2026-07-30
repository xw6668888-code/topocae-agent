from __future__ import annotations

import unittest

from run_p4_r3a_hybrid_evidence_resolver import _train_coverage


class P4R3AAuditTests(unittest.TestCase):
    def test_train_coverage_reads_train_without_exporting_text(self):
        audit = _train_coverage()
        self.assertGreater(audit["case_count"], 0)
        self.assertEqual(audit["extract_error_count"], 0)
        self.assertFalse(audit["raw_text_exported"])
        self.assertFalse(audit["development_smoke_used_for_rule_authoring"])


if __name__ == "__main__":
    unittest.main()
