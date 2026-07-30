from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProtectedInventoryTests(unittest.TestCase):
    def test_preflight_and_final_p1_b_inventory_are_identical(self) -> None:
        baseline = json.loads(
            (ROOT / "reports/p1_c_preflight/protected_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        final = json.loads(
            (ROOT / "reports/p1_c_final_acceptance_v1/protected_inventory_final.json").read_text(
                encoding="utf-8"
            )
        )
        diff = json.loads(
            (ROOT / "reports/p1_c_final_acceptance_v1/protected_inventory_diff.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(baseline["aggregate_sha256"], final["aggregate_sha256"])
        self.assertEqual(baseline["total_file_count"], final["total_file_count"])
        self.assertEqual(baseline["total_byte_count"], final["total_byte_count"])
        self.assertTrue(diff["passed"])
        self.assertEqual(
            (diff["added_count"], diff["removed_count"], diff["changed_count"]),
            (0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
