from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import unittest

import run_p1_c_synthetic_holdout_v3_once as formal


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "output_generation_p1_c_synthetic_holdout_v3"
REPORT = ROOT / "reports" / "p1_c_holdout_v3_once"


class HoldoutV3ProtocolTests(unittest.TestCase):
    def test_plan_freeze_predates_generation_and_has_frozen_1440_quota(self) -> None:
        plan = json.loads((HOLDOUT / "manifests/generation_plan_freeze.json").read_text(encoding="utf-8"))
        generated = json.loads((HOLDOUT / "holdout_generation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["sample_count"], 1440)
        self.assertFalse(plan["holdout_results_seen"])
        self.assertLess(datetime.fromisoformat(plan["frozen_at_utc"]), datetime.fromisoformat(generated["generated_at_utc"]))

    def test_generation_and_leakage_contracts(self) -> None:
        quality = json.loads((HOLDOUT / "reports/quality_report.json").read_text(encoding="utf-8"))
        leakage = json.loads((HOLDOUT / "reports/leakage_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(quality["automatic_pass"])
        self.assertEqual(quality["cleanup_pass_count"], 1440)
        self.assertTrue(leakage["all_zero"])
        self.assertFalse(leakage["holdout_v2_predictions_or_errors_read"])

    def test_exactly_once_and_review_archive(self) -> None:
        generated = json.loads((HOLDOUT / "holdout_generation_manifest.json").read_text(encoding="utf-8"))
        state = json.loads((REPORT / "formal_inference_state.json").read_text(encoding="utf-8"))
        self.assertEqual(generated["inference_pass_count"], 1)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_sample_count"], 1440)
        self.assertEqual(len(list((REPORT / "sample_predictions").glob("*.json"))), 1440)
        self.assertEqual(len(list((HOLDOUT / "human_review_36").glob("*.step"))), 36)

    def test_second_formal_run_is_rejected_before_inference(self) -> None:
        state = json.loads(formal.STATE.read_text(encoding="utf-8"))
        hashes = {key: state[key] for key in (
            "p1_b_selection_sha256", "p1_c_selection_sha256", "generation_manifest_sha256"
        )}
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            formal.start_or_resume_state(hashes, state["script_sha256"], 1440)


if __name__ == "__main__":
    unittest.main()
