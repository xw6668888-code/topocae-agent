from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_p4_api_r1b_schema_diagnosis as diagnosis


class R1BSchemaDiagnosisTests(unittest.TestCase):
    def test_transport_persists_count_and_enforces_cap(self):
        original_output = diagnosis.OUTPUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                diagnosis.OUTPUT = Path(directory)
                transport = diagnosis.CappedTransport(
                    transport=lambda *_: {"ok": True}
                )
                for _ in range(diagnosis.BACKEND_REQUEST_CAP):
                    self.assertEqual(transport(None, None, None, None), {"ok": True})
                with self.assertRaisesRegex(RuntimeError, "E_BACKEND_REQUEST_CAP"):
                    transport(None, None, None, None)
                saved = json.loads(
                    (diagnosis.OUTPUT / "backend_request_counter.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["backend_request_count"], diagnosis.BACKEND_REQUEST_CAP)
        finally:
            diagnosis.OUTPUT = original_output

    def test_validation_recorder_only_keeps_codes_and_paths(self):
        recorder = diagnosis.ValidationRecorder()
        result = recorder({"status": "VALID"})
        self.assertFalse(result["valid"])
        self.assertTrue(recorder.attempts[0]["errors"])
        for error in recorder.attempts[0]["errors"]:
            self.assertEqual(set(error), {"error_code", "field_path"})


if __name__ == "__main__":
    unittest.main()
