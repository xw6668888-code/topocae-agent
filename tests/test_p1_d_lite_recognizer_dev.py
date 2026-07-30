from __future__ import annotations

import unittest

from p1_d_lite.recognizer_dev import RecognizerDevError, error_payload, recognize_graph


class RecognizerDevContractTests(unittest.TestCase):
    def test_invalid_schema_returns_fail_closed_error_code(self) -> None:
        with self.assertRaises(RecognizerDevError) as raised:
            recognize_graph({"schema_version": "wrong", "nodes": [], "edges": []})
        self.assertEqual(raised.exception.code, "E_SCHEMA")

    def test_error_payload_contains_no_feature_objects(self) -> None:
        payload = error_payload(RecognizerDevError("E_SCHEMA", "bad graph"))
        self.assertEqual(payload["coverage_status"], "ERROR")
        self.assertEqual(payload["error"]["code"], "E_SCHEMA")
        self.assertEqual(payload["feature_objects"], [])


if __name__ == "__main__":
    unittest.main()

