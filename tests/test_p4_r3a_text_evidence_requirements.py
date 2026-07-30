from __future__ import annotations

import unittest

from utils.cae_text_evidence import CAETextEvidenceExtractor


class TextEvidenceRequirementTests(unittest.TestCase):
    def test_feature_operation_without_selector_records_missing_selector(self):
        result = CAETextEvidenceExtractor().extract({
            "text": "Apply 1 MPa pressure normal to the hole inner wall.",
            "language": "en-US",
        })
        missing = {
            item["normalized"]["parameter"]
            for item in result["evidence"]
            if item["category"] == "missing_required_parameter"
        }
        self.assertIn("selector", missing)


if __name__ == "__main__":
    unittest.main()
