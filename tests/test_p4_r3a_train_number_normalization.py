from __future__ import annotations

import unittest

from utils.cae_text_evidence import CAETextEvidenceExtractor


class TrainNumberNormalizationTests(unittest.TestCase):
    def test_chinese_digit_sequence_count_is_normalized(self):
        text = "\u5bf9\u4e00\u4e00\u4e2a\u5706\u5b54\u5b54\u58c1\u8bbe\u7f6e\u5c40\u90e8\u7f51\u683c\u5c3a\u5bf8 1 mm\u3002"
        result = CAETextEvidenceExtractor().extract({"text": text, "language": "zh-CN"})
        selectors = [item["normalized"] for item in result["evidence"] if item["category"] == "selector"]
        self.assertIn({"selector": "count", "count": 11}, selectors)


if __name__ == "__main__":
    unittest.main()
