from __future__ import annotations

import json
import unittest
from pathlib import Path

from utils.cae_query_ground.constants import ROOT
from utils.cae_query_ground.formal_planner import FINAL_DATASET_VERSION
from utils.cae_query_ground.formal_validation import validate_formal_record
from utils.cae_query_ground.io import read_json, sha256_file


OUTPUT = ROOT / "output_cae_query_ground_v1"
REPORT = ROOT / "reports" / "p3_a1_cae_queryground_v1"


def _jsonl_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


class FormalDatasetRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quality = read_json(REPORT / "quality_audit.json")
        cls.manifest = read_json(OUTPUT / "manifests" / "dataset_manifest.json")
        cls.h3 = read_json(REPORT / "human_review_package" / "h3_01_ingestion_summary.json")
        cls.acceptance = read_json(REPORT / "acceptance.json")
        cls.freeze = read_json(REPORT / "freeze" / "freeze_manifest.json")

    def test_gate_p3_and_scope(self) -> None:
        self.assertEqual(self.acceptance["status"], "PASS")
        self.assertTrue(self.acceptance["automatic_pass"])
        self.assertTrue(all(self.acceptance["gate_p3"].values()))
        self.assertFalse(any(self.quality["scope_guards"].values()))

    def test_formal_counts_and_bilingual_cardinality(self) -> None:
        self.assertEqual(self.manifest["dataset_version"], FINAL_DATASET_VERSION)
        self.assertGreaterEqual(self.quality["counts"]["canonical_records"], 20000)
        self.assertGreaterEqual(self.quality["counts"]["renderings"], 60000)
        self.assertEqual(
            self.quality["counts"]["languages"]["en-US"],
            2 * self.quality["counts"]["canonical_records"],
        )
        self.assertEqual(
            self.quality["counts"]["languages"]["zh-CN"],
            self.quality["counts"]["canonical_records"],
        )
        for split, expected in self.quality["counts"]["splits"].items():
            self.assertEqual(_jsonl_count(OUTPUT / "canonical" / f"{split}.jsonl"), expected)
            self.assertEqual(_jsonl_count(OUTPUT / "renderings" / f"{split}.jsonl"), expected * 3)

    def test_full_quality_proofs(self) -> None:
        self.assertTrue(self.quality["automatic_pass"])
        self.assertEqual(self.quality["record_schema_and_semantic_failures"], 0)
        self.assertEqual(self.quality["target_reconstruction"]["passed"], self.quality["target_reconstruction"]["total"])
        self.assertEqual(self.quality["p2_validation"]["passed"], self.quality["p2_validation"]["unique_intent_result_payloads"])
        self.assertEqual(self.quality["split_leakage"]["total_collisions"], 0)
        self.assertEqual(self.quality["duplicate_canonical_payloads"], 0)
        self.assertTrue(all(self.quality["coverage"].values()))

    def test_schema_round_trip_on_each_split(self) -> None:
        for split in ("train", "val", "test"):
            record = _first_jsonl(OUTPUT / "canonical" / f"{split}.jsonl")
            self.assertEqual(validate_formal_record(record), [])
            self.assertEqual(record, json.loads(json.dumps(record, ensure_ascii=False)))

    def test_dataset_manifest_file_hashes(self) -> None:
        for entry in self.manifest["files"]:
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, entry["bytes"], path)
            self.assertEqual(sha256_file(path), entry["sha256"], path)

    def test_h3_real_review_and_expression_decisions(self) -> None:
        self.assertEqual(self.h3["status"], "PASS")
        self.assertEqual(self.h3["review_rows"], 120)
        self.assertEqual(self.h3["effective_correct_rate"], 1.0)
        self.assertEqual(self.h3["expression_rows_populated"], 20)
        self.assertGreaterEqual(self.h3["expression_decisions"]["adopt_with_normalization"], 10)

    def test_freeze_assets_hashes(self) -> None:
        self.assertEqual(self.freeze["status"], "FROZEN")
        self.assertEqual(self.freeze["input_references"]["p2_hash_changes"], 0)
        for entry in self.freeze["assets"]:
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256_file(path), entry["sha256"], path)


if __name__ == "__main__":
    unittest.main()

