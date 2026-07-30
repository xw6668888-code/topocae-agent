from __future__ import annotations

import csv
import json
import unittest
from collections import Counter

from utils.cae_query_ground.constants import CANONICAL_FEATURES, OPERATIONS, ROOT
from utils.cae_query_ground.io import read_json, read_jsonl, sha256_file
from utils.cae_query_ground.validation_v4 import validate_manifest
from utils.cae_query_ground.validation import validate_record


REPORT = ROOT / "reports" / "p3_a1_cae_queryground_v1"
PILOT = ROOT / "output_cae_query_ground_v1" / "pilot"


class CaeQueryGroundPilotRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_audit = read_json(REPORT / "source_and_role_audit.json")
        cls.quality = read_json(REPORT / "quality_audit.json")
        cls.manifest = read_json(PILOT / "manifests" / "dataset_manifest.json")
        cls.canonical = [
            item for split in ("train", "val", "test")
            for item in read_jsonl(PILOT / "canonical" / f"{split}.jsonl")
        ]
        cls.renderings = [
            item for split in ("train", "val", "test")
            for item in read_jsonl(PILOT / "renderings" / f"{split}.jsonl")
        ]

    def test_source_crosswalk_and_frozen_hashes(self) -> None:
        self.assertTrue(self.source_audit["automatic_pass"])
        self.assertEqual(self.source_audit["source"]["sample_count"], 5000)
        self.assertEqual(self.source_audit["source"]["graph_hash_mismatches"], 0)
        self.assertEqual(self.source_audit["role_audit"]["hard_failure_count"], 0)
        self.assertEqual(self.source_audit["p2_freeze"]["mismatch_count"], 0)
        self.assertTrue(self.source_audit["p2_freeze"]["supervisor_added_runner"]["matches"])

    def test_schema_strictness_and_round_trip(self) -> None:
        self.assertEqual(validate_record(self.canonical[0]), [])
        extra = json.loads(json.dumps(self.canonical[0], ensure_ascii=False))
        extra["unexpected"] = True
        self.assertTrue(any("Additional properties" in error for error in validate_record(extra)))
        self.assertEqual(validate_manifest(self.manifest), [])
        self.assertEqual(self.canonical[0], json.loads(json.dumps(self.canonical[0], ensure_ascii=False)))

    def test_pilot_counts_coverage_and_proofs(self) -> None:
        self.assertEqual((len(self.canonical), len(self.renderings)), (1120, 3360))
        self.assertEqual(self.quality["counts"]["splits"], {"test": 140, "train": 840, "val": 140})
        self.assertEqual(self.quality["counts"]["languages"], {"en-US": 2240, "zh-CN": 1120})
        self.assertEqual(set(self.quality["counts"]["features"]), set(CANONICAL_FEATURES))
        self.assertEqual(set(self.quality["counts"]["operations"]), set(OPERATIONS))
        self.assertEqual(set(self.quality["counts"]["levels"]), {"L1", "L2", "L3", "L4", "L5"})
        self.assertEqual(set(self.quality["counts"]["statuses"]), {"VALID", "NEEDS_CLARIFICATION", "INVALID"})
        self.assertTrue(self.quality["automatic_pass"])
        self.assertEqual(self.quality["record_schema_and_semantic_failures"], 0)
        self.assertEqual(self.quality["target_reconstruction"]["passed"], self.quality["target_reconstruction"]["total"])
        self.assertEqual(self.quality["rendering_identity_failures"], 0)
        self.assertEqual(self.quality["rendering_cardinality_failures"], 0)
        self.assertEqual(self.quality["split_leakage"]["total_collisions"], 0)

    def test_three_renderings_per_canonical(self) -> None:
        counts = Counter(item["source_ref"]["canonical_record_id"] for item in self.renderings)
        self.assertEqual(len(counts), 1120)
        self.assertEqual(set(counts.values()), {3})

    def test_manifest_hashes(self) -> None:
        for entry in self.manifest["files"]:
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, entry["bytes"], path)
            self.assertEqual(sha256_file(path), entry["sha256"], path)

    def test_h3_package_blank_and_balanced(self) -> None:
        with (REPORT / "human_review_package" / "cae_query_review_120.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 120)
        self.assertEqual(Counter(row["language"] for row in rows), {"en-US": 60, "zh-CN": 60})
        self.assertEqual(Counter(row["difficulty_level"] for row in rows), {f"L{i}": 24 for i in range(1, 6)})
        self.assertEqual(set(row["canonical_feature"] for row in rows), set(CANONICAL_FEATURES))
        self.assertEqual(
            {op for row in rows for op in row["operation_type"].split("+")},
            set(OPERATIONS),
        )
        for row in rows:
            self.assertTrue(all(not row[field] for field in ("review_decision", "error_category", "corrected_query", "reviewer_note")))
        self.assertEqual(self.manifest["review_gate"]["reviewed_rows"], 0)
        self.assertEqual(self.manifest["review_gate"]["status"], "WAITING_FOR_REAL_USER_REVIEW")

    def test_prohibited_scope_and_no_freeze(self) -> None:
        self.assertFalse(any(self.quality["scope_guards"].values()))
        self.assertFalse((REPORT / "freeze").exists())
        self.assertFalse((REPORT / "acceptance.json").exists())
        self.assertFalse((REPORT / "final_acceptance.zh-CN.md").exists())
        self.assertLess(len(self.canonical), 20000)


if __name__ == "__main__":
    unittest.main()

