from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

import jsonschema

from utils.cae_query_ground.constants import CANONICAL_FEATURES, OPERATIONS, ROOT
from utils.cae_query_ground.io import read_json, read_jsonl, sha256_file


REPORT = ROOT / "reports" / "p3_a1_cae_queryground_v1"
PILOT = ROOT / "output_cae_query_ground_v1" / "pilot"


class CaeQueryGroundPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_audit = read_json(REPORT / "source_and_role_audit.json")
        cls.quality = read_json(REPORT / "quality_audit.json")
        cls.manifest = read_json(PILOT / "manifests" / "dataset_manifest.json")
        cls.canonical = [
            record
            for split in ("train", "val", "test")
            for record in read_jsonl(PILOT / "canonical" / f"{split}.jsonl")
        ]
        cls.renderings = [
            record
            for split in ("train", "val", "test")
            for record in read_jsonl(PILOT / "renderings" / f"{split}.jsonl")
        ]

    def test_source_and_p2_frozen_integrity(self) -> None:
        self.assertTrue(self.source_audit["automatic_pass"])
        self.assertEqual(self.source_audit["source"]["sample_count"], 5000)
        self.assertEqual(self.source_audit["source"]["graph_missing"], 0)
        self.assertEqual(self.source_audit["source"]["graph_hash_mismatches"], 0)
        self.assertEqual(self.source_audit["p2_freeze"]["mismatch_count"], 0)
        self.assertTrue(self.source_audit["p2_freeze"]["supervisor_added_runner"]["matches"])

    def test_role_crosswalk_and_coverage(self) -> None:
        self.assertEqual(self.source_audit["role_audit"]["hard_failure_count"], 0)
        self.assertEqual(set(self.quality["counts"]["features"]), set(CANONICAL_FEATURES))
        self.assertEqual(set(self.quality["counts"]["operations"]), set(OPERATIONS))
        self.assertTrue(all(self.quality["coverage"].values()))

    def test_counts_and_cardinality(self) -> None:
        self.assertEqual(len(self.canonical), 1120)
        self.assertEqual(len(self.renderings), 3360)
        self.assertEqual(self.quality["counts"]["splits"], {"test": 140, "train": 840, "val": 140})
        by_canonical = Counter(record["source_ref"]["canonical_record_id"] for record in self.renderings)
        self.assertEqual(set(by_canonical.values()), {3})
        languages = Counter(record["query"]["language"] for record in self.renderings)
        self.assertEqual(languages, {"en-US": 2240, "zh-CN": 1120})

    def test_record_and_manifest_schema_strictness(self) -> None:
        record_schema = read_json(ROOT / "schemas" / "cae_query_ground_record.schema.json")
        manifest_schema = read_json(ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json")
        store = {
            "https://topocae.local/schemas/cae_intent_result.schema.json": read_json(
                ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "schemas" / "cae_intent_result.schema.json"
            ),
            "https://topocae.local/schemas/cae_intent.schema.json": read_json(
                ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "schemas" / "cae_intent.schema.json"
            ),
        }
        resolver = jsonschema.RefResolver.from_schema(record_schema, store=store)
        jsonschema.Draft202012Validator(record_schema, resolver=resolver).validate(self.canonical[0])
        jsonschema.Draft202012Validator(manifest_schema).validate(self.manifest)
        extra = dict(self.canonical[0])
        extra["unexpected"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(record_schema, resolver=resolver).validate(extra)
        self.assertEqual(self.canonical[0], json.loads(json.dumps(self.canonical[0], ensure_ascii=False)))

    def test_full_quality_proof(self) -> None:
        self.assertTrue(self.quality["automatic_pass"])
        self.assertEqual(self.quality["record_schema_and_semantic_failures"], 0)
        self.assertEqual(self.quality["target_reconstruction"]["passed"], self.quality["target_reconstruction"]["total"])
        self.assertEqual(self.quality["rendering_identity_failures"], 0)
        self.assertEqual(self.quality["rendering_cardinality_failures"], 0)
        self.assertEqual(self.quality["split_leakage"]["total_collisions"], 0)

    def test_manifest_file_hashes(self) -> None:
        for entry in self.manifest["files"]:
            path = ROOT / Path(entry["path"])
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, entry["bytes"], path)
            self.assertEqual(sha256_file(path), entry["sha256"], path)

    def test_h3_is_real_user_gate_and_blank(self) -> None:
        path = REPORT / "human_review_package" / "cae_query_review_120.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 120)
        self.assertEqual(Counter(row["language"] for row in rows), {"en-US": 60, "zh-CN": 60})
        self.assertEqual(Counter(row["difficulty_level"] for row in rows), {f"L{i}": 24 for i in range(1, 6)})
        for row in rows:
            for field in ("review_decision", "error_category", "corrected_query", "reviewer_note"):
                self.assertEqual(row[field], "")
        self.assertEqual(self.manifest["review_gate"]["reviewed_rows"], 0)
        self.assertEqual(self.manifest["review_gate"]["status"], "WAITING_FOR_REAL_USER_REVIEW")

    def test_scope_guards_and_no_premature_freeze(self) -> None:
        self.assertFalse(any(self.quality["scope_guards"].values()))
        self.assertFalse((REPORT / "freeze").exists())
        self.assertFalse((REPORT / "acceptance.json").exists())
        self.assertFalse((REPORT / "final_acceptance.zh-CN.md").exists())
        self.assertLess(len(self.canonical), 20000)


if __name__ == "__main__":
    unittest.main()

