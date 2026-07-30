from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

from build_p3_a1r_pilot_queries import _audit_queries, read_jsonl
from utils.cae_query_ground_p3_a1r.frozen_assets import verify_freeze_manifest


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = ROOT / "output_generation_p3_a1r_multi_instance_v1"
REPORT_ROOT = ROOT / "reports" / "p3_a1r_multi_instance_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


class P3A1RPilotOutputTests(unittest.TestCase):
    def test_all_geometry_artifacts_exist_match_hash_and_are_real_multi_instance(self) -> None:
        records = read_jsonl(GEOMETRY_ROOT / "manifests" / "sample_manifest.jsonl")
        self.assertGreaterEqual(len(records), 60)
        self.assertLessEqual(len(records), 100)
        self.assertEqual(17, len({item["feature_type"] for item in records}))
        for record in records:
            self.assertIn(record["instance_count"], (2, 3, 4))
            self.assertTrue(record["geometry_audit"]["passed"])
            for path_key, hash_key in (
                ("step_path", "step_sha256"),
                ("brep_path", "brep_sha256"),
                ("graph_path", "graph_sha256"),
            ):
                path = Path(record[path_key])
                self.assertTrue(path.is_file(), path)
                self.assertEqual(record[hash_key], sha256_file(path), path)
            graph = json.loads(Path(record["graph_path"]).read_text(encoding="utf-8"))
            feature_sets = graph["feature_sets"]
            self.assertEqual(record["instance_count"], len(feature_sets))
            self.assertEqual({record["feature_type"]}, {item["macro_feature"] for item in feature_sets})
            self.assertEqual(
                set(range(1, record["instance_count"] + 1)),
                {int(item["instance_id"]) for item in feature_sets},
            )

    def test_query_contract_recomputes_targets_metrics_and_ties(self) -> None:
        records = read_jsonl(GEOMETRY_ROOT / "queries" / "canonical.jsonl")
        audit = _audit_queries(records)
        self.assertTrue(audit["automatic_pass"], audit["failure_examples"])
        self.assertEqual({"passed": 316, "total": 316}, audit["target_reconstruction"])
        self.assertEqual({"passed": 340, "total": 340}, audit["metric_recomputation"])
        self.assertEqual({"largest": 60, "nearest": 60, "smallest": 60}, audit["executable_L4"])
        self.assertEqual(24, audit["status_counts"]["NEEDS_CLARIFICATION"])

    def test_h3r_review_packages_are_human_completed_and_stratified(self) -> None:
        geometry_path = REPORT_ROOT / "human_review_package" / "geometry_review_20.csv"
        query_path = REPORT_ROOT / "human_review_package" / "query_review_60.csv"
        with geometry_path.open("r", encoding="utf-8-sig", newline="") as handle:
            geometry = list(csv.DictReader(handle))
        with query_path.open("r", encoding="utf-8-sig", newline="") as handle:
            queries = list(csv.DictReader(handle))
        self.assertEqual(20, len(geometry))
        self.assertEqual(17, len({item["feature_type"] for item in geometry}))
        self.assertEqual(6, sum(item["geometry_decision"] == "PASS" for item in geometry))
        self.assertEqual(14, sum(item["geometry_decision"] == "MINOR" for item in geometry))
        self.assertEqual(0, sum(item["geometry_decision"] == "BAD" for item in geometry))
        self.assertTrue(all(item["reviewer_reason"].strip() for item in geometry if item["geometry_decision"] == "MINOR"))
        self.assertEqual(60, len(queries))
        self.assertEqual(17, len({item["feature_type"] for item in queries}))
        self.assertEqual(
            {"all": 12, "count": 12, "largest": 12, "smallest": 12, "nearest": 12},
            {name: sum(item["quantifier"] == name for item in queries) for name in ("all", "count", "largest", "smallest", "nearest")},
        )
        self.assertGreater(sum(item["tie_expected"].lower() == "true" for item in queries), 0)
        self.assertTrue(all(item["review_decision"] == "approve" for item in queries))

    def test_old_p3_freeze_is_still_101_of_101(self) -> None:
        result = verify_freeze_manifest(
            ROOT,
            ROOT / "reports" / "p3_a1_cae_queryground_v1" / "freeze" / "freeze_manifest.json",
        )
        self.assertTrue(result["automatic_pass"], result["mismatches"])
        self.assertEqual(101, result["matching_asset_count"])
        self.assertEqual(0, result["hash_changes"])


if __name__ == "__main__":
    unittest.main()
