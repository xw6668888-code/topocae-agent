from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import evaluate_paper_v1_recognition as evaluation
from audit_paper_v1_provenance import ExcludedArtifactError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paper_v1_clean_candidate"
REPORTS = ROOT / "reports" / "paper_v1_p1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PaperV1P1CleanBaselineTests(unittest.TestCase):
    def test_forbidden_s3_path_is_rejected_before_read(self) -> None:
        config = load(CONFIG / "datasets.json")
        forbidden = ROOT / "test_part_s3" / "must_not_be_read.json"
        with self.assertRaises(ExcludedArtifactError):
            evaluation.assert_allowed_path(forbidden, config)

    def test_effective_formal_splits_are_disjoint_and_fixed(self) -> None:
        config = load(CONFIG / "datasets.json")
        splits = evaluation.load_effective_splits(config, CONFIG / "quality_exclusions.json")
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 4315, "val": 540, "test": 540})
        self.assertFalse(set(splits["train"]) & set(splits["val"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["val"]) & set(splits["test"]))

    def test_dataset_and_model_provenance_gates_pass(self) -> None:
        audit = load(REPORTS / "dataset_audit.json")
        self.assertEqual(audit["gate"]["status"], "pass")
        direct = audit["solid_validity"]["direct_synthetic_test_validation"]
        self.assertEqual(direct["status"], "pass")
        self.assertEqual(direct["step_count"], 540)
        self.assertEqual(audit["leakage"]["base_geometry_cross_split_collision_count"], 0)
        self.assertEqual(audit["leakage"]["rotation_invariant_exact_signature_cross_split_collision_count"], 0)
        provenance = load(REPORTS / "model_data_provenance.json")
        selected = {"frozen_gnn", "clean_candidate_classifier"}
        rows = {row["artifact"]: row for row in provenance["rows"]}
        for name in selected:
            self.assertTrue(rows[name]["allowed_in_clean_paper_baseline"])
            self.assertFalse(rows[name]["contains_real_data"])
            self.assertFalse(rows[name]["contains_s3_derived_data"])

    def test_metric_coverage_and_selection_contract(self) -> None:
        metrics = load(REPORTS / "metrics.json")["metrics"]
        for variant in ("clean_core", "clean_rib"):
            self.assertEqual(metrics[variant]["synthetic_test"]["face"]["status"], "computed")
            self.assertEqual(metrics[variant]["real_test_part"]["face"]["status"], "not_computable")
            self.assertEqual(metrics[variant]["real_test_part_s1"]["object"]["status"], "not_computable")
        selection = load(CONFIG / "selection.json")
        self.assertEqual(selection["selected_variant"], "clean_core")
        self.assertEqual(selection["synthetic_test_comparison"]["rib_object_f1_gain"], 0.0)
        self.assertFalse(selection["criteria"]["stable_rib_object_f1_gain"])
        self.assertFalse(selection["real_dev_used_for_selection"])

    def test_repeatability_is_exact(self) -> None:
        repeatability = load(REPORTS / "repeatability.json")
        self.assertEqual(repeatability["status"], "pass")
        self.assertEqual(repeatability["compared_file_count"], 1144)
        self.assertEqual(repeatability["mismatches"], [])
        self.assertTrue(repeatability["metrics_exact_match"])

    def test_human_review_package_is_complete_and_clean(self) -> None:
        synthetic = load(REPORTS / "human_review_synthetic_20.json")
        real = load(REPORTS / "human_review_real_dev_10.json")
        self.assertEqual(len(synthetic["samples"]), 20)
        self.assertEqual(len(real["samples"]), 10)
        self.assertGreaterEqual(len({sample["stock_type"] for sample in synthetic["samples"]}), 4)
        paths = json.dumps([synthetic, real], ensure_ascii=False).lower()
        self.assertNotIn("test_part_s3", paths)
        self.assertNotIn("real_step_s3", paths)

    def test_freeze_manifest_hashes(self) -> None:
        manifest = load(ROOT / "paper_v1_clean_candidate_freeze_manifest.json")
        self.assertEqual(manifest["selected_variant"], "clean_core")
        self.assertFalse(manifest["uses_real_dev_for_training_or_selection"])
        self.assertFalse(manifest["uses_s3_derived_training_data"])
        for artifact in manifest["artifacts"]:
            path = ROOT / artifact["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["path"])


if __name__ == "__main__":
    unittest.main()
