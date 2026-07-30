from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

from build_paper_v1_p1_a2_human_review_package import (
    ERROR_TYPE_OPTIONS,
    EXPECTED_FREEZE_MANIFEST_SHA256,
    FREEZE_MANIFEST,
    PACKAGE_ROOT,
    REVIEW_RESULT_OPTIONS,
    SYNTHETIC_CLASSES,
    build_synthetic_hints,
    build_truth_objects,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P1A2RUnitTests(unittest.TestCase):
    def test_truth_summary_contains_area_role_and_instance(self) -> None:
        graph = {
            "nodes": [
                {
                    "idx": 7,
                    "primitive_surface": "plane",
                    "semantic_label": 28,
                    "attributes": {"physical": {"area": 12.5}},
                }
            ],
            "feature_sets": [
                {
                    "instance_id": 3,
                    "macro_feature": "rib",
                    "node_indices": [7],
                    "source_generator": "RibGenerator",
                    "raw_parameters": {"placement_mode": "boundary"},
                }
            ],
            "label_maps": {
                "face_semantic_subtype": {"id_to_key": {"28": "rib_side_plane"}}
            },
        }
        objects = build_truth_objects(graph)
        self.assertEqual(objects[0]["macro"], "rib")
        self.assertEqual(objects[0]["instance_id"], 3)
        self.assertEqual(objects[0]["face_ids"], [7])
        self.assertEqual(objects[0]["total_area"], 12.5)
        self.assertEqual(objects[0]["semantic_roles"], ["rib_side_plane"])

    def test_overlap_hints_are_advisory_and_find_miss_false_positive(self) -> None:
        truth = [{"object_id": "t1", "macro": "rib", "face_ids": [1, 2]}]
        predicted = [{"object_id": "p1", "macro": "web", "face_ids": [1, 2]}]
        hints = build_synthetic_hints(truth, predicted)
        self.assertTrue(hints["not_a_human_conclusion"])
        self.assertEqual(hints["counts"]["POSSIBLE_MISS"], 1)
        self.assertEqual(hints["counts"]["POSSIBLE_FALSE_POSITIVE"], 1)

    def test_allowed_user_fields_are_frozen(self) -> None:
        self.assertEqual(
            REVIEW_RESULT_OPTIONS, ("PASS", "FAIL", "INVALID_TEST_GEOMETRY")
        )
        self.assertIn("ROLE_ERROR", ERROR_TYPE_OPTIONS)
        self.assertIn("GEOMETRY_LABEL_ERROR", ERROR_TYPE_OPTIONS)


class P1A2RArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        if not PACKAGE_ROOT.is_dir():
            self.skipTest("P1-A2R package has not been generated")
        self.synthetic = load_json(PACKAGE_ROOT / "human_review_synthetic_20.json")
        self.real = load_json(PACKAGE_ROOT / "human_review_real_dev_10.json")

    def test_synthetic_count_coverage_paths_and_rib_question(self) -> None:
        samples = self.synthetic["samples"]
        self.assertEqual(len(samples), 20)
        self.assertEqual({sample["target_class"] for sample in samples}, set(SYNTHETIC_CLASSES))
        rib = next(sample for sample in samples if sample["sample_id"] == "holdout_rib_000")
        self.assertIn("INVALID_TEST_GEOMETRY", rib["special_question"])
        for sample in samples:
            for key in ("step_path", "graph_path", "prediction_path", "feature_objects_path"):
                self.assertTrue(Path(sample[key]).is_file(), sample[key])

    def test_real_count_live_stp_paths_and_three_allowed_directories(self) -> None:
        samples = self.real["samples"]
        self.assertEqual(len(samples), 10)
        self.assertGreaterEqual(len({sample["dataset"] for sample in samples}), 3)
        self.assertTrue(
            {sample["dataset"] for sample in samples}.issubset(
                {"test_part", "test_part_s", "test_part_s1", "test_part_s2"}
            )
        )
        for sample in samples:
            step = Path(sample["step_path"])
            self.assertTrue(step.is_file(), step)
            self.assertEqual(step.suffix.lower(), ".stp")
        dead = {
            "test_part/part1",
            "test_part/part2",
            "test_part/part3",
            "test_part/part5",
        }
        selected = {f"{sample['dataset']}/{sample['sample_id']}" for sample in samples}
        self.assertTrue(dead.isdisjoint(selected))

    def test_every_object_summary_is_actionable(self) -> None:
        for sample in self.synthetic["samples"]:
            for item in sample["truth_objects"]:
                self._assert_object(item, predicted=False)
            for item in sample["prediction_objects"]:
                self._assert_object(item, predicted=True)
        for sample in self.real["samples"]:
            for item in sample["prediction_objects"]:
                self._assert_object(item, predicted=True)

    def _assert_object(self, item: dict, *, predicted: bool) -> None:
        self.assertTrue(item["macro"])
        self.assertTrue(item["face_ids"])
        self.assertGreater(item["total_area"], 0.0)
        self.assertTrue(item["semantic_roles"])
        for face in item["faces"]:
            self.assertIn("face_id", face)
            self.assertGreater(face["area"], 0.0)
            self.assertTrue(face["semantic_role"])
        if predicted:
            self.assertIn("macro_mean", item["confidence"])
            self.assertIn("semantic_mean", item["confidence"])

    def test_csvs_have_only_three_blank_user_fill_fields(self) -> None:
        for name, expected_count in (
            ("human_review_synthetic_20.csv", 20),
            ("human_review_real_dev_10.csv", 10),
        ):
            with (PACKAGE_ROOT / name).open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_count)
            for row in rows:
                self.assertIn("review_result", row)
                self.assertIn("error_type", row)
                self.assertIn("review_note", row)
                self.assertEqual(row["review_result"], "")
                self.assertEqual(row["error_type"], "")
                self.assertEqual(row["review_note"], "")

    def test_guide_html_manifest_and_no_s3_path(self) -> None:
        guide = (PACKAGE_ROOT / "human_review_guide.zh-CN.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("UG/NX", guide)
        self.assertIn("用户只填写三列", guide)
        self.assertIn("返回", guide)
        html_text = (PACKAGE_ROOT / "human_review_summary.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("只读摘要", html_text)
        self.assertNotIn("test_part_s3", (guide + html_text).lower())
        manifest = load_json(PACKAGE_ROOT / "review_package_manifest.json")
        self.assertEqual(manifest["synthetic_sample_count"], 20)
        self.assertEqual(manifest["real_sample_count"], 10)
        for record in manifest["files"]:
            path = PACKAGE_ROOT.parents[2] / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256_file(path), record["sha256"])
        self.assertEqual(len(manifest["implementation_files"]), 2)
        for record in manifest["implementation_files"]:
            path = PACKAGE_ROOT.parents[2] / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256_file(path), record["sha256"])
        package_text = "".join(
            path.read_text(encoding="utf-8-sig", errors="replace")
            for path in PACKAGE_ROOT.iterdir()
            if path.suffix.lower() in {".json", ".csv", ".md", ".html"}
        )
        self.assertNotIn("test_part_s3", package_text.lower())

    def test_p1_a2_freeze_manifest_is_unchanged(self) -> None:
        self.assertEqual(sha256_file(FREEZE_MANIFEST), EXPECTED_FREEZE_MANIFEST_SHA256)


if __name__ == "__main__":
    unittest.main()
