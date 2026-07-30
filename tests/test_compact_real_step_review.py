import json
import tempfile
import unittest
from pathlib import Path

from generate_compact_real_step_review import build_compact_review


class CompactRealStepReviewTest(unittest.TestCase):
    def test_build_compact_review_summarizes_feature_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir)
            (feature_dir / "part1.json").write_text(json.dumps({
                "sample_id": "part1",
                "source_step_path": "W:/part1.stp",
                "feature_object_count": 1,
                "stock_face_indices": [0, 1],
                "feature_objects": [
                    {
                        "feature_object_id": "part1__feature_000",
                        "macro_feature": "through_hole_round",
                        "face_indices": [2, 3],
                        "confidence": {
                            "macro_confidence_mean": 0.91,
                            "semantic_confidence_mean": 0.82,
                        },
                        "selectable_regions": [
                            {"role": "hole_cylindrical_wall", "face_indices": [2]},
                        ],
                        "fingerprint": {
                            "area_total": 12.5,
                            "area_weighted_centroid": [1.0, 2.0, 3.0],
                            "center_bbox_size": [4.0, 5.0, 6.0],
                            "primitive_surface_histogram": {"cylinder": 1, "plane": 1},
                        },
                    }
                ],
            }), encoding="utf-8")

            review = build_compact_review(feature_dir)

        self.assertEqual("cae_real_step_compact_review_v1", review["schema"])
        self.assertEqual(1, review["sample_count"])
        sample = review["samples"][0]
        self.assertEqual("part1", sample["sample_id"])
        self.assertEqual({"through_hole_round": 1}, sample["macro_counts"])
        item = sample["features"][0]
        self.assertEqual("through_hole_round", item["macro_feature"])
        self.assertEqual([2, 3], item["face_indices"])
        self.assertEqual("unreviewed", item["review_status"])
        self.assertEqual(["hole_cylindrical_wall:[2]"], item["selectable_regions"])


if __name__ == "__main__":
    unittest.main()
