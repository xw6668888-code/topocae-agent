import json
import tempfile
import unittest
from pathlib import Path

from train_candidate_topology_classifier import sample_weights_for_records, train_candidate_classifier


class TrainCandidateTopologyClassifierTest(unittest.TestCase):
    def test_sample_weights_prefer_explicit_record_weight(self):
        records = [
            {"source": "truth"},
            {"source": "hardcase"},
            {"source": "truth", "sample_weight": 7.5},
        ]

        weights = sample_weights_for_records(records, hard_source_weight=4.0)

        self.assertEqual(weights.tolist(), [1.0, 4.0, 7.5])

    def test_train_candidate_classifier_writes_model_and_metrics(self):
        records = [
            {"sample_id": "a", "label": "rib", "label_id": 1, "feature_vector": [0.1, 3.0]},
            {"sample_id": "b", "label": "rib", "label_id": 1, "feature_vector": [0.12, 3.0]},
            {"sample_id": "c", "label": "reject", "label_id": 0, "feature_vector": [0.85, 3.0]},
            {"sample_id": "d", "label": "reject", "label_id": 0, "feature_vector": [0.90, 3.0]},
        ]
        metadata = {
            "feature_schema": ["area_ratio", "face_count"],
            "label_to_id": {"reject": 0, "rib": 1},
            "id_to_label": {"0": "reject", "1": "rib"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()
            for split_name in ["train", "val", "test"]:
                with (dataset_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fp:
                    for record in records:
                        fp.write(json.dumps(record) + "\n")
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            output_dir = Path(tmp) / "runs"
            result = train_candidate_classifier(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                run_name="smoke",
                model_kind="random_forest",
                seed=7,
                n_estimators=20,
                max_depth=4,
                hard_source_weight=2.0,
            )

            self.assertTrue((output_dir / "smoke" / "model.joblib").exists())
            self.assertTrue((output_dir / "smoke" / "metrics.json").exists())
            self.assertIn("test", result["metrics"])
            self.assertIn("test_by_source", result["metrics"])
            self.assertGreaterEqual(result["metrics"]["test"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()


