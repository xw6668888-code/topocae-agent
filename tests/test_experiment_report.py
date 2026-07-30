import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_experiment_report import generate_report


class ExperimentReportTest(unittest.TestCase):
    def test_generate_report_writes_html_with_core_metrics(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            run_dir = dataset_root / "model_runs" / "run"
            reports_dir = dataset_root / "reports"
            run_dir.mkdir(parents=True)
            reports_dir.mkdir(parents=True)

            write_json(dataset_root / "dataset_meta.json", {"actual_samples_written": 2, "random_seed": 123})
            write_json(
                reports_dir / "dataset_report.json",
                {
                    "total_samples": 2,
                    "splits": {
                        "train": {"sample_count": 1, "feature_instance_distribution": {"blind_hole_round": 2}},
                        "val": {"sample_count": 1, "feature_instance_distribution": {"pocket_circular": 1}},
                    },
                },
            )
            write_json(
                run_dir / "metrics.json",
                {
                    "best_epoch": 1,
                    "best_val_selection_score": 0.4,
                    "final_test_metrics": {
                        "macro_macro_f1_active": 0.3,
                        "semantic_macro_f1_active": 0.2,
                    },
                    "history": [
                        {
                            "epoch": 1,
                            "train": {"loss": 1.0},
                            "val": {"loss": 1.2, "macro_macro_f1_active": 0.3, "semantic_macro_f1_active": 0.2},
                        }
                    ],
                },
            )
            write_json(
                reports_dir / "prediction_eval.json",
                {
                    "sample_count": 2,
                    "face_count": 10,
                    "macro": {
                        "global_accuracy": 0.8,
                        "macro_f1": 0.7,
                        "weighted_f1": 0.75,
                        "per_class": {
                            "blind_hole_round": {"support": 3, "precision": 1.0, "recall": 0.67, "f1": 0.8}
                        },
                    },
                    "semantic": {
                        "global_accuracy": 0.6,
                        "macro_f1": 0.5,
                        "weighted_f1": 0.55,
                        "per_class": {},
                    },
                },
            )
            output_path = reports_dir / "report.html"

            generate_report(
                dataset_root=dataset_root,
                run_dir=run_dir,
                prediction_eval_path=reports_dir / "prediction_eval.json",
                output_path=output_path,
            )

            html = output_path.read_text(encoding="utf-8")
            self.assertIn("CAE STEP Recognition Experiment Report", html)
            self.assertIn("Macro Feature", html)
            self.assertIn("Macro Classes to Inspect", html)
            self.assertIn("blind_hole_round", html)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
