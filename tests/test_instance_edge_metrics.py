import unittest

import torch

from train_gnn_baseline import build_edge_same_instance_metrics_report, compute_binary_edge_metrics
from utils.instance_edge_metrics import compute_edge_same_instance_metrics


class InstanceEdgeMetricsTest(unittest.TestCase):
    def test_ignores_unsupervised_mask_and_negative_labels(self):
        metrics = compute_edge_same_instance_metrics(
            scores=[0.9, 0.8, 0.1, 0.7, 0.2],
            labels=[1, 0, 0, -1, 1],
            mask=[1, 0, 1, 1, 0],
            threshold=0.5,
            thresholds=[0.5],
            scores_are_logits=False,
        )

        self.assertEqual(metrics["valid_edge_count"], 2)
        self.assertEqual(metrics["positive_edge_count"], 1)
        self.assertEqual(metrics["negative_edge_count"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_threshold_precision_recall_and_f1_are_computed_from_probabilities(self):
        metrics = compute_edge_same_instance_metrics(
            scores=[0.9, 0.6, 0.4, 0.2],
            labels=[1, 0, 1, 0],
            threshold=0.5,
            thresholds=[0.5],
            scores_are_logits=False,
        )

        self.assertEqual(metrics["valid_edge_count"], 4)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_no_valid_edges_returns_stable_empty_metrics(self):
        metrics = compute_edge_same_instance_metrics(
            scores=[0.9, 0.1],
            labels=[-1, 0],
            mask=[1, 0],
            threshold=0.5,
            thresholds=[0.3, 0.5],
            scores_are_logits=False,
        )

        self.assertEqual(metrics["valid_edge_count"], 0)
        self.assertEqual(metrics["positive_edge_count"], 0)
        self.assertEqual(metrics["negative_edge_count"], 0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertFalse(metrics["auc_available"])
        self.assertIsNone(metrics["best_f1_threshold"])
        self.assertEqual(len(metrics["threshold_scan"]), 2)

    def test_threshold_scan_selects_best_f1_threshold(self):
        metrics = compute_edge_same_instance_metrics(
            scores=[0.9, 0.65, 0.55, 0.4],
            labels=[1, 1, 0, 0],
            threshold=0.5,
            thresholds=[0.3, 0.5, 0.7],
            scores_are_logits=False,
        )

        self.assertAlmostEqual(metrics["best_f1"], 0.8)
        self.assertAlmostEqual(metrics["best_f1_threshold"], 0.5)
        self.assertEqual([row["threshold"] for row in metrics["threshold_scan"]], [0.3, 0.5, 0.7])

class TrainGNNEdgeMetricsIntegrationTest(unittest.TestCase):
    def test_compute_binary_edge_metrics_keeps_legacy_fields_and_adds_scan_fields(self):
        metrics = compute_binary_edge_metrics(
            logits=[torch.tensor([3.0, -3.0, 0.0], dtype=torch.float32)],
            targets=[torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)],
            masks=[torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)],
        )

        self.assertEqual(metrics["edge_supervised_count"], 3.0)
        self.assertAlmostEqual(metrics["edge_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["edge_micro_precision"], 1.0)
        self.assertAlmostEqual(metrics["edge_micro_recall"], 1.0)
        self.assertAlmostEqual(metrics["edge_micro_f1"], 1.0)
        self.assertEqual(metrics["edge_positive_count"], 2.0)
        self.assertEqual(metrics["edge_negative_count"], 1.0)
        self.assertIn("edge_roc_auc", metrics)
        self.assertIn("edge_pr_auc", metrics)
        self.assertIn("edge_threshold_scan", metrics)
        self.assertIsNotNone(metrics["edge_best_f1_threshold"])

    def test_build_edge_same_instance_report_uses_validation_threshold_selection(self):
        train_metrics = {
            "edge_supervised_count": 10.0,
            "edge_micro_f1": 0.60,
            "edge_best_f1_threshold": 0.45,
            "edge_best_f1": 0.62,
            "edge_threshold_scan": [{"threshold": 0.45, "f1": 0.62}],
        }
        val_metrics = {
            "edge_supervised_count": 8.0,
            "edge_micro_f1": 0.70,
            "edge_best_f1_threshold": 0.55,
            "edge_best_f1": 0.75,
            "edge_threshold_scan": [{"threshold": 0.55, "f1": 0.75}],
        }
        test_metrics = {
            "edge_supervised_count": 6.0,
            "edge_micro_f1": 0.65,
            "edge_best_f1_threshold": 0.50,
            "edge_best_f1": 0.68,
            "edge_threshold_scan": [{"threshold": 0.50, "f1": 0.68}],
        }

        report = build_edge_same_instance_metrics_report(
            best_epoch=3,
            best_val_selection_score=0.82,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        self.assertEqual(report["schema"], "edge_same_instance_metrics_v1")
        self.assertEqual(report["best_epoch"], 3)
        self.assertEqual(report["threshold_selection"]["source_split"], "val")
        self.assertEqual(report["threshold_selection"]["best_f1_threshold"], 0.55)
        self.assertEqual(report["threshold_selection"]["best_f1"], 0.75)
        self.assertEqual(set(report["splits"].keys()), {"train", "val", "test"})
        self.assertEqual(report["splits"]["val"]["micro_f1"], 0.70)
        self.assertEqual(report["splits"]["val"]["threshold_scan"], [{"threshold": 0.55, "f1": 0.75}])

if __name__ == "__main__":
    unittest.main()
