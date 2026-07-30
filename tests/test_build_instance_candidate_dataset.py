import json
import math
import tempfile
import unittest
from pathlib import Path

from build_instance_candidate_dataset import (
    build_dataset_from_paths,
    split_records,
    write_dataset_outputs,
)


def node(idx, primitive="plane", area=1.0):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "attributes": {
            "physical": {"area": area, "center_of_mass": [float(idx), 0.0, 0.0]},
            "specific": {"normal": [0.0, 0.0, 1.0]},
            "shape_descriptors": {
                "projected_long_extent": 10.0,
                "projected_short_extent": 3.0,
            },
        },
    }


def edge(src, dst, convexity="convex"):
    return [src, dst], {
        "source_face": src,
        "target_face": dst,
        "curve_type": "line",
        "curve_type_id": 1,
        "is_line": 1.0,
        "convexity": convexity,
        "convexity_id": 1 if convexity == "convex" else 2,
        "length": 1.0,
    }


class BuildInstanceCandidateDatasetTest(unittest.TestCase):
    def test_dataset_builder_writes_truth_positive_assembled_reject_schema_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "graphs"
            prediction_dir = root / "predictions"
            output_dir = root / "dataset"
            graph_dir.mkdir()
            prediction_dir.mkdir()

            edges = []
            edge_attributes = []
            for pair, attr in [
                edge(0, 1, "convex"),
                edge(1, 3, "convex"),
                edge(1, 2, "convex"),
            ]:
                edges.append(pair)
                edge_attributes.append(attr)

            graph_data = {
                "nodes": [node(0, area=20.0), node(1, area=20.0), node(2, area=4.0), node(3, area=50.0)],
                "edges": edges,
                "edge_attributes": edge_attributes,
                "feature_sets": [
                    {
                        "instance_id": 1,
                        "macro_feature": "rib",
                        "node_indices": [0, 1, 2],
                    }
                ],
            }
            prediction_data = {
                "sample_id": "toy",
                "faces": [
                    {"idx": 0, "macro_feature": "rib"},
                    {"idx": 1, "macro_feature": "rib"},
                    {"idx": 2, "macro_feature": "stock"},
                    {"idx": 3, "macro_feature": "rib"},
                ],
                "edge_same_instance": [0.95, 0.95, 0.10],
            }
            (graph_dir / "toy.json").write_text(json.dumps(graph_data), encoding="utf-8")
            (prediction_dir / "toy.json").write_text(json.dumps(prediction_data), encoding="utf-8")

            summary = build_dataset_from_paths(
                graph_dir=graph_dir,
                prediction_dir=prediction_dir,
                output_dir=output_dir,
                recursive=False,
                target_macros={"rib"},
                min_iou=0.6,
                min_candidate_coverage=0.70,
                min_truth_coverage=0.70,
                seed=123,
                train_ratio=0.5,
                val_ratio=0.25,
            )

            for name in [
                "records.jsonl",
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "feature_schema.json",
                "dataset_summary.json",
            ]:
                self.assertTrue((output_dir / name).exists(), name)

            records = [
                json.loads(line)
                for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            labels_by_source_faces = {
                (record["source"], tuple(record["face_indices"])): record["label"]
                for record in records
            }
            self.assertEqual(labels_by_source_faces[("truth", (0, 1, 2))], "rib")
            self.assertEqual(labels_by_source_faces[("assembled_instance_candidate", (0, 1, 3))], "reject")
            self.assertIn("features", records[0])
            self.assertIn("face_count", records[0]["features"])
            self.assertIn("topology_valid", records[0]["features"])

            schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
            self.assertIn("face_count", schema["features"])
            self.assertIn("same_instance_probability_mean", schema["features"])
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["feature_schema"], schema["features"])
            self.assertEqual(metadata["label_to_id"]["reject"], 0)
            self.assertEqual(metadata["id_to_label"]["0"], "reject")
            self.assertIn("rib", metadata["label_to_id"])
            self.assertEqual(summary["record_count"], len(records))
            self.assertEqual(summary["label_counts"]["rib"], 1)
            self.assertEqual(summary["label_counts"]["reject"], 1)

            split_records = []
            for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                split_records.extend(
                    json.loads(line)
                    for line in (output_dir / name).read_text(encoding="utf-8").splitlines()
                    if line
                )
            self.assertEqual(len(split_records), len(records))

    def test_write_outputs_rejects_empty_records_before_writing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"

            with self.assertRaisesRegex(ValueError, "record_count"):
                write_dataset_outputs(
                    [],
                    output_dir,
                    seed=1,
                    train_ratio=0.5,
                    val_ratio=0.0,
                    target_macros={"rib"},
                    skipped_prediction_count=0,
                )

            self.assertFalse((output_dir / "records.jsonl").exists())

    def test_write_outputs_requires_reject_positive_and_nonempty_feature_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "only_positive"
            only_positive = [
                {"sample_id": "a", "source": "truth", "label": "rib", "features": {"face_count": 1.0}},
                {"sample_id": "b", "source": "truth", "label": "rib", "features": {"face_count": 2.0}},
            ]

            with self.assertRaisesRegex(ValueError, "reject"):
                write_dataset_outputs(
                    only_positive,
                    output_dir,
                    seed=1,
                    train_ratio=0.5,
                    val_ratio=0.0,
                    target_macros={"rib"},
                    skipped_prediction_count=0,
                )
            self.assertFalse((output_dir / "records.jsonl").exists())

            empty_schema = [
                {"sample_id": "a", "source": "truth", "label": "rib", "features": {}},
                {"sample_id": "b", "source": "assembled_instance_candidate", "label": "reject", "features": {}},
            ]
            with self.assertRaisesRegex(ValueError, "feature_schema"):
                write_dataset_outputs(
                    empty_schema,
                    Path(tmp) / "empty_schema",
                    seed=1,
                    train_ratio=0.5,
                    val_ratio=0.0,
                    target_macros={"rib"},
                    skipped_prediction_count=0,
                )

    def test_split_records_validates_ratios_and_allows_empty_val_for_tiny_dataset(self):
        records = [{"id": idx} for idx in range(4)]
        invalid_ratios = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, -0.1),
            (0.5, 1.0),
            (0.8, 0.2),
        ]
        for train_ratio, val_ratio in invalid_ratios:
            with self.subTest(train_ratio=train_ratio, val_ratio=val_ratio):
                with self.assertRaises(ValueError):
                    split_records(records, seed=1, train_ratio=train_ratio, val_ratio=val_ratio)

        with self.assertRaisesRegex(ValueError, "train.*test"):
            split_records([{"id": 1}], seed=1, train_ratio=0.5, val_ratio=0.0)

        splits = split_records([{"id": 1}, {"id": 2}], seed=1, train_ratio=0.5, val_ratio=0.25)
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["val"]), 0)
        self.assertEqual(len(splits["test"]), 1)

    def test_write_outputs_sanitizes_feature_vector_and_standard_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            records = [
                {"sample_id": "a", "source": "truth", "label": "rib", "features": {"bad": float("nan"), "ok": 1.0}},
                {
                    "sample_id": "b",
                    "source": "assembled_instance_candidate",
                    "label": "reject",
                    "features": {"bad": float("inf"), "ok": 2.0},
                },
            ]

            write_dataset_outputs(
                records,
                output_dir,
                seed=1,
                train_ratio=0.5,
                val_ratio=0.0,
                target_macros={"rib"},
                skipped_prediction_count=0,
            )

            raw_records = (output_dir / "records.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw_records)
            self.assertNotIn("Infinity", raw_records)
            written_records = [json.loads(line) for line in raw_records.splitlines() if line]
            for record in written_records:
                self.assertTrue(all(math.isfinite(value) for value in record["feature_vector"]))

    def test_build_records_can_use_hybrid_proposals_and_preserve_sources(self):
        from unittest.mock import patch
        from pathlib import Path

        from build_instance_candidate_dataset import build_candidate_records_for_graph

        graph_data = {
            "nodes": [
                node(0, primitive="cylinder", area=20.0),
                node(1, primitive="plane", area=5.0),
            ],
            "edges": [],
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "through_hole_round",
                    "node_indices": [0],
                }
            ],
        }
        prediction_data = {
            "sample_id": "toy",
            "faces": [
                {"idx": 0, "macro_feature": "stock"},
                {"idx": 1, "macro_feature": "stock"},
            ],
        }
        hybrid_candidate = {
            "source": "fallback_rule",
            "proposal_sources": ["fallback_rule"],
            "proposal_macro": "through_hole_round",
            "face_indices": [0],
            "seed_face_indices": [0],
            "assembly_reasons": ["fallback_two_circular_plane_boundaries"],
        }

        with patch("build_instance_candidate_dataset.assemble_instance_candidates", return_value=[]), \
             patch("build_instance_candidate_dataset.generate_hybrid_object_proposals", return_value=[hybrid_candidate]):
            records = build_candidate_records_for_graph(
                sample_id="toy",
                graph_data=graph_data,
                prediction_data=prediction_data,
                graph_path=Path("toy.json"),
                prediction_path=Path("toy_pred.json"),
                target_macros={"through_hole_round"},
                min_iou=0.5,
                min_candidate_coverage=0.7,
                min_truth_coverage=0.7,
                proposal_mode="hybrid",
            )

        by_source = {record["source"]: record for record in records}
        self.assertIn("truth", by_source)
        self.assertIn("fallback_rule", by_source)
        self.assertEqual(by_source["fallback_rule"]["label"], "through_hole_round")
        self.assertEqual(by_source["fallback_rule"]["proposal_sources"], ["fallback_rule"])
        self.assertEqual(by_source["fallback_rule"]["assembly_reasons"], ["fallback_two_circular_plane_boundaries"])
        self.assertEqual(by_source["fallback_rule"]["features"]["proposal_source_fallback_rule"], 1.0)

    def test_parse_source_macro_pairs_accepts_csv_and_ignores_empty_items(self):
        from build_instance_candidate_dataset import parse_source_macro_pairs

        pairs = parse_source_macro_pairs(
            "fallback_rule:through_hole_round, topology_rule:rectangular_boss, "
        )

        self.assertEqual(
            pairs,
            {
                ("fallback_rule", "through_hole_round"),
                ("topology_rule", "rectangular_boss"),
            },
        )

    def test_parse_source_macro_pairs_rejects_malformed_items(self):
        from build_instance_candidate_dataset import parse_source_macro_pairs

        with self.assertRaisesRegex(ValueError, "source:macro"):
            parse_source_macro_pairs("fallback_rule-through_hole_round")

    def test_hybrid_rebalance_adds_positive_source_clone_and_downweights_reject(self):
        from unittest.mock import patch
        from pathlib import Path

        from build_instance_candidate_dataset import build_candidate_records_for_graph

        graph_data = {
            "nodes": [
                node(0, primitive="cylinder", area=20.0),
                node(1, primitive="cylinder", area=18.0),
            ],
            "edges": [],
            "feature_sets": [
                {
                    "instance_id": 1,
                    "macro_feature": "through_hole_round",
                    "node_indices": [0],
                }
            ],
        }
        prediction_data = {
            "sample_id": "toy",
            "faces": [
                {"idx": 0, "macro_feature": "stock"},
                {"idx": 1, "macro_feature": "stock"},
            ],
        }
        bad_fallback_candidate = {
            "source": "fallback_rule",
            "proposal_sources": ["fallback_rule"],
            "proposal_macro": "through_hole_round",
            "face_indices": [1],
            "seed_face_indices": [1],
            "assembly_reasons": ["fallback_two_circular_plane_boundaries"],
        }

        with patch("build_instance_candidate_dataset.assemble_instance_candidates", return_value=[]), \
             patch("build_instance_candidate_dataset.generate_hybrid_object_proposals", return_value=[bad_fallback_candidate]):
            records = build_candidate_records_for_graph(
                sample_id="toy",
                graph_data=graph_data,
                prediction_data=prediction_data,
                graph_path=Path("toy.json"),
                prediction_path=Path("toy_pred.json"),
                target_macros={"through_hole_round"},
                min_iou=0.5,
                min_candidate_coverage=0.7,
                min_truth_coverage=0.7,
                proposal_mode="hybrid",
                source_macro_positive_clone_pairs={("fallback_rule", "through_hole_round")},
                reject_downweight_source_macro_pairs={("fallback_rule", "through_hole_round")},
                default_sample_weight=1.0,
                positive_nontruth_weight=3.0,
                positive_clone_weight=5.0,
                downweighted_reject_weight=0.35,
            )

        by_source_faces = {
            (record["source"], tuple(record["face_indices"])): record
            for record in records
        }

        clone = by_source_faces[("fallback_rule", (0,))]
        self.assertEqual(clone["label"], "through_hole_round")
        self.assertEqual(clone["augmentation"], "source_macro_positive_clone")
        self.assertEqual(clone["rebalance_action"], "positive_source_macro_clone")
        self.assertAlmostEqual(clone["sample_weight"], 5.0)
        self.assertEqual(clone["features"]["proposal_source_fallback_rule"], 1.0)

        reject = by_source_faces[("fallback_rule", (1,))]
        self.assertEqual(reject["label"], "reject")
        self.assertEqual(reject["rebalance_action"], "downweighted_reject")
        self.assertAlmostEqual(reject["sample_weight"], 0.35)

    def test_write_outputs_reports_rebalance_action_and_weight_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            records = [
                {
                    "sample_id": "a",
                    "source": "truth",
                    "proposal_macro": "rib",
                    "label": "rib",
                    "features": {"face_count": 1.0},
                    "sample_weight": 1.0,
                    "rebalance_action": "default",
                },
                {
                    "sample_id": "b",
                    "source": "fallback_rule",
                    "proposal_macro": "through_hole_round",
                    "label": "through_hole_round",
                    "features": {"face_count": 1.0},
                    "sample_weight": 5.0,
                    "rebalance_action": "positive_source_macro_clone",
                    "augmentation": "source_macro_positive_clone",
                },
                {
                    "sample_id": "c",
                    "source": "fallback_rule",
                    "proposal_macro": "through_hole_round",
                    "label": "reject",
                    "features": {"face_count": 1.0},
                    "sample_weight": 0.35,
                    "rebalance_action": "downweighted_reject",
                },
            ]

            summary = write_dataset_outputs(
                records,
                output_dir,
                seed=1,
                train_ratio=0.5,
                val_ratio=0.0,
                target_macros={"rib", "through_hole_round"},
                skipped_prediction_count=0,
                proposal_mode="hybrid",
            )

            self.assertEqual(summary["rebalance_action_counts"]["default"], 1)
            self.assertEqual(summary["rebalance_action_counts"]["positive_source_macro_clone"], 1)
            self.assertEqual(summary["rebalance_action_counts"]["downweighted_reject"], 1)
            self.assertEqual(summary["source_macro_positive_clone_count"], 1)
            self.assertAlmostEqual(summary["sample_weight_stats"]["min"], 0.35)
            self.assertAlmostEqual(summary["sample_weight_stats"]["max"], 5.0)

if __name__ == "__main__":
    unittest.main()

