import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.topology_dataset_filter import filter_topology_valid_samples


class TopologyDatasetFilterTest(unittest.TestCase):
    def test_filter_topology_valid_samples_copies_only_samples_without_invalid_audit_records(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            (source / "graphs").mkdir(parents=True)
            (source / "steps").mkdir(parents=True)
            (source / "label_maps").mkdir()
            (source / "reports").mkdir()

            valid_graph = {"nodes": [], "edges": [], "edge_attributes": [], "feature_sets": []}
            invalid_graph = {"nodes": [], "edges": [], "edge_attributes": [], "feature_sets": []}
            write_json(source / "graphs" / "valid.json", valid_graph)
            write_json(source / "graphs" / "invalid.json", invalid_graph)
            (source / "steps" / "valid.step").write_text("STEP valid", encoding="utf-8")
            (source / "steps" / "invalid.step").write_text("STEP invalid", encoding="utf-8")
            write_json(source / "label_maps" / "macro_feature_label_map.json", {"ok": True})
            write_json(
                source / "reports" / "topology_rule_audit.json",
                {
                    "sample_count": 2,
                    "invalid_samples": [
                        {
                            "sample_id": "invalid",
                            "graph_path": str((source / "graphs" / "invalid.json").resolve()),
                            "invalid_feature_set_count": 1,
                        }
                    ],
                },
            )

            report = filter_topology_valid_samples(source, target)

            self.assertEqual(report["source_sample_count"], 2)
            self.assertEqual(report["accepted_sample_count"], 1)
            self.assertEqual(report["rejected_sample_count"], 1)
            self.assertTrue((target / "graphs" / "valid.json").exists())
            self.assertTrue((target / "steps" / "valid.step").exists())
            self.assertFalse((target / "graphs" / "invalid.json").exists())
            self.assertTrue((target / "label_maps" / "macro_feature_label_map.json").exists())
            self.assertTrue((target / "reports" / "topology_filter_report.json").exists())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
