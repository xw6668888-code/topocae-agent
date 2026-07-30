import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.topology_rule_audit import audit_graph_directory, audit_graph_topology, aggregate_topology_audits


def node(idx, primitive):
    primitive_ids = {"plane": 1, "cylinder": 2}
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "primitive_surface_id": primitive_ids.get(primitive, 0),
        "attributes": {
            "physical": {"area": 1.0, "center_of_mass": [float(idx), 0.0, 0.0]},
            "specific": {"normal": [0.0, 0.0, 1.0]} if primitive == "plane" else {"axis": [0.0, 0.0, 1.0], "radius": 2.0},
            "shape_descriptors": {},
        },
    }


class TopologyRuleAuditTest(unittest.TestCase):
    def test_audit_graph_reports_invalid_two_face_rectangular_pocket(self):
        graph = {
            "nodes": [node(0, "plane"), node(1, "plane")],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "line", "convexity": "concave", "convexity_id": 2},
                {"source_face": 1, "target_face": 0, "curve_type": "line", "convexity": "concave", "convexity_id": 2},
            ],
            "feature_sets": [
                {
                    "instance_id": 7,
                    "macro_feature": "pocket_rect",
                    "node_indices": [0, 1],
                }
            ],
        }

        report = audit_graph_topology(graph, graph_path="sample.json")

        self.assertEqual(report["feature_set_count"], 1)
        self.assertEqual(report["invalid_feature_set_count"], 1)
        self.assertEqual(report["invalid_feature_sets"][0]["macro_feature"], "pocket_rect")
        self.assertIn("min_5_faces_required", report["invalid_feature_sets"][0]["reasons"])

    def test_aggregate_topology_audits_counts_invalid_by_macro_and_reason(self):
        aggregate = aggregate_topology_audits([
            {
                "sample_id": "a",
                "feature_set_count": 1,
                "invalid_feature_set_count": 1,
                "invalid_feature_sets": [
                    {
                        "macro_feature": "pocket_rect",
                        "reasons": ["min_5_faces_required", "requires_planar_bottom_and_side_faces"],
                    }
                ],
            },
            {
                "sample_id": "b",
                "feature_set_count": 1,
                "invalid_feature_set_count": 0,
                "invalid_feature_sets": [],
            },
        ])

        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["feature_set_count"], 2)
        self.assertEqual(aggregate["invalid_feature_set_count"], 1)
        self.assertEqual(aggregate["invalid_by_macro"], {"pocket_rect": 1})
        self.assertEqual(aggregate["reason_counts"]["min_5_faces_required"], 1)

    def test_audit_graph_directory_writes_aggregate_report(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graphs_dir = root / "graphs"
            graphs_dir.mkdir()
            output_path = root / "reports" / "topology_rule_audit.json"
            graph = {
                "nodes": [node(0, "plane"), node(1, "plane")],
                "edges": [[0, 1]],
                "edge_attributes": [{"source_face": 0, "target_face": 1, "curve_type": "line"}],
                "feature_sets": [{"instance_id": 1, "macro_feature": "pocket_rect", "node_indices": [0, 1]}],
            }
            (graphs_dir / "sample.json").write_text(json.dumps(graph), encoding="utf-8")

            report = audit_graph_directory(graphs_dir, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["invalid_feature_set_count"], 1)


if __name__ == "__main__":
    unittest.main()
