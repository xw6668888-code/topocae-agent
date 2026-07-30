from __future__ import annotations

from copy import deepcopy
import unittest

from p1_d_lite.common import (
    one_to_one_area_match,
    strip_graph_truth,
    validate_graph,
)
from p1_d_lite.layered import _verified_objects
from p1_d_lite.oracle import _reconcile_sample


def graph_with_two_equal_ribs(scale: float = 1.0) -> dict:
    nodes = []
    for index in range(4):
        nodes.append({
            "idx": index,
            "primitive_surface": "plane",
            "macro_feature": "rib",
            "instance_id": 7,
            "attributes": {
                "physical": {"area": 1.0 * scale * scale},
                "topology_v2": {"outer_loop_count": 1, "inner_loop_count": 0},
                "macro_feature": "rib",
                "truth": True,
            },
        })
    return {
        "schema_version": "topocae_graph_v2_relative_1_1",
        "graph_v2_protocol": {"labels_provided": True},
        "nodes": nodes,
        "edges": [[0, 1], [1, 0], [2, 3], [3, 2]],
        "feature_sets": [{"macro_feature": "rib", "face_indices": [0, 1]}],
    }


class P1DLiteDiagnosticTests(unittest.TestCase):
    def test_normal_unique_truth_reconciliation(self) -> None:
        graph = graph_with_two_equal_ribs()
        truth = [{"truth_index": 0, "macro_feature": "rib", "area": 2.0, "face_count": 2}]
        pool = [{"source": "unit", "macro_feature": "rib", "face_indices": [0, 1], "area": 2.0}]
        result = _reconcile_sample("dev/sample", truth, graph, pool, [])
        self.assertEqual(result[0]["status"], "auto_unique")
        self.assertEqual(result[0]["face_indices"], [0, 1])

    def test_missing_truth_is_empty_not_fabricated(self) -> None:
        self.assertEqual(_reconcile_sample("dev/sample", [], graph_with_two_equal_ribs(), [], []), [])

    def test_ambiguous_truth_is_not_auto_selected(self) -> None:
        graph = graph_with_two_equal_ribs()
        truth = [{"truth_index": 0, "macro_feature": "rib", "area": 2.0, "face_count": 2}]
        pool = [
            {"source": "a", "macro_feature": "rib", "face_indices": [0, 1], "area": 2.0},
            {"source": "b", "macro_feature": "rib", "face_indices": [2, 3], "area": 2.0},
        ]
        result = _reconcile_sample("dev/sample", truth, graph, pool, [])
        self.assertEqual(result[0]["status"], "ambiguous")
        self.assertIsNone(result[0]["face_indices"])

    def test_no_candidate_is_unresolved(self) -> None:
        truth = [{"truth_index": 0, "macro_feature": "rib", "area": 2.0, "face_count": 2}]
        result = _reconcile_sample("dev/sample", truth, graph_with_two_equal_ribs(), [], [])
        self.assertEqual(result[0]["status"], "unresolved")

    def test_rotation_equivalent_relations_preserve_mapping(self) -> None:
        graph = graph_with_two_equal_ribs()
        rotated = deepcopy(graph)
        rotated["rotation_axis"] = [0.3, 0.4, 0.5]
        rotated["rotation_angle_radians"] = 1.234
        truth = [{"truth_index": 0, "macro_feature": "rib", "area": 2.0, "face_count": 2}]
        pool = [{"source": "unit", "macro_feature": "rib", "face_indices": [0, 1], "area": 2.0}]
        original = _reconcile_sample("dev/sample", truth, graph, pool, [])[0]
        transformed = _reconcile_sample("dev/sample", truth, rotated, pool, [])[0]
        self.assertEqual(original["status"], transformed["status"])
        self.assertEqual(original["face_indices"], transformed["face_indices"])

    def test_uniform_scale_equivalent_relations_preserve_mapping(self) -> None:
        graph = graph_with_two_equal_ribs(scale=3.0)
        truth = [{"truth_index": 0, "macro_feature": "rib", "area": 18.0, "face_count": 2}]
        pool = [{"source": "unit", "macro_feature": "rib", "face_indices": [0, 1], "area": 18.0}]
        result = _reconcile_sample("dev/sample", truth, graph, pool, [])
        self.assertEqual(result[0]["status"], "auto_unique")

    def test_truth_is_stripped_before_inference(self) -> None:
        clean = strip_graph_truth(graph_with_two_equal_ribs())
        self.assertNotIn("feature_sets", clean)
        self.assertFalse(clean["graph_v2_protocol"]["labels_provided"])
        for node in clean["nodes"]:
            self.assertNotIn("macro_feature", node)
            self.assertNotIn("instance_id", node)
            self.assertNotIn("truth", node["attributes"])

    def test_wrong_schema_fails_closed(self) -> None:
        graph = graph_with_two_equal_ribs()
        graph["schema_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Graph v2.1"):
            validate_graph(graph)

    def test_abstention_is_not_emitted_as_feature_object(self) -> None:
        graph = graph_with_two_equal_ribs()
        records = [
            {"candidate_id": "a", "status": "abstain", "abstain": True, "predicted_macro": "unknown", "refined_face_indices": [0, 1]},
            {"candidate_id": "b", "status": "accepted", "abstain": False, "predicted_macro": "rib", "refined_face_indices": [2, 3], "final_confidence": 0.9},
        ]
        output = _verified_objects(graph, records)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["macro_feature"], "rib")

    def test_area_match_is_strictly_one_to_one(self) -> None:
        truths = [
            {"macro_feature": "rib", "area": 10.0},
            {"macro_feature": "rib", "area": 10.0},
        ]
        predictions = [{"macro_feature": "rib", "area": 10.0}]
        result = one_to_one_area_match(truths, predictions)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()

