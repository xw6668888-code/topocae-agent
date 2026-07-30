import unittest

from utils.instance_candidate_assembly import (
    assemble_instance_candidates,
    assemble_seed_components,
    build_edge_probability_lookup,
    deduplicate_candidates,
    expand_weak_candidate_faces,
)


def face(idx, macro, area=1.0, confidence=0.9, primitive="plane"):
    return {
        "idx": idx,
        "macro_feature": macro,
        "macro_confidence": confidence,
        "area": area,
        "primitive_surface": primitive,
    }


def edge_attr(src, dst, curve_type="line", convexity="concave"):
    return {
        "source_face": src,
        "target_face": dst,
        "curve_type": curve_type,
        "is_line": 1.0 if curve_type == "line" else 0.0,
        "convexity": convexity,
    }


def graph(edges, faces, edge_attributes=None):
    result = {"edges": edges, "faces": faces}
    if edge_attributes is not None:
        result["edge_attributes"] = edge_attributes
    return result


class InstanceCandidateAssemblyTest(unittest.TestCase):
    def test_edge_probability_lookup_accepts_multiple_record_formats_and_is_undirected(self):
        graph_data = graph([[0, 1], [1, 2], [2, 3], [3, 4]], [face(i, "rib") for i in range(5)])
        prediction_data = {
            "edge_same_instance": [
                {"edge_index": 0, "probability": 0.91},
                {"src": 1, "dst": 2, "same_instance_probability": 0.72},
                {"face_pair": [3, 2], "probability": 0.63},
            ]
        }

        lookup = build_edge_probability_lookup(graph_data, prediction_data)

        self.assertAlmostEqual(lookup[(0, 1)], 0.91)
        self.assertAlmostEqual(lookup[(1, 0)], 0.91)
        self.assertAlmostEqual(lookup[(1, 2)], 0.72)
        self.assertAlmostEqual(lookup[(2, 3)], 0.63)
        self.assertNotIn((3, 4), lookup)

    def test_edge_probability_lookup_keeps_max_probability_for_duplicate_undirected_edge(self):
        graph_data = graph([[0, 1]], [face(0, "rib"), face(1, "rib")])
        prediction_data = {
            "edge_same_instance": [
                {"source_face": 0, "target_face": 1, "same_instance_probability": 0.9},
                {"source_face": 1, "target_face": 0, "same_instance_probability": 0.1},
            ]
        }

        lookup = build_edge_probability_lookup(graph_data, prediction_data)

        self.assertAlmostEqual(lookup[(0, 1)], 0.9)
        self.assertAlmostEqual(lookup[(1, 0)], 0.9)

    def test_strong_same_instance_edges_assemble_same_non_stock_faces(self):
        graph_data = graph([[0, 1], [1, 2]], [face(0, "rib"), face(1, "rib"), face(2, "stock")])
        prediction_data = {
            "faces": [face(0, "rib"), face(1, "rib"), face(2, "stock")],
            "edge_same_instance": [0.82, 0.99],
        }

        candidates = assemble_seed_components(graph_data, prediction_data, strong_threshold=0.55)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["proposal_macro"], "rib")
        self.assertEqual(candidates[0]["face_indices"], [0, 1])
        self.assertEqual(candidates[0]["seed_face_indices"], [0, 1])
        self.assertIn("strong_same_instance_edge", candidates[0]["assembly_reasons"])

    def test_edges_below_strong_threshold_do_not_merge_seed_components(self):
        graph_data = graph([[0, 1]], [face(0, "rib"), face(1, "rib")])
        prediction_data = {
            "faces": [face(0, "rib"), face(1, "rib")],
            "edge_same_instance": [0.54],
        }

        candidates = assemble_seed_components(graph_data, prediction_data, strong_threshold=0.55)

        self.assertEqual([candidate["face_indices"] for candidate in candidates], [[0], [1]])
        self.assertTrue(all("single_face_seed" in candidate["assembly_reasons"] for candidate in candidates))

    def test_graph_truth_macro_is_not_used_as_seed_when_prediction_macro_is_missing(self):
        graph_data = {
            "nodes": [face(0, "rib"), face(1, "rib")],
            "edges": [[0, 1]],
            "edge_attributes": [edge_attr(0, 1)],
        }
        prediction_data = {"edge_same_instance": [0.99]}

        candidates = assemble_instance_candidates(graph_data, prediction_data)

        self.assertEqual(candidates, [])

    def test_rib_weak_expansion_absorbs_small_stock_like_face_but_rejects_large_stock_face(self):
        graph_data = graph(
            [[0, 1], [1, 2], [1, 3]],
            [
                face(0, "rib", area=10.0),
                face(1, "rib", area=9.0),
                face(2, "stock", area=4.0, confidence=0.42),
                face(3, "stock", area=100.0, confidence=0.40),
            ],
            [edge_attr(0, 1), edge_attr(1, 2), edge_attr(1, 3)],
        )
        prediction_data = {
            "faces": [
                face(0, "rib", area=10.0),
                face(1, "rib", area=9.0),
                face(2, "stock", area=4.0, confidence=0.42),
                face(3, "stock", area=100.0, confidence=0.40),
            ],
            "edge_same_instance": [0.90, 0.41, 0.92],
        }
        seeds = assemble_seed_components(graph_data, prediction_data)

        expanded = expand_weak_candidate_faces(seeds, graph_data, prediction_data, weak_threshold=0.35)

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["face_indices"], [0, 1, 2])
        self.assertEqual(expanded[0]["seed_face_indices"], [0, 1])
        self.assertIn("weak_stock_like_expansion", expanded[0]["assembly_reasons"])
        self.assertIn("large_stock_area_guard", expanded[0]["assembly_reasons"])

    def test_rib_weak_expansion_uses_reversed_edge_attributes_for_undirected_key(self):
        graph_data = graph(
            [[0, 1]],
            [
                face(0, "rib", area=10.0),
                face(1, "stock", area=4.0, confidence=0.42),
            ],
            [edge_attr(1, 0, "line", "concave")],
        )
        prediction_data = {
            "faces": [
                face(0, "rib", area=10.0),
                face(1, "stock", area=4.0, confidence=0.42),
            ],
            "edge_same_instance": [0.60],
        }

        candidates = assemble_instance_candidates(graph_data, prediction_data, strong_threshold=0.70, weak_threshold=0.35)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["face_indices"], [0, 1])
        self.assertIn("weak_stock_like_expansion", candidates[0]["assembly_reasons"])

    def test_rib_weak_expansion_rejects_small_stock_like_cylinder(self):
        graph_data = {
            "nodes": [
                {**face(0, "stock", area=10.0), "primitive_surface": "plane"},
                {**face(1, "stock", area=3.0), "primitive_surface": "cylinder"},
            ],
            "edges": [[0, 1]],
            "edge_attributes": [edge_attr(0, 1, "line", "concave")],
        }
        prediction_data = {
            "faces": [
                {"idx": 0, "macro_feature": "rib"},
                {"idx": 1, "macro_feature": "stock"},
            ],
            "edge_same_instance": [0.60],
        }

        candidates = assemble_instance_candidates(graph_data, prediction_data, strong_threshold=0.70, weak_threshold=0.35)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["face_indices"], [0])

    def test_rib_weak_expansion_uses_nested_physical_area_for_area_guard(self):
        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "plane",
                    "attributes": {"physical": {"area": 10.0}},
                },
                {
                    "idx": 1,
                    "primitive_surface": "plane",
                    "attributes": {"physical": {"area": 4.0}},
                },
            ],
            "edges": [[0, 1]],
            "edge_attributes": [edge_attr(0, 1, "line", "concave")],
        }
        prediction_data = {
            "faces": [
                {"idx": 0, "macro_feature": "rib"},
                {"idx": 1, "macro_feature": "stock"},
            ],
            "edge_same_instance": [0.60],
        }

        candidates = assemble_instance_candidates(graph_data, prediction_data, strong_threshold=0.70, weak_threshold=0.35)

        self.assertEqual(candidates[0]["face_indices"], [0, 1])

    def test_slot_like_faces_weakly_expand_but_fillet_is_not_unconditionally_compatible(self):
        graph_data = graph(
            [[0, 1], [0, 2]],
            [
                face(0, "slot_u"),
                face(1, "slot_rect", confidence=0.48),
                face(2, "fillet", area=0.2, confidence=0.99),
            ],
            [edge_attr(0, 1, "line", "concave"), edge_attr(0, 2, "line", "concave")],
        )
        prediction_data = {
            "faces": [
                face(0, "slot_u"),
                face(1, "slot_rect", confidence=0.48),
                face(2, "fillet", area=0.2, confidence=0.99),
            ],
            "edge_same_instance": [0.42, 0.95],
        }
        seeds = assemble_seed_components(graph_data, prediction_data)

        expanded = expand_weak_candidate_faces(seeds, graph_data, prediction_data, weak_threshold=0.35)

        self.assertEqual(expanded[0]["face_indices"], [0, 1])
        self.assertIn("weak_slot_like_expansion", expanded[0]["assembly_reasons"])
        self.assertNotIn(2, expanded[0]["face_indices"])

    def test_slot_like_faces_do_not_weakly_expand_across_convex_edge(self):
        graph_data = graph(
            [[0, 1]],
            [face(0, "slot_u"), face(1, "slot_rect", confidence=0.48)],
            [edge_attr(0, 1, "line", "convex")],
        )
        prediction_data = {
            "faces": [face(0, "slot_u"), face(1, "slot_rect", confidence=0.48)],
            "edge_same_instance": [0.42],
        }
        seeds = assemble_seed_components(graph_data, prediction_data)

        expanded = expand_weak_candidate_faces(seeds, graph_data, prediction_data, weak_threshold=0.35)

        self.assertEqual([candidate["face_indices"] for candidate in expanded], [[0], [1]])

    def test_rib_faces_do_not_weakly_expand_across_convex_edge(self):
        graph_data = graph(
            [[0, 1]],
            [face(0, "rib"), face(1, "rib")],
            [edge_attr(0, 1, "line", "convex")],
        )
        prediction_data = {
            "faces": [face(0, "rib"), face(1, "rib")],
            "edge_same_instance": [0.60],
        }

        candidates = assemble_instance_candidates(graph_data, prediction_data, strong_threshold=0.70, weak_threshold=0.35)

        self.assertEqual([candidate["face_indices"] for candidate in candidates], [[0], [1]])
        self.assertTrue(all("weak_macro_compatible_expansion" not in candidate["assembly_reasons"] for candidate in candidates))

    def test_rib_faces_weakly_expand_across_concave_line_edge(self):
        graph_data = graph(
            [[0, 1]],
            [face(0, "rib"), face(1, "rib")],
            [edge_attr(0, 1, "line", "concave")],
        )
        prediction_data = {
            "faces": [face(0, "rib"), face(1, "rib")],
            "edge_same_instance": [0.60],
        }

        candidates = assemble_instance_candidates(graph_data, prediction_data, strong_threshold=0.70, weak_threshold=0.35)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["face_indices"], [0, 1])
        self.assertIn("weak_macro_compatible_expansion", candidates[0]["assembly_reasons"])

    def test_rib_faces_do_not_weakly_expand_across_explicit_protected_edge(self):
        protected_attributes = [
            {"edge_role": "protected_boundary"},
            {"weak_assembly_compatible": False},
        ]
        for protected_attribute in protected_attributes:
            with self.subTest(protected_attribute=protected_attribute):
                graph_data = graph(
                    [[0, 1]],
                    [face(0, "rib"), face(1, "rib")],
                    [{**edge_attr(0, 1, "line", "concave"), **protected_attribute}],
                )
                prediction_data = {
                    "faces": [face(0, "rib"), face(1, "rib")],
                    "edge_same_instance": [0.60],
                }

                candidates = assemble_instance_candidates(
                    graph_data,
                    prediction_data,
                    strong_threshold=0.70,
                    weak_threshold=0.35,
                )

                self.assertEqual([candidate["face_indices"] for candidate in candidates], [[0], [1]])
                self.assertTrue(
                    all("weak_macro_compatible_expansion" not in candidate["assembly_reasons"] for candidate in candidates)
                )
    def test_deduplicate_candidates_keeps_more_complete_record_for_same_face_set(self):
        candidates = [
            {
                "source": "seed",
                "proposal_macro": "rib",
                "face_indices": [1, 0],
                "seed_face_indices": [0, 1],
                "assembly_reasons": ["strong_same_instance_edge"],
            },
            {
                "source": "expanded",
                "proposal_macro": "rib",
                "face_indices": [0, 1],
                "seed_face_indices": [0, 1],
                "mean_same_instance_probability": 0.87,
                "min_same_instance_probability": 0.87,
                "assembly_reasons": ["strong_same_instance_edge", "weak_slot_like_expansion"],
            },
        ]

        deduped = deduplicate_candidates(candidates)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["source"], "expanded")
        self.assertAlmostEqual(deduped[0]["mean_same_instance_probability"], 0.87)

    def test_missing_edge_same_instance_returns_stable_single_seed_candidates(self):
        graph_data = graph([[0, 1]], [face(0, "rib"), face(1, "stock")])
        prediction_data = {"faces": [face(0, "rib"), face(1, "stock")]}

        candidates = assemble_instance_candidates(graph_data, prediction_data)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["face_indices"], [0])
        self.assertEqual(candidates[0]["proposal_macro"], "rib")
        self.assertIn("single_face_seed", candidates[0]["assembly_reasons"])


if __name__ == "__main__":
    unittest.main()