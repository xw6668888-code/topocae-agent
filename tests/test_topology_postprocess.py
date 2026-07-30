import unittest

from utils.topology_postprocess import (
    apply_topology_postprocess,
    find_counterbore_candidates,
    find_rectangular_prism_candidates,
)


def graph_node(idx, primitive="plane", normal=None, axis=None, radius=0.0, descriptors=None):
    primitive_ids = {"plane": 1, "cylinder": 2, "cone": 3}
    specific = {}
    if normal is not None:
        specific["normal"] = normal
    if axis is not None:
        specific["axis"] = axis
    if radius:
        specific["radius"] = radius
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "primitive_surface_id": primitive_ids.get(primitive, 0),
        "attributes": {
            "physical": {
                "area": descriptors.pop("area", 1.0) if descriptors else 1.0,
                "center_of_mass": descriptors.pop("center", [float(idx), 0.0, 0.0]) if descriptors else [float(idx), 0.0, 0.0],
            },
            "specific": specific,
            "shape_descriptors": descriptors or {},
        },
    }


def plane_node(idx):
    return graph_node(idx, "plane", normal=[0.0, 0.0, 1.0])


def face(idx, macro_id=11, macro="pocket_rect"):
    return {
        "idx": idx,
        "primitive_surface": "plane",
        "primitive_surface_id": 1,
        "macro_feature_id": macro_id,
        "macro_feature": macro,
        "macro_confidence": 0.91,
        "semantic_subtype_id": 24,
        "semantic_subtype": "rectangular_pocket_bottom",
        "semantic_confidence": 0.80,
    }


def edge(src, dst, curve_type="line", convexity="concave", length=1.0, dihedral_angle=1.57079632679):
    curve_ids = {"line": 1, "circle": 2}
    convexity_ids = {"convex": 1, "concave": 2, "smooth": 3}
    return (
        [src, dst],
        {
            "source_face": src,
            "target_face": dst,
            "curve_type": curve_type,
            "curve_type_id": curve_ids.get(curve_type, 0),
            "is_line": 1.0 if curve_type == "line" else 0.0,
            "is_circle": 1.0 if curve_type == "circle" else 0.0,
            "convexity": convexity,
            "convexity_id": convexity_ids.get(convexity, 0),
            "length": length,
            "dihedral_angle": dihedral_angle,
        },
    )


def graph(nodes, edge_defs):
    edges = []
    attrs = []
    for edge_def in edge_defs:
        edge_pair, attr = edge(*edge_def)
        edges.append(edge_pair)
        attrs.append(attr)
    return {"nodes": nodes, "edges": edges, "edge_attributes": attrs}


class TopologyPostprocessTest(unittest.TestCase):
    def test_invalid_two_face_rectangular_pocket_is_demoted_to_stock(self):
        graph = {
            "nodes": [plane_node(0), plane_node(1)],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "line", "convexity": "concave", "convexity_id": 2},
                {"source_face": 1, "target_face": 0, "curve_type": "line", "convexity": "concave", "convexity_id": 2},
            ],
        }
        prediction = {
            "sample_id": "sample",
            "faces": [face(0), face(1)],
        }

        corrected = apply_topology_postprocess(prediction, graph)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["stock", "stock"])
        self.assertEqual(corrected["topology_postprocess"]["invalid_component_count"], 1)
        self.assertEqual(corrected["topology_postprocess"]["demoted_face_count"], 2)
        self.assertIn("min_5_faces_required", corrected["topology_postprocess"]["invalid_components"][0]["reasons"])
        self.assertEqual(corrected["faces"][0]["raw_macro_feature"], "pocket_rect")

    def test_partial_rectangular_pocket_candidate_expands_to_five_concave_planes(self):
        graph_data = graph(
            [plane_node(idx) for idx in range(5)],
            [
                (0, 1, "line", "concave"),
                (1, 0, "line", "concave"),
                (0, 2, "line", "concave"),
                (2, 0, "line", "concave"),
                (0, 3, "line", "concave"),
                (3, 0, "line", "concave"),
                (0, 4, "line", "concave"),
                (4, 0, "line", "concave"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=11, macro="pocket_rect"),
                face(1, macro_id=11, macro="pocket_rect"),
                face(2, macro_id=11, macro="pocket_rect"),
                face(3, macro_id=0, macro="stock"),
                face(4, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["pocket_rect"] * 5)
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "rect_pocket_bottom_plane")
        self.assertEqual(corrected["faces"][3]["semantic_subtype"], "rect_pocket_side_plane")

    def test_noisy_rectangular_pocket_component_uses_local_candidate_search(self):
        nodes = [graph_node(idx, "plane", normal=[0, 0, 1], descriptors={"area": 10.0}) for idx in range(5)]
        nodes.extend(
            graph_node(idx, "plane", normal=[0, 0, 1], descriptors={"area": 50.0})
            for idx in range(5, 34)
        )
        edge_defs = []
        for side_idx in [1, 2, 3, 4]:
            edge_defs.extend([(0, side_idx, "line", "concave"), (side_idx, 0, "line", "concave")])
        for idx in range(4, 33):
            edge_defs.extend([(idx, idx + 1, "line", "concave"), (idx + 1, idx, "line", "concave")])
        graph_data = graph(nodes, edge_defs)
        prediction_faces = [
            face(0, macro_id=11, macro="pocket_rect"),
            face(1, macro_id=11, macro="pocket_rect"),
            face(2, macro_id=11, macro="pocket_rect"),
        ] + [face(idx, macro_id=0, macro="stock") for idx in range(3, 34)]

        candidates = find_rectangular_prism_candidates(
            prediction_faces,
            graph_data,
            set(),
            "pocket_rect",
            "concave",
        )

        self.assertIn([0, 1, 2, 3, 4], candidates)

    def test_partial_rectangular_boss_candidate_expands_to_five_convex_planes(self):
        graph_data = graph(
            [plane_node(idx) for idx in range(5)],
            [
                (0, 1, "line", "convex"),
                (1, 0, "line", "convex"),
                (0, 2, "line", "convex"),
                (2, 0, "line", "convex"),
                (0, 3, "line", "convex"),
                (3, 0, "line", "convex"),
                (0, 4, "line", "convex"),
                (4, 0, "line", "convex"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=24, macro="rectangular_boss"),
                face(1, macro_id=24, macro="rectangular_boss"),
                face(2, macro_id=24, macro="rectangular_boss"),
                face(3, macro_id=0, macro="stock"),
                face(4, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["rectangular_boss"] * 5)
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "pad_top_plane")
        self.assertEqual(corrected["faces"][3]["semantic_subtype"], "pad_side_plane")

    def test_exterior_two_face_cylinder_candidate_is_repaired_to_cylinder_boss_not_blind_hole(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1000.0, "center": [0.0, 0.0, 0.0]}),
                graph_node(1, "plane", normal=[0, 0, -1], descriptors={"area": 1000.0, "center": [0.0, 0.0, 10.0]}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=2.0, descriptors={"area": 75.4, "center": [0.0, 0.0, -1.5]}),
                graph_node(3, "plane", normal=[0, 0, -1], descriptors={"area": 12.56, "center": [0.0, 0.0, -3.0]}),
            ],
            [
                (2, 0, "circle", "smooth"),
                (0, 2, "circle", "smooth"),
                (2, 3, "circle", "smooth"),
                (3, 2, "circle", "smooth"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(2, macro_id=2, macro="blind_hole_round"),
                face(3, macro_id=2, macro="blind_hole_round"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["cylinder_boss", "cylinder_boss"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["cylinder_boss_side", "cylinder_boss_top"])

    def test_valid_transition_candidate_is_not_demoted(self):
        graph = {"nodes": [plane_node(0)], "edges": [], "edge_attributes": []}
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=25, macro="fillet")]}

        corrected = apply_topology_postprocess(prediction, graph)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "fillet")
        self.assertEqual(corrected["topology_postprocess"]["demoted_face_count"], 0)

    def test_single_cylinder_countersink_is_repaired_to_through_hole_when_open_at_both_ends(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1]),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=3.0),
                graph_node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (1, 0, "circle", "smooth"),
                (0, 1, "circle", "smooth"),
                (1, 2, "circle", "smooth"),
                (2, 1, "circle", "smooth"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(
                    1,
                    macro_id=5,
                    macro="countersink_hole",
                )
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["faces"][0]["macro_feature_id"], 1)
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "through_hole_cylinder")
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)
        self.assertEqual(corrected["topology_postprocess"]["demoted_face_count"], 0)

    def test_cone_and_cylinder_are_merged_into_countersink_even_when_initial_macros_disagree(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=5.0),
                graph_node(1, "cone", axis=[0, 0, 1]),
            ],
            [
                (0, 1, "circle", "smooth"),
                (1, 0, "circle", "smooth"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=1, macro="through_hole_round"),
                face(1, macro_id=5, macro="countersink_hole"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["countersink_hole", "countersink_hole"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["countersink_small_cylinder", "countersink_cone"])
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_counterbore_candidate_merges_large_cylinder_step_and_small_through_cylinder(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1000.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 120.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 150.0}),
                graph_node(3, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 400.0}),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 1000.0}),
            ],
            [
                (1, 0, "circle", "smooth", 50.0),
                (0, 1, "circle", "smooth", 50.0),
                (1, 2, "circle", "concave", 50.0),
                (2, 1, "circle", "concave", 50.0),
                (2, 3, "circle", "convex", 25.0),
                (3, 2, "circle", "convex", 25.0),
                (3, 4, "circle", "smooth", 25.0),
                (4, 3, "circle", "smooth", 25.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(1, macro_id=4, macro="counterbore_hole"),
                face(2, macro_id=0, macro="stock"),
                face(3, macro_id=1, macro="through_hole_round"),
            ],
        }

        candidates = find_counterbore_candidates(prediction["faces"], graph_data, set())
        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(candidates, [[1, 2, 3]])
        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["counterbore_hole"] * 3)
        self.assertEqual(
            [f["semantic_subtype"] for f in corrected["faces"]],
            ["counterbore_large_cylinder", "counterbore_step_plane", "counterbore_small_cylinder"],
        )
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_stepped_multi_radius_predictions_are_split_to_counterbore_candidates(self):
        graph_data = graph(
            [
                graph_node(7, "cylinder", axis=[0, 1, 0], radius=3.0, descriptors={"area": 84.823}),
                graph_node(8, "cylinder", axis=[0, 1, 0], radius=1.75, descriptors={"area": 38.4845}),
                graph_node(9, "cylinder", axis=[0, 1, 0], radius=3.0, descriptors={"area": 84.823}),
                graph_node(10, "cylinder", axis=[0, 1, 0], radius=1.75, descriptors={"area": 38.4845}),
                graph_node(13, "plane", normal=[0, -1, 0], descriptors={"area": 18.6532}),
                graph_node(14, "plane", normal=[0, -1, 0], descriptors={"area": 18.6532}),
            ],
            [
                (7, 14, "circle", "smooth", 18.85), (14, 7, "circle", "smooth", 18.85),
                (8, 14, "circle", "smooth", 11.00), (14, 8, "circle", "smooth", 11.00),
                (9, 13, "circle", "smooth", 18.85), (13, 9, "circle", "smooth", 18.85),
                (10, 13, "circle", "smooth", 11.00), (13, 10, "circle", "smooth", 11.00),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=3, macro="stepped_hole_round") for idx in [7, 8, 9, 10, 13, 14]],
        }

        candidates = find_counterbore_candidates(prediction["faces"], graph_data, set())
        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(sorted(candidates), [[7, 8, 14], [9, 10, 13]])
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [7, 8, 14]}, {"counterbore_hole"})
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [9, 10, 13]}, {"counterbore_hole"})
        self.assertEqual(by_idx[7]["semantic_subtype"], "counterbore_large_cylinder")
        self.assertEqual(by_idx[8]["semantic_subtype"], "counterbore_small_cylinder")
        self.assertEqual(by_idx[14]["semantic_subtype"], "counterbore_step_plane")

    def test_large_context_plane_with_stepped_semantic_is_not_counterbore_shoulder(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 2000.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 120.0}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 80.0}),
            ],
            [
                (0, 1, "circle", "smooth", 50.0), (1, 0, "circle", "smooth", 50.0),
                (0, 2, "circle", "smooth", 25.0), (2, 0, "circle", "smooth", 25.0),
            ],
        )
        plane = face(0, macro_id=0, macro="stock")
        plane["semantic_subtype"] = "stepped_hole_step_plane"
        large = face(1, macro_id=1, macro="through_hole_round")
        small = face(2, macro_id=3, macro="stepped_hole_round")
        prediction_faces = [plane, large, small]

        candidates = find_counterbore_candidates(prediction_faces, graph_data, set())

        self.assertEqual(candidates, [])

    def test_stock_cylinders_and_shoulder_are_not_repaired_to_counterbore_without_counterbore_seed(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1000.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 120.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 150.0}),
                graph_node(3, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 400.0}),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 1000.0}),
            ],
            [
                (1, 0, "circle", "smooth", 50.0),
                (0, 1, "circle", "smooth", 50.0),
                (1, 2, "circle", "concave", 50.0),
                (2, 1, "circle", "concave", 50.0),
                (2, 3, "circle", "convex", 25.0),
                (3, 2, "circle", "convex", 25.0),
                (3, 4, "circle", "smooth", 25.0),
                (4, 3, "circle", "smooth", 25.0),
            ],
        )
        prediction_faces = [
            face(1, macro_id=0, macro="stock"),
            face(2, macro_id=0, macro="stock"),
            face(3, macro_id=1, macro="through_hole_round"),
        ]

        candidates = find_counterbore_candidates(prediction_faces, graph_data, set())

        self.assertEqual(candidates, [])

    def test_chamfer_like_cone_and_stock_cylinder_are_not_repaired_to_countersink_without_countersink_seed(self):
        graph_data = graph(
            [
                graph_node(0, "cone", axis=[0, 0, 1], descriptors={"area": 20.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=10.0, descriptors={"area": 400.0}),
            ],
            [
                (0, 1, "circle", "smooth", 62.8),
                (1, 0, "circle", "smooth", 62.8),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=26, macro="chamfer"),
                face(1, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["chamfer", "stock"])

    def test_blind_hole_candidate_merges_cylinder_wall_with_misclassified_bottom_plane(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1]),
                graph_node(1, "cylinder", axis=[0, 0, -1], radius=4.0),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 50.27}),
            ],
            [
                (1, 0, "circle", "smooth"),
                (0, 1, "circle", "smooth"),
                (1, 2, "circle", "smooth"),
                (2, 1, "circle", "smooth"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(1, macro_id=2, macro="blind_hole_round"),
                face(2, macro_id=14, macro="cylinder_boss"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["blind_hole_round", "blind_hole_round"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["blind_hole_cylinder", "blind_hole_bottom"])
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_two_face_v_slot_predicted_as_rect_slot_is_repaired_to_slot_v(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
                graph_node(1, "plane", normal=[0, -0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
            ],
            [
                (0, 1, "line", "concave", 100.0, 0.75),
                (1, 0, "line", "concave", 100.0, 0.75),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=10, macro="slot_rect"),
                face(1, macro_id=10, macro="slot_rect"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_v", "slot_v"])
        self.assertEqual([f["macro_feature_id"] for f in corrected["faces"]], [33, 33])
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_split_two_face_v_slot_is_repaired_even_when_same_instance_score_is_low(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
                graph_node(1, "plane", normal=[0, -0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
            ],
            [
                (0, 1, "line", "concave", 100.0, 0.75),
                (1, 0, "line", "concave", 100.0, 0.75),
            ],
        )
        graph_data["prediction_edge_same_instance"] = [
            {"source_face": 0, "target_face": 1, "same_instance_probability": 0.1},
            {"source_face": 1, "target_face": 0, "same_instance_probability": 0.1},
        ]
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=10, macro="slot_rect"),
                face(1, macro_id=10, macro="slot_rect"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_v", "slot_v"])
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_outer_chamfer_like_obround_component_is_demoted(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 20.0}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"projected_aspect_ratio": 20.0}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=5.0),
                graph_node(3, "cylinder", axis=[0, 0, 1], radius=5.0),
            ],
            [
                (0, 2, "line", "convex"),
                (2, 0, "line", "convex"),
                (2, 1, "line", "convex"),
                (1, 2, "line", "convex"),
                (1, 3, "line", "convex"),
                (3, 1, "line", "convex"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=6, macro="obround_hole") for idx in range(4)],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["stock", "stock", "fillet", "fillet"])
        self.assertEqual(corrected["topology_postprocess"]["repaired_component_count"], 1)

    def test_u_slot_candidate_repairs_obround_like_planes_and_cylinder(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 1, 0], radius=2.0),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
            ],
            [
                (0, 1, "line", "concave", 100.0),
                (1, 0, "line", "concave", 100.0),
                (0, 2, "line", "concave", 100.0),
                (2, 0, "line", "concave", 100.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(0, macro_id=6, macro="obround_hole"), face(1, macro_id=6, macro="obround_hole"), face(2, macro_id=6, macro="obround_hole")],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_u", "slot_u", "slot_u"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["slot_u_bottom_plane", "slot_u_side_plane", "slot_u_side_plane"])

    def test_candidate_repair_is_not_overwritten_by_larger_invalid_obround_component(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"projected_aspect_ratio": 20.0}),
                graph_node(1, "cylinder", axis=[0, 1, 0], radius=2.0),
                graph_node(2, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(3, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
            ],
            [
                (1, 2, "line", "concave", 100.0),
                (2, 1, "line", "concave", 100.0),
                (1, 3, "line", "concave", 100.0),
                (3, 1, "line", "concave", 100.0),
                (0, 1, "line", "convex", 1.0),
                (1, 0, "line", "convex", 1.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=6, macro="obround_hole") for idx in range(4)],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["stock", "slot_u", "slot_u", "slot_u"])

    def test_o_slot_candidate_repairs_large_angle_cylindrical_fillet_prediction(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 1, 0], radius=2.5, descriptors={"area": 392.699}),
                graph_node(1, "plane", normal=[1, 0, 0]),
                graph_node(2, "plane", normal=[-1, 0, 0]),
            ],
            [
                (0, 1, "line", "convex", 50.0),
                (1, 0, "line", "convex", 50.0),
                (0, 2, "line", "convex", 50.0),
                (2, 0, "line", "convex", 50.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(0, macro_id=25, macro="fillet")],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "slot_o")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "slot_o_bottom_plane")

    def test_stock_transition_candidates_are_repaired_to_fillet_and_chamfer(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 50000.0}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 42000.0}),
                graph_node(2, "cylinder", axis=[1, 0, 0], radius=20.0, descriptors={"area": 1570.8}),
                graph_node(
                    3,
                    "plane",
                    normal=[0, 0.7, 0.7],
                    descriptors={
                        "area": 1096.0,
                        "projected_aspect_ratio": 22.0,
                        "projected_long_extent": 155.0,
                        "projected_short_extent": 7.1,
                    },
                ),
            ],
            [
                (2, 0, "circle", "smooth", 31.4),
                (0, 2, "circle", "smooth", 31.4),
                (2, 1, "line", "convex", 50.0),
                (1, 2, "line", "convex", 50.0),
                (3, 0, "line", "concave", 155.0, 0.785),
                (0, 3, "line", "concave", 155.0, 0.785),
                (3, 1, "line", "concave", 155.0, 0.785),
                (1, 3, "line", "concave", 155.0, 0.785),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=0, macro="stock") for idx in range(4)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][2]["macro_feature"], "fillet")
        self.assertEqual(corrected["faces"][2]["semantic_subtype"], "fillet_blend_surface")
        self.assertEqual(corrected["faces"][3]["macro_feature"], "chamfer")
        self.assertEqual(corrected["faces"][3]["semantic_subtype"], "chamfer_bevel_plane")

    def test_stock_v_slot_pair_is_repaired_to_slot_v_before_chamfer(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0.48, 0.87], descriptors={"area": 8790.3, "projected_aspect_ratio": 25.0}),
                graph_node(1, "plane", normal=[0, 0.35, -0.94], descriptors={"area": 8219.4, "projected_aspect_ratio": 26.0}),
            ],
            [
                (0, 1, "line", "concave", 470.0, 0.868),
                (1, 0, "line", "concave", 470.0, 0.868),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock"), face(1, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_v", "slot_v"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["slot_v_side_plane", "slot_v_side_plane"])

    def test_chamfer_predicted_v_slot_pair_is_repaired_to_slot_v_before_chamfer_objects(self):
        graph_data = graph(
            [
                graph_node(18, "plane", normal=[-0.41, 0.91, 0.0], descriptors={"area": 47.0363, "projected_aspect_ratio": 57.49, "projected_long_extent": 52.0, "projected_short_extent": 0.90}),
                graph_node(19, "plane", normal=[-0.53, -0.85, 0.0], descriptors={"area": 50.6622, "projected_aspect_ratio": 53.37, "projected_long_extent": 52.0, "projected_short_extent": 0.97}),
            ],
            [
                (18, 19, "line", "concave", 52.0, 0.977),
                (19, 18, "line", "concave", 52.0, 0.977),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(18, macro_id=26, macro="chamfer"),
                face(19, macro_id=26, macro="chamfer"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([item["macro_feature"] for item in corrected["faces"]], ["slot_v", "slot_v"])
        self.assertEqual([item["semantic_subtype"] for item in corrected["faces"]], ["slot_v_side_plane", "slot_v_side_plane"])

    def test_stock_bspline_slot_surface_is_repaired_to_slot_o(self):
        graph_data = graph(
            [
                graph_node(0, "bspline_surface", descriptors={"area": 35991.1}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 540000.0}),
                graph_node(2, "plane", normal=[0, 1, 0], descriptors={"area": 358000.0}),
            ],
            [
                (0, 1, "bspline", "smooth", 486.0),
                (1, 0, "bspline", "smooth", 486.0),
                (0, 2, "bspline", "smooth", 489.0),
                (2, 0, "bspline", "smooth", 489.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "slot_o")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "slot_o_bottom_plane")

    def test_stock_three_face_thin_wall_is_repaired_to_rib(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0.95, -0.31],
                    descriptors={
                        "area": 6261.8,
                        "projected_aspect_ratio": 62.0,
                        "projected_long_extent": 626.0,
                        "projected_short_extent": 10.0,
                        "center": [281.0, 96.7, -318.8],
                    },
                ),
                graph_node(
                    1,
                    "plane",
                    normal=[1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [286.0, 64.5, -219.6],
                    },
                ),
                graph_node(
                    2,
                    "plane",
                    normal=[-1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [276.0, 64.5, -219.6],
                    },
                ),
            ],
            [
                (0, 1, "line", "convex", 626.0),
                (1, 0, "line", "convex", 626.0),
                (0, 2, "line", "convex", 626.0),
                (2, 0, "line", "convex", 626.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=0, macro="stock") for idx in range(3)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["rib", "rib", "rib"])
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "rib_top_plane")
        self.assertEqual(corrected["faces"][1]["semantic_subtype"], "rib_side_plane")
        self.assertEqual(corrected["faces"][2]["semantic_subtype"], "rib_side_plane")

    def test_stock_slab_planes_are_not_repaired_to_rib(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0, -1],
                    descriptors={
                        "area": 134080.0,
                        "projected_aspect_ratio": 1.58,
                        "projected_long_extent": 472.0,
                        "projected_short_extent": 299.0,
                        "center": [206.0, 153.0, -21.0],
                    },
                ),
                graph_node(
                    1,
                    "plane",
                    normal=[0, 0, 1],
                    descriptors={
                        "area": 159940.0,
                        "projected_aspect_ratio": 1.37,
                        "projected_long_extent": 471.0,
                        "projected_short_extent": 343.0,
                        "center": [206.0, 128.0, 29.0],
                    },
                ),
                graph_node(
                    2,
                    "plane",
                    normal=[0, 1, 0],
                    descriptors={
                        "area": 21500.0,
                        "projected_aspect_ratio": 8.6,
                        "projected_long_extent": 430.0,
                        "projected_short_extent": 50.0,
                        "center": [206.0, 300.0, 4.0],
                    },
                ),
            ],
            [
                (2, 0, "line", "convex", 430.0),
                (0, 2, "line", "convex", 430.0),
                (2, 1, "line", "convex", 430.0),
                (1, 2, "line", "convex", 430.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=0, macro="stock") for idx in range(3)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["stock", "stock", "stock"])

    def test_moderate_aspect_stock_end_face_is_not_repaired_to_chamfer(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0, -1],
                    descriptors={
                        "area": 18060.0,
                        "projected_aspect_ratio": 10.24,
                        "projected_long_extent": 430.0,
                        "projected_short_extent": 42.0,
                    },
                ),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 540000.0}),
                graph_node(2, "plane", normal=[1, 0, 0], descriptors={"area": 60000.0}),
            ],
            [
                (0, 1, "line", "convex", 430.0),
                (1, 0, "line", "convex", 430.0),
                (0, 2, "line", "convex", 42.0),
                (2, 0, "line", "convex", 42.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")

    def test_u_slot_candidate_accepts_web_side_planes_before_o_slot_repair(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 1, 0], radius=1.25, descriptors={"area": 1963.4954}),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 200.0}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 200.0}),
            ],
            [
                (0, 1, "line", "concave", 500.0),
                (1, 0, "line", "concave", 500.0),
                (0, 2, "line", "concave", 500.0),
                (2, 0, "line", "concave", 500.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=25, macro="fillet"),
                face(1, macro_id=21, macro="web"),
                face(2, macro_id=21, macro="web"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_u", "slot_u", "slot_u"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["slot_u_bottom_plane", "slot_u_side_plane", "slot_u_side_plane"])

    def test_all_stock_five_convex_planes_can_be_repaired_to_rectangular_boss(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1457.0}),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"area": 94.0, "projected_aspect_ratio": 23.5}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"area": 94.0, "projected_aspect_ratio": 23.5}),
                graph_node(3, "plane", normal=[0, 1, 0], descriptors={"area": 62.0, "projected_aspect_ratio": 15.5}),
                graph_node(4, "plane", normal=[0, -1, 0], descriptors={"area": 62.0, "projected_aspect_ratio": 15.5}),
                graph_node(5, "plane", normal=[0, 0, 1], descriptors={"area": 9807.0}),
            ],
            [
                (0, 1, "line", "convex", 47.0),
                (1, 0, "line", "convex", 47.0),
                (0, 2, "line", "convex", 47.0),
                (2, 0, "line", "convex", 47.0),
                (0, 3, "line", "convex", 31.0),
                (3, 0, "line", "convex", 31.0),
                (0, 4, "line", "convex", 31.0),
                (4, 0, "line", "convex", 31.0),
                (1, 5, "line", "concave", 47.0),
                (5, 1, "line", "concave", 47.0),
                (2, 5, "line", "concave", 47.0),
                (5, 2, "line", "concave", 47.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=0, macro="stock") for idx in range(6)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["rectangular_boss"] * 5 + ["stock"])
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "pad_top_plane")
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"][1:5]], ["pad_side_plane"] * 4)

    def test_rectangular_boss_completion_accepts_web_side_predictions(self):
        graph_data = graph(
            [graph_node(idx, "plane", descriptors={"projected_aspect_ratio": 20.0}) for idx in range(5)]
            + [graph_node(5, "plane", descriptors={"area": 9807.0})],
            [
                (0, 1, "line", "convex", 47.0),
                (1, 0, "line", "convex", 47.0),
                (0, 2, "line", "convex", 47.0),
                (2, 0, "line", "convex", 47.0),
                (0, 3, "line", "convex", 31.0),
                (3, 0, "line", "convex", 31.0),
                (0, 4, "line", "convex", 31.0),
                (4, 0, "line", "convex", 31.0),
                (1, 5, "line", "concave", 47.0),
                (5, 1, "line", "concave", 47.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=24, macro="rectangular_boss"),
                face(2, macro_id=21, macro="web"),
                face(3, macro_id=24, macro="rectangular_boss"),
                face(4, macro_id=21, macro="web"),
                face(5, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["rectangular_boss"] * 5 + ["stock"])

    def test_three_plane_stock_web_rectangular_slot_is_repaired_to_slot_rect(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
            ],
            [
                (0, 1, "line", "concave", 435.0),
                (1, 0, "line", "concave", 435.0),
                (0, 2, "line", "concave", 435.0),
                (2, 0, "line", "concave", 435.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=21, macro="web"),
                face(2, macro_id=21, macro="web"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_rect", "slot_rect", "slot_rect"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["slot_rect_bottom_plane", "slot_rect_side_plane", "slot_rect_side_plane"])

    def test_low_confidence_hole_cylinder_transition_can_be_repaired_to_fillet(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 100000.0}),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={"area": 100000.0}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=20.0, descriptors={"area": 1570.7963}),
            ],
            [
                (0, 2, "line", "convex", 50.0),
                (2, 0, "line", "convex", 50.0),
                (1, 2, "line", "convex", 50.0),
                (2, 1, "line", "convex", 50.0),
                (0, 2, "circle", "smooth", 31.416),
                (2, 0, "circle", "smooth", 31.416),
                (1, 2, "circle", "smooth", 31.416),
                (2, 1, "circle", "smooth", 31.416),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
                {**face(2, macro_id=1, macro="through_hole_round"), "macro_confidence": 0.5002},
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][2]["macro_feature"], "fillet")
        self.assertEqual(corrected["faces"][2]["semantic_subtype"], "fillet_blend_surface")

    def test_plate_side_face_with_right_angle_edges_is_not_repaired_to_chamfer(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, -1, 0],
                    descriptors={
                        "area": 835.0,
                        "projected_aspect_ratio": 33.4,
                        "projected_long_extent": 167.0,
                        "projected_short_extent": 5.0,
                    },
                ),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 9807.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 12408.0}),
            ],
            [
                (0, 1, "line", "convex", 167.0, 1.57079632679),
                (1, 0, "line", "convex", 167.0, 1.57079632679),
                (0, 2, "line", "convex", 167.0, 1.57079632679),
                (2, 0, "line", "convex", 167.0, 1.57079632679),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")

    def test_u_slot_candidate_accepts_web_bottom_prediction(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 1, 0], radius=1.75, descriptors={"area": 2391.5}),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 145.0}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 145.0}),
            ],
            [
                (0, 1, "line", "concave", 435.0),
                (1, 0, "line", "concave", 435.0),
                (0, 2, "line", "concave", 435.0),
                (2, 0, "line", "concave", 435.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=21, macro="web"),
                face(1, macro_id=21, macro="web"),
                face(2, macro_id=21, macro="web"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["slot_u", "slot_u", "slot_u"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["slot_u_bottom_plane", "slot_u_side_plane", "slot_u_side_plane"])

    def test_exterior_two_face_stepped_hole_candidate_is_repaired_to_cylinder_boss(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1200.0, "center": [0.0, 0.0, 0.0]}),
                graph_node(1, "plane", normal=[0, 0, -1], descriptors={"area": 1200.0, "center": [0.0, 0.0, 20.0]}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=2.7, descriptors={"area": 101.2, "center": [0.0, 0.0, 23.0]}),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 22.7, "center": [0.0, 0.0, 26.0]}),
            ],
            [
                (2, 1, "circle", "smooth", 16.96),
                (1, 2, "circle", "smooth", 16.96),
                (2, 3, "circle", "smooth", 16.96),
                (3, 2, "circle", "smooth", 16.96),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                {**face(2, macro_id=3, macro="stepped_hole_round"), "semantic_subtype": "stepped_hole_small_hole_cylinder"},
                {**face(3, macro_id=3, macro="stepped_hole_round"), "semantic_subtype": "stepped_hole_step_plane"},
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"]], ["cylinder_boss", "cylinder_boss"])
        self.assertEqual([f["semantic_subtype"] for f in corrected["faces"]], ["cylinder_boss_side", "cylinder_boss_top"])

    def test_large_base_plate_faces_are_not_repaired_to_rib(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0, 1],
                    descriptors={
                        "area": 9807.45,
                        "projected_aspect_ratio": 2.49,
                        "projected_long_extent": 177.09,
                        "projected_short_extent": 71.25,
                        "center": [13.3, 2.6, 0.0],
                    },
                ),
                graph_node(
                    1,
                    "plane",
                    normal=[0, -1, 0],
                    descriptors={
                        "area": 835.0,
                        "projected_aspect_ratio": 33.4,
                        "projected_long_extent": 167.0,
                        "projected_short_extent": 5.0,
                        "center": [13.5, -33.0, 2.5],
                    },
                ),
                graph_node(
                    2,
                    "plane",
                    normal=[0, 0, 1],
                    descriptors={
                        "area": 12408.45,
                        "projected_aspect_ratio": 2.48,
                        "projected_long_extent": 177.21,
                        "projected_short_extent": 71.58,
                        "center": [13.8, 2.4, 5.0],
                    },
                ),
                graph_node(3, "plane", normal=[1, 0, 0], descriptors={"area": 2600.0}),
                graph_node(4, "plane", normal=[-1, 0, 0], descriptors={"area": 2600.0}),
            ],
            [
                (0, 1, "line", "convex", 167.0),
                (1, 0, "line", "convex", 167.0),
                (1, 2, "line", "convex", 167.0),
                (2, 1, "line", "convex", 167.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=0, macro="stock") for idx in range(5)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([f["macro_feature"] for f in corrected["faces"][:3]], ["stock", "stock", "stock"])

    def test_obround_like_large_cylindrical_slot_surface_is_repaired_to_slot_o_not_fillet(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, -1, 0], radius=2.5, descriptors={"area": 3416.28}),
                graph_node(1, "plane", normal=[0, 0, 1]),
                graph_node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (0, 1, "line", "convex", 435.0),
                (1, 0, "line", "convex", 435.0),
                (0, 2, "line", "convex", 435.0),
                (2, 0, "line", "convex", 435.0),
                (0, 1, "circle", "smooth", 7.85),
                (1, 0, "circle", "smooth", 7.85),
                (0, 2, "circle", "smooth", 7.85),
                (2, 0, "circle", "smooth", 7.85),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=6, macro="obround_hole")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "slot_o")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "slot_o_bottom_plane")

    def test_multi_face_obround_hole_is_not_split_into_slot_o_end_cylinders(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 100.0}),
                graph_node(2, "cylinder", axis=[0, 1, 0], radius=1.0, descriptors={"area": 314.159}),
                graph_node(3, "cylinder", axis=[0, 1, 0], radius=1.0, descriptors={"area": 314.159}),
            ],
            [
                (0, 2, "line", "concave", 100.0),
                (2, 0, "line", "concave", 100.0),
                (1, 2, "line", "concave", 100.0),
                (2, 1, "line", "concave", 100.0),
                (0, 3, "line", "concave", 100.0),
                (3, 0, "line", "concave", 100.0),
                (1, 3, "line", "concave", 100.0),
                (3, 1, "line", "concave", 100.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=6, macro="obround_hole") for idx in range(4)]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([item["macro_feature"] for item in corrected["faces"]], ["obround_hole"] * 4)


if __name__ == "__main__":
    unittest.main()
