import unittest

from utils.topology_rules import validate_feature_topology


def node(idx, primitive, area=1.0, normal=None, axis=None, radius=0.0, center=None, descriptors=None):
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
            "physical": {"area": area, "center_of_mass": center or [0.0, 0.0, 0.0]},
            "specific": specific,
            "shape_descriptors": descriptors or {},
        },
    }


def directed_edge(src, dst, curve_type="line", convexity="unknown", length=1.0, dihedral_angle=1.57079632679):
    curve_ids = {"unknown": 0, "line": 1, "circle": 2}
    convexity_ids = {"unknown": 0, "convex": 1, "concave": 2, "smooth": 3}
    return (
        [src, dst],
        {
            "source_face": src,
            "target_face": dst,
            "curve_type": curve_type,
            "curve_type_id": curve_ids[curve_type],
            "length": length,
            "is_line": 1.0 if curve_type == "line" else 0.0,
            "is_circle": 1.0 if curve_type == "circle" else 0.0,
            "convexity": convexity,
            "convexity_id": convexity_ids[convexity],
            "dihedral_angle": dihedral_angle,
        },
    )


def graph(nodes, edge_pairs):
    edges = []
    attrs = []
    for edge_pair in edge_pairs:
        src, dst, curve_type, convexity, *rest = edge_pair
        length = rest[0] if len(rest) >= 1 else 1.0
        dihedral_angle = rest[1] if len(rest) >= 2 else 1.57079632679
        edge, attr = directed_edge(
            src,
            dst,
            curve_type=curve_type,
            convexity=convexity,
            length=length,
            dihedral_angle=dihedral_angle,
        )
        edges.append(edge)
        attrs.append(attr)
    return {"nodes": nodes, "edges": edges, "edge_attributes": attrs}


class TopologyRulesTest(unittest.TestCase):
    def test_blind_hole_requires_cylindrical_wall_and_bottom_face(self):
        g = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "cylinder", axis=[0, 0, 1], radius=5.0),
                node(2, "plane", normal=[0, 0, 1]),
            ],
            [
                (0, 1, "circle", "convex"),
                (1, 0, "circle", "convex"),
                (1, 2, "circle", "concave"),
                (2, 1, "circle", "concave"),
            ],
        )

        valid = validate_feature_topology("blind_hole_round", [1, 2], g)
        invalid = validate_feature_topology("blind_hole_round", [1], g)

        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])
        self.assertIn("missing_bottom_face", invalid["reasons"])

    def test_coaxial_multi_radius_hole_accepts_two_cylinders_and_shoulder(self):
        g = graph(
            [
                node(0, "cylinder", axis=[0, 0, 1], radius=8.0),
                node(1, "plane", normal=[0, 0, 1]),
                node(2, "cylinder", axis=[0, 0, 1], radius=4.0),
            ],
            [
                (0, 1, "circle", "concave"),
                (1, 0, "circle", "concave"),
                (1, 2, "circle", "convex"),
                (2, 1, "circle", "convex"),
            ],
        )

        stepped = validate_feature_topology("stepped_hole_round", [0, 1, 2], g)
        counterbore = validate_feature_topology("counterbore_hole", [0, 1, 2], g)

        self.assertTrue(stepped["valid"])
        self.assertTrue(counterbore["valid"])
        self.assertEqual(stepped["normalized_family"], "coaxial_multi_radius_hole")

    def test_counterbore_requires_large_cylinder_shoulder_and_small_through_cylinder_without_bottom(self):
        g = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "cylinder", axis=[0, 0, 1], radius=8.0),
                node(2, "plane", normal=[0, 0, 1]),
                node(3, "cylinder", axis=[0, 0, 1], radius=4.0),
                node(4, "plane", normal=[0, 0, -1]),
                node(5, "plane", normal=[0, 0, 1]),
            ],
            [
                (1, 0, "circle", "smooth"),
                (0, 1, "circle", "smooth"),
                (1, 2, "circle", "concave"),
                (2, 1, "circle", "concave"),
                (2, 3, "circle", "convex"),
                (3, 2, "circle", "convex"),
                (3, 4, "circle", "smooth"),
                (4, 3, "circle", "smooth"),
                (3, 5, "circle", "concave"),
                (5, 3, "circle", "concave"),
            ],
        )

        valid = validate_feature_topology("counterbore_hole", [1, 2, 3], g)
        invalid_with_bottom = validate_feature_topology("counterbore_hole", [1, 2, 3, 5], g)

        self.assertTrue(valid["valid"])
        self.assertIn("large_cylinder_shoulder_small_through_cylinder", valid["evidence"])
        self.assertFalse(invalid_with_bottom["valid"])
        self.assertIn("counterbore_should_not_include_bottom_face", invalid_with_bottom["reasons"])

    def test_through_hole_accepts_single_cylinder_with_two_circular_openings(self):
        g = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "cylinder", axis=[0, 0, 1], radius=3.0),
                node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (1, 0, "circle", "smooth"),
                (0, 1, "circle", "smooth"),
                (1, 2, "circle", "smooth"),
                (2, 1, "circle", "smooth"),
            ],
        )

        result = validate_feature_topology("through_hole_round", [1], g)

        self.assertTrue(result["valid"])
        self.assertIn("single_cylindrical_wall_with_two_openings", result["evidence"])

    def test_countersink_requires_cone_plus_cylinder(self):
        cylinder_only = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "cylinder", axis=[0, 0, 1], radius=3.0),
                node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (1, 0, "circle", "smooth"),
                (1, 2, "circle", "smooth"),
            ],
        )
        valid_graph = graph(
            [
                node(0, "cone", axis=[0, 0, 1]),
                node(1, "cylinder", axis=[0, 0, 1], radius=3.0),
            ],
            [
                (0, 1, "circle", "smooth"),
                (1, 0, "circle", "smooth"),
            ],
        )

        invalid = validate_feature_topology("countersink_hole", [1], cylinder_only)
        valid = validate_feature_topology("countersink_hole", [0, 1], valid_graph)

        self.assertFalse(invalid["valid"])
        self.assertIn("missing_conical_countersink_face", invalid["reasons"])
        self.assertTrue(valid["valid"])

    def test_obround_hole_requires_concave_internal_wall_junctions(self):
        valid_obround = graph(
            [
                node(0, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 8.0}),
                node(1, "cylinder", axis=[0, 0, 1], radius=5.0),
                node(2, "cylinder", axis=[0, 0, 1], radius=5.0),
                node(3, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 8.0}),
            ],
            [
                (0, 1, "line", "concave"),
                (1, 0, "line", "concave"),
                (0, 2, "line", "concave"),
                (2, 0, "line", "concave"),
                (1, 3, "line", "concave"),
                (3, 1, "line", "concave"),
                (2, 3, "line", "concave"),
                (3, 2, "line", "concave"),
            ],
        )
        outer_chamfer_like = graph(
            [
                node(0, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 20.0}),
                node(1, "plane", normal=[0, 1, 0], descriptors={"projected_aspect_ratio": 20.0}),
                node(2, "cylinder", axis=[0, 0, 1], radius=5.0),
                node(3, "cylinder", axis=[0, 0, 1], radius=5.0),
            ],
            [
                (0, 2, "line", "convex"),
                (2, 0, "line", "convex"),
                (1, 3, "line", "convex"),
                (3, 1, "line", "convex"),
            ],
        )

        self.assertTrue(validate_feature_topology("obround_hole", [0, 1, 2, 3], valid_obround)["valid"])
        invalid = validate_feature_topology("obround_hole", [0, 1, 2, 3], outer_chamfer_like)
        self.assertFalse(invalid["valid"])
        self.assertIn("missing_concave_obround_wall_junctions", invalid["reasons"])

    def test_rectangular_pocket_rejects_two_face_components(self):
        g = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "plane", normal=[1, 0, 0]),
                node(2, "plane", normal=[-1, 0, 0]),
                node(3, "plane", normal=[0, 1, 0]),
                node(4, "plane", normal=[0, -1, 0]),
            ],
            [
                (0, 1, "line", "concave"),
                (0, 2, "line", "concave"),
                (0, 3, "line", "concave"),
                (0, 4, "line", "concave"),
            ],
        )

        invalid = validate_feature_topology("pocket_rect", [0, 1], g)
        valid = validate_feature_topology("pocket_rect", [0, 1, 2, 3, 4], g)

        self.assertFalse(invalid["valid"])
        self.assertIn("min_5_faces_required", invalid["reasons"])
        self.assertTrue(valid["valid"])

    def test_rectangular_boss_requires_five_planar_faces_with_convex_internal_edges(self):
        g = graph(
            [
                node(0, "plane", normal=[0, 0, 1]),
                node(1, "plane", normal=[1, 0, 0]),
                node(2, "plane", normal=[-1, 0, 0]),
                node(3, "plane", normal=[0, 1, 0]),
                node(4, "plane", normal=[0, -1, 0]),
            ],
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

        invalid = validate_feature_topology("rectangular_boss", [0, 1, 2], g)
        valid = validate_feature_topology("rectangular_boss", [0, 1, 2, 3, 4], g)

        self.assertFalse(invalid["valid"])
        self.assertIn("min_5_faces_required", invalid["reasons"])
        self.assertTrue(valid["valid"])
        self.assertIn("top_plus_four_side_faces", valid["evidence"])

    def test_slot_rect_requires_bottom_and_side_face_topology(self):
        two_face_v_slot = graph(
            [
                node(0, "plane", normal=[0, 0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
                node(1, "plane", normal=[0, -0.9, -0.4], descriptors={"projected_aspect_ratio": 80.0}),
            ],
            [
                (0, 1, "line", "concave", 100.0, 0.75),
                (1, 0, "line", "concave", 100.0, 0.75),
            ],
        )
        rect_slot = graph(
            [
                node(0, "plane", normal=[0, 0, 1], descriptors={"projected_aspect_ratio": 20.0}),
                node(1, "plane", normal=[1, 0, 0], descriptors={"projected_aspect_ratio": 20.0}),
                node(2, "plane", normal=[-1, 0, 0], descriptors={"projected_aspect_ratio": 20.0}),
            ],
            [
                (0, 1, "line", "concave", 100.0, 1.57079632679),
                (1, 0, "line", "concave", 100.0, 1.57079632679),
                (0, 2, "line", "concave", 100.0, 1.57079632679),
                (2, 0, "line", "concave", 100.0, 1.57079632679),
            ],
        )

        invalid_rect = validate_feature_topology("slot_rect", [0, 1], two_face_v_slot)
        valid_v = validate_feature_topology("slot_v", [0, 1], two_face_v_slot)
        valid_rect = validate_feature_topology("slot_rect", [0, 1, 2], rect_slot)

        self.assertFalse(invalid_rect["valid"])
        self.assertIn("rect_slot_requires_bottom_plus_two_sides", invalid_rect["reasons"])
        self.assertTrue(valid_v["valid"])
        self.assertIn("two_inclined_planes_with_long_concave_valley", valid_v["evidence"])
        self.assertTrue(valid_rect["valid"])

    def test_rib_web_candidate_uses_thin_parallel_face_pair(self):
        g = graph(
            [
                node(
                    0,
                    "plane",
                    area=300.0,
                    normal=[1, 0, 0],
                    center=[0.0, 0.0, 0.0],
                    descriptors={"projected_long_extent": 60.0, "projected_short_extent": 20.0},
                ),
                node(
                    1,
                    "plane",
                    area=300.0,
                    normal=[-1, 0, 0],
                    center=[3.0, 0.0, 0.0],
                    descriptors={"projected_long_extent": 60.0, "projected_short_extent": 20.0},
                ),
            ],
            [(0, 1, "line", "smooth"), (1, 0, "line", "smooth")],
        )

        result = validate_feature_topology("rib_web_candidate", [0, 1], g)

        self.assertTrue(result["valid"])
        self.assertIn("thin_parallel_face_pair", result["evidence"])

    def test_rib_web_candidate_accepts_high_long_to_thickness_ratio_even_when_short_extent_is_moderate(self):
        g = graph(
            [
                node(
                    0,
                    "plane",
                    area=510.0,
                    normal=[1, 0, 0],
                    center=[0.0, 0.0, 0.0],
                    descriptors={"projected_long_extent": 44.0, "projected_short_extent": 11.6},
                ),
                node(
                    1,
                    "plane",
                    area=510.0,
                    normal=[-1, 0, 0],
                    center=[4.1, 0.0, 0.0],
                    descriptors={"projected_long_extent": 44.0, "projected_short_extent": 11.6},
                ),
            ],
            [(0, 1, "line", "smooth"), (1, 0, "line", "smooth")],
        )

        result = validate_feature_topology("rib_web_candidate", [0, 1], g)

        self.assertTrue(result["valid"])

    def test_transition_features_are_valid_as_single_faces_and_can_attach_to_parent(self):
        g = graph([node(0, "plane", area=5.0, normal=[0, 0, 1])], [])

        result = validate_feature_topology("chamfer", [0], g)

        self.assertTrue(result["valid"])
        self.assertTrue(result["may_attach_to_parent_feature"])


if __name__ == "__main__":
    unittest.main()
