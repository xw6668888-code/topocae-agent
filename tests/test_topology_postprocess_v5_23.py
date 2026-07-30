from __future__ import annotations

import unittest

from utils.topology_postprocess import (
    find_review_targeted_chamfer_candidates,
    find_review_targeted_fillet_candidates,
    find_review_targeted_rib_candidates,
)


def plane_node(idx: int, area: float, aspect: float, long_extent: float, short_extent: float, normal=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0)):
    return {
        "idx": idx,
        "primitive_surface": "plane",
        "area": area,
        "attributes": {
            "physical": {"area": area, "center_of_mass": list(center)},
            "specific": {"normal": list(normal)},
            "shape_descriptors": {
                "projected_aspect_ratio": aspect,
                "projected_long_extent": long_extent,
                "projected_short_extent": short_extent,
            },
        },
    }


def cylinder_node(idx: int, area: float, aspect: float, long_extent: float, short_extent: float):
    return {
        "idx": idx,
        "primitive_surface": "cylinder",
        "area": area,
        "attributes": {
            "physical": {"area": area, "center_of_mass": [0.0, 0.0, 0.0]},
            "shape_descriptors": {
                "projected_aspect_ratio": aspect,
                "projected_long_extent": long_extent,
                "projected_short_extent": short_extent,
            },
        },
    }


def graph_data(nodes, edges, edge_attributes):
    return {"nodes": nodes, "edges": edges, "edge_attributes": edge_attributes}


class TopologyPostprocessV523Test(unittest.TestCase):
    def test_review_targeted_rib_finds_parallel_thin_sides_with_cap(self):
        nodes = [
            plane_node(1, area=500.0, aspect=8.0, long_extent=100.0, short_extent=12.0, normal=(0.0, 1.0, 0.0), center=(0.0, -1.0, 0.0)),
            plane_node(2, area=500.0, aspect=8.0, long_extent=100.0, short_extent=12.0, normal=(0.0, 1.0, 0.0), center=(0.0, 1.0, 0.0)),
            plane_node(3, area=120.0, aspect=6.0, long_extent=100.0, short_extent=15.0, normal=(0.0, 0.0, 1.0), center=(0.0, 0.0, 6.0)),
            plane_node(10, area=5000.0, aspect=1.0, long_extent=100.0, short_extent=50.0, normal=(1.0, 0.0, 0.0), center=(0.0, 0.0, -1.0)),
        ]
        faces = [
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 3, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 10, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph_data(
            nodes,
            edges=[[1, 3], [2, 3], [1, 10], [2, 10]],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 100.0},
                {"curve_type": "line", "convexity": "convex", "length": 100.0},
                {"curve_type": "line", "convexity": "concave", "length": 100.0},
                {"curve_type": "line", "convexity": "concave", "length": 100.0},
            ],
        )

        candidates = find_review_targeted_rib_candidates(faces, data, set())

        self.assertIn([1, 2, 3], candidates)

    def test_review_targeted_fillet_finds_single_curved_transition_face(self):
        nodes = [
            cylinder_node(4, area=235.0, aspect=12.0, long_extent=50.0, short_extent=4.0),
            plane_node(1, area=2000.0, aspect=1.5, long_extent=80.0, short_extent=50.0),
            plane_node(2, area=1800.0, aspect=1.4, long_extent=80.0, short_extent=45.0),
        ]
        faces = [
            {"idx": 4, "macro_feature": "stock", "semantic_subtype": "stock_face", "macro_confidence": 0.35},
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph_data(
            nodes,
            edges=[[4, 1], [4, 2]],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 50.0},
                {"curve_type": "line", "convexity": "convex", "length": 50.0},
            ],
        )

        candidates = find_review_targeted_fillet_candidates(faces, data, set())

        self.assertIn([4], candidates)

    def test_review_targeted_chamfer_finds_narrow_bevel_plane(self):
        nodes = [
            plane_node(5, area=16.0, aspect=9.0, long_extent=40.0, short_extent=4.0),
            plane_node(1, area=2000.0, aspect=1.5, long_extent=80.0, short_extent=50.0),
            plane_node(2, area=1800.0, aspect=1.4, long_extent=80.0, short_extent=45.0),
        ]
        faces = [
            {"idx": 5, "macro_feature": "stock", "semantic_subtype": "stock_face", "macro_confidence": 0.30},
            {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            {"idx": 2, "macro_feature": "stock", "semantic_subtype": "stock_face"},
        ]
        data = graph_data(
            nodes,
            edges=[[5, 1], [5, 2]],
            edge_attributes=[
                {"curve_type": "line", "convexity": "convex", "length": 40.0, "dihedral_angle": 0.785},
                {"curve_type": "line", "convexity": "convex", "length": 40.0, "dihedral_angle": 0.785},
            ],
        )

        candidates = find_review_targeted_chamfer_candidates(faces, data, set())

        self.assertIn([5], candidates)


if __name__ == "__main__":
    unittest.main()
