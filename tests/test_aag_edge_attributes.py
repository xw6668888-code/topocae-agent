import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

from core.AAGExtractor import extract_aag_graph


class AAGEdgeAttributesTest(unittest.TestCase):
    def test_box_edges_include_aligned_brep_edge_attributes(self):
        shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()

        graph = extract_aag_graph(shape, include_label_fields=False)

        self.assertIn("edges", graph)
        self.assertIn("edge_attributes", graph)
        self.assertEqual(len(graph["edges"]), len(graph["edge_attributes"]))
        self.assertGreater(len(graph["edge_attributes"]), 0)

        first_attr = graph["edge_attributes"][0]
        required_keys = {
            "curve_type",
            "curve_type_id",
            "length",
            "is_line",
            "is_circle",
            "circle_radius",
            "convexity",
            "convexity_id",
            "dihedral_angle",
            "adjacent_face_normal_angle",
            "source_face",
            "target_face",
        }
        self.assertTrue(required_keys.issubset(first_attr.keys()))

        lengths = [attr["length"] for attr in graph["edge_attributes"]]
        self.assertTrue(any(abs(length - 10.0) < 1e-5 for length in lengths))
        self.assertTrue(any(abs(length - 20.0) < 1e-5 for length in lengths))
        self.assertTrue(any(abs(length - 30.0) < 1e-5 for length in lengths))
        self.assertTrue(all(attr["curve_type"] == "line" for attr in graph["edge_attributes"]))
        self.assertTrue(all(attr["is_line"] == 1.0 for attr in graph["edge_attributes"]))
        self.assertTrue(all(attr["length"] > 0.0 for attr in graph["edge_attributes"]))

    def test_blind_rectangular_pocket_edges_include_concavity(self):
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCC.Core.gp import gp_Pnt

        stock = BRepPrimAPI_MakeBox(20.0, 20.0, 10.0).Shape()
        cutter = BRepPrimAPI_MakeBox(gp_Pnt(5.0, 5.0, 6.0), 10.0, 10.0, 8.0).Shape()
        shape = BRepAlgoAPI_Cut(stock, cutter).Shape()

        graph = extract_aag_graph(shape, include_label_fields=False)

        self.assertTrue(
            any(edge_attr["convexity"] == "concave" for edge_attr in graph["edge_attributes"]),
            "A blind rectangular pocket should expose at least one concave internal edge.",
        )


if __name__ == "__main__":
    unittest.main()
